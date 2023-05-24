import argparse
import pickle
import time
import pandas as pd
import numpy as np
from utils import Data
from utils import *
from model_dipsgnn import * 
from scipy import optimize


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-100k', help='dataset name: ml-100k/ml-1m/yelp/tmall')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size of item/node embedding in graph')
parser.add_argument('--userhiddenSize', type=int, default=50, help='hidden state size of user embedding')
parser.add_argument('--usersideinfo', action='store_false', help='default True, use user side information, if False just use onehot of userid')

parser.add_argument('--epsilon1',type=float,default=20.0,help='privacy budget for protecting user features')
parser.add_argument('--epsilon',type=float,default=5.0,help='privacy budget for protecting edges in every epoch')
parser.add_argument('--delta',type=float,default=0.00000025,help='relaxation of every epoch update, default value is for ml-1m')
parser.add_argument('--clip_norm', type=float, default=0.5, help='default clip_norm for clipping')

parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--logfile',type=str,default="./log/dipsgnn.log",help='path to store the log file')
opt = parser.parse_args()


#Setting the logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s- %(filename)s[line:%(lineno)d]- %(message)s')
# FileHandler
file_handler = logging.FileHandler(f"{opt.logfile}")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# StreamHandler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info('Start running: ----- DRAPGNN ----- with following parameters')
logger.info(opt)


def cal_sigma(x,T,C,epsilon,delta):
    '''
    given:
    T: number of propogation steps
    C: clip norm
    epsilon: privacy budget of one epoch
    delta: relaxation parameter of guassian dp

    x: the standard deviation of guassian noise we want to get
    '''
    return T*C**2/(2*x**2)+C*np.sqrt(2*T*np.log(1/delta))/x - epsilon 


if opt.epsilon1==20:
    suffix1='/train.txt'
    suffix2='/test.txt'
elif opt.epsilon1==10:
    suffix1='/train_fe10.0.txt'
    suffix2='/test_fe10.0.txt'
else:
    suffix1='/train_fe30.0.txt'
    suffix2='/test_fe30.0.txt'

logger.info(f'Training dataset: {opt.dataset + suffix1}')


def main():    
    train_data = pickle.load(open('./datasets/' + opt.dataset + suffix1, 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/' + opt.dataset + suffix2, 'rb'))

    train_data = Data(train_data, shuffle=True) #shuffle=True for train_data
    test_data = Data(test_data, shuffle=False)  #shuffle=False for test_data

    #n_node: num of unique items after filtering and preprocessing in original dataset
    if opt.dataset == 'sample':
        n_node = 171  
        n_user = 88  
        user_featureSize = 29
    elif opt.dataset == 'ml-1m':
        n_node = 2811
        n_user = 5950
        user_featureSize = 30
    elif opt.dataset == 'ml-100k':
        n_node = 1153
        n_user = 944
        user_featureSize = 84
    elif opt.dataset=='tmall':   #tmall_nov1to7
        n_node = 98227
        n_user = 132725
        user_featureSize = 12
    else: #yelp
        n_node = 56429
        n_user = 99012
        user_featureSize = 6

    # initialize the gnn model
    model = trans_to_cuda(SequenceGraph(opt, n_node, n_user, user_featureSize))

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        logger.info('-------------------------------------------------------')
        logger.info('epoch: %s' % epoch)   
        clip_norm = opt.clip_norm    
        logger.info(f'clip norm = {clip_norm} ') 

        root= optimize.fsolve(cal_sigma,x0=1,args=(opt.step,clip_norm,opt.epsilon,opt.delta))
        sigma = root[0]
        logger.info(f'sigma in drapgnn = {sigma}')

        # clip and add noise evey epoch
        hit20, mrr20, hit10, mrr10, hit5, mrr5 = train_test(model, train_data, test_data, clip_norm, sigma)
        logger.info('hit20 %s, mrr20 %s, hit10 %s, mrr10 %s, hit5 %s, mrr5 %s' %(hit20, mrr20, hit10, mrr10, hit5, mrr5))
        flag = 0
        if hit20 >= best_result[0]:
            best_result[0] = hit20
            best_epoch[0] = epoch
            flag = 1
        if mrr20 >= best_result[1]:
            best_result[1] = mrr20
            best_epoch[1] = epoch
            flag = 1
        if hit10 >= best_result[2]:
            best_result[2] = hit10
            best_epoch[2] = epoch
            flag = 1
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch[3] = epoch
            flag = 1
        if hit5 >= best_result[4]:
            best_result[4] = hit5
            best_epoch[4] = epoch
            flag = 1
        if mrr5 >= best_result[5]:
            best_result[5] = mrr20
            best_epoch[5] = epoch
            flag = 1
        logger.info('Best Result:')
        logger.info('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        logger.info('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
        logger.info('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[4], best_result[5], best_epoch[4], best_epoch[5]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    logger.info('-------------------------------------------------------')
    end = time.time()
    logger.info("Run time: %f s\n\n" % (end - start))
    best_result= ['%.2f' % i for i in best_result]
    best_epoch=['%.0f' % i for i in best_epoch]
    output=pd.DataFrame([best_result, best_epoch],columns=['Recall@20','MMR@20','Recall@10','MMR@10','Recall@5','MMR@5'])
    filename= 'results1/'+ opt.logfile[4:-4] +'.csv'
    output.to_csv(filename,index=False)


if __name__ == '__main__':
    main()