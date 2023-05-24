import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import logging
from main_dipsgnn import logger

from utils_edgerand import seed
torch.manual_seed(seed)
logger.info(f'random seed for this experiement is: {seed}')


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class GNN(Module):
    def __init__(self, hidden_size,user_hidden_size, usersideinfo, user_embedding, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.user_embedding= user_embedding
        self.user_side_info = usersideinfo
        self.hidden_size = hidden_size
        self.user_hidden_size= user_hidden_size
        self.input_size = hidden_size * 2                                    # for outgoing and income
        self.gate_size = 3 * hidden_size                                     # for three weight matrix in reset/input/new gate
        
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size)) # input_size only used here to define w_ih
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        self.linear_edge_in = nn.Linear(self.hidden_size+self.user_hidden_size, self.hidden_size, bias=True)   # W_in
        self.linear_edge_out = nn.Linear(self.hidden_size+self.user_hidden_size, self.hidden_size, bias=True)  # W_out



    def GNNCell(self, A, hidden, users, user_fe, clip_norm, sigma):
        '''
        A: batch_size * max_n_node * 2max_n_node
        hidden: node embedding, batch_size * max_n_node * hidden_size , if not specified "hidden" means item/node embedding
        user_hidden: batch_size * user_hidden_size
        '''
        sigma=trans_to_cuda(torch.Tensor([sigma]).float())
        if self.user_side_info:
            user_hidden = self.user_embedding(user_fe)  # get user embedding, batch_size * user_hidden_size
        else:
            user_hidden = self.user_embedding(users)    # batch_size * user_hidden_size
        user_hidden= user_hidden.view(hidden.shape[0],1,-1).expand(-1,hidden.shape[1],-1)  # batch_size * max_node * user_hidden_size
        H = torch.cat([hidden,user_hidden],2)                                # batch_size * max_n_node * (hidden_size + user_hidden_size)
         
        H_bar = F.normalize(H,dim=2) * clip_norm                             # make the norm of every row equal to clip_norm                  

        input_in = torch.matmul(A[:, :, :A.shape[1]], H_bar)                 # aggregate, batch_size * max_n_node * (hidden_size + user_hidden_size)
        input_in += sigma ** 2 * trans_to_cuda(torch.randn(input_in.shape))  # perturb with gaussian noise
        input_in = self.linear_edge_in(input_in) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], H_bar) 
        input_out += sigma ** 2 * trans_to_cuda(torch.randn(input_out.shape))
        input_out = self.linear_edge_out(input_out) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)           # batch_size * max_n_node * (2 hidden_size)

        # GRU update
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        hidden= H_bar[:,:,:hidden.shape[2]]                    #important the keep the scale of node_embed and item_embed at the same level?
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        e_it = hidden + inputgate * (newgate - hidden)        # batch_size * max_n_node * hidden_size

        return e_it


    def forward(self, A, hidden, users, user_fe, clip_norm, sigma):
        '''
        Note: the output of last GNNCell is the input of next GNNCell
        '''
        for i in range(self.step):
            hidden= self.GNNCell(A, hidden, users, user_fe, clip_norm, sigma)
        return hidden


# SequenceGraph is contrsuct on the above simple GNN Modules
class SequenceGraph(Module):
    def __init__(self, opt, n_node, n_user, user_featureSize):
        super(SequenceGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.user_hidden_size=opt.userhiddenSize
        self.user_feature_size=user_featureSize
        self.user_side_info= opt.usersideinfo
        self.n_node = n_node
        self.n_user = n_user
        self.batch_size = opt.batchSize
        self.user_embedding=nn.Embedding(self.user_feature_size,self.user_hidden_size)
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)                   # a embedding lookup table is updated during learning, input is lookup index
        if self.user_side_info:
            self.user_embedding = nn.Linear(self.user_feature_size, self.user_hidden_size, bias=False)   # E_U, linear layer to transfer user feature(side information) to user embedding
        else:
            self.user_embedding = nn.Embedding(self.n_user, self.user_hidden_size)     # without user side info, just onehot vector of userid             
        self.gnn = GNN(self.hidden_size, self.user_hidden_size, self.user_side_info, self.user_embedding, step=opt.step)       # opt.step is gnn propogation steps
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)                           
        self.linear_transform = nn.Linear(self.hidden_size * 2+self.user_hidden_size, self.hidden_size, bias=True) # B, from (2d+d_prime)-dim to d-dim
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) #initialize parameters
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def compute_scores(self, hidden, users, user_fe, mask):
        '''
        hidden: aka. seq_hidden, batch_size * seq_len x hidden_size
        user_hidden: seq_user_hidde, batch_size *  user_hidden_size
        mask: batch_size x seq_len (seq_len=max_seq_len, masking every seq with length as max_seq_len)
        '''
        if self.user_side_info:
            user_hidden = self.user_embedding(user_fe)  # batch_size * user_hidden_size
        else:
            user_hidden = self.user_embedding(users)    # batch_size * user_hidden_size
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]    # embedding of last item of every seq (s_l), batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])                 # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)                                               # batch_size x seq_len x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))                          # alpha: batch_size x seq_len x 1, q1+q2: batch_size x seq_len x latent_size
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1) # element_wise product then sum, a: batch_size x latent_size
        a = self.linear_transform(torch.cat([a, ht, user_hidden], 1))
        b = self.embedding.weight[1:]                                              # n_node x latent_size, start from 1, because item_index start from 1
        scores = torch.matmul(a, b.transpose(1, 0))                                # batch_size x n_node
        return scores


    def forward(self, inputs, users, user_fe, A, clip_norm, sigma):
        '''
        inputs: batch_size * max_n_node
        A: batch_size * max_n_node * 2 max_n_node
        '''
        hidden = self.embedding(inputs)                 # intialize node embedding, batch_size * max_n_node * hidden_size
        hidden = self.gnn(A, hidden, users,user_fe, clip_norm, sigma)
        return hidden


def forward(model, i, data, clip_norm, sigma):
    '''
    i: list of index in a batch

    targets: batch_size * 1, list of y/label for each seq in a batch 
    scores: batch_size x n_node
    '''
    alias_inputs, A, items, mask, targets, users, user_fe = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())  # batch_size * len_max
    items = trans_to_cuda(torch.Tensor(items).long())                # batch_size * max_n_node
    A = trans_to_cuda(torch.Tensor(A).float())                       # batch_size * max_n_node * 2 max_n_node
    mask = trans_to_cuda(torch.Tensor(mask).long())                  # batch_size * len_max
    users = trans_to_cuda(torch.Tensor(users).long())                # batch_size * 1
    user_fe = trans_to_cuda(torch.Tensor(user_fe).float())           # batch_size * user_featureSize

    hidden = model(items, users, user_fe, A, clip_norm, sigma)            # hidden: batch_size * max_n_node * hidden_size, user_hidden: batch_size * user_hidden_size
    get = lambda i: hidden[i][alias_inputs[i]] 
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) # batch_size x len_max x hidden_size
    return targets, model.compute_scores(seq_hidden, users, user_fe, mask)


# train test for one epoch
def train_test(model, train_data, test_data, clip_norm, sigma):
    model.scheduler.step()
    logger.info('start training: %s'% datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)  
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data, clip_norm, sigma)
        targets = trans_to_cuda(torch.Tensor(targets).long())      # tensor, batch_size * 1
        loss = model.loss_function(scores, targets - 1)            # in CL loss must use softmax; in targets, item index start from 1, so -1
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            logger.info('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    logger.info('\tTotal Loss:\t%.3f' % total_loss)


    logger.info('start predicting: %s'% datetime.datetime.now())
    model.eval()
    hit20, mrr20, hit10, mrr10, hit5, mrr5=[],[],[],[],[],[]
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        # on testset, we use the private parameters and embeddings learned from train set, so no need to clip and add noise
        targets, scores = forward(model, i, test_data, clip_norm, 0)   
        top20_scores_index = scores.topk(20)[1]    
        top20_scores_index = trans_to_cpu(top20_scores_index).detach().numpy()
        top10_scores_index = scores.topk(10)[1]    
        top10_scores_index = trans_to_cpu(top10_scores_index).detach().numpy()
        top5_scores_index = scores.topk(5)[1]    
        top5_scores_index = trans_to_cpu(top5_scores_index).detach().numpy()

        for index20, index10, index5 ,target in zip(top20_scores_index, top10_scores_index, top5_scores_index, targets):
            hit20.append(np.isin(target - 1, index20))
            hit10.append(np.isin(target - 1, index10))
            hit5.append(np.isin(target - 1, index5))
            if len(np.where(index20 == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(index20 == target - 1)[0][0] + 1))

            if len(np.where(index10 == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(index10 == target - 1)[0][0] + 1))

            if len(np.where(index5 == target - 1)[0]) == 0:
                mrr5.append(0)
            else:
                mrr5.append(1 / (np.where(index5 == target - 1)[0][0] + 1))
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    hit5 = np.mean(hit5) * 100
    mrr5 = np.mean(mrr5) * 100
    return hit20, mrr20, hit10, mrr10, hit5, mrr5