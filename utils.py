import networkx as nx
import numpy as np
import pandas as pd
import random
seed=0
np.random.seed(seed)


def data_masks(all_usr_pois, item_tail):
    '''
    all_usr_pois: nested list, inner list is a seq
    item_tail: list, if seq_len<=len_max, mask it with item_tail to make all seqs same length

    us_pois: nested list, inner list is masked seq 
    us_msks: nested list, inner list is 1-0 list 
    len_max: max length of all seq in all_usr_pois
    '''
    us_lens = [len(upois) for upois in all_usr_pois] #[len of every seq in train_set]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]  # nested list
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] #nested list[[without mask part is 1, with mask part is 0]]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y ,train_set_users,train_user_fe= train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_user=[train_set_users[s] for s in sidx[n_train:]]
    valid_set_user_fe=[train_user_fe[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_user = [train_set_users[s] for s in sidx[:n_train]]
    train_set_user_fe = [train_user_fe[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y,train_set_user,train_set_user_fe), (valid_set_x, valid_set_y,valid_set_user,valid_set_user_fe)



class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]  # get x part from train,(test/valid) 
        inputs, mask, len_max = data_masks(inputs, [0])  #inputs is us_pois, mask is us_msks from data_masks()
        self.inputs = np.asarray(inputs) 
        self.mask = np.asarray(mask)           #transfer nested list to array
        self.len_max = len_max
        self.targets = np.asarray(data[1])     # y/label for each seq in inputs
        self.users = np.asarray(data[2])       # userid for each seq in inputs 
        self.user_fe=np.asarray(data[3])       # user_feature for each seq in inputs
        self.length = len(inputs)              # number of train_seqs with labels
        self.shuffle = shuffle
        self.graph = graph
    
    
    # we should analyze the influence of batch on privacy account
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices


    def get_slice(self, i): #i is a slice(list)
        '''
        input:
        i: batch_size, list of index for a batch from output of generate_batch

        output:
        alias_inputs: batch_size * len_max, each sublist is positions of each item in seq in unique nodes(sorted)
        A: batch_size * max_n_node * 2max_n_node
        items: batch_size * max_n_node, each sublist is unique nodes in a seq with mask 0 to match max_n_node
        mask: batch_size * len_max, mask vector for each seq in a batch
        targets: batch_size * 1, list of y/label for each seq in a batch 
        '''
        inputs, mask, targets, users, user_fe = self.inputs[i], self.mask[i], self.targets[i], self.users[i], self.user_fe[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)  # max unique node num in a seq for all seqs
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0]) 
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1): 
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] += 1 
            u_A_in=u_A
            u_A_out=u_A.transpose()
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) #position of each item in u_input in unique nodes(sorted)
        return alias_inputs, A, items, mask, targets, users, user_fe