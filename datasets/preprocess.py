#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on June, 2022

@author: Wentao Hu
"""
import argparse
from re import U, sub
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd
import numpy as np
import random
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', help='dataset name: ml-1m/yelp/tmall')
parser.add_argument('--epsilon', type=float, default=20, help='privacy budget to perturb user features, 999 means budget is infinity/no perturbation')
opt = parser.parse_args()
print(opt)

dataset = opt.dataset
print("-- Starting @ %ss" % datetime.datetime.now())


if dataset=="ml-1m":
    df=pd.read_csv(f"{dataset}/ratings.dat",sep="::",header=None,engine="python")
    df=df.rename(columns={0:"user",1:"item",2:"rating",3:"time"})
    #in ml-1m, rating no less than 4 is transfered to implicit feedback
    df=df[df['rating']>=4]
elif dataset=="ml-100k":
    df=pd.read_csv(f"{dataset}/u.data",sep="\t",header=None,engine="python")
    df=df.rename(columns={0:"user",1:"item",2:"rating",3:"time"})
    #in ml-100k, rating no less than 1 is transfered to implicit feedback
    df=df[df['rating']>=1]
elif dataset=='tmall': 
    df=pd.read_csv(f"{dataset}/tmall_simplified_nov1to7.csv",engine="python")
    df=df.rename(columns={"user_id":"user","item_id":"item","time_stamp":"time"})
else:
    df=pd.read_csv(f"{dataset}/yelp_review_encoded.csv",engine="python")
    df=df.rename(columns={"user_id":"user","business_id":"item","stars":"rating","date":"time"})
if dataset !='tmall': #tmall dataset preserve origin order
    df=df.sort_values(by=["user","time"])
df.to_csv(f"{dataset}/{dataset}-numerical.dat",sep='\t', index=False, header=False) 


# 10 core filtering of inactive users and items
l1=len(df)
user_grouped_df=df.groupby('user')
df=user_grouped_df.filter(lambda x:x['item'].count()>=10)
l2=len(df)

item_grouped_df=df.groupby('item')
df=item_grouped_df.filter(lambda x:x['user'].count()>=10)
l3=len(df)
print("length in 1 round filtering:", l1,l2,l3)
while l3<l1 or l2<l1:
    l1=len(df)
    user_grouped_df=df.groupby('user')
    df=user_grouped_df.filter(lambda x:x['item'].count()>=10)
    l2=len(df)

    item_grouped_df=df.groupby('item')
    df=item_grouped_df.filter(lambda x:x['user'].count()>=10)
    l3=len(df)

if dataset=='tmall':
    df=df.sort_values(['user','time','rank'])
df.to_csv(f"{dataset}/{dataset}-10core.dat",sep='\t', index=False, header=False) 

print("-- finish 10-core filtering @ %ss" % datetime.datetime.now())
print("length after all rounds of 10-core fitlering:",l1,l2,l3,len(df))
# get average user seq after 10-core
print(f"--average length of {dataset} after 10-core--\n",df.groupby('user').count().iloc[:,0].mean())


if dataset=="ml-1m":
    max_length=100
    df_cutted=df.groupby('user').tail(max_length)
    df_cutted.to_csv(f"{dataset}/{dataset}-cutted.dat",sep='\t', index=False, header=False) 
elif dataset=="ml-100k":
    max_length=120
    df_cutted=df.groupby('user').tail(max_length)
    df_cutted.to_csv(f"{dataset}/{dataset}-cutted.dat",sep='\t', index=False, header=False) 
elif dataset=="tmall":
    max_length=50
    df_cutted=df.groupby('user').tail(max_length)
    df_cutted.to_csv(f"{dataset}/{dataset}-cutted.dat",sep='\t', index=False, header=False) 
else: #yelp
    max_length=30
    df_cutted=df.groupby('user').tail(max_length)
    df_cutted.to_csv(f"{dataset}/{dataset}-cutted.dat",sep='\t', index=False, header=False)
    
print(f"max length of {dataset}: %s"% max_length)
print("-- finish cutting @ %ss" % datetime.datetime.now())
print("n_node(unique items) after cutting: ",len(df_cutted['item'].unique()))
print("n_user(unique users) after cutting: ",len(df_cutted['user'].unique()))


#split train sequence and test sequence for each user
tra_seqs=[]
tes_seqs=[]
user_seqs=[] #userid of each tra_seq and tes_seq
grouped=df_cutted.groupby('user')
train_user=[]
test_user=[]
train_item=[]
test_item=[]
train_rating=[]
test_rating=[]
for user in df_cutted['user'].unique():
    user_seqs.append(user)
    subgroup=grouped.get_group(user)
    tra_length=int(0.8*len(subgroup))
    tes_length=len(subgroup)-tra_length
    tra_seq=list(subgroup['item'].head(tra_length))
    tes_seq=list(subgroup['item'].tail(tes_length))
    tra_seqs.append(tra_seq)  #nested list
    tes_seqs.append(tes_seq)
    # for generate train_df/test_df
    train_item+=tra_seq    #a very long list
    test_item+=tes_seq
    train_user+=[user]*tra_length
    test_user+=[user]*tes_length
    if dataset!='tmall':
        train_rating+=list(subgroup['rating'].head(tra_length))
        test_rating+=list(subgroup['rating'].tail(tes_length))

if dataset !='tmall':
    train_df=pd.DataFrame({'user':train_user,'item':train_item,'rating':train_rating})
    test_df=pd.DataFrame({'user':test_user,'item':test_item,'rating':test_rating})
else:
    train_df=pd.DataFrame({'user':train_user,'item':train_item})
    test_df=pd.DataFrame({'user':test_user,'item':test_item,})
train_df.to_csv(f'{dataset}/train_df.csv',index=False)
test_df.to_csv(f'{dataset}/test_df.csv',index=False)

print('\n----number of training instances',len(train_df))
print('\n----number of test instances',len(test_df))

print("\nfirst 2 elements of tra_seqs before reindexing",tra_seqs[:2])
print("\nfirst 2 elements of tes_seqs before reindexing",tes_seqs[:2])
print("\nlast 10 elements of user_seqs before reindexing",user_seqs[-10:])


# renumber users to start from 1, as some users are filtered out in 10-core
user_dict={}
def obtain_user():
    out_index=[]
    user_index=1
    for user in user_seqs:
        if user in user_dict:
            out_index+=[user_dict[user]]
        else:
            out_index+=[user_index]
            user_dict[user]=user_index
            user_index+=1
    return out_index, user_dict
encoded_user_seqs, user_dict=obtain_user()


# renumber items to start from 1 and convert training seqs
item_dict={}  #{item_id:renumbered index}
def obtain_tra():
    train_seqs=[]
    item_ctr=1
    for s in tra_seqs:
        outseq=[]
        for i in s:
            if i in item_dict:
                outseq+=[item_dict[i]]
            else:
                outseq+=[item_ctr]
                item_dict[i]=item_ctr
                item_ctr+=1
        train_seqs+=[outseq]
    return train_seqs
train_seqs=obtain_tra()


# Convert test seq with item_dict, ignoring items that do not appear in training set
def obtain_tes():
    test_seqs=[]
    for s in tes_seqs:
        outseq=[]
        for i in s:
            if i in item_dict:
                outseq+=[item_dict[i]]
        test_seqs+=[outseq]
    return test_seqs
test_seqs=obtain_tes()

print("\nfirst 2 of train_seqs after reindexing", train_seqs[:2])
print("\nfirst 2 of test_seqs after reindexing", test_seqs[:2])
print("\nlast 10 elements of user_seqs after reindexing", encoded_user_seqs[-10:])


#process a train_seq/test_seq into many subseqs with labels for training
def process_seqs(iseqs):
    userids=[]
    orig_userids=[]
    out_seqs=[]
    labs=[]
    for i in range(len(iseqs)):
        seq=iseqs[i]
        user=encoded_user_seqs[i]
        orig_user=user_seqs[i]
        for j in range(1,len(seq)):
            tar=seq[-j]
            labs+=[tar]
            out_seqs+=[seq[:-j]]
            userids+=[user]
            orig_userids+=[orig_user]
    return out_seqs,labs,userids, orig_userids

tr_seqs,tr_labs,tr_users, orig_tr_users=process_seqs(train_seqs)
te_seqs,te_labs,te_users, orig_te_users=process_seqs(test_seqs)
print("-- finish process seqs to get labels @ %ss" % datetime.datetime.now())
print("number of tr_seqs with labels",len(tr_seqs),len(tr_labs),len(tr_users),len(orig_tr_users))
print("number of te_seqs with labels",len(te_seqs),len(te_labs),len(te_users),len(orig_te_users))


# Perturb numerical user features
def perturb_numerical(df,colname,epsilon):
    '''
    df: DataFrame containing the target column 
    colname: string , name of target numerical colnums in perturbation
    epsilon:perturbation privacy budget
    '''
    min_num=np.min(df[colname])
    max_num=np.max(df[colname])
    f=lambda x:(x-(min_num+max_num)/2)/((max_num-min_num)/2) 
    df[colname]=df[colname].apply(f) #make values of the series in [-1,1]
    ksi=np.random.sample((len(df),))
    threshold=np.exp(epsilon/2)/(np.exp(epsilon/2)+1)
    C=(np.exp(epsilon/2)+1)/(np.exp(epsilon/2)-1)

    tmpname="p"+colname  #name of newly added perturbed values
    df[tmpname]=np.zeros((len(df),1))
    for i in range(len(df)):
        l=(C+1)/2*df.iloc[i, df.columns.get_loc(colname)]-(C-1)/2
        pi=l+C-1
        intervals=[[-C,l],[pi,C]]
        if ksi[i]<threshold:
            df.iloc[i, df.columns.get_loc(tmpname)]=random.uniform(l,pi)
        else:
            df.iloc[i, df.columns.get_loc(tmpname)]=random.uniform(*random.choices(intervals,weights=[r[1]-r[0] for r in intervals])[0])
    return df


# Perturb categorical user features
def perturb_onehot(c,budget):
    '''
    c: 0 or 1 in a onehot vector
    budget: privacy budget
    '''
    c_prime=0
    if c==1:
        c_prime=np.random.choice([1,0],1)[0]
    else:
        pb=1/(np.exp(budget)+1)
        c_prime=np.random.choice([1,0],p=[pb,1-pb])
    return c_prime


def perturb_categorical(df,colname,epsilon):
    '''
    df: DataFrame containing the target column 
    colnames:string, name of target categorical colnum for perturbation
    epsilon:perturbation privacy budget  
    '''
    onehot_names=list(filter(lambda x:x.startswith(colname),df.columns))[1:]
    onehot_index=[df.columns.get_loc(colname) for colname in onehot_names]
    for i in range(len(df)):
        c_vector=df.iloc[i,onehot_index]
        df.iloc[i,onehot_index]=c_vector.apply(perturb_onehot,budget=epsilon)
    return df


def encode_df(df,col_types):
    for i in range(1,len(df.columns)):
        colname=df.columns[i]
        if col_types[i]==1: # 1 for numerical column, we should normalize numerical feature into [-1,1]
            min_num=np.min(df[colname])
            max_num=np.max(df[colname])
            f=lambda x:(x-(min_num+max_num)/2)/((max_num-min_num)/2) 
            p_colname="p"+colname
            df[p_colname]=df[colname].apply(f)    
        else:
            onehot_cols=pd.get_dummies(df.iloc[:,df.columns.get_loc(colname)],prefix=colname)
            df=pd.concat([df,onehot_cols],axis=1)
    return df


def perturb_feature(df,col_types,epsilon):
    '''
    df: DataFrame containing the target numerical or categorical columns to perturb
    col_types: list containing datatypes of columns, 1 for numerical columns, 0 for object columns
    epsilon: perturbation privacy budget
    '''
    d_prime=len(df.columns)-1 # num of features, -1 for user_index column
    df= encode_df(df,col_types)  
    zeta=int(np.max([1,np.min([d_prime,np.floor(epsilon/2.5)])]))
    cols_select=np.random.choice(np.arange(1,d_prime+1),zeta,replace=False)
    cols_select.sort()
    for col_pos in cols_select:
        col_type=col_types[col_pos]
        colname=df.columns[col_pos]
        if col_type==1:  # 1 means numerical column
            df=perturb_numerical(df,colname,epsilon/zeta)
            p_colname="p"+colname
            df[p_colname]=df[p_colname]*d_prime/zeta
        else:
            df=perturb_categorical(df,colname,epsilon/zeta)   
    return df


epsilon= opt.epsilon
if dataset == "ml-1m":
    feature_num=3
    user_df=pd.read_csv(f"{dataset}/users.dat",delimiter="::",header=None, usecols=[0,1,2,3])
    user_df=user_df.rename(columns={0:"user",1:"gender",2:"age",3:"occupation"})
    user_df=user_df[user_df['user'].isin(user_seqs)]
    if epsilon==999:
        user_df=encode_df(user_df,col_types=[0,0,0,0])
    else:
        user_df=perturb_feature(user_df,col_types=[0,0,0,0],epsilon=epsilon)
elif dataset == "ml-100k":
    feature_num=3
    user_df=pd.read_csv(f"{dataset}/u.user",delimiter="|",header=None, usecols=[0,1,2,3],engine='python')
    user_df=user_df.rename(columns={0:"user",1:"gender",2:"age",3:"occupation"})
    user_df=user_df[user_df['user'].isin(user_seqs)]
    if epsilon==999:
        user_df=encode_df(user_df,col_types=[0,0,0,0])
    else:
        user_df=perturb_feature(user_df,col_types=[0,0,0,0],epsilon=epsilon)
elif dataset=="tmall":
    feature_num=2
    user_df=pd.read_csv(f"{dataset}/user_info_format1.csv")
    user_df=user_df[user_df['user'].isin(user_seqs)]
    if epsilon==999:
        user_df=encode_df(user_df,col_types=[0,0,0])
    else:
        user_df=perturb_feature(user_df,col_types=[0,0,0],epsilon=epsilon) 
else:
    feature_num=6
    user_df=pd.read_csv(f"{dataset}/yelp_user_encoded.csv")
    user_df=user_df[user_df['user'].isin(user_seqs)]
    if epsilon==999:
        user_df=encode_df(user_df,col_types=[0,1,1,1,1,1,1])
    else:
        user_df=perturb_feature(user_df,col_types=[0,1,1,1,1,1,1],epsilon=epsilon)


print("-- finish perturbing features @ %ss" % datetime.datetime.now())
print("-- length of user_df after perturbing: %s" % len(user_df))


userslist=list(user_df['user'])
print('num of users in user_df',len(userslist))
tr_user_fe=[]
te_user_fe=[]
tr_length=len(orig_tr_users)
te_length=len(orig_te_users)
print("tr_length,te_length",tr_length,te_length)
for i in range(tr_length):
    if i%int(tr_length/20)==0:
        print("now processing tr_users",i)
    if i>=1 and orig_tr_users[i]==orig_tr_users[i-1]:
        tr_user_fe.append(feature)
    else:
        user=orig_tr_users[i]
        ind=userslist.index(user)
        feature = list(user_df.iloc[ind,:][feature_num+1:])
        tr_user_fe.append(feature)
   
for i in range(te_length):
    if i%int(te_length/20)==0:
        print("now processing te_users",i)
    if i>=1 and orig_te_users[i]==orig_te_users[i-1]:
        te_user_fe.append(feature)
    else:
        user=orig_te_users[i]
        ind=userslist.index(user)
        feature = list(user_df.iloc[ind,:][feature_num+1:])
        te_user_fe.append(feature)


print("-- finish getting features for each subseq with label @ %ss" % datetime.datetime.now())
print(len(tr_seqs),len(tr_labs),len(tr_users),len(tr_user_fe))
print("\nfirst 2 train_seqs with labels:\n",tr_seqs[:2],tr_labs[:2],tr_users[:2],tr_user_fe[:2])
print("\nfirst 2 test_seqs with labels:\n",te_seqs[:2],te_labs[:2],te_users[:2],te_user_fe[:2])
print("\nuser feature size:\n",len(te_user_fe[0]))


#store preprocessed data with pickle
tra=(tr_seqs,tr_labs,tr_users,tr_user_fe)
tes=(te_seqs,te_labs,te_users,te_user_fe)


if epsilon==999:
    pickle.dump(tra, open(f'{dataset}/train_fe_nonp.txt', 'wb'))
    pickle.dump(tes, open(f'{dataset}/test_fe_nonp.txt', 'wb'))
    pickle.dump(tra_seqs, open(f'{dataset}/all_train_seq_fe_nonp.txt', 'wb'))
else:
    pickle.dump(tra, open(f'{dataset}/train_fe{epsilon}.txt', 'wb'))
    pickle.dump(tes, open(f'{dataset}/test_fe{epsilon}.txt', 'wb'))
    pickle.dump(tra_seqs, open(f'{dataset}/all_train_seq_fe{epsilon}.txt', 'wb'))

print('Done.')