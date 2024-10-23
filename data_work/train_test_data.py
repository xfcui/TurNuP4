import os
import pickle
import hashlib
import argparse
import requests
import sys
import numpy as np
import torch
from rdkit import Chem
from zeep import Client
from os.path import join
from loading import *
from rdkit.Chem import Crippen, AllChem
from rdkit.Chem import Descriptors
from urllib.request import urlopen, Request
from bioservices import *
import warnings
from rdkit.Chem import Draw
import pandas as pd
import random
from os.path import join
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from drfp import DrfpEncoder
CURRENT_DIR = os.getcwd()




def split_dataframe_enzyme(frac, df):
    df1 = pd.DataFrame(columns = list(df.columns))
    df2 = pd.DataFrame(columns = list(df.columns))
    
    #n_training_samples = int(cutoff * len(df))
    
    df.reset_index(inplace = True, drop = True)
    
    #frac = int(1/(1- cutoff))
    
    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) +len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            # 根据给定的比例将索引分配到训练集或测试集
            if ind % frac != 0:
                # 添加到训练集索引列表
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))
                 # 扩展训练集索引，确保包含相同序列的所有行
                while n_old != len(train_indices):
                    n_old = len(train_indices)
                    # 获取训练集中所有序列的唯一值
                    training_seqs= list(set(df["Sequence"].loc[train_indices]))
                    # 将包含这些序列的所有行的索引添加到训练集索引中
                    train_indices = train_indices + (list(df.loc[df["Sequence"].isin(training_seqs)].index))
                    train_indices = list(set(train_indices))
                
            else:
                # 添加到测试集索引列表
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices)) 
                # 扩展测试集索引，确保包含相同序列的所有行
                while n_old != len(test_indices):
                    n_old = len(test_indices)
                    # 获取测试集中所有序列的唯一值
                    testing_seqs= list(set(df["Sequence"].loc[test_indices]))
                     # 将包含这些序列的所有行的索引添加到测试集索引中
                    test_indices = test_indices + (list(df.loc[df["Sequence"].isin(testing_seqs)].index))
                    test_indices = list(set(test_indices))
                
        ind +=1
    
    
    df1 = df.loc[train_indices]
    df2 = df.loc[test_indices]
    
    return(df1, df2)



if __name__ == "__main__":
    
    try:
        seed = int(sys.argv[1])
    except:
        seed = 123
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""


    datadir = './kcat/TurNuP4/data/'
   


    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')
    kcat_data = pd.read_pickle(datadir+"kcat_data/finish_propress.pkl")
    #UID, Sequence, kcat, sub_ids, pro_ids, log10_kcat, reaction_max, UID_max, EC_max,kcat_max

    
    
    #创建训练集和测试集
    all_data = kcat_data.copy()
    all_data = all_data.sample(frac = 1, random_state = seed)
    all_data.reset_index(drop= True, inplace = True)
    train_df, test_df = split_dataframe_enzyme(frac = 5, df = all_data.copy())
    print('seed = ',seed)
    print("Test set size: %s" % len(test_df))
    print("Training set size: %s" % len(train_df))
    print("Size of test set in percent: %s" % np.round(100*len(test_df)/ (len(test_df) + len(train_df))))


    train_df.reset_index(inplace = True, drop = True)
    test_df.reset_index(inplace = True, drop = True)
    
    train_df.to_pickle(datadir+"kcat_data/splits/train_kcat.pkl")
    test_df.to_pickle(datadir+"kcat_data/splits/test_kcat.pkl")
    
    
    data_train2 = train_df.copy()
    data_train2["index"] = list(data_train2.index)

    data_train2, df_fold = split_dataframe_enzyme(df = data_train2, frac=5)
    indices_fold1 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold1))#

    data_train2, df_fold = split_dataframe_enzyme(df = data_train2, frac=4)
    indices_fold2 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold2))

    data_train2, df_fold = split_dataframe_enzyme(df = data_train2, frac=3)
    indices_fold3 = list(df_fold["index"])
    print(len(data_train2), len(indices_fold3))

    data_train2, df_fold = split_dataframe_enzyme(df = data_train2, frac=2)
    indices_fold4 = list(df_fold["index"])
    indices_fold5 = list(data_train2["index"])
    print(len(data_train2), len(indices_fold4))


    fold_indices = [indices_fold1, indices_fold2, indices_fold3, indices_fold4, indices_fold5]


    
    for i in range(5):
        train = []
        for j in range(5):
            if i != j:
                train = train + fold_indices[j]
        np.save(datadir+"kcat_data/splits/train_without_valid"+str(i),train)
        np.save(datadir+"kcat_data/splits/valid"+str(i), fold_indices[i])

    print('train/test data prepared')

















