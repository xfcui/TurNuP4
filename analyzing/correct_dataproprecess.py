import os
import pickle
import hashlib
import argparse
import requests
import numpy as np
import torch
from rdkit import Chem
from zeep import Client
from os.path import join
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
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb
CURRENT_DIR = os.getcwd()

def compare_floats(a, b, tolerance=0.0000001):
    return abs(a - b) < tolerance

def save_list_to_binary_file(lst, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lst, f)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./kcat/data/', help='data文件夹位置')#'../data/'
    opt = parser.parse_args()


    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')
    kcat_data_me = pd.read_pickle(opt.datadir+"kcat_data/finish_propress.pkl")
    print('复现数据量： ',len(kcat_data_me))
    #UID, Sequence,ESM1b,ESM1b_ts, kcat, sub_ids, pro_ids, log10_kcat, reaction_max, UID_max, EC_max,kcat_max
    
    kcat_data = pd.read_pickle(opt.datadir+"kcat_data/final_kcat_dataset.pkl")
    print('源代码数据量： ',len(kcat_data))
    #Uniprot IDs  Sequence  substrates products

    
    Seq_set = set(kcat_data['Sequence'])
    Seq_me_set = set(kcat_data_me['Sequence'])
    print("两边的序列相同： ",Seq_set==Seq_me_set)
    
    diff_1 = Seq_set.difference(Seq_me_set)
    print("在源代码不在复现里面的Seq： ",len(diff_1))
    
    diff_2 = Seq_me_set.difference(Seq_set)
    print("在复现不在源代码里面的Seq： ",len(diff_2))
    

    kcat_data_more = kcat_data[kcat_data.apply(lambda row: row['Sequence'] in diff_1,axis = 1)]
    kcat_data_me_more = kcat_data_me[kcat_data_me.apply(lambda row: row['Sequence'] in diff_2,axis = 1)]
    
    pass
    
    kcat_data['substrates_set'] = kcat_data['substrates'].apply(lambda x: tuple(sorted(x)))
    kcat_data['products_set'] = kcat_data['products'].apply(lambda x: tuple(sorted(x)))

    # 去重并统计不同反应的数量
    unique_reactions = kcat_data.drop_duplicates(subset=['substrates_set', 'products_set'])

    # 输出结果
    print("Number of unique reactions:", len(unique_reactions))
    
    kcat_data_me['substrates_set'] = kcat_data_me['sub_ids'].apply(lambda x: tuple(sorted(x)))
    kcat_data_me['products_set'] = kcat_data_me['pro_ids'].apply(lambda x: tuple(sorted(x)))

    # 去重并统计不同反应的数量
    unique_reactions = kcat_data_me.drop_duplicates(subset=['substrates_set', 'products_set'])

    # 输出结果
    print("Number of unique reactions:", len(unique_reactions))
    
    
    
    
    
    
    
    #kcat_data_me_more.to_pickle(opt.datadir+"kcat_data/kcat_data_me.pkl")
    
    #排序先
    
    # kcat_data = kcat_data.sort_values(by='Sequence')

    # kcat_data_me = kcat_data_me.sort_values(by='Sequence')
    
    
    pass

    # kcat_data = pd.read_pickle(opt.datadir+"kcat_data/max_max_max.pkl")
    
    # kcat_data_me = pd.read_pickle(opt.datadir+"kcat_data/preprocessing_kcat_data.pkl")
    
    print('复现数据量： ',len(kcat_data_me))

    print('源代码数据量： ',len(kcat_data))
    
    error_kcat = 0
    error_U = 0
    error_R = 0
    error_EC = 0
    error_DRFP = 0
    right = 0
    index_same_me = []
    index_same_he = []
    index_unsame_he = []
    question_R = []
    for i in kcat_data.index:
        seq_i = kcat_data['Sequence'][i]
        sub_ids_i = set(kcat_data["substrates"][i])
        pro_ids_i = set(kcat_data["products"][i])
        same_data = kcat_data_me[kcat_data_me.apply(lambda row: row["Sequence"]==seq_i and set(row["sub_ids"])==sub_ids_i and set(row["pro_ids"])==pro_ids_i,axis=1)]

        if len(same_data)==0:
            index_unsame_he.append(i)
            pass
        else:
            correct = True
            index = list(same_data.index)[0]
            if not compare_floats(kcat_data['geomean_kcat'][i],same_data["log10_kcat"][index]):
                error_kcat += 1
                correct = False
                pass
            
            
            if kcat_data['max_kcat_for_UID'][i] != same_data["UID_max"][index]:
                error_U += 1
                #correct = False
                pass
            
            if kcat_data['max_kcat_for_RID'][i] != same_data["reaction_max"][index]:
                if(kcat_data['substrates'][i]==''):
                    pass
                else:
                    error_R += 1
                    if sub_ids_i not in question_R:
                        question_R.append(sub_ids_i)
                    #correct = False
                    pass
            
            if pd.isnull(kcat_data['max_kcat_for_EC'][i]) and pd.isnull(same_data["EC_max"][index]):
                pass
            elif kcat_data['max_kcat_for_EC'][i] != same_data["EC_max"][index]:
                error_EC += 1
                #correct = False
                pass
            if correct:
                index_same_he.append(i)
                index_same_me.append(index)
                right += 1
            else:
                index_unsame_he.append(i)
                pass
    save_list_to_binary_file(index_same_me,opt.datadir+'kcat_data/index_same_me.pkl')
    save_list_to_binary_file(index_same_he,opt.datadir+'kcat_data/index_same_he.pkl')
    
    only_me = kcat_data_me[~kcat_data_me.index.isin(index_same_me)]
    
    
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    