import os
import pickle
import hashlib
import argparse
import requests
import esm
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
import gc
from rdkit import Chem
from rdkit.Chem import AllChem
from drfp import DrfpEncoder
CURRENT_DIR = os.getcwd()





if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./kcat/TurNuP4/data/', help='data文件夹位置')#'../data/'
    opt = parser.parse_args()



    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')

    #UID, Sequence, kcat, sub_ids, pro_ids, log10_kcat, reaction_max, UID_max, EC_max,kcat_max

    # model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    # #model.to(device)
    # model.eval()

    seq_data = pd.read_pickle(opt.datadir + "kcat_data/fp_kcat_data.pkl")


    '''rep_esm1b_dict = torch.load(opt.datadir+"enzyme_data/all_sequences_kcat_esm1b.pt")
    rep_esm1b_ts_dict = torch.load(opt.datadir+"enzyme_data/all_sequences_kcat_esm1b_ts.pt")
    
    
    for i in seq_data.index:
        if not pd.isnull(seq_data["Sequence"][i]):        
            seq_data["ESM1b"][i] = rep_esm1b_dict[str(i)+".pt"]     
            seq_data["ESM1b_ts"][i] = rep_esm1b_ts_dict[str(i)+".pt"]'''
            
            
    seq_vec_data = pd.read_pickle(opt.datadir+"enzyme_data/all_sequences_with_IDs_and_ESM1b_ts.pkl")        
    
    seq_data["ESM1b"],seq_data["ESM1b_ts"] = '' , ''
    for i in seq_data.index:
        if seq_data['ESM1b'][i]=='':
            seq_i = seq_data["Sequence"][i]
            same_seq = seq_vec_data.loc[seq_vec_data["Sequence"]==seq_i]
            if len(same_seq)>0:
                esm1b = same_seq["ESM1b"][same_seq.index[0]]
                esm1b_ts = same_seq["ESM1b_ts"][same_seq.index[0]]
                seq_data['ESM1b'][i] = esm1b
                seq_data['ESM1b_ts'][i] = esm1b_ts
        
    
        
    seq_data = seq_data.loc[seq_data["ESM1b"] != ""].loc[seq_data["ESM1b_ts"] != ""]  
    print("匹配完酶向量： ",len(seq_data))
    
    
    
    '''    
    seq_data["ESM2"] = ''
    sequences = list(seq_data['Sequence'])

    # 准备数据
    batch_converter = alphabet.get_batch_converter()
    data = [("sequence_{}".format(i), seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)


    # 获取特征向量
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]

    # 提取每个序列的特征向量
    sequence_representations = []
    for i, seq in enumerate(sequences):
        seq_rep = token_representations[i, 1: len(seq) + 1].mean(0)
        sequence_representations.append(seq_rep)

    seq_data["ESM2"] = sequence_representations
     seq_data = seq_data.loc[seq_data["ESM2"] != ""]
    print("匹配完esm2： ",len(seq_data))
    '''
    
        
    
   
    
    
    
    seq_data.to_pickle(opt.datadir+"kcat_data/finish_propress.pkl")
    














