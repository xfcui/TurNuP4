
import os
import pickle
import hashlib
import argparse
import requests
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from zeep import Client
from os.path import join
from loading import *
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from urllib.request import urlopen, Request
from bioservices import *
import warnings


def Count_mw(metabolites):
    mw = 0
    for met in metabolites:
        if met != "":
            mol = Chem.inchi.MolFromInchi(met)
            mw = mw + Descriptors.MolWt(mol)
        
    return(mw)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./kcat/TurNuP4/data/', help='data文件夹位置')#../../data/
    parser.add_argument('--ifdatapreprocess', type=bool, default=True, help='是否需要重新加载数据')
    
    
    opt = parser.parse_args()


    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')

    if opt.ifdatapreprocess:
    
        #loading
        data_Brenda = load_BRENDA(opt.datadir+'kcat_data/BRENDA_kcat.pkl')
        data_Sabio = load_Sabio(opt.datadir+"kcat_data/Sabio_kcat.pkl")
        data_UniProt = load_UniProt(opt.datadir+"kcat_data/Uniprot_kcat.pkl")
        data_kcat = pd.concat([pd.concat([data_Sabio, data_Brenda], ignore_index=True), data_UniProt], ignore_index=True)

        
        #merging
        
        data_kcat.rename(columns={"Uniprot ID":"UID"}, inplace=True)

        data_kcat['kcat'] = pd.to_numeric(data_kcat['kcat'],errors='coerce')
        data_kcat['kcat'] = data_kcat['kcat'].astype(object)

        all_data = data_kcat.copy()
        data_kcat = data_kcat.loc[~pd.isnull(data_kcat["kcat"])].loc[~pd.isnull(data_kcat["pro_ids"])].loc[~pd.isnull(data_kcat["sub_ids"])]
        
        # for i in data_kcat.index:  #过滤空项
        #     if isinstance(data_kcat["sub_ids"][i],list):
        #         data_kcat["sub_ids"][i] = list(filter(None, data_kcat["sub_ids"][i]))

        #     if isinstance(data_kcat["pro_ids"][i],list):
        #         data_kcat["pro_ids"][i] = list(filter(None, data_kcat["pro_ids"][i]))


        
        print("merge后数据量: %s" % len(data_kcat))
        #UID , kcat , sub_ids , pro_ids , Substrates	Products
        

        #把反应信息修正一下
        drop = []
        for i in data_kcat.index:
            #把生成物反应物中中 nan [] [''] '' [nan]去掉   
            if isinstance(data_kcat['sub_ids'][i],list):
                temp = []
                for s in data_kcat['sub_ids'][i]:
                    if isinstance(s,str) and len(s)>0 :
                        temp.append(s)
                    else: 
                        drop.append(i)
                if len(temp)>0:
                    data_kcat['sub_ids'][i] = temp
                else:
                    drop.append(i)
            else:
                drop.append(i)

                
            if isinstance(data_kcat['pro_ids'][i],list):
                temp = []
                for s in data_kcat['pro_ids'][i]:
                    if isinstance(s,str) and len(s)>0 :
                        temp.append(s)
                    else: 
                        drop.append(i)

                if len(temp)>0:
                    data_kcat['pro_ids'][i] = temp

                else:
                    drop.append(i)
            else:
                drop.append(i)
        data_kcat.drop(list(set(drop)),inplace = True)
        
        
        
        #删除重复项  将酶标识符和kcat相等的值拿下  ？按说应该酶标识符 kcat 【反应信息】 反应信息太乱
        drop = []
        for i in data_kcat.index:
            UID_i, kcat_i = data_kcat["UID"][i], data_kcat["kcat"][i]
            same_data = data_kcat.loc[data_kcat["UID"] == UID_i].loc[data_kcat["kcat"] == kcat_i]        
            if len(same_data) > 1:
                drop = drop + list(same_data.index)[1:]
        data_kcat.drop(list(set(drop)), inplace=True)
        print("删除重复后数据量： %s" % len(data_kcat))

        #将酶的UID匹配出氨基酸序列
        UID_Seq_data = pd.read_csv(opt.datadir+"enzyme_data/UNIPROT_results.tab", sep="\t")
        UID_Seq_data.drop(columns=["Entry"], inplace=True)
        UID_Seq_data.rename(columns={"Uniprot ID":"UID"}, inplace=True)
        data_kcat = data_kcat.merge(UID_Seq_data, how="left", on="UID")
        data_kcat = data_kcat.loc[~pd.isnull(data_kcat["UID"])]
        data_kcat = data_kcat.loc[~pd.isnull(data_kcat["Sequence"])]
        print("匹配完氨基酸序列的: %s" % len(data_kcat["Sequence"]))

        #添加EC号  添加个头
        # EC_data = pd.read_csv(opt.datadir+"enzyme_data/Uniprot_results_EC.tab", sep = "\t")
        # EC_data.drop(columns=["Entry"], inplace=True)
        # EC_data.rename(columns={"Uniprot ID":"UID","EC number":"EC"}, inplace=True)
        # data_kcat = data_kcat.merge(EC_data, how="left", on="UID")

        
        
        #把反应前后全部匹配为InChI字符串
        CDids = []
        CD_data = data_kcat[data_kcat.apply(lambda row: any('=' not in s for s in row['sub_ids']+row['pro_ids']), axis=1)]
        for i in CD_data.index:
            if data_kcat["sub_ids"][i]!='I':
                CDids = CDids + data_kcat["sub_ids"][i]
            if data_kcat["pro_ids"][i] != 'I':
                CDids = CDids + data_kcat["pro_ids"][i]

        sCDid = set(CDids)
        CD_InChI_match = {}

        
        #KEGG_conn = KEGG()
        #ChEBI_conn = ChEBI()
        
        CD_InChI_dic = {}
        with open(opt.datadir+"reaction_data/CD_InChI_dict",'rb') as file:
            CD_InChI_dic = pickle.load(file)  
        
        
        for CDid in CDids:
            
            '''
            CD_InChI_match[CDid] = np.nan
            
            try:#   从数据库找
                KEGG_entry = KEGG_conn.parse(KEGG_conn.get(CDid))
                ChEBI_entry = ChEBI_conn.getCompleteEntity('CHEBI:' + KEGG_entry['DBLINKS']['ChEBI'])
                CD_InChI_match[CDid] = ChEBI_entry.inchi
            except:
                pass'''
            
            CD_InChI_match[CDid] = CD_InChI_dic.get(CDid,np.nan)
                
            if pd.isnull(CD_InChI_match[CDid]):
                try:#   从文件找
                    if(CDid[0]=='D'):
                        mol = Chem.MolFromMolFile(join(opt.datadir,"metabolite_data",
                                                "mol-files","mol-files_2", CDid + '.mol'))
                    elif(int(CDid[1:])<17382):
                        mol = Chem.MolFromMolFile(join(opt.datadir,"metabolite_data",
                                                "mol-files","mol-files_1", CDid + '.mol'))
                    else:
                        mol = Chem.MolFromMolFile(join(opt.datadir, "metabolite_data",
                                                "mol-files","mol-files_2", CDid + '.mol'))
                    
                    CD_InChI_match[CDid] = Chem.MolToInchi(mol)
                    

                except:
                    pass

            '''if pd.isnull(CD_InChI_match[CDid]):
                try:
                    mol = Chem.MolFromMolFile(opt.datadir+"metabolite_data/mol-files/"+CDid + '.mol')
                    CD_InChI_match[CDid] = Chem.MolToInchi(mol)
                except:
                    pass'''
                    

        drop = []
        for i in CD_data.index:
            InChI_subs = []
            for sub in data_kcat["sub_ids"][i]:
                if len(sub)>0 and sub[0]!='I':
                    inchi = CD_InChI_match[sub]
                    if pd.isnull(inchi) or inchi == '':
                        drop.append(i)
                    InChI_subs.append(inchi)
                else:
                    InChI_subs.append(sub)
            data_kcat["sub_ids"][i] = set(InChI_subs)
            InChI_pros = []
            for pro in data_kcat["pro_ids"][i]:
                if len(pro)>0 and pro[0]!='I':
                    inchi = CD_InChI_match[pro]
                    if pd.isnull(inchi) or inchi == '':
                        drop.append(i)
                    InChI_pros.append(inchi)
                else:
                    InChI_pros.append(pro)
            data_kcat["pro_ids"][i] = set(InChI_pros)

        data_kcat.drop(list(set(drop)),inplace = True)


        drop = []
        #把同反应，同seq的整合一下        
        for i in data_kcat.index:
            if data_kcat["kcat"][i]==-1:
                continue
            sub_ids_i = set(data_kcat["sub_ids"][i])
            pro_ids_i = set(data_kcat["pro_ids"][i])
            
            #UID_i = data_kcat["UID"][i]
            #same_data = data_kcat[data_kcat.apply(lambda row: row["UID"]==UID_i and set(row["sub_ids"])==sub_ids_i and set(row["pro_ids"])==pro_ids_i,axis=1)]
            Seq_i = data_kcat["Sequence"][i]
            same_data = data_kcat[data_kcat.apply(lambda row: row["Sequence"]==Seq_i and set(row["sub_ids"])==sub_ids_i and set(row["pro_ids"])==pro_ids_i,axis=1)]
            
            UIDs = list(set(list(same_data["UID"])))
            
            kcat = []
            if len(same_data)>1:
                drop = drop + list(same_data.index)[1:]
                for index in same_data.index:
                    kcat.append(same_data["kcat"][index])
                    data_kcat["kcat"][index] = -1
                data_kcat["kcat"][list(same_data.index)[0]] = kcat
            else:
                kcat.append(data_kcat["kcat"][i])
                data_kcat["kcat"][i] = kcat
            data_kcat["UID"][i] = UIDs
                 
                 
                    
        data_kcat.drop(list(set(drop)),inplace = True)

        print("把反应信息修正一下,将新反应信息和新酶序列整理后： ",len(data_kcat))
        #kcat值：取最大值 删除异常值 计算几何平均  同反应计算最大，同酶计算最大，
        # 同EC号计算最大  计算前后分子量
        data_kcat["log10_kcat"], data_kcat["log2_kcat"],data_kcat["reaction_max"], data_kcat["Seq_max"],data_kcat["UID_max"], data_kcat["EC_max"],data_kcat["kcat_max"] = -1,-1, -1,-1, -1,np.nan,-1
        data_kcat["sub_mw"], data_kcat["pro_mw"] = 0,0
        for i in data_kcat.index:
            #计算同UID最大值
            if data_kcat["UID_max"][i]==-1:
                for UID_i in data_kcat["UID"][i]:
                    same_UID = all_data.loc[all_data["UID"]==UID_i]
                    UID_max = same_UID['kcat'].apply(lambda x: max(x) if isinstance(x, list) else x).max()
                    data_kcat["UID_max"][i] = max(UID_max,data_kcat["UID_max"][i])
            

                    
        for i in data_kcat.index:    
            #计算同reaction最大值
            if data_kcat["reaction_max"][i]==-1:
                sub_ids_i = set(data_kcat["sub_ids"][i])
                pro_ids_i = set(data_kcat["pro_ids"][i])
                same_reaction = data_kcat[data_kcat.apply(lambda row: set(row["sub_ids"])==sub_ids_i and set(row["pro_ids"])==pro_ids_i,axis=1)]
                reaction_max = same_reaction['kcat'].apply(lambda x: max(x) if isinstance(x, list) else x).max()
                for index in same_reaction.index:
                    data_kcat["reaction_max"][index] = reaction_max
        
        #添加EC号
        EC_data = pd.read_csv(opt.datadir+"enzyme_data/Uniprot_results_EC.tab", sep = "\t")
        data_kcat["EC"] = ""
        for i in data_kcat.index:
            UID0 = data_kcat["UID"][i][0]
            try:
                data_kcat["EC"][i] = list(EC_data["EC number"].loc[EC_data["Uniprot ID"] == UID0])[0].split("; ")
            except:
                data_kcat["EC"][i] = []
        
        
        
        for i in data_kcat.index:    
            #计算同EC最大值
            EC_max_data = pd.read_pickle(opt.datadir+"enzyme_data/df_EC_max_kcat.pkl")
            ECs_i = data_kcat["EC"][i]

            EC_max = 0
            for EC_i in ECs_i:
                if EC_i == 'I0ZIE0':
                    pass
                try:
                    EC_max = max(EC_max,list(EC_max_data["max_kcat"].loc[EC_max_data["EC"] == EC_i])[0])
                except:
                    pass
            if EC_max!=0:
                data_kcat['EC_max'][i] = EC_max
            else:
                data_kcat['EC_max'][i] = np.nan
            
            '''if pd.isnull(EC_max):
                same_UID = data_kcat.loc[data_kcat["EC"]==EC_i]
                EC_max = same_UID['kcat'].apply(lambda x: max(x) if isinstance(x, list) else x).max()
                data_kcat['EC_max'][i] = EC_max
                pass
            ######
            else:
                same_UID = data_kcat.loc[data_kcat["EC"]==EC_i]
                EC_max = same_UID['kcat'].apply(lambda x: max(x) if isinstance(x, list) else x).max()
                print(data_kcat['EC_max'][i],EC_max)'''

            #计算log10 
        Drop_index = []
        for i in data_kcat.index:
            
            if data_kcat["log10_kcat"][i]==-1:
                # sub_ids_i = set(data_kcat["sub_ids"][i])
                # pro_ids_i = set(data_kcat["pro_ids"][i])
                # Seq_i = data_kcat["Sequence"][i]
                # same_data = data_kcat[data_kcat.apply(lambda row: row["Sequence"]==Seq_i and set(row["sub_ids"])==sub_ids_i and set(row["pro_ids"])==pro_ids_i,axis=1)]  
                # UID_max = data_kcat["UID_max"][i]
                # if len(same_data) > 1:
                #     Drop_index = Drop_index + list(same_data.index)[1:]
                # for same_index in same_data.index:
                #     data_kcat["log10_kcat"][same_index] = 0
                #     UID_max = max(UID_max,data_kcat["UID_max"][same_index])
                #     if isinstance(data_kcat["kcat"][same_index],list):
                #         kcats = kcats + data_kcat["kcat"][same_index]
                #     else:
                #         kcats.append(data_kcat["kcat"][same_index])
                        
                # data_kcat["UID_max"][i] = UID_max  
                
                kcats = data_kcat["kcat"][i]
                max_kcat = max(kcats)
                data_kcat["kcat_max"][i] = max_kcat
                kcats = [kcat for kcat in kcats if kcat / max_kcat > 0.01]
                data_kcat["log10_kcat"][i] = np.mean(np.log10(kcats))
                data_kcat["log2_kcat"][i] = np.mean(np.log2(kcats))
                
        data_kcat.drop(list(set(Drop_index)), inplace=True)

        for i in data_kcat.index:    
            #计算前后分子量
            sub_i = set(data_kcat['sub_ids'][i])
            pro_i = set(data_kcat['pro_ids'][i])
            
            if data_kcat['sub_mw'][i]==0:
                same_data = data_kcat[data_kcat.apply(lambda row: set(row["sub_ids"])==sub_i,axis=1)]
                mw_subs = Count_mw(sub_i)
                for index in same_data.index:
                    data_kcat['sub_mw'][index] = mw_subs
                    
            if data_kcat['pro_mw'][i]==0:
                same_data = data_kcat[data_kcat.apply(lambda row: set(row["pro_ids"])==pro_i,axis=1)]
                mw_pros = Count_mw(pro_i)
                for index in same_data.index:
                    data_kcat['pro_mw'][index] = mw_pros

        
        print("data preprocessed： ",len(data_kcat))
        data_kcat.to_pickle(opt.datadir+"kcat_data/preprocessing_kcat_data.pkl")
    
    
    
    
    kcat_data = pd.read_pickle(opt.datadir+"kcat_data/preprocessing_kcat_data.pkl")
    
    #删除不合理/不natural的值
    kcat_data["EC_max"].loc[pd.isnull(kcat_data["EC_max"])] = kcat_data['kcat_max']
    #kcat_data = kcat_data[kcat_data.apply(lambda row: row["kcat_max"]/row["UID_max"]>0.1 and row["kcat_max"]/row["reaction_max"]>0.01 and row["kcat_max"]/row["EC_max"]>0.01 and row["kcat_max"]/row["EC_max"]<10 ,axis=1)]
    kcat_data = kcat_data[kcat_data.apply(lambda row: row["kcat_max"]/row["UID_max"]>0.1 and row["kcat_max"]/row["reaction_max"]>0.01 and row["kcat_max"]/row["EC_max"]>0.01 and row["kcat_max"]/row["EC_max"]<10 ,axis=1)]
    print("删除不合理/不natural的值后： ",len(kcat_data))
    
    
    #删除前后分子量之比过大的值
    kcat_data = kcat_data[kcat_data.apply(lambda row: row['pro_mw']!=0,axis = 1)]
    kcat_data['mw_reaction'] = kcat_data['sub_mw'] / kcat_data['pro_mw']
    kcat_data = kcat_data.loc[kcat_data["mw_reaction"] < 3]
    kcat_data = kcat_data.loc[kcat_data["mw_reaction"] > 1/3]
    print("保留前后分子量之比正常的值后： ",len(kcat_data))
    
    #删除过大过小的值
    kcat_data = kcat_data.loc[~(kcat_data["log10_kcat"]>5)]
    kcat_data = kcat_data.loc[~(kcat_data["log10_kcat"]<-2.5)]
    print("删除过大过小的值后： ",len(kcat_data))
    
    kcat_data.to_pickle(opt.datadir+"kcat_data/preprocessed_kcat_data.pkl")
    
    print('data  preprocessing finished')































