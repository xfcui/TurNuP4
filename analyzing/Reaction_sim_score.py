import os
import pickle
import hashlib
import argparse
import requests
import numpy as np
import sys
import torch
from rdkit import Chem
from zeep import Client
from os.path import join
from rdkit.Chem import Crippen, AllChem
from rdkit.Chem import Descriptors
from urllib.request import urlopen, Request
from bioservices import *
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import Draw
import pandas as pd
import random
from os.path import join
from rdkit import Chem
from rdkit.Chem import AllChem
from drfp import DrfpEncoder
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb
from sklearn import metrics
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import LinearRegression
import scipy



def img_4_load(loadnames,savename):
    
    datadir = "./kcat/data/"

    data_train = pd.read_pickle(datadir+"kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle(datadir+"kcat_data/splits/test_kcat.pkl")

    data_train = data_train.iloc[:, [7,18,20,35,39,40,50]]
    data_test = data_test.iloc[:, [7,18,20,35,39,40,50]]

    train_fps = [np.array(list(data_train["DRFP"][ind])).reshape(1,-1).astype(int) for ind in data_train.index]
    test_fps = [np.array(list(data_test["DRFP"][ind])).reshape(1,-1).astype(int) for ind in data_test.index]


    max_sim = []

    for fp in test_fps:
        jaccard_sim = np.array([1- scipy.spatial.distance.cdist(fp,train_fp, metric='jaccard')[0][0] for train_fp in train_fps])
        max_sim.append(np.max(jaccard_sim))
        
    data_test["reaction_sim"] = max_sim

    data_test["reaction_sim"]= (data_test["reaction_sim"] - np.min(data_test["reaction_sim"]))
    data_test["reaction_sim"] = data_test["reaction_sim"]/np.max(data_test["reaction_sim"])
    

    
    r2 = []
    pearson = []
    mse = []
    num = []
    
    for loadname in loadnames:
        
        pred_y = np.load(datadir+"training_results_1/y_test_pred_xgboost_"+loadname+".npy")
        test_y = np.load(datadir+"training_results_1/y_test_true_xgboost_"+loadname+".npy")

        data_test["y_pred"] = pred_y
        data_test["y_true"] = test_y

        sim04 = data_test[data_test.apply(lambda row: row["reaction_sim"]<0.4,axis=1)]
        sim48 = data_test[data_test.apply(lambda row: row["reaction_sim"]>=0.4 and row["reaction_sim"]<0.8,axis=1)]
        sim81 = data_test[data_test.apply(lambda row: row["reaction_sim"]>=0.8 and row["reaction_sim"]<1,axis=1)]
        sim1 = data_test[data_test.apply(lambda row: row["reaction_sim"]>=1,axis=1)]

        r2_scores = []
        pearson_r = []
        mse_scores = []
        point_num = []

        true = sim04['y_true']
        pred = sim04['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim04))


        true = sim48['y_true']
        pred = sim48['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim48))


        true = sim81['y_true']
        pred = sim81['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim81))


        true = sim1['y_true']
        pred = sim1['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim1))

        r2.append(r2_scores)
        pearson.append(pearson_r)
        mse.append(mse_scores)
        num = point_num
    
    # 创建横坐标列表，等距分布的值
    x_coords = [1, 2, 3, 4]

    # 对应横坐标的名称
    x_labels = ['0.0 - 0.4', '0.4 - 0.8', '0.8 - 1','same reaction']
    
    color = ['red','blue','green','orange','yellow','purple']
    
    
    annotations = num
    plt.figure(figsize=(8, 8))
    for index in range(len(r2)):
        plt.scatter(x_coords, r2[index], color=color[index], s=10)
            
    plt.xlabel('Reaction similarity score', fontsize=14)
    plt.ylabel('R2 Score', fontsize=14)
    plt.title('Scatter Plot of Data Points', fontsize=16)
    plt.xticks(ticks=x_coords, labels=x_labels, fontsize=12)
    plt.ylim(-0.2, 0.8)

    for i in range(len(x_coords)):
        plt.text(x_coords[i], r2[0][i], str(annotations[i]), fontsize=8, ha='right')
        for index in range(len(r2)):
            plt.text(x_coords[i], r2[index][i], str(r2[index][i]), fontsize=10, ha='left')

    plt.savefig(datadir+"sim_img/sim_r2_"+savename+'_4.png')
    # plt.savefig(datadir+"sim_r2_FP_DRFP_DRFPaddsub_2571_3.png")
    
    
    plt.figure(figsize=(8, 8))    

    for index in range(len(pearson)):
        plt.scatter(x_coords, pearson[index], color=color[index], s=10)
            
    plt.xlabel('Reaction similarity score', fontsize=14)
    plt.ylabel('Pearson r', fontsize=14)
    plt.title('Scatter Plot of Data Points', fontsize=16)
    plt.xticks(ticks=x_coords, labels=x_labels, fontsize=12)
    plt.ylim(0, 1)

    for i in range(len(x_coords)):
        plt.text(x_coords[i], pearson[0][i], str(annotations[i]), fontsize=8, ha='right')
        for index in range(len(pearson)):
            plt.text(x_coords[i], pearson[index][i], str(pearson[index][i]), fontsize=10, ha='left')

    plt.savefig(datadir+"sim_img/sim_pearson_"+savename+'_4.png')
    # plt.savefig(datadir+"sim_r2_FP_DRFP_DRFPaddsub_2571_3.png")
    
    
    plt.figure(figsize=(10, 10))    
    for index in range(len(r2)):
            plt.scatter(x_coords, mse[index], color=color[index], s=10)
            
    plt.xlabel('Reaction similarity score', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Scatter Plot of Data Points', fontsize=16)
    plt.xticks(ticks=x_coords, labels=x_labels, fontsize=12)
    plt.ylim(0.4, 2)

    for i in range(len(x_coords)):
        plt.text(x_coords[i], mse[0][i], str(annotations[i]), fontsize=8, ha='right')
        for index in range(len(mse)):
            plt.text(x_coords[i], mse[index][i], str(mse[index][i]), fontsize=10, ha='left')

    plt.savefig(datadir+"sim_img/sim_mse_"+savename+'_4.png')
    # plt.savefig(datadir+"sim_r2_FP_DRFP_DRFPaddsub_2571_3.png")


    
def img_3_load(loadnames,savename):
    
    datadir = "./kcat/data/"

    data_train = pd.read_pickle(datadir+"kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle(datadir+"kcat_data/splits/test_kcat.pkl")

    data_train = data_train.iloc[:, [7,18,20,35,39,40,50]]
    data_test = data_test.iloc[:, [7,18,20,35,39,40,50]]

    train_fps = [np.array(list(data_train["DRFP"][ind])).reshape(1,-1).astype(int) for ind in data_train.index]
    test_fps = [np.array(list(data_test["DRFP"][ind])).reshape(1,-1).astype(int) for ind in data_test.index]


    max_sim = []

    for fp in test_fps:
        jaccard_sim = np.array([1- scipy.spatial.distance.cdist(fp,train_fp, metric='jaccard')[0][0] for train_fp in train_fps])
        max_sim.append(np.max(jaccard_sim))
        
    data_test["reaction_sim"] = max_sim

    data_test["reaction_sim"]= (data_test["reaction_sim"] - np.min(data_test["reaction_sim"]))
    data_test["reaction_sim"] = data_test["reaction_sim"]/np.max(data_test["reaction_sim"])
    

    
    r2 = []
    pearson = []
    mse = []
    num = []
    
    for loadname in loadnames:
        
        pred_y = np.load(datadir+"training_results_1/y_test_pred_xgboost_"+loadname+".npy")
        test_y = np.load(datadir+"training_results_1/y_test_true_xgboost_"+loadname+".npy")

        data_test["y_pred"] = pred_y
        data_test["y_true"] = test_y

        sim04 = data_test[data_test.apply(lambda row: row["reaction_sim"]<0.4,axis=1)]
        sim48 = data_test[data_test.apply(lambda row: row["reaction_sim"]>=0.4 and row["reaction_sim"]<0.8,axis=1)]
        sim81 = data_test[data_test.apply(lambda row: row["reaction_sim"]>=0.8 and row["reaction_sim"]<=1,axis=1)]
        # sim1 = data_test[data_test.apply(lambda row: row["reaction_sim"]>=1,axis=1)]

        r2_scores = []
        pearson_r = []
        mse_scores = []
        point_num = []

        true = sim04['y_true']
        pred = sim04['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim04))


        true = sim48['y_true']
        pred = sim48['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim48))


        true = sim81['y_true']
        pred = sim81['y_pred']
        r2_scores.append(np.round(r2_score(true, pred),3))
        pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        point_num.append(len(sim81))


        # true = sim1['y_true']
        # pred = sim1['y_pred']
        # r2_scores.append(np.round(r2_score(true, pred),3))
        # pearson_r.append(np.round(stats.pearsonr(true, pred)[0],3))
        # mse_scores.append(np.round(np.mean(abs(np.reshape(true, (-1)) -  pred)**2),3))
        # point_num.append(len(sim81))

        r2.append(r2_scores)
        pearson.append(pearson_r)
        mse.append(mse_scores)
        num = point_num
    
    # 创建横坐标列表，等距分布的值
    x_coords = [1, 2, 3]

    # 对应横坐标的名称
    x_labels = ['0.0 - 0.4', '0.4 - 0.8', '0.8 - 1']
    
    color = ['red','blue','green','orange','yellow','purple']
    
    
    annotations = num
    plt.figure(figsize=(8, 8))
    for index in range(len(r2)):
        plt.scatter(x_coords, r2[index], color=color[index], s=10)
            
    plt.xlabel('Reaction similarity score', fontsize=14)
    plt.ylabel('R2 Score', fontsize=14)
    plt.title('Scatter Plot of Data Points', fontsize=16)
    plt.xticks(ticks=x_coords, labels=x_labels, fontsize=12)
    plt.ylim(-0.2, 0.8)

    for i in range(len(x_coords)):
        plt.text(x_coords[i], r2[0][i], str(annotations[i]), fontsize=8, ha='right')
        for index in range(len(r2)):
            plt.text(x_coords[i], r2[index][i], str(r2[index][i]), fontsize=10, ha='left')

    plt.savefig(datadir+"sim_img/sim_r2_"+savename+'_4.png')
    # plt.savefig(datadir+"sim_r2_FP_DRFP_DRFPaddsub_2571_3.png")
    
    
    plt.figure(figsize=(8, 8))    

    for index in range(len(pearson)):
        plt.scatter(x_coords, pearson[index], color=color[index], s=10)
            
    plt.xlabel('Reaction similarity score', fontsize=14)
    plt.ylabel('Pearson r', fontsize=14)
    plt.title('Scatter Plot of Data Points', fontsize=16)
    plt.xticks(ticks=x_coords, labels=x_labels, fontsize=12)
    plt.ylim(0, 1)

    for i in range(len(x_coords)):
        plt.text(x_coords[i], pearson[0][i], str(annotations[i]), fontsize=8, ha='right')
        for index in range(len(pearson)):
            plt.text(x_coords[i], pearson[index][i], str(pearson[index][i]), fontsize=10, ha='left')

    plt.savefig(datadir+"sim_img/sim_pearson_"+savename+'_4.png')
    # plt.savefig(datadir+"sim_r2_FP_DRFP_DRFPaddsub_2571_3.png")
    
    
    plt.figure(figsize=(10, 10))    
    for index in range(len(r2)):
            plt.scatter(x_coords, mse[index], color=color[index], s=10)
            
    plt.xlabel('Reaction similarity score', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Scatter Plot of Data Points', fontsize=16)
    plt.xticks(ticks=x_coords, labels=x_labels, fontsize=12)
    plt.ylim(0.4, 2)

    for i in range(len(x_coords)):
        plt.text(x_coords[i], mse[0][i], str(annotations[i]), fontsize=8, ha='right')
        for index in range(len(mse)):
            plt.text(x_coords[i], mse[index][i], str(mse[index][i]), fontsize=10, ha='left')

    plt.savefig(datadir+"sim_img/sim_mse_"+savename+'_4.png')
    # plt.savefig(datadir+"sim_r2_FP_DRFP_DRFPaddsub_2571_3.png")
    

if __name__ == "__main__":
    img_4_load(['DRFP','DRFPaddsub','DRFP_AE_512s','DRFP_AE_512d','DRFP_AE_1048'],'FP_DRFP_DRFP_159')
 