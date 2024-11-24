import os
import pickle
import numpy as np
import sys
import torch
from bioservices import *
import warnings
import pandas as pd                     
import os
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb
CURRENT_DIR = os.getcwd()

def with_DRFP(data_train,data_test,train_indices, test_indices)                                                                      :
    train_X = np.array(list(data_train["DRFP"]))
    #train_X = np.array(list(data_train["DRFP"]))
    train_Y = np.array(list(data_train["log10_kcat"]))

    test_X = np.array(list(data_test["DRFP_imb"]))
    #test_X = np.array(list(data_test["DRFP"]))
    test_Y = np.array(list(data_test["log10_kcat"]))
    
    param = {'learning_rate': 0.08987247189322463,
         'max_delta_step': 1.1939737318908727,
         'max_depth': 11.268531225242574,
         'min_child_weight': 2.8172720953826302,
         'num_rounds': 109.03643430746544,
         'reg_alpha': 1.9412226989868904,
         'reg_lambda': 4.950543905603358}
    # print("train with DRFP:")
    
    return train_valid_test(train_X,train_Y,test_X,test_Y,param,train_indices, test_indices,'DRFP')
    

def with_esm1b_ts(data_train,data_test,train_indices, test_indices):
    train_ESM1b = np.array(list(data_train["ESM1b_ts"]))
    train_X = train_ESM1b
    train_Y = np.array(list(data_train["log10_kcat"]))

    test_ESM1b = np.array(list(data_test["ESM1b_ts"]))
    test_X = test_ESM1b
    test_Y = np.array(list(data_test["log10_kcat"]))


    param = {'learning_rate': 0.2831145406836757,
         'max_delta_step': 0.07686715986169101, 
         'max_depth': 4.96836783761305,
          'min_child_weight': 6.905400087083855,
           'num_rounds': 313.1498988074061,
            'reg_alpha': 1.717314107718892,
             'reg_lambda': 2.470354543039016}
    # print("train with esm1b_ts:")
    return train_valid_test(train_X,train_Y,test_X,test_Y,param,train_indices, test_indices,"ESM1b_ts")


    

def with_DRFP_esm1b_ts_mean(data_train,data_test,train_indices, test_indices,y_valid_pred_DRFP,y_valid_pred_esm1b_ts,y_test_pred_drfp, y_test_pred_esm1b_ts):


    train_Y = np.array(list(data_train["log10_kcat"]))
    test_Y = np.array(list(data_test["log10_kcat"]))
    train_Y = np.concatenate([train_Y, test_Y])
    
    print("train with DRFP+esm1b_ts:")
    
    R2 = []
    MSE = []
    Pearson = []

    
    y_test_pred = np.mean([y_test_pred_drfp, y_test_pred_esm1b_ts], axis =0)

    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred)**2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)
    print('MSE: ',np.round(MSE_dif_fp_test,3),'R2: ', np.round(R2_dif_fp_test,3),'Pearson: ',np.round(Pearson[0],3) )
    

    
    return y_test_pred
    

def with_DRFP_esm2_mean(data_train,data_test,train_indices, test_indices,y_valid_pred_DRFP,y_valid_pred_esm1b_ts,y_test_pred_drfp, y_test_pred_esm1b_ts):


    train_Y = np.array(list(data_train["log10_kcat"]))
    test_Y = np.array(list(data_test["log10_kcat"]))
    train_Y = np.concatenate([train_Y, test_Y])
    
    print("train with DRFP+esm2:")
    
    R2 = []
    MSE = []
    Pearson = []


    y_test_pred = np.mean([y_test_pred_drfp, y_test_pred_esm1b_ts], axis =0)

    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred)**2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)
    print(np.round(Pearson[0],3) ,np.round(MSE_dif_fp_test,3), np.round(R2_dif_fp_test,3))

    
    return y_test_pred

def print_sim_score(true,pred):
    with open('./data/sim_index/sim04_indices.pkl', 'rb') as f:
        sim04_indices = pickle.load(f)

    with open('./data/sim_index/sim48_indices.pkl', 'rb') as f:
        sim48_indices = pickle.load(f)

    with open('./data/sim_index/sim81_indices.pkl', 'rb') as f:
        sim81_indices = pickle.load(f)

    with open('./data/sim_index/sim1_indices.pkl', 'rb') as f:
        sim1_indices = pickle.load(f)
    
    sim04_true = true[sim04_indices]
    sim04_pred = pred[sim04_indices]
    print("04 r2:",np.round(r2_score(sim04_true, sim04_pred),3))
    # print("04 pearson:",np.round(stats.pearsonr(sim04_true, sim04_pred)[0],3))
    # print("04 mse:",np.round(np.mean(abs(np.reshape(sim04_true, (-1)) -  sim04_pred)**2),3))
    # print("04 num:",len(sim04_indices))


    sim48_true = true[sim48_indices]
    sim48_pred = pred[sim48_indices]
    print("48 r2:",np.round(r2_score(sim48_true, sim48_pred),3))
    # print("48 pearson:",np.round(stats.pearsonr(sim48_true, sim48_pred)[0],3))
    # print("48 mse:",np.round(np.mean(abs(np.reshape(sim48_true, (-1)) -  sim48_pred)**2),3))
    # print("48 num:",len(sim48_indices))
    
    sim81_true = true[sim81_indices]
    sim81_pred = pred[sim81_indices]
    print("81 r2:",np.round(r2_score(sim81_true, sim81_pred),3))
    # print("81 pearson:",np.round(stats.pearsonr(sim81_true, sim81_pred)[0],3))
    # print("81 mse:",np.round(np.mean(abs(np.reshape(sim81_true, (-1)) -  sim81_pred)**2),3))
    # print("81 num:",len(sim81_indices))
    
    sim1_true = true[sim1_indices]
    sim1_pred = pred[sim1_indices]
    print("1 r2:",np.round(r2_score(sim1_true, sim1_pred),3))
    # print("1 pearson:",np.round(stats.pearsonr(sim1_true, sim1_pred)[0],3))
    # print("1 mse:",np.round(np.mean(abs(np.reshape(sim1_true, (-1)) -  sim1_pred)**2),3))
    # print("1 num:",len(sim1_indices))
    
def print_sim_score3(true,pred):
    with open('./data/sim_index/sim03_indices.pkl', 'rb') as f:
        sim04_indices = pickle.load(f)

    with open('./data/sim_index/sim36_indices.pkl', 'rb') as f:
        sim48_indices = pickle.load(f)


    with open('./data/sim_index/sim1_indices.pkl', 'rb') as f:
        sim1_indices = pickle.load(f)
    
    sim04_true = true[sim04_indices]
    sim04_pred = pred[sim04_indices]
    print("03 r2:",np.round(r2_score(sim04_true, sim04_pred),3))
    print("03 pearson:",np.round(stats.pearsonr(np.array(sim04_true), np.reshape(np.array(sim04_pred), (-1)))[0], 4))
    # print("04 pearson:",np.round(stats.pearsonr(sim04_true, sim04_pred)[0],3))
    # print("04 mse:",np.round(np.mean(abs(np.reshape(sim04_true, (-1)) -  sim04_pred)**2),3))
    # print("04 num:",len(sim04_indices))


    sim48_true = true[sim48_indices]
    sim48_pred = pred[sim48_indices]
    print("36 r2:",np.round(r2_score(sim48_true, sim48_pred),3))
    print("36 pearson:",np.round(stats.pearsonr(np.array(sim48_true), np.reshape(np.array(sim48_pred), (-1)))[0], 4))
    # print("48 pearson:",np.round(stats.pearsonr(sim48_true, sim48_pred)[0],3))
    # print("48 mse:",np.round(np.mean(abs(np.reshape(sim48_true, (-1)) -  sim48_pred)**2),3))
    # print("48 num:",len(sim48_indices))
    
    sim1_true = true[sim1_indices]
    sim1_pred = pred[sim1_indices]
    print("1 r2:",np.round(r2_score(sim1_true, sim1_pred),3))
    print("1 pearson:",np.round(stats.pearsonr(np.array(sim1_true), np.reshape(np.array(sim1_pred), (-1)))[0], 4))
    # print("1 pearson:",np.round(stats.pearsonr(sim1_true, sim1_pred)[0],3))
    # print("1 mse:",np.round(np.mean(abs(np.reshape(sim1_true, (-1)) -  sim1_pred)**2),3))
    # print("1 num:",len(sim1_indices))
                 
    print(np.round(r2_score(sim04_true, sim04_pred),3),np.round(r2_score(sim48_true, sim48_pred),3),np.round(r2_score(sim1_true, sim1_pred),3))
    print(np.round(stats.pearsonr(np.array(sim04_true), np.reshape(np.array(sim04_pred), (-1)))[0], 4),np.round(stats.pearsonr(np.array(sim48_true), np.reshape(np.array(sim48_pred), (-1)))[0], 4),np.round(stats.pearsonr(np.array(sim1_true), np.reshape(np.array(sim1_pred), (-1)))[0], 4))
         
    
def train_valid_test(train_X,train_Y,test_X,test_Y,param,train_indices, test_indices,savename):
    num_round = param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))

    del param["num_rounds"]
    R2 = []
    MSE = []
    Pearson = []
    y_valid_pred_list = []


    

    dtrain = xgb.DMatrix(train_X, label = train_Y)
    dtest = xgb.DMatrix(test_X)

    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

    
    y_test_pred = bst.predict(dtest)
    MSE_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred)**2)
    R2_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)


    
    return y_valid_pred_list,y_test_pred





if __name__ == "__main__":
    
    try:
        seed = int(sys.argv[1])
    except:
        seed = 0
    
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    """Hyperparameters."""
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datadir', type=str, default='./data/', help='data文件夹位置')#'../data/'
    # opt = parser.parse_args()

    datadir = "./data/"

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')
    

    
    #UID, Sequence,ESM1b,ESM1b_ts,  kcat, sub_ids, pro_ids, log10_kcat, reaction_max, UID_max, EC_max,kcat_max
    
    
    data_train = pd.read_pickle(datadir+"kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle(datadir+"kcat_data/splits/test_kcat.pkl")

    data_train_balance = data_train[data_train.apply(lambda row: row["balance"]==True,axis=1)]
    data_train_imbalance = data_train[data_train.apply(lambda row: row["balance"]==False,axis=1)]

    data_test_balance = data_test[data_test.apply(lambda row: row["balance"]==True,axis=1)]
    data_test_imbalance = data_test[data_test.apply(lambda row: row["balance"]==False,axis=1)]
    
    b_test_index = data_test.index[data_test['balance'] == True].tolist()
    imb_test_index = data_test.index[data_test['balance'] == False].tolist()
    
    
    #data_train = data_train_balance
    if seed > 0:
        if seed ==2:
            print("balance as train")
            data_train = data_train_balance
        if seed ==3:
            print("imbalance as train")
            data_train = data_train_imbalance
    
    
    #data_test = data_test_imbalance
    
    # new_df = data_test.iloc[:, [0, 1, 2,3,4,5,6,7,9,10,11,13,17,30,31,32]]
    
    
    train_indices = []
    test_indices = []
    for i in range(5):
        train_indice = list(np.load(datadir+"kcat_data/splits/train_without_valid"+str(i)+".npy", allow_pickle = True))
        test_indice = list(np.load(datadir+"kcat_data/splits/valid"+str(i)+".npy", allow_pickle = True))
        train_indices.append(train_indice)
        test_indices.append(test_indice)
        pass
    

    y_valid_pred_DRFP,y_test_pred_drfp= with_DRFP(data_train,data_test,train_indices, test_indices)
    
    
    y_valid_pred_esm1b_ts, y_test_pred_esm1b_ts = with_esm1b_ts(data_train,data_test,train_indices, test_indices)
    

    
    print('TurNuP(product_absent):')
    with_DRFP_esm1b_ts_mean(data_train,data_test,train_indices, test_indices,y_valid_pred_DRFP,y_valid_pred_esm1b_ts,y_test_pred_drfp, y_test_pred_esm1b_ts)
    

    
    
    print('\n\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    