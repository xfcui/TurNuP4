import os
import pickle
import json
import hashlib
import argparse
import requests
import numpy as np
import torch
import timeit
import math
from rdkit import Chem
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from tqdm import tqdm




def save_model(model, path):
    state = {
        'model': model.state_dict()
    }
    torch.save(state, path)


def print_sim_score(true,pred):
    with open('./kcat/TurNuP4/data/sim_index/sim04_indices.pkl', 'rb') as f:
        sim04_indices = pickle.load(f)

    with open('./kcat/TurNuP4/data/sim_index/sim48_indices.pkl', 'rb') as f:
        sim48_indices = pickle.load(f)

    with open('./kcat/TurNuP4/data/sim_index/sim81_indices.pkl', 'rb') as f:
        sim81_indices = pickle.load(f)

    with open('./kcat/TurNuP4/data/sim_index/sim1_indices.pkl', 'rb') as f:
        sim1_indices = pickle.load(f)
    
    true = np.array(true)
    pred = np.array(pred)
    
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

# 加载模型参数
def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    return model

def save_hyperparameters(opt, path):
    with open(path, 'w') as f:
        for key, value in vars(opt).items():
            f.write(f'{key}: {value}\n')

def log_training_process(epoch, train_loss, valid_loss, test_loss, train_mse, valid_mse, test_mse, train_r2, valid_r2, test_r2, train_pearson, valid_pearson, test_pearson,bestonce,path):
    with open(path, 'a') as f:
        f.write(f'Epoch {epoch}:\n')
        f.write(f'Train Loss: {train_loss}, Train MSE: {train_mse}, Train R2: {train_r2},Train Pearson: {train_pearson}\n')
        f.write(f'Valid Loss: {valid_loss}, Valid MSE: {valid_mse}, Valid R2: {valid_r2},Valid Pearson: {valid_pearson}\n')
        f.write(f'Test Loss: {test_loss}, Test MSE: {test_mse}, Test R2: {test_r2}, Test Pearson: {test_pearson},Best: {bestonce}\n\n')
       



class MoleculeProteinModel(nn.Module):
    def __init__(self, mol_input_dim, prot_input_dim, mol_embed_dim,prot_embed_dim, mlp_hidden_dim, output_dim):
        super(MoleculeProteinModel, self).__init__()


        self.mol_fc2 = nn.Linear(mol_input_dim, mol_embed_dim)
        self.mol_ln2 = nn.LayerNorm(mol_embed_dim)
        self.mol_fc3 = nn.Linear(mol_embed_dim, 1024)
        self.mol_ln3 = nn.LayerNorm(1024)
        self.mol_fc4 = nn.Linear(1024, output_dim)
        

        self.prot_fc2 = nn.Linear(prot_input_dim, prot_embed_dim)
        self.prot_ln2 = nn.LayerNorm(prot_embed_dim)
        self.prot_fc5 = nn.Linear(prot_embed_dim, output_dim)
        

        self.mol_dropout = nn.Dropout(opt.dropout_mol)
        self.prot_dropout = nn.Dropout(opt.dropout_prot)
        
        self.dropout = nn.Dropout(0.5)


    def forward(self, mol_input, prot_input):
        # Molecule feature extraction
        mol_features = F.relu(self.mol_fc2(mol_input))
        
        mol_features = self.mol_ln2(mol_features)
        mol_features = F.relu(self.mol_fc3(mol_features))
        mol_features = self.mol_ln3(mol_features)

        mol_output = self.mol_fc4(mol_features)
        
        # Protein feature extractionm
        prot_features = F.relu(self.prot_fc2(prot_input))
        prot_features = self.prot_dropout(prot_features)  #
        prot_output = self.prot_fc5(prot_features)
        

        
        return mol_output , prot_output

def train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, num_epochs, device):
    model.to(device)
    criterion = nn.MSELoss()

    max_test_pearson = 0  # 用以保存最好的结果
    max_valid_pearson = 0  #显示追踪最好结果
    max_valid_R2 = 0
    pearson_epoch = 0
    R2_epoch = 0
    
    max_valid_pearsonb = 0  #显示追踪最好结果
    max_valid_R2b = 0

    
    max_valid_pearsonimb = 0  #显示追踪最好结果
    max_valid_R2imb = 0

    max_valid_DRFP_pearson = 0 
    max_valid_DRFP_R2 = 0

    
    max_valid_DRFP_pearsonb = 0  #显示追踪最好结果
    max_valid_DRFP_R2b = 0

    
    max_valid_DRFP_pearsonimb = 0  #显示追踪最好结果
    max_valid_DRFP_R2imb = 0
    
    update = True
    
    true = []
    best_pred = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        y_true, y_pred_mol, y_pred_prot = [], [], []
        
        for mol_inputs, prot_inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            mol_inputs, prot_inputs, labels = mol_inputs.to(device), prot_inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            mol_outputs, prot_outputs = model(mol_inputs, prot_inputs)
            loss_mol = criterion(mol_outputs, labels.view(-1, 1))
            loss_prot = criterion(prot_outputs, labels.view(-1, 1))
            loss = (0.1)*loss_mol + (0.2)*loss_prot
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            y_true.extend(labels.detach().cpu().numpy())
            y_pred_mol.extend(mol_outputs.detach().cpu().numpy())
            y_pred_prot.extend(prot_outputs.detach().cpu().numpy())
        

        y_pred_avg = [(m + p) / 2 for m, p in zip(y_pred_mol, y_pred_prot)]
        train_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_avg])
        train_r2 = r2_score(y_true, y_pred_avg)
        train_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_avg), (-1)))[0], 4)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}, MSE: {train_mse}, R2: {train_r2}, Pearson: {train_pearson}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        y_true, y_pred_mol, y_pred_prot = [], [], []
        
        with torch.no_grad():
            for mol_inputs, prot_inputs, labels in tqdm(val_loader, desc='Validation'):
                mol_inputs, prot_inputs, labels = mol_inputs.to(device), prot_inputs.to(device), labels.to(device)
                mol_outputs, prot_outputs = model(mol_inputs, prot_inputs)
                loss_mol = criterion(mol_outputs, labels.view(-1, 1))
                loss_prot = criterion(prot_outputs, labels.view(-1, 1))
                loss = (0.1)*loss_mol + (0.2)*loss_prot
                val_loss += loss.item()
                
                y_true.extend(labels.detach().cpu().numpy())
                y_pred_mol.extend(mol_outputs.detach().cpu().numpy())
                y_pred_prot.extend(prot_outputs.detach().cpu().numpy())

        
        y_pred_avg = [(m + p) / 2 for m, p in zip(y_pred_mol, y_pred_prot)]
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_avg])
        val_r2 = r2_score(y_true, y_pred_avg)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_avg), (-1)))[0], 4)
        if val_r2>max_valid_R2:
            update = True
        else:
            update = False
        
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_mol])
        val_r2 = r2_score(y_true, y_pred_mol)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_mol), (-1)))[0], 4)
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson}')
        
        if update:
            max_valid_DRFP_R2 = val_r2
            max_valid_DRFP_pearson = val_pearson
            # best_pred = list(y_pred_mol)
            true = list(y_true)
        
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_prot])
        val_r2 = r2_score(y_true, y_pred_prot)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_prot), (-1)))[0], 4)
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson}')
        
        
        y_pred_avg = [(m + p) / 2 for m, p in zip(y_pred_mol, y_pred_prot)]
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_avg])
        val_r2 = r2_score(y_true, y_pred_avg)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_avg), (-1)))[0], 4)
        
        
        if update:
            max_valid_R2 = val_r2
            max_valid_pearsonb = val_pearson
            R2_epoch = epoch+1
            pearson_epoch = epoch+1
            best_pred = list(y_pred_avg)
            # torch.save(model.state_dict(), opt.modeldir+"emb_encoder_down.pth")

        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson},best: ({R2_epoch}) R2: {max_valid_R2}, ({pearson_epoch}) ,Pearson: {max_valid_pearson}')
        print('')
        
        
        
        y_pred_mol_copy  = y_pred_mol
        y_pred_prot_copy  = y_pred_prot
        y_true_copy  = y_true 
        
        print("balance")
        # Validation phase
        
        y_pred_mol  = [y_pred_mol[i] for i in b_test_index] 
        y_pred_prot  = [y_pred_prot[i] for i in b_test_index] 
        y_true  = [y_true[i] for i in b_test_index] 

    
    
        
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_mol])
        val_r2 = r2_score(y_true, y_pred_mol)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_mol), (-1)))[0], 4)
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson}')
        
        if update:
            max_valid_DRFP_R2b = val_r2
            max_valid_DRFP_pearsonb = val_pearson
        
        
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_prot])
        val_r2 = r2_score(y_true, y_pred_prot)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_prot), (-1)))[0], 4)
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson}')
        
        
        y_pred_avg = [(m + p) / 2 for m, p in zip(y_pred_mol, y_pred_prot)]
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_avg])
        val_r2 = r2_score(y_true, y_pred_avg)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_avg), (-1)))[0], 4)
        
        if update:
            max_valid_R2b = val_r2
            max_valid_pearsonb = val_pearson
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson},best:  R2: {max_valid_R2b} ,Pearson: {max_valid_pearsonb}')
        print('')
        
        
        y_pred_mol  = y_pred_mol_copy
        y_pred_prot  = y_pred_prot_copy
        y_true  = y_true_copy
        print("imbalance")
        # Validation phase
        y_pred_mol  = [y_pred_mol[i] for i in imb_test_index]
        y_pred_prot  = [y_pred_prot[i] for i in imb_test_index] 
        y_true  = [y_true[i] for i in imb_test_index] 
    
    
        
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_mol])
        val_r2 = r2_score(y_true, y_pred_mol)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_mol), (-1)))[0], 4)
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson}')
        
        if update:
            max_valid_DRFP_R2imb = val_r2
            max_valid_DRFP_pearsonimb = val_pearson
        
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_prot])
        val_r2 = r2_score(y_true, y_pred_prot)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_prot), (-1)))[0], 4)
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson}')
        
        
        y_pred_avg = [(m + p) / 2 for m, p in zip(y_pred_mol, y_pred_prot)]
        val_mse = mean_squared_error([np.log10(pow(2, x)) for x in y_true], [np.log10(pow(2, y)) for y in y_pred_avg])
        val_r2 = r2_score(y_true, y_pred_avg)
        val_pearson = np.round(stats.pearsonr(np.array(y_true), np.reshape(np.array(y_pred_avg), (-1)))[0], 4)
        
        if update:
            max_valid_R2imb = val_r2
            max_valid_pearsonimb = val_pearson
        
        
        
        print(f'MSE: {val_mse}, R2: {val_r2}, Pearson: {val_pearson},best:  R2: {max_valid_R2imb},Pearson: {max_valid_pearsonimb}')
        print('\n')

        scheduler.step()
    
    
    print_sim_score(true,best_pred)
    print("max_valid_DRFP_R2 =", max_valid_DRFP_R2)
    print("max_valid_DRFP_R2b =", max_valid_DRFP_R2b)
    print("max_valid_DRFP_R2imb =", max_valid_DRFP_R2imb)  
    
    print("max_valid_R2 =", max_valid_R2)
    print("max_valid_R2b =", max_valid_R2b)
    print("max_valid_R2imb =", max_valid_R2imb)





if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./kcat/TurNuP4/data/', help='data文件夹位置')
    parser.add_argument('--resultdir', type=str, default='./kcat/kcat_2024/result/', help='结果保存文件夹位置')
    parser.add_argument('--modeldir', type=str, default='./kcat/kcat_2024/save_model/', help='模型和超参数文件夹位置')
    parser.add_argument('--para', type=str, default='', help='模型参数名称')
    parser.add_argument('--load_dict', default=False, help='是否载入模型参数')
    parser.add_argument('--save_name', default='0608_21', help='模型保存名字')
    
    parser.add_argument('--mol_input_dim', nargs='+', type=int, default=1024, help='分子指纹维度')
    parser.add_argument('--prot_input_dim', nargs='+',type=int, default=2560, help='氨基酸序列维度')
    parser.add_argument('--mol_embed_dim', nargs='+',type=int, default=1024, help='分子隐层维度')
    parser.add_argument('--prot_embed_dim', nargs='+',type=int, default=512, help='氨基酸序列隐层维度')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='MLP隐层维度')
    parser.add_argument('--mlp_hidden2', type=int, default=256, help='MLP隐层维度')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--dropout_mol',  default=0.1, help='mol dropout系数')
    parser.add_argument('--dropout_prot',  default=0.8, help='prot dropout系数 ')
    parser.add_argument('--dropout_mlp', default=0.2, help='mlp dropout系数')
    parser.add_argument('--layer_mol', type=int, default=2, help='mol学习层数')
    parser.add_argument('--layer_prot', type=int, default=2, help='prot学习层数')
    
    parser.add_argument('--batch_size', nargs='+',type=int, default=32, help='batch大小')
    parser.add_argument('--lr',default=0.00005, help='学习率，控制模型参数更新的步长')
    parser.add_argument('--lr_decay', default=0.9,help='学习率衰减因子，控制学习率在训练过程中的衰减')
    parser.add_argument('--decay_interval', type=int, default=15, help='学习率衰减 的间隔，表示在多少个epoch之后对学习率进行衰减')
    parser.add_argument('--weight_decay_1', default=0.005, help=' L1正则化')
    parser.add_argument('--weight_decay_2', default=0.001, help=' L2正则化')
    parser.add_argument('--epoch', type=int, default=100, help='训练的迭代次数，表示训练过程中循环的epoch数')
    

    opt = parser.parse_args()


    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')


    (mol_input_dim, prot_input_dim,batch_size, mol_embed_dim,prot_embed_dim, mlp_hidden_dim, output_dim) = map(int, [opt.mol_input_dim, opt.prot_input_dim,opt.batch_size, opt.mol_embed_dim,opt.prot_embed_dim, opt.mlp_hidden_dim, opt.output_dim])
    

    data_train = pd.read_pickle(opt.datadir+"kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle(opt.datadir+"kcat_data/splits/test_kcat.pkl")

    data_train_balance = data_train[data_train.apply(lambda row: row["balance"]==True,axis=1)]
    data_train_imbalance = data_train[data_train.apply(lambda row: row["balance"]==False,axis=1)]

    data_test_balance = data_test[data_test.apply(lambda row: row["balance"]==True,axis=1)]
    data_test_imbalance = data_test[data_test.apply(lambda row: row["balance"]==False,axis=1)]
    


    b_test_index = data_test.index[data_test['balance'] == True].tolist()
    imb_test_index = data_test.index[data_test['balance'] == False].tolist()


    # train_prot = torch.tensor(np.array(list(data_train["ESM1b"])), dtype=torch.float32)
    train_prot = data_train["ESM2"].tolist()
    train_prot = torch.stack(train_prot)
    train_mol = torch.tensor(np.array(list(data_train["DRFP_ae_2d_fl_4_ft_all2"])), dtype=torch.float32)
    train_kcat = torch.tensor(np.array(list(data_train["log2_kcat"])), dtype=torch.float32)


    # test_prot = torch.tensor(np.array(list(data_test["ESM1b"])), dtype=torch.float32)
    test_prot = data_test["ESM2"].tolist()
    test_prot = torch.stack(test_prot)
    test_mol = torch.tensor(np.array(list(data_test["DRFP_ae_2d_fl_4_ft_all2"])), dtype=torch.float32)
    test_kcat = torch.tensor(np.array(list(data_test["log2_kcat"])), dtype=torch.float32)
    
    
    train_dataset = TensorDataset(train_mol, train_prot, train_kcat)
    test_dataset = TensorDataset(test_mol, test_prot, test_kcat)
    valid_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - valid_size
    
    #test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset, [test_size, valid_size])
    valid_dataset =test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)


    # 初始化模型
    model = MoleculeProteinModel(mol_input_dim, prot_input_dim, mol_embed_dim,prot_embed_dim, mlp_hidden_dim, output_dim)
    model = model.to(device)

    if opt.load_dict:
        model = load_model(model, opt.modeldir+opt.para+'.pth')
    
    num_epochs = opt.epoch
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr = opt.lr,weight_decay=opt.weight_decay_2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_interval, gamma=opt.lr_decay)
    criterion = nn.MSELoss()

    train_model(model, train_loader, valid_loader, test_loader, optimizer, scheduler,num_epochs, device)
    
    # save_hyperparameters(opt,opt.modeldir+opt.save_name+'.txt')

    print('dl_DRFP_ae_2d_fl_4_ft_all_sim')






























