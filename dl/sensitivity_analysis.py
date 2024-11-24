import argparse
import numpy as np
import torch
from bioservices import *
import warnings
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset




def l1_penalty(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

def save_model(model, path):
    state = {
        'model': model.state_dict()
    }
    torch.save(state, path)


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

        # Molecule feature extraction
        # self.mol_fc1 = nn.Linear(mol_input_dim, mol_input_dim)
        # self.mol_ln1 = nn.LayerNorm(mol_input_dim)
        self.mol_fc2 = nn.Linear(mol_input_dim, mol_embed_dim)
        self.mol_ln2 = nn.LayerNorm(mol_embed_dim)
        self.mol_fc3 = nn.Linear(mol_embed_dim, 1024)
        self.mol_ln3 = nn.LayerNorm(1024)
        self.mol_fc4 = nn.Linear(1024, output_dim)
        
        # Protein feature extraction
        # self.prot_fc1 = nn.Linear(prot_input_dim, prot_input_dim)
        # self.prot_ln1 = nn.LayerNorm(prot_input_dim)
        self.prot_fc2 = nn.Linear(prot_input_dim, prot_embed_dim)
        self.prot_ln2 = nn.LayerNorm(prot_embed_dim)
        self.prot_fc3 = nn.Linear(prot_embed_dim, prot_embed_dim)
        self.prot_ln3 = nn.LayerNorm(prot_embed_dim)
        self.prot_fc4 = nn.Linear(prot_embed_dim, prot_embed_dim)
        self.prot_ln4 = nn.LayerNorm(prot_embed_dim)
        self.prot_fc5 = nn.Linear(prot_embed_dim, output_dim)
        
        # self.prot_conv1 = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, padding=1)
        # self.prot_ln1 = nn.LayerNorm(embed_dim)
        # self.prot_conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        # self.prot_ln2 = nn.LayerNorm(embed_dim)

        self.mol_dropout = nn.Dropout(opt.dropout_mol)
        self.prot_dropout = nn.Dropout(opt.dropout_prot)
        
        self.dropout = nn.Dropout(0.5)

        
        self.encoder_fc = nn.Sequential(
            nn.Linear(2048 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024)  # 最终压缩到1024维
        )
        
        # MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(prot_embed_dim,opt.mlp_hidden2),
            #nn.Linear(mol_embed_dim+prot_embed_dim,opt.mlp_hidden2),

            nn.Linear(opt.mlp_hidden2, output_dim)
        )

    def mol_forward(self, x):
        # Molecule feature extraction
        
        batch_size, num_vectors, vector_length = x.size()
        
        # x = x.view(batch_size * num_vectors, vector_length)  # 展平
         
        # x = self.embedding_bag(x)
        
        x = x.view(batch_size, -1)  # 恢复批次大小维度
        
        mol_input = self.encoder_fc(x)
        
        mol_features = F.relu(self.mol_fc2(mol_input))
        
        mol_features = self.mol_ln2(mol_features)
        mol_features = self.mol_dropout(mol_features)
        mol_features = F.relu(self.mol_fc3(mol_features))
        mol_features = self.mol_ln3(mol_features)
        # mol_features = F.relu(self.mol_fc3(mol_features))
        # # # mol_features = self.mol_dropout(mol_features)
        # mol_features = self.mol_ln3(mol_features)
        mol_features = self.mol_dropout(mol_features)
        mol_output = self.mol_fc4(mol_features)
        
        return mol_output

    def forward(self, x, prot_input):
        # Molecule feature extraction
        
        batch_size, num_vectors, vector_length = x.size()
        
        # x = x.view(batch_size * num_vectors, vector_length)  # 展平
         
        # x = self.embedding_bag(x)
        
        x = x.view(batch_size, -1)  # 恢复批次大小维度
        
        mol_input = self.encoder_fc(x)
        
        mol_features = F.relu(self.mol_fc2(mol_input))
        
        mol_features = self.mol_ln2(mol_features)
        mol_features = self.mol_dropout(mol_features)
        mol_features = F.relu(self.mol_fc3(mol_features))
        mol_features = self.mol_ln3(mol_features)
        # mol_features = F.relu(self.mol_fc3(mol_features))
        # # # mol_features = self.mol_dropout(mol_features)
        # mol_features = self.mol_ln3(mol_features)
        mol_features = self.mol_dropout(mol_features)
        mol_output = self.mol_fc4(mol_features)
        
        # Protein feature extraction
        
        # prot_features = F.relu(self.prot_fc1(prot_input))
        # prot_features = self.prot_ln1(prot_features)  # 应用 LayerNorm
        
        prot_features = F.relu(self.prot_fc2(prot_input))
        # prot_features = self.prot_ln2(prot_features)  # 应用 LayerNorm
        prot_features = self.prot_dropout(prot_features)  # 应用 Dropout
        
        # prot_features = F.relu(self.prot_fc3(prot_features))
        # prot_features = self.prot_ln3(prot_features)  # 应用 LayerNorm
        # prot_features = self.prot_dropout(prot_features)
        prot_output = self.prot_fc5(prot_features)
        

        
        # Concatenate features
        #combined_features = torch.cat((mol_features, prot_features), dim=-1)
        combined_features = prot_features
        #combined_features = attn_output_rev
        
        # MLP for regression
        output = self.mlp(combined_features)
        
        return mol_output , prot_output,output


def model_predict(input):
    input1 = torch.tensor(input, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model.mol_forward(input1)
        output = output.numpy()
        return output

def analysis_w(model,data_test):
    model.eval()
    point_sub = 0
    point_pro = 0
    point_all = 0
    point_none = 0
    
    list_sub = []
    list_pro = []
    list_all = []
    list_none = []
    
    labels_mol = np.array(list(data_test["lab_pos"]))
    for i in range(len(data_test)):
        
        label = labels_mol[i,:,0].astype(int)
        
        input1 = test_mol[i].unsqueeze(0) # 添加一个维度以表示 batch
        # input2 = test_prot[50].unsqueeze(0).numpy()
        input1.requires_grad = True 
        
        output = model.mol_forward(input1)
        output.backward()
        
        input_grad = input1.grad
        input_grad_np = input_grad.detach().numpy()
        
        input_grad_abs = np.abs(input_grad_np)
        input_grad_abs = np.sum(input_grad_abs,axis=2).squeeze()
        
        # input_grad_norm = input_grad_abs - min(input_grad_abs)
        # input_grad_abs = input_grad_norm/max(input_grad_norm)
        
        
        # input_grad_sum = np.sum(input_grad_np,axis=2)
        # input_grad_abs = np.abs(input_grad_sum).squeeze()
        
        indices_2048 = np.where(label == 2048)[0]
        indices_2049 = np.where(label == 2049)[0]
        indices_2050 = np.where(label == 2050)[0]
        indices_0 = np.where(label == 0)[0]

        if len(indices_2048)==0:
            continue
        if len(indices_2049)==0:
            continue
        
        
        # 获取另一个向量的对应子部分
        sub_vector_2048 = input_grad_abs[indices_2048]
        sub_vector_2049 = input_grad_abs[indices_2049]
        sub_vector_2050 = input_grad_abs[indices_2050]
        sub_vector_0 = input_grad_abs[indices_0]
        
        mean_2048 = np.mean(sub_vector_2048)
        mean_2049 = np.mean(sub_vector_2049)
        mean_2050 = np.mean(sub_vector_2050)
        mean_0 = np.mean(sub_vector_0)
        

        
        # print('sub_only:   ',sum_2048)
        # print('pro_only:   ',sum_2049)
        # print('all_have:   ',sum_2050,'\n\n')
        
        list_sub.append(mean_2048)
        list_pro.append(mean_2049)
        list_all.append(mean_2050)
        list_none.append(mean_0)
        
        # max_point = max(mean_2048,mean_2049,mean_2050)
        max_point = max(mean_2048,mean_2049,mean_2050,mean_0)
        if mean_2048==max_point:
            point_sub+=1
        elif mean_2049==max_point:
            point_pro+=1
        else:
            point_all+=1
        

    none_contri = sum(list_none)/len(list_none)
    print('sub_only:   ',(sum(list_sub)/len(list_sub))/none_contri)
    print('pro_only:   ',(sum(list_pro)/len(list_pro))/none_contri)
    # print('none:   ',sum(list_none)/len(list_none))
    print('all_have:   ',(sum(list_all)/len(list_all))/none_contri,'\n')


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/', help='data文件夹位置')
    parser.add_argument('--resultdir', type=str, default='./result/', help='结果保存文件夹位置')
    parser.add_argument('--modeldir', type=str, default='./save_model/', help='模型和超参数文件夹位置')
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
    parser.add_argument('--dropout_mol',  default=0.5, help='mol dropout系数')
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
    
    
    
    # data_train = data_train_balance
    # data_test = data_test_imbalance

    b_test_index = data_test.index[data_test['balance'] == True].tolist()
    imb_test_index = data_test.index[data_test['balance'] == False].tolist()
    

    
    
    # train_prot = torch.tensor(np.array(list(data_train["ESM1b"])), dtype=torch.float32)
    train_prot = data_train["ESM2"].tolist()
    train_prot = torch.stack(train_prot)
    train_mol = torch.tensor(np.array(list(data_train["DRFP_ae_2d_fl_ft_all2_emb"])), dtype=torch.float32)
    train_kcat = torch.tensor(np.array(list(data_train["log2_kcat"])), dtype=torch.float32)


    # test_prot = torch.tensor(np.array(list(data_test["ESM1b"])), dtype=torch.float32)
    test_prot = data_test["ESM2"].tolist()
    test_prot = torch.stack(test_prot)
    test_mol = torch.tensor(np.array(list(data_test["DRFP_ae_2d_fl_ft_all2_emb"])), dtype=torch.float32)
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

    
    model.load_state_dict(torch.load(opt.modeldir+"emb_encoder.pth"))
    
    device = torch.device('cpu')
    model.to(device)
    
    # print('测试集分析')
    # analysis_w(model,data_test)
    
    print('sensitivity analysis on balance data')
    analysis_w(model,data_test_balance)
    
    print('sensitivity analysis on imbalance data')
    analysis_w(model,data_test_imbalance)
































