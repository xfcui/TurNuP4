import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

# 选择可用的倒数第二张GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        device = torch.device(f'cuda:{3}')
        print(f'使用GPU {3}')
    else:
        device = torch.device('cuda:0')
        print('只有一个GPU，使用GPU 0')
else:
    device = torch.device('cpu')
    print('CPU！！')

# reactions = pd.read_pickle('./kcat/data/' + "kcat_data/reaction_scp.pkl")
# df = pd.read_pickle("./kcat/data/kcat_data/finish_propress.pkl")

# reactions = pd.read_pickle("./kcat/data/kcat_data/finish_propress.pkl")
reactions = pd.read_pickle('./kcat/data/'+"kcat_data/splits/train_kcat.pkl")
df = pd.read_pickle('./kcat/data/'+"kcat_data/splits/test_kcat.pkl")

reactions = reactions[reactions.apply(lambda row: row["balance"]==True,axis=1)]
df = df[df.apply(lambda row: row["balance"]==True,axis=1)]

data = torch.tensor(np.array(list(reactions["lab_pos"])), dtype=torch.int)
label = torch.tensor(np.array(list(reactions["four"])), dtype=torch.float32)
data_tensor = data
label_tensor = label

# 定义测试集
data_test = torch.tensor(np.array(list(df["lab_pos"])), dtype=torch.int)
label_test = torch.tensor(np.array(list(df["four"])), dtype=torch.float32)
data_test_tensor = data_test.to(device)
label_test_tensor = label_test.to(device)

# 创建数据加载器
batch_size = 256
dataset = TensorDataset(data_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义分类任务模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.embedding_bag = nn.EmbeddingBag(2051, 4, mode='mean')
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(2048 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048 * 4)  # 输出2048 * 4的向量
        )

    def forward(self, x):
        batch_size, num_vectors, vector_length = x.size()
        
        x = x.view(batch_size * num_vectors, vector_length)  # 展平
         
        x = self.embedding_bag(x)
        
        x = x.view(batch_size, -1)  # 恢复批次大小维度
        
        FPs = self.encoder_fc[0:4](x)
        
        x = self.encoder_fc(x)
        
        return x.view(batch_size, 2048, 4),FPs

# 实例化模型、损失函数和优化器
model = Classifier()

model.embedding_bag.load_state_dict(torch.load('./kcat/kcat_2024/AE/encoder/embedding_bag_0727_4.pth', map_location=device))
model.encoder_fc.load_state_dict(torch.load('./kcat/kcat_2024/AE/encoder/encoder_fc_0727_4.pth', map_location=device))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)

def calculate_metrics(output, target):
    model.eval()
    with torch.no_grad():
        output = output.argmax(dim=-1)
        target = target.argmax(dim=-1)
        
        recalls = recall_score(target.cpu().flatten(), output.cpu().flatten(), average=None)
        conf_matrix = confusion_matrix(target.cpu().flatten(), output.cpu().flatten())
        
    model.train()
    return recalls, conf_matrix

# 计算测试集指标的函数
def evaluate_model(model, data_test_tensor, label_test_tensor):
    model.eval()
    with torch.no_grad():
        output,_ = model(data_test_tensor)
        output = output.view(-1, 4)
        label_test_tensor = label_test_tensor.view(-1, 4)
        
        loss = criterion(output, label_test_tensor.argmax(dim=-1)).item()
        recalls, conf_matrix = calculate_metrics(output, label_test_tensor)
        
    model.train()
    return loss, recalls, conf_matrix

min_loss = float('inf')
num_epochs = 1001
for epoch in range(num_epochs):
    epoch_loss = 0.0
    train_recalls = np.zeros(4)
    cm_TP = np.zeros(4)
    cm_FP = np.zeros(4)
    cm_FN = np.zeros(4)
    cm_TN = np.zeros(4)
    cal_train = False
    if epoch % 50 == 0 and epoch > 45:
        cal_train = True
    for data_batch, data_label in dataloader:
        data_batch = data_batch.to(device)
        data_label = data_label.to(device)

        # 前向传播
        output,_ = model(data_batch)
        loss = criterion(output.view(-1, 4), data_label.view(-1, 4).argmax(dim=-1))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if cal_train:
            recalls, cm = calculate_metrics(output, data_label)
            train_recalls += recalls
            cm_TP += np.diag(cm)
            cm_FP += cm.sum(axis=0) - np.diag(cm)
            cm_FN += cm.sum(axis=1) - np.diag(cm)
            cm_TN += cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
        
    if cal_train:    
        train_recalls /= len(dataloader)
        cm_all = cm_TP + cm_FP + cm_FN + cm_TN
        cm_TP /= cm_all
        cm_FP /= cm_all
        cm_FN /= cm_all
        cm_TN /= cm_all   
    epoch_loss /= len(dataloader)     
  

    
        
    # 测试集评估
    test_loss, test_recalls, cm = evaluate_model(model, data_test_tensor, label_test_tensor)
    
    
    # if epoch > 200 and test_loss < min_loss:
    #     min_loss = test_loss
    #     model.eval()
    #     with torch.no_grad():
    #         # df_now = pd.read_pickle("./kcat/data/kcat_data/finish_propress.pkl")
    #         # data_now = torch.tensor(np.array(list(df_now["lab_pos"])), dtype=torch.int)
    #         # data_now = data_now.to(device)
    #         # _,encoded_data_test = model(data_now)
    #         # encoded_data_test = encoded_data_test.cpu().detach().numpy()
    #         # df_now['DRFP_ae_2d_fl_4_ft_all2'] = [np.array(encoded_data_test[i, :]) for i in range(encoded_data_test.shape[0])]
    #         # df_now.to_pickle("./kcat/data/kcat_data/finish_propress.pkl")
    #         torch.save(model.embedding_bag.state_dict(), './kcat/kcat_2024/AE/encoder/embedding_bag_0802.pth')
    #         torch.save(model.encoder_fc.state_dict(), './kcat/kcat_2024/AE/encoder/encoder_fc_0802.pth')
    #     model.train()
    
    
    if cal_train:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.10f}, Train Recalls: {train_recalls}')
        print(f'Train TP: {cm_TP}, Train FP: {cm_FP}, Train FN: {cm_FN}, Train TN: {cm_TN}')
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.10f}')
    
    cm_TP = np.diag(cm)
    cm_FP = cm.sum(axis=0) - np.diag(cm)
    cm_FN = cm.sum(axis=1) - np.diag(cm)
    cm_TN = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    cm_all = cm_TP + cm_FP + cm_FN + cm_TN

    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.10f}, Test Recalls: {test_recalls}')
    for i in range(4):
        print(f'Test Class {i} - TP: {cm_TP[i]/cm_all[i]:.10f}, FP: {cm_FP[i]/cm_all[i]:.10f}, FN: {cm_FN[i]/cm_all[i]:.10f}, TN: {cm_TN[i]/cm_all[i]:.10f}')
    
    
    scheduler.step()

print('训练结束')






























