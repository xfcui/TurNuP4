import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# 选择可用的倒数第二张GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        device = torch.device(f'cuda:{4}')
        print(f'使用GPU {4}')
    else:
        device = torch.device('cuda:0')
        print('只有一个GPU，使用GPU 0')
else:
    device = torch.device('cpu')
    print('CPU！！')

# 定义编码器模型
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.embedding_bag = nn.EmbeddingBag(2051, 4, mode='mean')
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(2048 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024)  # 最终压缩到1024维
        )

    def forward(self, x):
        batch_size, num_vectors, vector_length = x.size()
        
        x = x.view(batch_size * num_vectors, vector_length)  # 展平
         
        x = self.embedding_bag(x)
        
        x = x.view(batch_size, -1)  # 恢复批次大小维度
        
        x = self.encoder_fc(x)
        
        return x

# 加载测试数据
df = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/finish_propress.pkl")
data_test = torch.tensor(np.array(list(df["lab_pos"])), dtype=torch.int)
data_test_tensor = data_test.to(device)

# 实例化编码器模型
encoder = Encoder().to(device)

# 加载预训练权重
embedding_bag_state_dict = torch.load('./kcat/TurNuP4/AE/encoder/embedding_bag_0802.pth')
encoder_fc_state_dict = torch.load('./kcat/TurNuP4/AE/encoder/encoder_fc_0802.pth')

# 手动加载encoder_fc部分的权重
new_encoder_fc_state_dict = encoder.encoder_fc.state_dict()

# 加载第一个线性层的权重
new_encoder_fc_state_dict['0.weight'] = encoder_fc_state_dict['0.weight']
new_encoder_fc_state_dict['0.bias'] = encoder_fc_state_dict['0.bias']

# 加载第二个线性层的权重
new_encoder_fc_state_dict['2.weight'] = encoder_fc_state_dict['2.weight']
new_encoder_fc_state_dict['2.bias'] = encoder_fc_state_dict['2.bias']

encoder.embedding_bag.load_state_dict(embedding_bag_state_dict)
encoder.encoder_fc.load_state_dict(new_encoder_fc_state_dict)


# 对测试集数据进行编码
encoder.eval()
with torch.no_grad():
    encoded_data_test = encoder(data_test_tensor).cpu().detach().numpy()
    df['DRFP_ae_2d_fl_4_ft_all2'] = [np.array(encoded_data_test[i, :]) for i in range(encoded_data_test.shape[0])]

# 保存编码后的测试数据
df.to_pickle("./kcat/TurNuP4//data/kcat_data/finish_propress.pkl")

print('编码完成')


















