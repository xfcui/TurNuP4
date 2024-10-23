
'''import torch
import esm
import os

# 下载预训练的ESM2模型


# 创建保存模型的目录
model_dir = "save_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 保存模型和alphabet
torch.save(model.state_dict(), os.path.join(model_dir, "esm2_model.pth"))
with open(os.path.join(model_dir, "alphabet.pkl"), "wb") as f:
    import pickle
    pickle.dump(alphabet, f)

print("Model and alphabet saved successfully.")'''



import torch
import esm
import os
import pandas as pd

#model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model.eval()

#seq_data = pd.read_pickle('./kcat/data/' + "kcat_data/fp_kcat_data.pkl")
seq_data = pd.read_pickle('./kcat/data/' + "kcat_data/finish_propress.pkl")


seq_data['ESM2'] = ''

for i in seq_data.index:

    # 未知的氨基酸序列
    unknown_sequence = seq_data['Sequence'][i]

    # 准备数据，用 "unknown" 作为名称
    batch_converter = alphabet.get_batch_converter()
    data = [("sequence", unknown_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # 获取特征向量
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]

    # 提取序列的特征向量
    # 注意: token_representations 是 (batch_size, seq_len, hidden_dim) 的形状
    sequence_representation = token_representations[0, 1: len(unknown_sequence) + 1].mean(0)
    seq_data['ESM2'][i] = sequence_representation  #2560  1280

seq_data = seq_data.loc[seq_data["ESM2"] != ""]

print("匹配完esm2： ",len(seq_data))
    
#seq_data.to_pickle('./kcat/data/'+"kcat_data/fp_kcat_data.pkl")
seq_data.to_pickle('./kcat/data/'+"kcat_data/finish_propress.pkl")
pass
















