
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



seq_data = pd.read_pickle('./kcat/TurNuP4/data/' + "kcat_data/finish_propress.pkl")
esm2 = pd.read_pickle('./kcat/TurNuP4/data/' + "kcat_data/with_esm.pkl")

seq_data['ESM2_1280'] = esm2['ESM2_1280']
seq_data['ESM2'] = esm2['ESM2']


print("匹配完esm2_1280： ",len(seq_data))

    
seq_data.to_pickle('./kcat/TurNuP4/data/'+"kcat_data/finish_propress.pkl")
pass
















