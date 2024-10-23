
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import colorbar
from matplotlib.colors import LinearSegmentedColormap
 
# #练习的数据：
# #data=np.array([[0.3076,0.271,0.2786],[0.3636,0.3052,0.2932],[0.0282,0.0974,0.2036]])
# #data=np.array([[0.1172,0.1476,0.203],[0.083,0.111,0.1588],[0.2588,0.3044,0.3958]])
# #data=np.array([[0.1836,0.1782,0.1738],[0.0764,0.0322,0.0646],[0.206,0.2116,0.197]])
# #data=np.array([[0.3604,0.3042,0.3006],[0.3666,0.3208,0.2904],[0.3278,0.2934,0.3384]])
# #data=np.array([[0.2146,0.2308,0.2384],[0.1138,0.191,0.197],[0.2374,0.2388,0.2464]])
# #data=np.array([[0.2986,0.324,0.3724],[0.328,0.353,0.4088],[0.1286,0.1576,0.1666]])
# #data=np.array([[0.3464,0.3436,0.4012],[0.3222,0.3316,0.4006],[0.3634,0.398,0.395]])

# #data=np.array([[0.313,0.3056,0.354],[0.3712,0.3362,0.3944],[0.0232,0.152,0.153]])
# data=np.array([[0.3924,0.389,0.4108],[0.4376,0.426,0.4498],[0.1686,0.204,0.2148]])

# data=pd.DataFrame(data,columns=['reaction(mean)', 'sub(mean)', 'reaction+sub(mean)'], index=['all', 'balance', 'imbalance'])
 
# #绘制热度图：
# tick_=np.arange(0,1,5).astype(float)
# dict_={'orientation':'vertical',"label":"color  \
# scale","drawedges":True,"ticklocation":"right","extend":"min", \
# "filled":True,"alpha":5.8,"cmap":"cmap","ticks":tick_,"spaci,linewidths=0.5ng":'proportional'}

# #绘制添加数值和线条的热度图：
# cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'red'])
# sns.heatmap(data, cmap=cmap, linewidths=0.8, annot=True,vmin=0, vmax=0.5,annot_kws={"size": 15})
# plt.title("balance train set",size=20)
 
# #调整色带的标签：
# cbar = plt.gcf().axes[-1]  # 获取颜色条轴
# cbar.tick_params(labelsize=10, labelcolor="blue")
# cbar.set_ylabel(ylabel="Color Scale", size=10, color="red", loc="center")

# datadir = './kcat/data/training_results_same/' 
# plt.savefig(datadir+"b_rmsmrr+sm.png")


# 数据
data = {
    'DRFP': [0.2974, 0.3563, 0.006],
    'Substrate': [0.2796, 0.31, 0.1229],
    'Product': [0.2939, 0.3397, 0.0626],
    'NewFP': [0.3007, 0.3344, 0.1274]
}

index = ['complete', 'balance', 'imbalance']

# 将数据转换为DataFrame
df = pd.DataFrame(data, index=index)

# 创建热度图
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5,annot_kws={"size": 13},fmt=".3f")

# 添加标题和轴标签
plt.title('train with balance data')
plt.xlabel('FPS')
plt.ylabel('test Set part')

# 显示热度图
plt.tight_layout()
plt.show()

plt.savefig('./kcat/data/img_paper/img4/'+"b_train.png")
























