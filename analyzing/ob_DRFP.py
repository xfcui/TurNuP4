from rdkit import Chem
from rdkit.Chem  import  Draw, BRICS, AllChem
from rdkit.Chem.Draw  import  IPythonConsole 
from rdkit.Chem  import  rdChemReactions 
import torch as t
import pandas as pd
import numpy as np
from rdkit.Chem.Lipinski import RotatableBondSmarts
from drfp import DrfpEncoder




# sub = 'CC(=O)OC1=CC=CC=C1C(=O)O'
# pro = 'CC(=O)O.CC(=O)C'


# D = DrfpEncoder.encode(sub + ">>" + pro)[0]



# B = DrfpEncoder.encode(sub + ">>" + '')[0]



# P = DrfpEncoder.encode('' + ">>" + pro)[0]

data = pd.read_pickle('./kcat/data/' + "kcat_data/finish_propress.pkl")

data['2A2b2p'] = data.apply(lambda row: row['DRFP_del2'] + row['DRFP_del'] + row['DRFP'], axis=1)

# wdf= A2[np.any(A2 == 1, axis=1)]

# A2 = A2.T
# wdf2= A2[np.any(A2 == 1, axis=1)]


sub = np.array(list(data['DRFP_del']))
dr = np.array(list(data['DRFP']))
pro = np.array(list(data['DRFP_del2']))
all2 = np.array(list(data['2A2b2p']))

#将有问题的位提取出来
error = np.zeros_like(all2)
error[all2 == 3] = 1

all2[all2 == 3] = 2
all1 = (0.5)*all2

sub_only = all1 - pro
pro_only = all1 - sub
all_have = all1 - dr

#确定将这部分（结构）归到哪一类
# sub_only = sub_only + error
# pro_only = pro_only + error
# all_have = all_have + error

label = 2048*sub_only + 2049*pro_only + 2050*all_have


rows, cols = label.shape
row_pattern = np.arange(0, cols)

# 使用 numpy.tile 将该一维数组在行方向上复制多次
pos = np.tile(row_pattern, (rows, 1))

lab_pos = np.dstack((label, pos))

data['lab_pos'] = list(lab_pos)

data.to_pickle('./kcat/data/'+"kcat_data/finish_propress.pkl")













pass
























