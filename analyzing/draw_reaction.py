from drfp import DrfpEncoder
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
from rdkit.Chem.Draw import rdMolDraw2D
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
from rdkit.Chem import rdFMCS

# reaction_data = pd.read_pickle("./kcat/data/kcat_data/finish_propress.pkl")

# def get_reaction_site_smiles(metabolites):
#     reaction_site = ""
#     for met in metabolites:
#         mol = Chem.inchi.MolFromInchi(met)
#         if mol is not None:
#             Smiles = Chem.MolToSmiles(mol)
#         else:
#             return (np.nan)
#         reaction_site = reaction_site + "." + Smiles
#     return (reaction_site[1:])

# index_list = list(reaction_data.index)
# i=index_list[466]

# substrates = list(reaction_data["sub_ids"][i])
# products = list(reaction_data["pro_ids"][i])
# reation = (tuple(substrates),tuple(products))

# left_site = get_reaction_site_smiles(substrates)
# right_site = get_reaction_site_smiles(products)

# reaction_smiles = left_site + ">>" + right_site

# 定义反应SMILES
# reaction_smiles = "O=CC1=CC=CC=C1>>O=C(O)C1=CC=CC=C1"
reaction_smiles = "CCOC(=O)C.O >> CC(=O)O.CCO"

# 解析反应SMILES
rxn = Chem.rdChemReactions.ReactionFromSmarts(reaction_smiles)



# # 获取反应物、试剂和产物
# reactants = [rxn.GetReactants()[i] for i in range(rxn.GetNumReactantTemplates())]
# products = [rxn.GetProducts()[i] for i in range(rxn.GetNumProductTemplates())]

# 创建一个新的绘图对象
drawer = rdMolDraw2D.MolDraw2DCairo(500, 200)
drawer.DrawReaction(rxn)
drawer.FinishDrawing()

# 保存图像
with open("reaction.png", "wb") as f:
    f.write(drawer.GetDrawingText())

print("反应结构图已保存为reaction.png")


# # 设置绘图参数
# drawOptions = rdMolDraw2D.MolDrawOptions()
# drawOptions.addAtomIndices = True
# drawOptions.includeAtomNumbers = False
# drawOptions.clearBackground = False

# drawOptions.atomLabelFontSize = 0.5
# drawOptions.bondLineWidth = 1

# # 创建一个新的绘图对象
# drawer = rdMolDraw2D.MolDraw2DSVG(100, 50)
# drawer.SetDrawOptions(drawOptions)
# drawer.DrawReaction(rxn)
# drawer.FinishDrawing()

# # 获取SVG文本
# svg = drawer.GetDrawingText()

# # 将SVG文本保存到文件
# with open("reaction2.svg", "w") as f:
#     f.write(svg)

# print("反应结构图已保存为reaction.svg")



# common_structures_smiles = ['[CH3][CH2]', 'C=O', 'C-C-O']
# unique_reactants_smiles = ["COO", 'C-O-C', 'OCC']
# unique_products_smiles = ["O[H]", 'O=C-O']

# # 合并所有结构以便迭代
# all_structures = common_structures_smiles + unique_reactants_smiles + unique_products_smiles

# # 创建子结构图并保存
# for idx, smiles in enumerate(all_structures):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol:
#         img = Draw.MolToImage(mol, size=(300, 300))  # 绘制300x300的图片
#         img.save(f"structure_{idx+1}.png")
#     else:
#         print(f"Invalid SMILES: {smiles}")

# print("所有子结构图像已保存。")



# # # 定义反应物和产物的SMILES
# # reactant_smiles = ["O=CC1=CC=CC=C1"]
# # product_smiles = ["O=C(O)C1=CC=CC=C1"]

# # # 生成反应物和产物分子对象
# # reactants = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles]
# # products = [Chem.MolFromSmiles(smiles) for smiles in product_smiles]

# # # 手动定义共同结构和独有结构的SMILES
# common_structure_smiles = ["CCCl"]  # 苯环
# # unique_reactants_smiles = ["O=CC"]  # 甲酰基
# # unique_products_smiles = ["O=C(O)C"]  # 羧基

# # # 生成共同结构和独有结构的分子对象
# common_structure = [Chem.MolFromSmiles(smiles) for smiles in common_structure_smiles]
# # unique_reactants = [Chem.MolFromSmiles(smiles) for smiles in unique_reactants_smiles]
# # unique_products = [Chem.MolFromSmiles(smiles) for smiles in unique_products_smiles]

# # 设置绘图参数
# def draw_molecules(mols, filename):
#     # 创建一个新的绘图对象，尺寸小一点
#     drawer = rdMolDraw2D.MolDraw2DSVG(400, 150)
#     # 设置绘图参数
#     drawOptions = drawer.drawOptions()
#     drawOptions.bondLineWidth = 1  # 线条更细
#     drawOptions.padding = 0.1  # 边距更小
#     drawOptions.additionalAtomLabelPadding = 0.01  # 原子标签间距更小
#     drawer.SetDrawOptions(drawOptions)
#     drawer.DrawMolecules(mols)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()
#     with open(filename, "w") as f:
#         f.write(svg)

# # # 绘制反应图
# # drawer = rdMolDraw2D.MolDraw2DSVG(400, 150)
# # drawOptions = drawer.drawOptions()
# # drawOptions.bondLineWidth = 1
# # drawOptions.padding = 0.1
# # drawOptions.additionalAtomLabelPadding = 0.01
# # drawer.SetDrawOptions(drawOptions)
# # drawer.DrawReaction(rxn)
# # drawer.FinishDrawing()
# # svg_reaction = drawer.GetDrawingText()
# # with open("reaction.svg", "w") as f:
# #     f.write(svg_reaction)

# # # # 绘制反应物
# # # draw_molecules(reactants, "reactants.svg")

# # # # 绘制产物
# # # draw_molecules(products, "products.svg")

# # # 绘制反应物独有结构
# # draw_molecules(unique_reactants, "unique_reactants.svg")

# # # 绘制产物独有结构
# # draw_molecules(unique_products, "unique_products.svg")

# # 绘制共同结构
# draw_molecules(common_structure, "common_structure.svg")

# print("反应物、产物、独有结构和共同结构的图已保存为SVG文件。")





# import nglview as nv
# from Bio.PDB import PDBList, PDBParser
# import matplotlib.pyplot as plt
# import io

# # 下载PDB文件
# pdb_code = '1A4Y'  # 示例PDB ID，可以替换成其他酶的PDB ID
# pdbl = PDBList()
# pdb_file = pdbl.retrieve_pdb_file(pdb_code, file_format='pdb', pdir='.')

# # 解析PDB文件
# parser = PDBParser(QUIET=True)
# structure = parser.get_structure(pdb_code, pdb_file)

# # 可视化
# view = nv.show_biopython(structure)
# view.add_cartoon(color='spectrum')

# # 渲染图像并保存为PNG格式
# image_data = view.render_image(factor=4, trim=True, transparent=True)

# # 使用Matplotlib保存PNG图像
# image_stream = io.BytesIO(image_data)
# image = plt.imread(image_stream, format='png')

# # 保存为PNG文件
# plt.imsave('enzyme_structure.png', image)
# print("PNG file 'enzyme_structure.png' has been saved.")



















