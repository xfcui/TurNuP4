import os
import pickle
import hashlib
import argparse
import requests
import numpy as np
import torch
from rdkit import Chem
from zeep import Client
from os.path import join
from loading import *
from rdkit.Chem import Crippen, AllChem
from rdkit.Chem import Descriptors
from urllib.request import urlopen, Request
from bioservices import *
import warnings
from rdkit.Chem import Draw
import pandas as pd
import random
from os.path import join
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from drfp import DrfpEncoder
CURRENT_DIR = os.getcwd()

def get_reaction_site_smarts(metabolites):
    reaction_site = ""
    for met in metabolites:
        mol = Chem.inchi.MolFromInchi(met)
        if mol is not None:
            Smarts = Chem.MolToSmarts(mol)
        else:
            return (np.nan)
        reaction_site = reaction_site + "." + Smarts
    return (reaction_site[1:])


def get_reaction_site_smiles(metabolites):
    reaction_site = ""
    for met in metabolites:
        mol = Chem.inchi.MolFromInchi(met)
        if mol is not None:
            Smiles = Chem.MolToSmiles(mol)
        else:
            return (np.nan)
        reaction_site = reaction_site + "." + Smiles
    return (reaction_site[1:])


def convert_fp_to_array(difference_fp_dict):
    fp = np.zeros(2048)
    for key in difference_fp_dict.keys():
        fp[key] = difference_fp_dict[key]
    return (fp)



def del_small(metabolites):
    if (len(metabolites) < 1):
        return metabolites
    biggest = []
    big_mw = 0
    for index in range(len(metabolites)):
        met = metabolites[index]
        if met != "":
            mol = Chem.inchi.MolFromInchi(met)
            mw = Descriptors.MolWt(mol)
            if (mw > big_mw):
                big_mw = mw
                biggest.append(met)
    if (big_mw == 0):
        return metabolites
    return [biggest[-1]]

    pass


def count_atoms(molecule_str):
    # 将分子字符串转换为 RDKit Mol 对象
    mol = Chem.MolFromSmiles(molecule_str)

    if mol is not None:
        # 获取每类原子的数目
        atom_counts = {}
        atom_nums = {}
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()

            # 将氢离子抹去
            if atom_symbol == 'H':
                continue

            if atom_symbol in atom_counts:
                atom_counts[atom_symbol] += 1

            else:
                atom_counts[atom_symbol] = 1
                atom_nums[atom_symbol] = 1

        return atom_counts, atom_nums
    else:
        return None


def is_reaction_balanced(atom_counts_l, atom_counts_r):
    if atom_counts_l is not None:
        return atom_counts_l == atom_counts_r
    else:
        return False



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.options.mode.chained_assignment = None
    """Hyperparameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./kcat/TurNuP4/data/', help='data文件夹位置')#'../data/'
    opt = parser.parse_args()


    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU！！')
    reaction_data = pd.read_pickle(opt.datadir+"kcat_data/preprocessed_kcat_data.pkl")
    #UID, Sequence, kcat, sub_ids, pro_ids, log10_kcat, reaction_max, UID_max, EC_max,kcat_max
    
    reaction_data["structural_fp"], reaction_data["difference_fp"], reaction_data["DRFP"] = "", "", ""
    reaction_data["structural_fp_del"], reaction_data["difference_fp_del"], reaction_data["DRFP_del"] = "", "", ""
    reaction_data["structural_fp_del2"], reaction_data["difference_fp_del2"], reaction_data["DRFP_del2"] = "", "", ""
    reaction_data["structural_fp_balance"], reaction_data["difference_fp_balance"], reaction_data[
        "DRFP_balance"] = "", "", ""
    reaction_data["structural_fp_imbalance"], reaction_data["difference_fp_imbalance"], reaction_data[
        "DRFP_imbalance"] = "", "", ""
    reaction_data["#substrates"], reaction_data["#products"] = "", ""
    reaction_data["balance"] = True
    #构建反应集合
    rea_str_dic = {}
    rea_diff_dic = {}
    rea_DRFP_dic = {}

    for i in reaction_data.index:
        substrates = list(reaction_data["sub_ids"][i])
        products = list(reaction_data["pro_ids"][i])
        reation = (tuple(substrates),tuple(products))
        try:
            left_site = get_reaction_site_smarts(substrates)
            right_site = get_reaction_site_smarts(products)
            if not pd.isnull(left_site) and not pd.isnull(right_site):

                rxn_forward = AllChem.ReactionFromSmarts(left_site + ">>" + right_site)
                if reation in rea_str_dic:
                    str_fp = rea_str_dic[reation]
                else:
                    str_fp = Chem.rdChemReactions.CreateStructuralFingerprintForReaction(
                        rxn_forward).ToBitString()
                    rea_str_dic[reation] = str_fp

                if reation in rea_diff_dic:
                    diff_fp = rea_diff_dic[reation]
                else:
                    diff_fp = Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_forward)
                    diff_fp = convert_fp_to_array(diff_fp.GetNonzeroElements())
                    rea_diff_dic[reation] = diff_fp


                left_site = get_reaction_site_smiles(substrates)
                right_site = get_reaction_site_smiles(products)
                if reation in rea_DRFP_dic:
                    drfp = rea_DRFP_dic[reation]
                else:
                    drfp = DrfpEncoder.encode(left_site + ">>" + right_site)[0]
                    rea_DRFP_dic[reation] = drfp

                atom_counts_l,_ = count_atoms(left_site)
                atom_counts_r,_ = count_atoms(right_site)
                if(is_reaction_balanced(atom_counts_l,atom_counts_r)):
                    reaction_data["DRFP_balance"][i] = drfp
                    reaction_data["structural_fp_balance"][i] = str_fp
                    reaction_data["difference_fp_balance"][i] = diff_fp
                else:
                    reaction_data["DRFP_imbalance"][i] = drfp
                    reaction_data["structural_fp_imbalance"][i] = str_fp
                    reaction_data["difference_fp_imbalance"][i] = diff_fp
                    reaction_data["balance"][i] = False
                
                reaction_data["DRFP"][i] = drfp
                reaction_data["structural_fp"][i] = str_fp
                reaction_data["difference_fp"][i] = -1 * diff_fp
                reaction_data["#substrates"][i] = len(substrates)
                reaction_data["#products"][i] = len(products)
        except IndexError:
            pass



    #######################################
        # substrates = del_small(substrates)
        # products = del_small(products)
        substrates = list(reaction_data["sub_ids"][i])
        products  = []
        reation = (tuple(substrates),tuple(products))
        try:

            left_site = get_reaction_site_smarts(substrates)
            right_site = get_reaction_site_smarts(products)
            if not pd.isnull(left_site) and not pd.isnull(right_site):
                rxn_forward = AllChem.ReactionFromSmarts(left_site + ">>" + right_site)
                if reation in rea_str_dic:
                    str_fp = rea_str_dic[reation]
                else:
                    str_fp = Chem.rdChemReactions.CreateStructuralFingerprintForReaction(
                        rxn_forward).ToBitString()
                    rea_str_dic[reation] = str_fp

                if reation in rea_diff_dic:
                    diff_fp = rea_diff_dic[reation]
                else:
                    diff_fp = Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_forward)
                    diff_fp = convert_fp_to_array(diff_fp.GetNonzeroElements())
                    rea_diff_dic[reation] = diff_fp

                left_site = get_reaction_site_smiles(substrates)
                right_site = get_reaction_site_smiles(products)
                if reation in rea_DRFP_dic:
                    drfp = rea_DRFP_dic[reation]
                else:
                    drfp = DrfpEncoder.encode(left_site + ">>" + right_site)[0]
                    rea_DRFP_dic[reation] = drfp

                reaction_data["DRFP_del"][i] = drfp
                reaction_data["structural_fp_del"][i] = str_fp
                reaction_data["difference_fp_del"][i] = diff_fp

        except IndexError:
            pass
        
        
        substrates = []
        products  =list(reaction_data["pro_ids"][i])
        reation = (tuple(substrates),tuple(products))
        try:

            left_site = get_reaction_site_smarts(substrates)
            right_site = get_reaction_site_smarts(products)
            if not pd.isnull(left_site) and not pd.isnull(right_site):
                rxn_forward = AllChem.ReactionFromSmarts(left_site + ">>" + right_site)
                if reation in rea_str_dic:
                    str_fp = rea_str_dic[reation]
                else:
                    str_fp = Chem.rdChemReactions.CreateStructuralFingerprintForReaction(
                        rxn_forward).ToBitString()
                    rea_str_dic[reation] = str_fp

                if reation in rea_diff_dic:
                    diff_fp = rea_diff_dic[reation]
                else:
                    diff_fp = Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_forward)
                    diff_fp = convert_fp_to_array(diff_fp.GetNonzeroElements())
                    rea_diff_dic[reation] = diff_fp

                left_site = get_reaction_site_smiles(substrates)
                right_site = get_reaction_site_smiles(products)
                if reation in rea_DRFP_dic:
                    drfp = rea_DRFP_dic[reation]
                else:
                    drfp = DrfpEncoder.encode(left_site + ">>" + right_site)[0]
                    rea_DRFP_dic[reation] = drfp

                reaction_data["DRFP_del2"][i] = drfp
                reaction_data["structural_fp_del2"][i] = str_fp
                reaction_data["difference_fp_del2"][i] = diff_fp

        except IndexError:
            pass

    reaction_data = reaction_data.loc[reaction_data["structural_fp"] != ""]
    print("反应指纹添加后： ",len(reaction_data))
    
    reaction_data.to_pickle(opt.datadir + "kcat_data/fp_kcat_data.pkl")
    print('reaction fingerprint finished')
    
    pass



























































