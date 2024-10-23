
import os
import pickle
import hashlib
import argparse
import requests
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from zeep import Client
from os.path import join
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from urllib.request import urlopen, Request
from bioservices import *
import warnings


def load_BRENDA(path):
    kcat_data = pd.read_pickle(path)
    # adding reaction information:
    kcat_data.rename(columns={"correct reaction ID": "BRENDA reaction ID"})

    kcat_data["Uniprot ID"] = np.nan
    for index in kcat_data.index:  #Dataframe的index有时候/经常？是残缺的
        try:
            kcat_data["Uniprot ID"][index] = kcat_data["UNIPROT_list"][index][0]
        except IndexError:
            pass

    kcat_data = kcat_data.loc[~pd.isnull(kcat_data["Uniprot ID"])]

    kcat_data.drop(columns=["index","ID","checked", "ORGANISM","comment","PMID", "#UIDs","kcat", "kcat_new", "enzyme",
                            "new", "LITERATURE", "UNIPROT_list","EC", "new enzyme","correct reaction ID"],
                   inplace=True)

    kcat_data.rename(columns={"correct kcat": "kcat",
                              "substrate_ID_list": "sub_ids",
                              "product_ID_list": "pro_ids"}, inplace=True)

    # EC 酶分类号 , Uniprot ID酶标识号 , kcat , sub_ids pro_ids (InChI)

    #print("Number of data points: %s" % len(kcat_data))
    #print("Number of UniProt IDs: %s" % len(set(kcat_data["Uniprot ID"])))
    #print("Number of checked data points: %s" % len(kcat_data.loc[kcat_data["checked"]]))
    #print("Number of unchecked data points: %s" % len(kcat_data.loc[~kcat_data["checked"]]))
    return kcat_data


def load_Sabio(path):
    kcat_data = pd.read_pickle(path)
    kcat_data.drop(columns=["unit","PMID", "complete", "KEGG ID"], inplace=True)
    kcat_data.rename(columns={"substrate_IDs":"sub_ids","products_IDs": "pro_ids"}, inplace=True)
    # Uniprot ID , kcat , sub_ids pro_ids (标识符 C0000) ,Substrates	Products(名称？)

    return kcat_data


def load_UniProt(path):
    kcat_data = pd.read_pickle(path)

    kcat_data.drop(columns=["unit","complete", "reaction ID"], inplace=True)
    kcat_data.rename(columns={"substrate CHEBI IDs": "Substrates", "product CHEBI IDs": "Products",
                               "substrate InChIs": "sub_ids", "product InChIs": "pro_ids",
                               "kcat [1/sec]": "kcat"}, inplace=True)

    return kcat_data






































