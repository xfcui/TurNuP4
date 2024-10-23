import numpy as np
import scipy
import pandas as pd
import pickle

def save_simindex():
    
    data_train = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/test_kcat.pkl")
    
    train_fps = [np.array(list(data_train["DRFP"][ind])).reshape(1,-1).astype(int) for ind in data_train.index]
    test_fps = [np.array(list(data_test["DRFP"][ind])).reshape(1,-1).astype(int) for ind in data_test.index]


    max_sim = []

    for fp in test_fps:
        jaccard_sim = np.array([1- scipy.spatial.distance.cdist(fp,train_fp, metric='jaccard')[0][0] for train_fp in train_fps])
        max_sim.append(np.max(jaccard_sim))
        
    data_test["reaction_sim"] = max_sim

    data_test["reaction_sim"]= (data_test["reaction_sim"] - np.min(data_test["reaction_sim"]))
    data_test["reaction_sim"] = data_test["reaction_sim"]/np.max(data_test["reaction_sim"])
    
    sim04_indices = data_test[data_test.apply(lambda row: row["reaction_sim"] < 0.4, axis=1)].index.tolist()
    sim48_indices = data_test[data_test.apply(lambda row: row["reaction_sim"] >= 0.4 and row["reaction_sim"] < 0.8, axis=1)].index.tolist()
    sim81_indices = data_test[data_test.apply(lambda row: row["reaction_sim"] >= 0.8 and row["reaction_sim"] < 1, axis=1)].index.tolist()
    sim1_indices = data_test[data_test.apply(lambda row: row["reaction_sim"] >= 1, axis=1)].index.tolist()

    print("num04:",len(sim04_indices))
    print("num48:",len(sim48_indices))
    print("num81:",len(sim81_indices))
    print("num1:",len(sim1_indices))
    
    # 将列表保存到文件
    with open('./kcat/TurNuP4/data/sim_index/sim04_indices.pkl', 'wb') as f:
        pickle.dump(sim04_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim48_indices.pkl', 'wb') as f:
        pickle.dump(sim48_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim81_indices.pkl', 'wb') as f:
        pickle.dump(sim81_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim1_indices.pkl', 'wb') as f:
        pickle.dump(sim1_indices, f)
    
def save_simindex2():
    data_train = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/test_kcat.pkl")
    
    train_fps = [np.array(list(data_train["DRFP"][ind])).reshape(1, -1).astype(int) for ind in data_train.index]
    test_fps = [np.array(list(data_test["DRFP"][ind])).reshape(1, -1).astype(int) for ind in data_test.index]

    max_sim = []

    for fp in test_fps:
        jaccard_sim = np.array([1 - scipy.spatial.distance.cdist(fp, train_fp, metric='jaccard')[0][0] for train_fp in train_fps])
        max_sim.append(np.max(jaccard_sim))
        
    data_test["reaction_sim"] = max_sim
    data_test["reaction_sim"] = (data_test["reaction_sim"] - np.min(data_test["reaction_sim"]))
    data_test["reaction_sim"] = data_test["reaction_sim"] / np.max(data_test["reaction_sim"])
    
    sim1_indices = data_test[data_test["reaction_sim"] == 1].index.tolist()
    remaining_indices = data_test[data_test["reaction_sim"] != 1].sort_values(by="reaction_sim").index.tolist()
    
    # 将剩余的indices按顺序均匀分成3个子集
    chunk_size = len(remaining_indices) // 3
    sim04_indices = remaining_indices[:chunk_size]
    sim48_indices = remaining_indices[chunk_size:2*chunk_size]
    sim81_indices = remaining_indices[2*chunk_size:]

    print("num04:", len(sim04_indices))
    print("num48:", len(sim48_indices))
    print("num81:", len(sim81_indices))
    print("num1:", len(sim1_indices))
    
    # 将列表保存到文件
    with open('./kcat/TurNuP4/data/sim_index/sim04_indices.pkl', 'wb') as f:
        pickle.dump(sim04_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim48_indices.pkl', 'wb') as f:
        pickle.dump(sim48_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim81_indices.pkl', 'wb') as f:
        pickle.dump(sim81_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim1_indices.pkl', 'wb') as f:
        pickle.dump(sim1_indices, f)

def save_simindex_str():
    data_train = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/test_kcat.pkl")
    
    train_fps = [np.array(list(data_train["structural_fp"][ind][:3276])).reshape(1, -1).astype(int) for ind in data_train.index]
    test_fps = [np.array(list(data_test["structural_fp"][ind][:3276])).reshape(1, -1).astype(int) for ind in data_test.index]

    max_sim = []

    for fp in test_fps:
        jaccard_sim = np.array([1 - scipy.spatial.distance.cdist(fp, train_fp, metric='jaccard')[0][0] for train_fp in train_fps])
        max_sim.append(np.max(jaccard_sim))
        
    data_test["reaction_sim"] = max_sim
    data_test["reaction_sim"] = (data_test["reaction_sim"] - np.min(data_test["reaction_sim"]))
    data_test["reaction_sim"] = data_test["reaction_sim"] / np.max(data_test["reaction_sim"])
    
    sim1_indices = data_test[data_test["reaction_sim"] == 1].index.tolist()
    remaining_indices = data_test[data_test["reaction_sim"] != 1].sort_values(by="reaction_sim").index.tolist()
    
    # 将剩余的indices按顺序均匀分成3个子集
    chunk_size = len(remaining_indices) // 3
    sim04_indices = remaining_indices[:chunk_size]
    sim48_indices = remaining_indices[chunk_size:2*chunk_size]
    sim81_indices = remaining_indices[2*chunk_size:]

    print("num04:", len(sim04_indices))
    print("num48:", len(sim48_indices))
    print("num81:", len(sim81_indices))
    print("num1:", len(sim1_indices))
    
    # 将列表保存到文件
    with open('./kcat/TurNuP4/data/sim_index/sim04_indices.pkl', 'wb') as f:
        pickle.dump(sim04_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim48_indices.pkl', 'wb') as f:
        pickle.dump(sim48_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim81_indices.pkl', 'wb') as f:
        pickle.dump(sim81_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim1_indices.pkl', 'wb') as f:
        pickle.dump(sim1_indices, f)
    


def save_simindex3():
    data_train = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/train_kcat.pkl")
    data_test = pd.read_pickle("./kcat/TurNuP4/data/kcat_data/splits/test_kcat.pkl")
    
    train_fps = [np.array(list(data_train["DRFP"][ind])).reshape(1, -1).astype(int) for ind in data_train.index]
    test_fps = [np.array(list(data_test["DRFP"][ind])).reshape(1, -1).astype(int) for ind in data_test.index]

    max_sim = []

    for fp in test_fps:
        jaccard_sim = np.array([1 - scipy.spatial.distance.cdist(fp, train_fp, metric='jaccard')[0][0] for train_fp in train_fps])
        max_sim.append(np.max(jaccard_sim))
        
    data_test["reaction_sim"] = max_sim
    data_test["reaction_sim"] = (data_test["reaction_sim"] - np.min(data_test["reaction_sim"]))
    data_test["reaction_sim"] = data_test["reaction_sim"] / np.max(data_test["reaction_sim"])
    
    sim1_indices = data_test[data_test["reaction_sim"] == 1].index.tolist()
    remaining_indices = data_test[data_test["reaction_sim"] != 1].sort_values(by="reaction_sim").index.tolist()
    
    # 将剩余的indices按顺序均匀分成3个子集
    chunk_size = len(remaining_indices) // 2
    sim04_indices = remaining_indices[:chunk_size]
    sim81_indices = remaining_indices[chunk_size:]

    print("num04:", len(sim04_indices))
    print("num81:", len(sim81_indices))
    print("num1:", len(sim1_indices))
    
    # 将列表保存到文件
    with open('./kcat/TurNuP4/data/sim_index/sim03_indices.pkl', 'wb') as f:
        pickle.dump(sim04_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim36_indices.pkl', 'wb') as f:
        pickle.dump(sim81_indices, f)

    with open('./kcat/TurNuP4/data/sim_index/sim1_indices.pkl', 'wb') as f:
        pickle.dump(sim1_indices, f)



if __name__ == "__main__":
    
    save_simindex3()