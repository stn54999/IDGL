import os

import dgl
import torch
import numpy as np
from dgl import load_graphs
from torch.utils.data import Dataset



class CustomDGLDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        self.num_graphs = len(graphs)

        # 按照8-1-1的比例划分数据集
        train_size = int(0.8 * self.num_graphs)
        val_size = int(0.1 * self.num_graphs)
        test_size = self.num_graphs - train_size - val_size

        # 创建索引并随机排列
        indices = np.arange(self.num_graphs)
        np.random.shuffle(indices)

        # 根据索引分配数据集
        self.train_idx = indices[:train_size]
        self.val_idx = indices[train_size:train_size + val_size]
        self.test_idx = indices[train_size + val_size:]

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return self.num_graphs

    def train_set(self):
        return torch.utils.data.Subset(self, self.train_idx)

    def val_set(self):
        return torch.utils.data.Subset(self, self.val_idx)

    def test_set(self):
        return torch.utils.data.Subset(self, self.test_idx)



def load_cg(cg_path):
    """
    Load cell graphs
    """
    #Cell_Graph = load_graphs(cg_path)
    # paths=(os.path.join(cg_path,"AllCell.bin"))
    # print(paths)
    Cell_Graph=load_graphs((os.path.join(cg_path,"AllCell.bin")))

    return Cell_Graph

def load_dgl(data_dir='',dataset_str='',form=None):
    '''
    load raw datasets.
    :return: a list of DGL graphs, plus additional info if needed
    '''
    if form is not None:
        dataset_dir = data_dir+"/"+dataset_str+"/"+form
    else:
        dataset_dir = data_dir + "/" + dataset_str
    graphs = []
    if not os.path.exists(dataset_dir):
        return None
    patients = os.listdir(dataset_dir) #CellGraph/patient
    for i in range(len(patients)):
        path = os.path.join(dataset_dir, patients[i])  #所有patch所在的路径
        if os.path.isfile(path):
            continue
        patches = os.listdir(path)
        for patch in patches:
            graph = load_cg(os.path.join(path, patch))
            if graph is not None:
                graphs.append(graph[0])
    return graphs


def load_data(config):
    dataset_dir = config['data_dir']
    dataset_name = config['dataset_name']
    train_set = load_dgl(data_dir=dataset_dir, dataset_str=dataset_name, form='train')
    dev_set = load_dgl(data_dir=dataset_dir, dataset_str=dataset_name, form='val')
    test_set = load_dgl(data_dir=dataset_dir, dataset_str=dataset_name, form='test')
    return train_set, dev_set, test_set

