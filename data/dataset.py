from typing import Union
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader, Dataset
import torch
from torch_geometric.data.data import BaseData

# # Load the dataset
# dataset = PygNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')

 
# # Get split indices
# split_idx = dataset.get_idx_split()

# # dataset_dict = dict(dataset[0])
# # print(dataset_dict)
# # train_set = {prop: d['paper'][split_idx["train"]['paper']] for prop, d in dataset[0]}
# num_nodes_dict = dataset[0]['num_nodes_dict']
# edge_index_dict = dataset[0]['edge_index_dict']
# x_dict = dataset[0]['x_dict']
# node_year_dict = dataset[0]['node_year']
# edge_reltype_dict = dataset[0]['edge_reltype']
# y_dict = dataset[0]['y_dict']

class CoAuthorPredDataset(Dataset):
    def __init__(self, name='ogbn-mag', root = 'dataset/', split = 'train'):
        self.dataset = PygNodePropPredDataset(name = name, root = root)
        split_idx = self.dataset.get_idx_split()
        self.num_nodes_dict = self.dataset[0]['num_nodes_dict']
        self.edge_index_dict = self.dataset[0]['edge_index_dict']
        self.x_dict = self.dataset[0]['x_dict']
        self.node_year_dict = self.dataset[0]['node_year']
        self.edge_reltype_dict = self.dataset[0]['edge_reltype']
        self.y_dict = self.dataset[0]['y_dict']
        self.split_idx = torch.tensor(split_idx[split]['paper'])
        self._extract_split()

    def _extract_split(self):
        self.node_year_dict = {key: value[self.split_idx] for key, value in self.node_year_dict.items()}
        self.x_dict = {key: value[self.split_idx] for key, value in self.x_dict.items()}
        self.y_dict = {key: value[self.split_idx] for key, value in self.y_dict.items()}
        self.edge_index_dict = {key: value[:, self.split_idx] for key, value in self.edge_index_dict.items()}
        self.edge_reltype_dict = {key: value[self.split_idx] for key, value in self.edge_reltype_dict.items()}
    
    def len(self):
        return len(self.split_idx)
    
    def get(self, idx):
        return self.x_dict, self.edge_index_dict, self.edge_reltype_dict, self.y_dict, self.node_year_dict
    

dataset = CoAuthorPredDataset(name = "ogbn-mag", root = 'dataset/')
print(dataset[0])
    

