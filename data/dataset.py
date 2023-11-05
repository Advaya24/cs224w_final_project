from typing import Union
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader, Dataset
import torch
from torch_geometric.data.data import BaseData
import torch_geometric
from copy import deepcopy
# # Load the dataset
# dataset = PygNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')


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
    

# dataset = PygNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')
# graph=dataset[0]
# graph.to('cpu')
# import pdb;pdb.set_trace()
# print(graph.edge_index.device)
# train_idx = dataset.get_idx_split()["train"]
# subgraph = torch_geometric.utils.subgraph(train_idx, graph)

# print(dataset[0])
    

from ogb.nodeproppred import DglNodePropPredDataset
import dgl

dataset = DglNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')
graph,labels=dataset[0]
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]



# g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),('user', 'follows', 'user'): ([0, 1], [1, 3])})
# sg, inverse_indices = dgl.khop_subgraph(g, {'game': 0}, k=2)



train_graph= dgl.remove_nodes(graph, valid_idx['paper'],ntype='paper')

k_hops=1
paper_node_id=2
# out_sampled_graph = dgl.sampling.sample_neighbors(train_graph, {'paper':[paper_node_id]},k_hops,edge_dir ='out'  )
# in_sampled_graph = dgl.sampling.sample_neighbors(train_graph, {'paper':[paper_node_id]},k_hops,edge_dir ='in' )
# merged_graph=dgl.merge([out_sampled_graph, in_sampled_graph])

sampled_graph,inv_edges=dgl.khop_subgraph(train_graph,{'paper':[paper_node_id]},k_hops)



