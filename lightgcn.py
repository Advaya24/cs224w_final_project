import networkx as nx
import matplotlib.pyplot as plt
import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import numpy as np
import random

from evaluate import metrics


class LightGCN(nn.Module):
    def __init__(self, args, userNum, itemNum, hide_dim, layerNum=1):
        super(LightGCN, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        self.layerNum = layerNum
        self.embedding_dict = self.init_weight(userNum, itemNum, hide_dim)
        self.args = args

        self.layers = nn.ModuleList()
        for i in range(self.layerNum):
            self.layers.append(GCNLayer())
    
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    

    def forward(self, graph):

        res_user_embedding = self.embedding_dict['user_emb']
        res_item_embedding = self.embedding_dict['item_emb']

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, res_user_embedding, res_item_embedding)
            else:
                embeddings = layer(graph, embeddings[: self.userNum], embeddings[self.userNum: ])
            
            res_user_embedding = res_user_embedding + embeddings[: self.userNum]*(1/(i+2))
            res_item_embedding = res_item_embedding + embeddings[self.userNum: ]*(1/(i+2))

        user_embedding = res_user_embedding# / (len(self.layers)+1)

        item_embedding = res_item_embedding# / (len(self.layers)+1)

        return user_embedding, item_embedding


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            node_f = t.cat([u_f, v_f], dim=0)
            # D^-1/2
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            return rst
        

def sample_negative_edges(bipartite_graph, num_neg_samples, pos_u, pos_v):
    u = bipartite_graph.nodes('author')
    v = bipartite_graph.nodes('paper')
    
    # Sample negative edges
    negative_edges = []
    for i in range(len(pos_u)):
        # Randomly select a negative node
        negative_node = random.choice(v if i % 2 == 0 else u)
        
        # Append the negative edge
        negative_edges.append((pos_u[i], negative_node) if i % 2 == 0 else (negative_node, pos_v[i]))
    
    # Convert to DGL graph
    negative_edges = list(zip(*negative_edges))
    return t.tensor(negative_edges[0]), t.tensor(negative_edges[1])

if __name__ == '__main__':
    print("Loading data...")
    # load ogb data
    dataset = DglNodePropPredDataset(name='ogbn-mag')
    graph, label = dataset[0]
    # only keep ("author", "writes", "paper") relation
    graph = dgl.edge_type_subgraph(graph, [('author', 'writes', 'paper')])
    # split edges into train/valid/test
    u, v = graph.edges()
    eids = t.randperm(graph.number_of_edges())
    train_percent, valid_percent = 0.8, 0.1
    train_size, valid_size, test_size = int(train_percent * len(eids)), int(valid_percent * len(eids)), len(eids) - int(train_percent * len(eids)) - int(valid_percent * len(eids))
    train_eids, valid_eids, test_eids = t.split(eids, [train_size, valid_size, test_size])
    neg_u, neg_v = sample_negative_edges(graph, len(train_eids), u, v)
    train_pos_u, train_pos_v = u[train_eids], v[train_eids]
    valid_pos_u, valid_pos_v = u[valid_eids], v[valid_eids]
    test_pos_u, test_pos_v = u[test_eids], v[test_eids]
    train_neg_u, train_neg_v = neg_u[train_eids], neg_v[train_eids]
    valid_neg_u, valid_neg_v = neg_u[valid_eids], neg_v[valid_eids]
    test_neg_u, test_neg_v = neg_u[test_eids], neg_v[test_eids]

    author_ids = graph.nodes('author')
    paper_ids = graph.nodes('paper')

    # get the embedding ids
    train_pos_u = author_ids[train_pos_u]
    train_pos_v = paper_ids[train_pos_v]
    train_neg_u = author_ids[train_neg_u]
    train_neg_v = paper_ids[train_neg_v]
    valid_pos_u = author_ids[valid_pos_u]
    valid_pos_v = paper_ids[valid_pos_v]
    valid_neg_u = author_ids[valid_neg_u]
    valid_neg_v = paper_ids[valid_neg_v]
    test_pos_u = author_ids[test_pos_u]
    test_pos_v = paper_ids[test_pos_v]
    test_neg_u = author_ids[test_neg_u]
    test_neg_v = paper_ids[test_neg_v]


    # train graph
    train_graph = dgl.edge_subgraph(graph, train_eids, relabel_nodes=False)
    # valid graph
    valid_graph = dgl.edge_subgraph(graph, t.cat([train_eids, valid_eids], dim=0), relabel_nodes=False)
    # test graph
    test_graph = graph

    print("Training...")

    # model
    num_author = train_graph.number_of_nodes('author')
    num_paper = train_graph.number_of_nodes('paper')
    print(f"num_author: {num_author}, num_paper: {num_paper}")
    model = LightGCN(None, num_author, num_paper, 64, 1)
    print(f"model: {[param.shape for param in model.parameters()]}")
    # optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.05)
    train_graph = dgl.to_homogeneous(train_graph)
    valid_graph = dgl.to_homogeneous(valid_graph)
    test_graph = dgl.to_homogeneous(test_graph)
    train_loss = []
    valid_loss = []

    # train
    for epoch in range(10):
        print(f"Epoch: {epoch}")
        model.train()
        optimizer.zero_grad()
        author_embeddings, paper_embeddings = model(train_graph)
        
        # convert pos and neg ids to embeddings
        pos_author_embeddings = author_embeddings[train_pos_u]
        pos_paper_embeddings = paper_embeddings[train_pos_v]
        neg_author_embeddings = author_embeddings[train_neg_u]
        neg_paper_embeddings = paper_embeddings[train_neg_v]

        # BPR loss
        pos_score = t.sum(pos_author_embeddings * pos_paper_embeddings, dim=1)
        neg_score = t.sum(neg_author_embeddings * neg_paper_embeddings, dim=1)
        loss = -t.mean(t.log(t.sigmoid(pos_score - neg_score)))
        
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
        train_loss.append(loss.item())
        # validation
        model.eval()
        with t.no_grad():
            author_embeddings, paper_embeddings = model(valid_graph)
            # cosine similarity
            pos_score = t.sum(author_embeddings[valid_pos_u] * paper_embeddings[valid_pos_v], dim=1)
            neg_score = t.sum(author_embeddings[valid_neg_u] * paper_embeddings[valid_neg_v], dim=1)
            loss = -t.mean(t.log(t.sigmoid(pos_score - neg_score)))
            print(f"Valid Loss: {loss.item()}")
            valid_loss.append(loss.item())
    
    # plot loss
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    # test
    model.eval()
    with t.no_grad():
        print(metrics(model, test_graph))
