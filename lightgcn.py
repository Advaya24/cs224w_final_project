import dgl.function as fn
from torch import nn
import torch as t


class LightGCN2(nn.Module):
    def __init__(self, args, userNum, item_feat_dim, hide_dim, layerNum=1):
        super(LightGCN2, self).__init__()
        self.userNum = userNum
        self.feat_dim = item_feat_dim
        self.hide_dim = hide_dim
        self.layerNum = layerNum
        self.item_mlp = nn.Sequential(
            nn.Linear(item_feat_dim, hide_dim),
            nn.ReLU(),
            nn.Linear(hide_dim, hide_dim)
        )
        self.embedding_dict = self.init_weight(userNum, hide_dim)
        self.args = args

        self.layers = nn.ModuleList()
        for i in range(self.layerNum):
            self.layers.append(GCNLayer())
    
    def init_weight(self, userNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
        })
        return embedding_dict
    

    def forward(self, graph, item_feat):

        res_user_embedding = self.embedding_dict['user_emb']
        # detach the item_feat
        item_feat = item_feat.detach()
        res_item_embedding = self.item_mlp(item_feat)

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, res_user_embedding, res_item_embedding)
            else:
                embeddings = layer(graph, embeddings[: self.userNum], embeddings[self.userNum: ])
            
            res_user_embedding = res_user_embedding + embeddings[: self.userNum]*(1/(i+2))
            res_item_embedding = res_item_embedding + embeddings[self.userNum: ]*(1/(i+2))

        user_embedding = res_user_embedding

        item_embedding = res_item_embedding

        return user_embedding, item_embedding


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

        user_embedding = res_user_embedding

        item_embedding = res_item_embedding

        return user_embedding, item_embedding


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            node_f = t.cat([u_f, v_f], dim=0)
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
