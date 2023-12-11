import random
import torch as t
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

def sample_negative_edges(bipartite_graph, pos_u, pos_v):
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

def load_data():
    print("Loading data...")
    # load ogb data
    dataset = DglNodePropPredDataset(name='ogbn-mag', root='data/dataset/')
    graph, label = dataset[0]
    _, feat_dim = graph.ndata['feat']['paper'].shape
    paper_feat = graph.ndata['feat']['paper']
    # only keep ("author", "writes", "paper") relation
    graph = dgl.edge_type_subgraph(graph, [('author', 'writes', 'paper')])
    # split edges into train/valid/test
    u, v = graph.edges()
    eids = t.randperm(graph.number_of_edges())
    train_percent, valid_percent = 0.7, 0.15
    train_size, valid_size, test_size = int(train_percent * len(eids)), int(valid_percent * len(eids)), len(eids) - int(train_percent * len(eids)) - int(valid_percent * len(eids))
    train_eids, valid_eids, test_eids = t.split(eids, [train_size, valid_size, test_size])
    print(f"train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}")
    neg_u, neg_v = sample_negative_edges(graph, u, v)
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


    valid_authors,b1=t.unique(valid_pos_u,return_counts=True)
    # for i in range(5):
    #     print((b1>i).sum())
    valid_papers,b1=t.unique(valid_pos_v,return_counts=True)

    test_authors,b1=t.unique(test_pos_u,return_counts=True)
    # for i in range(5):
    #     print((b1>i).sum())

    test_papers,b1=t.unique(test_pos_v,return_counts=True)
    # for i in range(5):
    #     print((b1>i).sum())

    # train graph
    train_graph = dgl.edge_subgraph(graph, train_eids, relabel_nodes=False)
    # valid graph
    valid_graph = dgl.edge_subgraph(graph, t.cat([train_eids, valid_eids], dim=0), relabel_nodes=False)
    # test graph
    test_graph = graph
    num_author = train_graph.number_of_nodes('author')
    num_paper = train_graph.number_of_nodes('paper')
    print(f"num_author: {num_author}, num_paper: {num_paper}")
    train_graph = dgl.to_homogeneous(train_graph)
    train_graph = dgl.add_reverse_edges(train_graph)
    valid_graph = dgl.to_homogeneous(valid_graph)
    valid_graph = dgl.add_reverse_edges(valid_graph)
    test_graph = dgl.to_homogeneous(test_graph)
    test_graph = dgl.add_reverse_edges(test_graph)

    return_dict = {
        'train_graph': train_graph,
        'valid_graph': valid_graph,
        'test_graph': test_graph,
        'train_pos_u': train_pos_u,
        'train_pos_v': train_pos_v,
        'train_neg_u': train_neg_u,
        'train_neg_v': train_neg_v,
        'valid_pos_u': valid_pos_u,
        'valid_pos_v': valid_pos_v,
        'valid_neg_u': valid_neg_u,
        'valid_neg_v': valid_neg_v,
        'test_pos_u': test_pos_u,
        'test_pos_v': test_pos_v,
        'test_neg_u': test_neg_u,
        'test_neg_v': test_neg_v,
        'paper_feat': paper_feat,
        'num_author': num_author,
        'num_paper': num_paper,
        'feat_dim': feat_dim,
        'valid_authors': valid_authors,
        'test_authors': test_authors, 
        'valid_papers': valid_papers,
        'test_papers': test_papers
    }
    return return_dict