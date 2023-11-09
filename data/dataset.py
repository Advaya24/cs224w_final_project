from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch
import numpy as np

dataset = DglNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')
graph,labels=dataset[0]
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]



# g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),('user', 'follows', 'user'): ([0, 1], [1, 3])})
# sg, inverse_indices = dgl.khop_subgraph(g, {'game': 0}, k=2)


k_hops=3
paper_node_id=10
author_node_id=100
# Apply the mask to the original graph
train_graph= dgl.remove_nodes(graph, valid_idx['paper'],ntype='paper')
train_graph= dgl.remove_nodes(train_graph, train_graph.nodes('institution'),ntype='institution')
train_graph= dgl.remove_nodes(train_graph, train_graph.nodes('field_of_study'),ntype='field_of_study')

# remove institute nodes
# print(train_graph)
# train_graph = dgl.remove_nodes(graph, graph.nodes('institution'),ntype='institution')
# train_graph= dgl.remove_nodes(train_graph, graph


# out_sampled_graph = dgl.sampling.sample_neighbors(train_graph, {'paper':[paper_node_id]},k_hops,edge_dir ='out'  )
# in_sampled_graph = dgl.sampling.sample_neighbors(train_graph, {'paper':[paper_node_id]},k_hops,edge_dir ='in' )
# merged_graph=dgl.merge([out_sampled_graph, in_sampled_graph])

def sample_author_pairs_full_random(graph,num_samples=100):
    positive_pairs=torch.zeros(num_samples,2)
    negative_pairs=torch.zeros(num_samples,2)
    cfiller=0
    anodes=graph.nodes(ntype='author')
    while cfiller<(num_samples):
        source=np.random.choice(anodes)
        papers=graph.successors(source,etype='writes')
        paper=np.random.choice(papers)
        writers=graph.predecessors(paper,etype='writes')
        for writer in writers:
            if writer!=source:
                positive_pairs[cfiller,0]=source
                positive_pairs[cfiller,1]=writer
                cfiller+=1
                break

    negative_pairs[:,0]=torch.tensor(np.random.choice(anodes,num_samples))
    negative_pairs[:,1]=torch.tensor(np.random.choice(anodes,num_samples))

    return positive_pairs,negative_pairs

def sample_author_pairs(graph,num_samples=100):
    positive_pairs=torch.zeros(num_samples,2)
    negative_pairs=torch.zeros(num_samples,2)
    cfiller=0
    anodes=graph.nodes(ntype='author')
    while cfiller<(num_samples):
        source=np.random.choice(anodes)
        papers=graph.successors(source,etype='writes')
        for paper in papers:
            if np.random.uniform()<0.8:
                continue
            writers=graph.predecessors(paper,etype='writes')
            for writer in writers:
                if np.random.uniform()<0.8:
                    continue
                if writer!=source:
                    positive_pairs[cfiller,0]=source
                    positive_pairs[cfiller,1]=writer
                    cfiller+=1
                    if cfiller==num_samples:
                        break
            if cfiller==num_samples:
                break

    negative_pairs[:,0]=torch.tensor(np.random.choice(anodes,num_samples))
    negative_pairs[:,1]=torch.tensor(np.random.choice(anodes,num_samples))

    # e1,e2=graph.edges(etype='writes')
    # cnter=0
    # for x in range(num_samples):
    #     ps1=e2[e1==negative_pairs[x][0]]
    #     ps2=e2[e1==negative_pairs[x][1]]
    #     cp=np.intersect1d(ps1,ps2)
    #     if len(cp)!=0:
    #         cnter+=1
    return positive_pairs,negative_pairs


# sample_author_pairs_full_random(dgl.khop_subgraph(graph,{'paper':[paper_node_id]},4)[0])



def sample_positive_negative_author(graph,author_node_id, k_hops=3, khop_weights = [0, 0.75, 0.25]):
    author_nodes = {}
    sampled_graphs = {}
    for khop in range(1,k_hops+1):
        sampled_graph,inv_edges=dgl.khop_subgraph(graph,{'author':[author_node_id]},khop)
        sampled_graphs[khop] = sampled_graph
        nodes = set(sampled_graph.nodes('author').tolist())
        # nodes.remove(author_node_id)
        for hop in range(1,khop):
            nodes = nodes.difference(author_nodes[hop])
        author_nodes[khop] = nodes

    # sample an author node with weight by hop
    hop = np.random.choice(range(1,k_hops+1),p=khop_weights)
    other_author_node_id = np.random.choice(list(author_nodes[hop]))
    positive = sampled_graphs[khop].ndata[dgl.NID]['author'][other_author_node_id]
    negatives = set(graph.nodes('author').tolist()).difference(sampled_graph.ndata[dgl.NID]['author'])
    negative = np.random.choice(list(negatives))
    return (author_node_id, positive.item()), (author_node_id, negative)






print(sample_positive_negative_author(graph,author_node_id))




