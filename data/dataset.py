from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch
import numpy as np
from dgl import AddReverse, Compose, ToSimple

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
    print(sampled_graphs)
    # sample an author node with weight by hop
    hop = np.random.choice(range(1,k_hops+1),p=khop_weights)
    other_author_node_id = np.random.choice(list(author_nodes[hop]))
    positive = sampled_graphs[khop].ndata[dgl.NID]['author'][other_author_node_id]
    negatives = set(graph.nodes('author').tolist()).difference(sampled_graph.ndata[dgl.NID]['author'])
    negative = np.random.choice(list(negatives))
    return (author_node_id, positive.item()), (author_node_id, negative)



class PaperNbrSampler(dgl.dataloading.Sampler):
    def __init__(self, num_author_pair,khops):
        super().__init__()
        self.num_author_pair = num_author_pair #This is per graph
        self.khops=khops


    def sample_positive_author_pairs(self,graph,num_samples_per_graph):
        num_graphs=graph.batch_size
        positive_pairs=torch.zeros(num_samples_per_graph*num_graphs,2)
        
        cfiller=0
        anodes=graph.nodes(ntype='author')
        while cfiller<(positive_pairs.shape[0]):
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

        return positive_pairs

    def sample_negative_author_pairs(self,graph,sample_per_graph):
        
        anodes=graph.nodes(ntype='author')
        cntauth=graph.batch_num_nodes('author')
        num_graphs=cntauth.shape[0]
        negative_pairs=torch.zeros(sample_per_graph*num_graphs,2)
        idx2fillfrom=0
        for i in range(num_graphs):
            negative_pairs[sample_per_graph*i:sample_per_graph*(i+1),0]=torch.tensor(np.random.choice(anodes[idx2fillfrom:idx2fillfrom+cntauth[i]],sample_per_graph))
            negative_pairs[sample_per_graph*i:sample_per_graph*(i+1),1]=torch.tensor(np.random.choice(anodes[idx2fillfrom:idx2fillfrom+cntauth[i]],sample_per_graph))
            idx2fillfrom+=cntauth[i]
        return negative_pairs


    def sample(self,graph,indices):
        #g is full graph. indices are the train paper nodes in curent mini batch
        subgraphs=[]
        for paper in indices:
            sg=dgl.khop_subgraph(graph,{'paper':[paper]},self.khops)[0]
            subgraphs.append(sg)
        mini_batch=dgl.batch(subgraphs)
        positive_pairs=self.sample_positive_author_pairs(mini_batch,self.num_author_pair)
        negative_pairs=self.sample_negative_author_pairs(mini_batch,self.num_author_pair)
        return mini_batch,positive_pairs,negative_pairs


coauth_train_loader = dgl.dataloading.DataLoader(
        train_graph,
        train_graph.nodes('paper'),
        PaperNbrSampler(2,2),
        batch_size=16,
        shuffle=True,
        num_workers=0,
        device='cpu',
    )


# print(sample_positive_negative_author(train_graph,author_node_id))
# print(sample_positive_negative_author(train_graph,author_node_id, 4, [0, 0.7, 0.2, 0.1]))


# dgl dataloader using graph and train_idx
# def prepare_data(device):
#     dataset = DglNodePropPredDataset(name="ogbn-mag")
#     split_idx = dataset.get_idx_split()
#     # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
#     g, labels = dataset[0]
#     labels = labels["paper"].flatten()

#     transform = Compose([ToSimple(), AddReverse()])
#     g = transform(g)

#     # print("Loaded graph: {}".format(g))

#     # train sampler
#     negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
#     sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
#     # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
#     sampler = dgl.dataloading.as_edge_prediction_sampler(
#         sampler, negative_sampler=negative_sampler
#     )
#     num_workers = 0
#     train_masks = {etype: (torch.randperm(g.number_of_edges(etype)) < 0.8 * g.number_of_edges()).to(torch.int64) for etype in g.etypes}
#     train_loader = dgl.dataloading.DataLoader(
#         g,
#         train_masks,
#         sampler,
#         batch_size=128,
#         shuffle=True,
#         num_workers=num_workers,
#         device=device,
#     )

#     return g, labels, dataset.num_classes, split_idx, train_loader

# g, labels, dataset.num_classes, split_idx, train_loader = prepare_data('cpu')
# print(g)
count = 0
for i, (subg, ppair, npair) in enumerate(coauth_train_loader):
    # print(input_nodes)
    # print(positive_graph)
    # print(negative_graph)
    # print(blocks)
    import pdb;pdb.set_trace()
    # count += 1
    # if count >= 1:
    #     break

