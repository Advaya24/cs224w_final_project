from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch
import numpy as np
from dgl import AddReverse, Compose, ToSimple



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
        #Nodes of individual graphs are ordered by (graph,nodeID in graph) and then given one grand ordering which is what we use here. 
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




if __name__=='__main__':
    dataset = DglNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')
    graph,labels=dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    k_hops=2
    pairs_per_graph=2
    batch_size=16
    train_graph= dgl.remove_nodes(graph, valid_idx['paper'],ntype='paper')
    train_graph= dgl.remove_nodes(train_graph, train_graph.nodes('institution'),ntype='institution')
    train_graph= dgl.remove_nodes(train_graph, train_graph.nodes('field_of_study'),ntype='field_of_study')
    coauth_train_loader = dgl.dataloading.DataLoader(
        train_graph,
        train_graph.nodes('paper'),
        PaperNbrSampler(pairs_per_graph,k_hops),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        device='cpu',
    )

    for i, (subg, ppair, npair) in enumerate(coauth_train_loader):
        print(subg)
        print(ppair)
        print(npair)
        break
