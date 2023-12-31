from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch
import numpy as np
from dgl import AddReverse, Compose, ToSimple
from dgl.nn import SAGEConv
from tqdm import tqdm
from numpy.random import default_rng
from copy import deepcopy
import networkx as nx
dataset = DglNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')
graph,labels=dataset[0]
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

valid_idx['paper']=[x for x in valid_idx['paper'] if len(graph.predecessors(x,etype='writes'))>1]
# g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),('user', 'follows', 'user'): ([0, 1], [1, 3])})
# sg, inverse_indices = dgl.khop_subgraph(g, {'game': 0}, k=2)


k_hops=5
# Apply the mask to the original graph
train_graph= dgl.remove_nodes(graph, valid_idx['paper'],ntype='paper')
# train_graph= dgl.remove_nodes(train_graph, train_graph.nodes('institution'),ntype='institution')
# train_graph= dgl.remove_nodes(train_graph, train_graph.nodes('field_of_study'),ntype='field_of_study')

# remove institute nodes
# print(train_graph)
# train_graph = dgl.remove_nodes(graph, graph.nodes('institution'),ntype='institution')
# train_graph= dgl.remove_nodes(train_graph, graph


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
            if len(papers)==0:
                continue
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
        positive_pairs=None#self.sample_positive_author_pairs(mini_batch,self.num_author_pair)
        negative_pairs=None#self.sample_negative_author_pairs(mini_batch,self.num_author_pair)
        return mini_batch#,positive_pairs,negative_pairs




class ValPaperNbrSampler(dgl.dataloading.Sampler):
    def __init__(self,khops):
        super().__init__()
        self.khops=khops

    def sample(self,graph,indices):
        #g is full graph. It returns the index in indices and the author nodes with which edge exists in reality
        #returns, OG subgraph, sub graph to do MSG passing on, correct edges 
        assert indices.shape[0]==1
        sg,invlabel=dgl.khop_subgraph(graph,{'paper':[indices[0]]},self.khops)
        paper_in_subg=invlabel['paper']
        authors=sg.predecessors(paper_in_subg[0],etype='writes')
        author_to_del=authors[:-1]
        edge_ids_to_del=sg.edge_ids(author_to_del,paper_in_subg.repeat(len(author_to_del)),etype='writes')
        sg_model_inp=dgl.remove_edges(sg, edge_ids_to_del,etype='writes')
        # sg_model_inp=deepcopy(sg)
        sg_to_pred=deepcopy(sg)
        for etype in sg.etypes:
            sg_to_pred=dgl.remove_edges(sg_to_pred, sg.edges(form='eid',etype=etype),etype=etype)
        all_authors=sg.nodes('author')
        non_paper_authors=np.setdiff1d(all_authors,author_to_del)
        neg_authors=np.random.choice(non_paper_authors,min(len(author_to_del)*2,non_paper_authors.shape[0]),replace=False)
        authors_to_add=np.concatenate([neg_authors,author_to_del])
        sg_to_pred.add_edges(authors_to_add,paper_in_subg.repeat(len(authors_to_add)),etype='writes')
        return sg_model_inp,sg_to_pred,invlabel['paper'][0],author_to_del,authors[-1]
        #we want to pass sg_del through the model and see performance on edge_ids_to_del edes of sg

class ValPaperNbrSamplerLoss(dgl.dataloading.Sampler):
    def __init__(self,khops):
        super().__init__()
        self.khops=khops
        
    def sample(self,graph,indices):
        #g is full graph. It returns the index in indices and the author nodes with which edge exists in reality
        #returns, OG subgraph, sub graph to do MSG passing on, correct edges 
        assert indices.shape[0]==1
        sg,invlabel=dgl.khop_subgraph(graph,{'paper':[indices[0]]},self.khops)
        paper_in_subg=invlabel['paper']
        authors=sg.predecessors(paper_in_subg[0],etype='writes')
        author_to_del=authors[:-1]
        edge_ids_to_del=sg.edge_ids(author_to_del,paper_in_subg.repeat(len(author_to_del)),etype='writes')
        sg_model_inp=dgl.remove_edges(sg, edge_ids_to_del,etype='writes')
        # sg_model_inp=deepcopy(sg)
        sg_pos=deepcopy(sg)
        sg_neg=deepcopy(sg)
        for etype in sg.etypes:
            sg_pos=dgl.remove_edges(sg_pos, sg.edges(form='eid',etype=etype),etype=etype)
            sg_neg=dgl.remove_edges(sg_neg, sg.edges(form='eid',etype=etype),etype=etype)
        all_authors=sg.nodes('author')
        neg_authors=np.random.choice(np.setdiff1d(all_authors,author_to_del),len(author_to_del))
        sg_pos.add_edges(author_to_del,paper_in_subg.repeat(len(author_to_del)),etype='writes')
        sg_neg.add_edges(neg_authors,paper_in_subg.repeat(len(neg_authors)),etype='writes')
        return sg_model_inp,sg_pos,sg_neg,#invlabel['paper'][0],author_to_del


coauth_train_loader = dgl.dataloading.DataLoader(
        train_graph,
        train_graph.nodes('paper'),
        PaperNbrSampler(2,k_hops),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        device='cpu',
    )

coauth_val_loader = dgl.dataloading.DataLoader(
        graph,
        valid_idx['paper'],
        ValPaperNbrSampler(k_hops),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        device='cpu',
    )

coauth_val_loader_loss = dgl.dataloading.DataLoader(
        graph,
        valid_idx['paper'],
        ValPaperNbrSamplerLoss(k_hops),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        device='cpu',
    )


def find_node_ids(node_type_list,num_node_func,node_type,node_ids):
    #the corresponding node ID after homogenisatio
    i=0
    ans=torch.clone(node_ids)
    while node_type_list[i]!=node_type:
        ans+=num_node_func(node_type_list[i])
        i+=1
    return ans

def get_avg_num_shortest_paths(graph,source,targets):
    g1=graph.to_networkx()
    return np.mean([len([x for x in nx.all_shortest_paths(g1,source=source,target=i)]) for i in targets])

def validate(model,val_loader):
    #We predict the as many authors as there are actually, and then we see how many are correct
    model.eval()
    recalls=[]
    for i, (subg_inp,sg_to_pred,paper,authors,conn_author) in enumerate(val_loader):
        # print(input_nodes)
        if i>50:
            break
        paper_node_homo=find_node_ids(subg_inp.ntypes,subg_inp.num_nodes,'paper',paper)
        author_node_homo=find_node_ids(subg_inp.ntypes,subg_inp.num_nodes,'author',authors)

        conn_author_homo=find_node_ids(subg_inp.ntypes,subg_inp.num_nodes,'author',conn_author)

        subg_pred_homo=dgl.to_homogeneous(sg_to_pred)
        subg_inp_homo=dgl.to_homogeneous(subg_inp)
        sub_pred_undir=dgl.add_reverse_edges(subg_pred_homo)
        sub_inp_undir=dgl.add_reverse_edges(subg_inp_homo)

        node_feats=torch.zeros((subg_pred_homo.num_nodes(),len(subg_inp.ntypes))) #one hot encoding of node type
        node_feats[torch.arange(subg_pred_homo.num_nodes()),subg_pred_homo.ndata['_TYPE']]=1
        # edge_ids_to_predict=sub_pred_undir.edge_ids(author_node_homo,paper_node_homo.repeat(len(author_node_homo)))

        op=model(sub_inp_undir,None,node_feats,sub_pred_undir)
        op=op[:(op.shape[0]//2)] #because symmetric edges
        recall_k=min(int(np.ceil(authors.shape[0]*1.5)),10)
        ###Recall@K#####
        # import pdb;pdb.set_trace()
        if len(op.shape)>1:
            pred_edges=torch.topk(op[:,0],recall_k)[1]
        else:
            pred_edges=torch.topk(op,recall_k)[1]
        pred_authors=sub_pred_undir.edges()[0][pred_edges]
        # import pdb;pdb.set_trace()
        # recalls.append(get_avg_num_shortest_paths(sub_inp_undir,conn_author_homo.item(),pred_authors))        

        # recalls.append(np.intersect1d(pred_authors,authors).shape[0]/authors.shape[0])

        # shortest_dist=dgl.shortest_dist(sub_inp_undir,conn_author_homo)
        # shortest_dist[shortest_dist==-1]=shortest_dist.max()+1
        # recalls.append(torch.mean((shortest_dist[pred_authors]).double()).detach().numpy())
        recalls.append(np.intersect1d(pred_authors,author_node_homo).shape[0]/authors.shape[0])

    return np.mean(recalls)

def validate_w_loss(model,val_loader,loss_func):
    #We predict the as many authors as there are actually, and then we see how many are correct

    model.eval()
    losses=[]
    for i, (subg_inp,sg_pos,sg_neg) in enumerate(val_loader):
        # print(input_nodes)
        if i>50:
            break
        # paper_node_homo=find_node_ids(subg_inp.ntypes,subg_inp.num_nodes,'paper',paper)
        # author_node_homo=find_node_ids(subg_inp.ntypes,subg_inp.num_nodes,'author',authors)

        subg_pos_homo=dgl.to_homogeneous(sg_pos)
        subg_neg_homo=dgl.to_homogeneous(sg_neg)
        subg_inp_homo=dgl.to_homogeneous(subg_inp)
        sub_pos_undir=dgl.add_reverse_edges(subg_pos_homo)
        sub_neg_undir=dgl.add_reverse_edges(subg_neg_homo)
        sub_inp_undir=dgl.add_reverse_edges(subg_inp_homo)

        node_feats=torch.zeros((subg_inp_homo.num_nodes(),len(subg_inp.ntypes))) #one hot encoding of node type
        node_feats[torch.arange(subg_inp_homo.num_nodes()),subg_inp_homo.ndata['_TYPE']]=1
        # edge_ids_to_predict=sub_pred_undir.edge_ids(author_node_homo,paper_node_homo.repeat(len(author_node_homo)))

        pos_score,neg_score=model(sub_inp_undir,sub_neg_undir,node_feats,sub_pos_undir)
        loss = loss_func(pos_score, neg_score)
        losses.append(loss.item())
        

    return np.mean(losses)
    

class DotProductPredictor(torch.nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class MLPPredictor(torch.nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = torch.nn.Linear(h_feats * 2, h_feats)
        self.W2 = torch.nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(torch.nn.functional.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def construct_input_positive_graph(graph,k):
    #expects undirected graph
    #remove k distinct edges, so total 2k removal counting reverse edge
    rng = default_rng()
    eids = rng.choice(graph.number_of_edges()//2, size=k, replace=False)
    eids=np.concatenate([eids,eids+graph.number_of_edges()//2])
    graph_inp=dgl.remove_edges(graph, eids)
    graph_pos=dgl.remove_edges(graph,torch.arange(graph.number_of_edges())) #remove all edges
    src,dst=graph.edges()
    graph_pos=dgl.add_edges(graph_pos,src[eids],dst[eids])
    return graph_inp,graph_pos

def construct_all_inputs(graph,authors,papers,authors_neg,papers_neg):
    author_paper_homo_edge_ids=graph.edge_ids(authors,papers)
    author_paper_homo_edge_ids=torch.concatenate([author_paper_homo_edge_ids,author_paper_homo_edge_ids+graph.number_of_edges()//2])    
    graph_inp=dgl.remove_edges(graph, author_paper_homo_edge_ids)

    graph_pos=dgl.remove_edges(graph,torch.arange(graph.number_of_edges())) #remove all edges
    graph_pos=dgl.add_edges(graph_pos,authors,papers)
    graph_pos=dgl.add_edges(graph_pos,papers,authors)

    graph_neg=dgl.remove_edges(graph,torch.arange(graph.number_of_edges())) #remove all edges
    graph_neg=dgl.add_edges(graph_neg,authors_neg,papers_neg)
    graph_neg=dgl.add_edges(graph_neg,papers_neg,authors_neg)
    # import pdb;pdb.set_trace()
    return graph_inp,graph_pos,graph_neg   
    

class Model(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        # self.sage = SAGE(in_features, hidden_features, out_features)
        self.sage1=SAGEConv(in_features, hidden_features, 'mean')
        self.sage2=SAGEConv(hidden_features, hidden_features, 'mean')
        self.sage3=SAGEConv(hidden_features, hidden_features, 'mean')
        # self.pred = DotProductPredictor()
        self.pred=MLPPredictor(hidden_features)
        self.bn=torch.nn.BatchNorm1d(hidden_features, eps=1)
    def forward(self, g, neg_g, x,pos_g=None):
         #g should ont have the edges we want to predict which are there in pos_g. 
        if pos_g is None:
            pos_g=g
        h = self.sage1(g, x)
        h=self.sage2(g,h)
        h=self.sage3(g,h)
        # h=self.bn(h)

        if pos_g is not None and neg_g is not None:
            return self.pred(pos_g, h), self.pred(neg_g, h)
        elif neg_g is not None:
            return self.pred(neg_g,h)
        else:
            return self.pred(pos_g,h)

def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def compute_loss_CE(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)


model = Model(4, 4)
opt = torch.optim.Adam(model.parameters())

# validate(model,coauth_val_loader)
# (subg, ppair, npair)    

def get_author_paper_pairs(graph,neg=False):
    #graph is hetero
    src,dst=graph.edges(etype='writes')
    num_writes=src.shape[0]
    rng = default_rng()
    eids = rng.choice(num_writes, size=num_writes//2, replace=False)
    authors=src[eids]
    if neg:
        papers=dst[np.random.permutation(eids)]
    else:
        papers=dst[eids]
    return authors,papers



for i, (subg) in (pbar:= tqdm(enumerate(coauth_train_loader))):
    # print(input_nodes)
    model.train()
    author_pred,paper_pred=get_author_paper_pairs(subg)
    author_pred_neg,paper_pred_neg=get_author_paper_pairs(subg,neg=True)

    sub_homo=dgl.to_homogeneous(subg)
    sub_homo_undir=dgl.add_reverse_edges(sub_homo)
    node_feats=torch.zeros((subg.num_nodes(),len(subg.ntypes))) #one hot encoding of node type
    node_feats[torch.arange(subg.num_nodes()),sub_homo_undir.ndata['_TYPE']]=1
    
    author_pred_homo=find_node_ids(subg.ntypes,subg.num_nodes,'author',author_pred)
    paper_pred_homo=find_node_ids(subg.ntypes,subg.num_nodes,'paper',paper_pred)

    author_pred_neg_homo=find_node_ids(subg.ntypes,subg.num_nodes,'author',author_pred_neg)
    paper_pred_neg_homo=find_node_ids(subg.ntypes,subg.num_nodes,'paper',paper_pred_neg)
    
    # num_edges_for_loss=author_pred_homo.shape[0]
    input_graph,positive_graph,negative_graph=construct_all_inputs(sub_homo_undir,author_pred_homo,paper_pred_homo,author_pred_neg_homo,paper_pred_neg_homo)
    


    # num_edges_for_loss=sub_homo_undir.number_of_edges()//6 #removing 2/6 of distinct edges using 2/3rd of distinct edegs to classify
    # negative_graph = construct_negative_graph(sub_homo_undir,  num_edges_for_loss)
    # input_graph,positive_graph=construct_input_positive_graph(sub_homo_undir,num_edges_for_loss)
    pos_score, neg_score = model(input_graph, negative_graph, node_feats,positive_graph)
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()
    pbar.set_description(f"Loss:{loss.item():.4f},Edges:{input_graph.number_of_edges()}")
    if (i+1)%50==0:
        print(validate(model,coauth_val_loader))
        # print(validate_w_loss(model,coauth_val_loader_loss,compute_loss))
    
    #convert graph to homogenous for link prediction, say subg=dgl.to_hetero(subh)
    #subg.ntypes gives ['author', 'field_of_study', 'institution', 'paper'] this is mapped to [0,1,2,3]
    #this info for each node is in subh.ndata['_TYPE'][65]
    #subr=dgl.add_reverse_edges(subh)
    #this we do link pred on

