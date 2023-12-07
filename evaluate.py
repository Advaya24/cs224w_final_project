import numpy as np
import torch

def metrics(model, test_graph):
    model.eval()
    auth_emb, paper_emb = model(test_graph)
    auth_emb = auth_emb.detach().cpu().numpy()
    paper_emb = paper_emb.detach().cpu().numpy()
	
    # get recall@k
    k = 10
    recall = []
    for u in range(model.userNum):
        # compute the similarity scores
        auth_emb_u = auth_emb[u]
        score = np.dot(auth_emb_u, paper_emb.T)
        # get the top k indices
        pred = np.argpartition(score, -k)[-k:]
        # find how many of the top k are in test_graph
        hit = len(set(pred).intersection(set(test_graph.successors(u).numpy())))
        recall.append(hit / k)


    recall = np.array(recall).mean()
    return recall

    
	