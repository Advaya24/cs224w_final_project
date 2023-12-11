import numpy as np
import torch as t
from tqdm import tqdm


def recallK_author_to_paper(valid_authors, pos_score, valid_pos_u, neg_score, valid_neg_u, k):
    # u is author, v is paper
    recs = []
    for author in tqdm(valid_authors):
        pos_papers = (valid_pos_u == author)
        neg_papers = (valid_neg_u == author)
        curr_pos_score = pos_score[pos_papers]
        curr_neg_score = neg_score[neg_papers]

        num_pos = curr_pos_score.shape[0]
        all_scores = t.cat([curr_pos_score, curr_neg_score], dim=0)

        # assert all_scores.shape == (t.sum(pos_papers) + t.sum(neg_papers),)
        if k > all_scores.shape[0]:
            continue

        topk_indices = t.topk(all_scores, k)[1]
        recs.append((topk_indices < num_pos).sum() / num_pos)
    print(f"Fraction of authors used {len(recs)/valid_authors.shape[0]}")
    return np.mean(recs)

def recallK(valid_papers, pos_score, valid_pos_v, neg_score, valid_neg_v, k):
    # u is author, v is paper
    recs = []
    for paper in tqdm(valid_papers):
        pos_authors = (valid_pos_v == paper)
        neg_authors = (valid_neg_v == paper)
        curr_pos_score = pos_score[pos_authors]
        curr_neg_score = neg_score[neg_authors]

        num_pos = curr_pos_score.shape[0]
        all_scores = t.cat([curr_pos_score, curr_neg_score], dim=0)

        # # assert all_scores.shape == (t.sum(pos_papers) + t.sum(neg_papers),)
        # if k > all_scores.shape[0]:
        #     continue

        # recall at k only makes sense if there are at least k positive examples
        if num_pos < k:
            continue

        topk_indices = t.topk(all_scores, k)[1]
        recs.append((topk_indices < num_pos).sum() / num_pos)
    print(f"Fraction of papers used {len(recs)/valid_papers.shape[0]}")
    return np.mean(recs)