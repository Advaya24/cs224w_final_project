import matplotlib.pyplot as plt
import torch as t
import random
from tqdm import tqdm
from data.data_lightgcn import load_data
from evaluate import recallK
from lightgcn import LightGCN2
import argparse

if __name__ == '__main__':
    # see if arguments are passed
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=1)
    args = parser.parse_args()
    num_layers = args.num_layers
    print(f"Number of layers: {num_layers}")

    data_dict = load_data()

    # unpack data
    train_graph = data_dict['train_graph']
    valid_graph = data_dict['valid_graph']
    test_graph = data_dict['test_graph']
    paper_feat = data_dict['paper_feat']
    train_pos_u = data_dict['train_pos_u']
    train_pos_v = data_dict['train_pos_v']
    train_neg_u = data_dict['train_neg_u']
    train_neg_v = data_dict['train_neg_v']
    valid_pos_u = data_dict['valid_pos_u']
    valid_pos_v = data_dict['valid_pos_v']
    valid_neg_u = data_dict['valid_neg_u']
    valid_neg_v = data_dict['valid_neg_v']
    test_pos_u = data_dict['test_pos_u']
    test_pos_v = data_dict['test_pos_v']
    test_neg_u = data_dict['test_neg_u']
    test_neg_v = data_dict['test_neg_v']
    num_author = data_dict['num_author']
    num_paper = data_dict['num_paper']
    feat_dim = data_dict['feat_dim']
    valid_authors = data_dict['valid_authors']
    valid_papers = data_dict['valid_papers']
    test_authors = data_dict['test_authors']
    test_papers = data_dict['test_papers']

    print("Training...")

    # model
    model = LightGCN2(None, num_author, feat_dim, 64, num_layers)
    # optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.005)
    train_loss = []
    valid_loss = []
    valid_recall = {
        1: [],
        2: [],
        4: [],
        6: [],
        8: [],
        10: []
    }

    # train
    for epoch in tqdm(range(10)):
        print(f"Epoch: {epoch}")
        model.train()
        optimizer.zero_grad()
        author_embeddings, paper_embeddings = model(train_graph, paper_feat)

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
            # cosine similarity
            pos_score = t.sum(author_embeddings[valid_pos_u] * paper_embeddings[valid_pos_v], dim=1)
            neg_score = t.sum(author_embeddings[valid_neg_u] * paper_embeddings[valid_neg_v], dim=1)
            loss = -t.mean(t.log(t.sigmoid(pos_score - neg_score)))

            valid_loss.append(loss.item())
            print(f"Valid Loss: {loss.item()}")
            val_sample = random.sample(range(valid_papers.shape[0]), 5000)
            for k in valid_recall.keys():
                recK = recallK(valid_papers[val_sample], pos_score, valid_pos_u, neg_score, valid_neg_u, k)
                valid_recall[k].append(recK)
                print(f"Recall@{k}: {recK}")

    # plot loss
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    for k in valid_recall.keys():
        plt.plot(valid_recall[k], label=f'Valid Recall@{k}')
    plt.title(f'Loss curve for LightGCN with {num_layers} GCN layers')
    plt.legend()
    plt.savefig(f'loss_{num_layers}.png')

    # test
    model.eval()
    with t.no_grad():
        # recall K
        author_embeddings, paper_embeddings = model(valid_graph, paper_feat)
        # cosine similarity
        pos_score = t.sum(author_embeddings[test_pos_u] * paper_embeddings[test_pos_v], dim=1)
        neg_score = t.sum(author_embeddings[test_neg_u] * paper_embeddings[test_neg_v], dim=1)
        loss = -t.mean(t.log(t.sigmoid(pos_score - neg_score)))
        print(f"Test Loss: {loss.item()}")
        test_sample = random.sample(range(test_papers.shape[0]), min(10000, test_papers.shape[0]))
        recK = recallK(test_papers[test_sample], pos_score, test_pos_u, neg_score, test_neg_u, 10)
        print(f"Test Recall: {recK}")
