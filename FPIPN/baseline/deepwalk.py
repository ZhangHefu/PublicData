
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pickle

def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = 'data/format/adjmat_a_p_t.pkl'
    adjmat = pickle.load(open(path, "rb" ))   
    edges = [(edge[0], edge[2]) for edge in adjmat]
    print(edges[2:5])
    # edges = [('1','2'),('3','4')]
    G = nx.Graph()
    G.add_edges_from(edges)
    # print(G.edges)
    
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=2, workers=10)
    embed_size = 128
    model.train(embed_size=embed_size, window_size=5, iter=2)
    embeddings = model.get_embeddings()
    # pickle.dump(embeddings)
    # print(embeddings)
    pickle.dump(embeddings, open("data/embedding/deepwalk_embed_{}.pkl".format(embed_size), "wb" )) 
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
