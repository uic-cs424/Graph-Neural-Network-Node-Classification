from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt

def visualize(h, color, path):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.title("t-SNE of Node Embeddings")
    plt.savefig(path)
