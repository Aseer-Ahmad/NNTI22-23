from tsne_torch import TorchTSNE as TSNE

def TSNEreduce(inp, n_comp, perplexity=30, n_iter = 1000 ):
    X_emb = TSNE(n_components=n_comp, perplexity=perplexity, n_iter=n_iter, verbose=True).fit_transform(inp)
    return X_emb


def plot2D(inp, pred_labels, true_labels):
    pass

def plot3D(inp, pred_labels, true_labels):
    pass

