from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def prepareDataforTensorBoard(inp, pred_labels, filename):
    pass


def TSNEreduce(inp, n_comp, perplexity=3 ):
    X_emb = TSNE(n_components=n_comp, learning_rate='auto',init='random', perplexity=perplexity).fit_transform(inp)
    return X_emb 


def plot2D_tsne(inp, true_labels, pred_labels, name):

    # flatten    
    tot_samples = inp.shape[0]
    a, b        = inp[0].shape
    inp_flat    = np.zeros((tot_samples, a*b))

    for i, item in enumerate(inp):
        inp_flat[i, :] =  np.array(item).reshape(-1, )

    #reduce
    inp_emb = TSNEreduce(inp_flat, 2)
    print(f'TSNE reduced from {inp_flat.shape} to {inp_emb.shape}')
    
    #plot
    for cls in range(10):
        x = inp_emb[pred_labels == cls, 0]
        y = inp_emb[pred_labels == cls, 1]
        plt.scatter(x, y, label = cls)

    plt.legend()
    plt.savefig(name)
    plt.show()
        
def plot3D_tsne(inp, true_labels, pred_labels, name):
        # flatten    
    tot_samples = inp.shape[0]
    a, b        = inp[0].shape
    inp_flat    = np.zeros((tot_samples, a*b))

    for i, item in enumerate(inp):
        inp_flat[i, :] =  np.array(item).reshape(-1, )

    #reduce
    inp_emb = TSNEreduce(inp_flat, 3)
    print(f'TSNE reduced from {inp_flat.shape} to {inp_emb.shape}')
    
    #plot

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for cls in range(10):
        x = inp_emb[pred_labels == cls, 0]
        y = inp_emb[pred_labels == cls, 1]
        z = inp_emb[pred_labels == cls, 2]
        
        ax.scatter(x, y, z, label = cls)

    plt.legend()
    plt.savefig(name)
    plt.show()

