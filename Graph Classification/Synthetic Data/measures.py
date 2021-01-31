import torch
from torch_geometric.utils import to_networkx
import numpy as np

def inertia(dataset, mode='structure'):
    def avg_degree(data):
        g = to_networkx(data)
        deg = np.array([d[1] for d in g.degree])
        return deg.mean()

    if mode == 'structure':
        x = torch.tensor([avg_degree(d) for d in dataset])
    elif mode == 'attributes':
        x = torch.stack([d.x.mean(axis=0) for d in dataset])

    g = x.mean(axis=0)  # cluster centroid

    d = torch.tensor(
        [ torch.dist(deg, g)**2 for deg in x ]  # squared Euclidean distance btwn each degree and the centroid
    ).sum()     # sum of all distances

    return d


def get_communities(dataset):
    ds_labels = torch.tensor([d.y for d in dataset])

    communities = [
        torch.where(ds_labels==i)[0]
        .tolist()

        for i in ds_labels.unique()
    ]

    return communities


def within_inertia(dataset, mode='structure', communities=None):
    if communities == None:
        communities = get_communities(dataset)

    inter_cluster_inertia = torch.tensor([
            inertia( [dataset[ci] for ci in c] , mode=mode)
            for c in communities
        ]).sum()

    wi = inter_cluster_inertia / inertia(dataset, mode=mode)

    return wi.item()
