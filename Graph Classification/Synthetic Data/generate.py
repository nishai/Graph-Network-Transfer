import itertools
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from copy import deepcopy

def damage_data(data, percentage):
    """
    Helper method for damaging a portion of the node attribute matrix.
    """
    N = data.num_nodes
    N_damage = int(N * percentage)
    new_data = deepcopy(data)
    # new_data.x = new_data.x.cpu()

    idx_damage = np.random.choice(N, N_damage, replace=False)
    new_data.x[idx_damage] = torch.randn(N_damage, data.num_features, dtype=new_data.x.dtype)

    return new_data


def generate_graph(n, m, label):
    """
    Generates a single graph using the Barabasi-Albert generator in PyG Data format without attributes.

    Params:
        n: Number of nodes
        m: Number of edges to attach from a new node to existing nodes
        label: Class label associated with graph

    Returns:
        data: PyTorch Geometric Data object of the generator
    """
    g = nx.generators.random_graphs.barabasi_albert_graph(n, m)

    data = from_networkx(g)
    data.y = label

    return data


def create_graphs_for_class(dataframe, label, m):
    """
    Creates graphs and assigns attributes from dataframe. For use within generate_dataset.

    Params:
        dataframe: The dataframe containing node attributes as rows
        label: Class label associated with the set of graphs
        m: Barabasi-Albert m parameter (see generate_graph)

    Returns:
        datapoints: A list containing PyG graphs with node attributes
    """
    n_samples = len(dataframe)
    datapoints = []
    curr_idx = 0

    #assign node attributes as per dataframe until rows run out
    while curr_idx < n_samples:
        num_nodes = 30 #np.random.randint(25,35)
        if curr_idx + num_nodes > n_samples: break

        g = generate_graph(num_nodes, m, label)
        attr = dataframe.iloc[curr_idx:curr_idx+num_nodes, dataframe.columns != 'y'].values # slices next unused region in df
        g.x = torch.tensor(attr)

        datapoints.append(g)
        curr_idx += num_nodes

    return datapoints


def generate_dataset(n_classes=10, n_per_class=100, n_features=10, n_informative=8, percent_swap=0, percent_damage=0):
    """
    Generates a dataset for graph classification.
    Generates each class using create_graphs_for_class and attributes using sklearn.datasets.make_classification.

    Params:
        n_classes:
            Number of classes in dataset.
            Default 10
        n_per_class:
            Number of graphs per class.
            Default 100
        n_features:
            The length of the node attribute vectors.
            Default 10
        n_informative:
            make_classification's n_informative parameter
        percent_swap:
            The portion of graphs to swap.
            Default 0.
            Range = [0, 1].
            Higher value = higher structural inertia.
        percent_damage:
            Whether to replace node attributes with random values.
            Default False.
            Higher value = higher attribute inertia.

    Returns:
        data: PyTorch Geometric Data object of the generator
    """
    # create attribute-level classification task
    X, y = make_classification(
        n_samples=n_classes * n_per_class * 30,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
    )

    df = pd.DataFrame(data=X)
    df['y'] = y

    # assign attributes to various classes per make_classification
    data_list = [create_graphs_for_class(df[df.y==i], i, m=i+1) for i in range(n_classes)]

    dataset = list(itertools.chain.from_iterable(data_list)) #

    # swap graph adjacencies
    if percent_swap > 0:
        num_to_swap = int(percent_swap * len(dataset))

        to_swap = zip(
            np.random.choice(len(dataset), num_to_swap, replace=False),
            np.random.choice(len(dataset), num_to_swap, replace=False),
        )

        for a,b in to_swap:
            dataset[a].edge_index, dataset[b].edge_index = dataset[b].edge_index, dataset[a].edge_index

    # damage attribute matrix
    if percent_damage > 0:
        for i, d in enumerate(dataset):
            dataset[i] = damage_data(d, percent_damage)

    return dataset
