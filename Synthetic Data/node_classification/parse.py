import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx

def read_graph(path, save_to=None):
    '''
    Method to read a .graph file, parse it and return a PyTorch-Geometric graph

    Parameters:
        path: path to .graph file
        save_to (optional): path to save PyG graph to
    Returns:
        data: the parsed PyG graph
    '''
    # read file
    f = open(path, 'r')
    file = f.read().splitlines()

    mode = 'vertex'

    node_attrs={}
    node_labels={}
    edges=[]

    for line in file[1:]:
        if line == '#':
            pass

        elif line == '# Edges':
            mode = 'edge'

        else:
            if mode == 'vertex':
                # parse node
                node_num, node_attr, node_class = line.split(';')

                node_num = int(node_num)
                node_class = int(node_class)
                node_attr = [float(attr) for attr in node_attr.split('|')]

                node_attrs[node_num] = node_attr
                node_labels[node_num] = node_class

            elif mode == 'edge':
                # parse edge
                i, j = line.split(';')
                i = int(i); j = int(j)

                edges.append( [i, j] )

    # construct node mapping to consecutive indices
    node_idx = np.array(list(node_attrs.keys()))
    new_idxs = np.arange(len(node_idx))
    mapping = { idx: new_idx for idx, new_idx in zip(node_idx, new_idxs) }

    # create attribute matrix X, and label vector y
    X = torch.tensor( [node_attrs[i] for i in node_idx] )
    y = torch.tensor( [node_labels[i] for i in node_idx] )

    # apply mapping to get edge_index
    edge_index = torch.tensor( [ [mapping[i], mapping[j]] for i, j in edges] ).T

    # create PyG data object
    data = Data(
        x=X,
        edge_index=edge_index,
        y=y
    )

    # save PyG graph
    if save_to != None:
        torch.save(data, save_to)

    return data
