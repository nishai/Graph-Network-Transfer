import itertools
import torch
import networkx as nx
from torch_geometric.utils import from_networkx


def generate_data(n, p, label, attr_len=10, mean=0, std=1):
    """
    Generates a single graph using the Erdős-Renyi generator in PyG Data format.
    Attribute values are generated using a Normal distribution with specified mean and std.

    Params:
        n: Number of nodes
        p: Probability for edge creation
        label: Class label associated with graph
        attr_len: Length of node attribute vector [default: 10]
        mean: Mean for Normal attribute generator [default: 0]
        std: Standard Deviation for Normal attribute generator [default: 1]

    Returns:
        data: PyTorch Geometric Data object of the generator
    """
    g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    data = from_networkx(g)
    data.x = torch.normal(mean=mean, std=std, size=(data.num_nodes, attr_len))
    data.y = label

    return data


def generate_dataset(params):
    """
    Generates a dataset with specified parameters

    Parameters:
        params: a list of parameters specifying the dataset properties in the following format
            [
                {'N': a, 'n': b, 'p': c, 'mean': d, 'std': e},
                ...
            ]
    """
    data_list = [

        # generate a list of a single class defined by params
        [
         generate_data(p['n'], p['p'], mean=p['mean'], std=p['std'], label=i)
         for _ in range(p['N'])
        ]

        # for all sets of params
        for i, p in enumerate(params)
    ]

    num_classes = len(params)
    dataset = list(itertools.chain.from_iterable(data_list))

    return dataset, num_classes
