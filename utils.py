import torch
from sklearn.decomposition import PCA

def reduce_dim(data, dim):
    """
    A method that reduces a graph's feature vectors to a given dimensionality.
    """
    pca = PCA(n_components=dim)
    new_data = pca.fit_transform( data.x.cpu().numpy() )
    data.x = torch.Tensor(new_data)

    return data


def reduce_dim_collect_dataset(datasets, dim):
    """
    A method that reduces dimensionality for multiple graphs or datasets and returns a single list with all the transformed graphs.
    """
    new_dataset = []

    for ds in datasets:
        for data in ds:
            new_dataset.append(reduce_dim(data, dim))

    return new_dataset
