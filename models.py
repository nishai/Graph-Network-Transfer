import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, VGAE


class GCN(torch.nn.Module):
    """
    Module for a two-layer Graph Convolution Network in PyTorch Geometric
    """
    def __init__(self, hidden_dim, n_features, n_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GVAE_Encoder(torch.nn.Module):
    """
    Module for a two GCN layer Encoder to be passed to the PyTorch Geometric VGAE Module.
    """
    def __init__(self, hidden_dim, n_features, n_classes):
        super(GVAE_Encoder, self).__init__()
        self.conv1 = GCNConv(n_features, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, n_classes)
        self.conv_logvar = GCNConv(hidden_dim, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.train_pos_edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class GraphSAGE(torch.nn.Module):
    """
    Module for a two-layer GraphSAGE network in PyTorch Geometric
    """
    def __init__(self, hidden_dim, n_features, n_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(n_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """
    Module for a two-layer Graph Attention Network in PyTorch Geometric
    """
    def __init__(self, hidden_dim, n_features, n_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(n_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
