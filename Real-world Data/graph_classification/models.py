import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_conv_layers, dropout=0.5):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_conv_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.pooling = global_mean_pool

        self.classify = Linear(hidden_channels, out_channels)

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.classify.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        # convs -> node embedding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # pooling -> graph embedding
        x = self.pooling(x, data.batch)

        # linear -> classification
        x = self.classify(x)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_conv_layers, dropout=0.5):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_conv_layers-1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.pooling = global_mean_pool

        self.classify = Linear(hidden_channels, out_channels)

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.classify.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        # convs -> node embedding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # pooling -> graph embedding
        x = self.pooling(x, data.batch)

        # linear -> classification
        x = self.classify(x)

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_conv_layers, dropout=0.5):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(Linear(in_channels, hidden_channels))
        )

        for _ in range(num_conv_layers-1):
            self.convs.append(
                GINConv(Linear(hidden_channels, hidden_channels))
            )

        self.pooling = global_mean_pool

        self.classify = Linear(hidden_channels, out_channels)

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.classify.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        # convs -> node embedding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # pooling -> graph embedding
        x = self.pooling(x, data.batch)

        # linear -> classification
        x = self.classify(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_conv_layers, dropout=0.5):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))

        for _ in range(num_conv_layers-1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        self.pooling = global_mean_pool

        self.classify = Linear(hidden_channels, out_channels)

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.classify.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        # convs -> node embedding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # pooling -> graph embedding
        x = self.pooling(x, data.batch)

        # linear -> classification
        x = self.classify(x)

        return x
