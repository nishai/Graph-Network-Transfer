import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels)) # input layer

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels)) # hidden layers

        self.convs.append(GCNConv(hidden_channels, out_channels)) # output layer

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # input layer

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels)) # hidden layers

        self.convs.append(SAGEConv(hidden_channels, out_channels)) # output layer

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels)) # input layer

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels)) # hidden layers

        self.convs.append(GATConv(hidden_channels, out_channels)) # output layer

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)



class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()

        # input layer
        self.convs.append(
            GINConv(Linear(in_channels, hidden_channels), train_eps=True)
        )

        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(Linear(hidden_channels, hidden_channels), train_eps=True)
            )

        # output layer
        self.convs.append(
            GINConv(Linear(hidden_channels, out_channels), train_eps=True)
        )

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
