from comet_ml import Experiment
import argparse
import torch
from torch import optim, nn
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from pytorchtools import EarlyStopping
from models import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from tqdm import tqdm
import learn2learn as l2l

device = 'cuda' if torch.cuda.is_available() else 'cpu'

networks  = {
    'gcn': GCN,
    'sage': SAGE,
    'gat': GAT,
}

layers  = {
    'gcn': GCNConv,
    'sage': SAGEConv,
    'gat': GATConv,
}

"""
Parsing arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--type', type=str, default='base')
parser.add_argument('--runs', type=int, default=30)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=3)

args = parser.parse_args()
assert args.model in ['gcn', 'sage', 'gat']
assert args.type in ['base', 'transfer', 'meta']
assert args.runs >= 1
assert args.epochs >= 1
assert args.lr > 0
assert args.hidden_dim > 0
assert args.num_layers > 0


"""
Data
"""
dataset = PygNodePropPredDataset(name="ogbn-mag")
rel_data = dataset[0]

data = Data(
    x=rel_data.x_dict['paper'],
    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
    y=rel_data.y_dict['paper']
).to(device)

if args.type in ['base', 'transfer']:
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']['paper'].to(device)
valid_idx = split_idx['valid']['paper'].to(device)

evaluator = Evaluator(name="ogbn-mag")


# ---------------------------------------------------
# Base Experiment: MAG - Random Seed
# ---------------------------------------------------
if  args.type == 'base':
    print('Node Classification Experiment')
    print('MAG task with random initialisation')
    print('----------------------------------------------')
    print('Model: {}'.format(args.model))
    print('Number of runs: {}'.format(args.runs))
    print('Number of epochs: {}'.format(args.epochs))
    print('Learning rate: {}'.format(args.lr))
    print()

    model = networks[args.model](
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers
    ).to(device)

    print(model)

    for run in range(args.runs):
        print()
        print('Run #{}'.format(run + 1))

        experiment = Experiment(project_name='node-classification', display_summary_level=0)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : data.num_features,
            'num_classes' : dataset.num_classes,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        model.reset_parameters()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        early_stopping = EarlyStopping(patience=25, verbose=False)

        for epoch in tqdm(range(args.epochs)):
            #-----------------------------------
            #   TRAIN
            # ----------------------------------
            model.train()
            optimiser.zero_grad()

            out = model(data.x, data.adj_t)[train_idx]
            loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
            loss.backward()
            optimiser.step()

            experiment.log_metric('train_loss', loss.item(), step=epoch)

            #-----------------------------------
            #   EVAL
            # ----------------------------------
            with torch.no_grad():
                model.eval()

                out = model(data.x, data.adj_t)
                y_pred = out.argmax(dim=-1, keepdim=True)

                train_acc = evaluator.eval({
                    'y_true': data.y[split_idx['train']['paper']],
                    'y_pred': y_pred[split_idx['train']['paper']],
                })['acc']

                valid_acc = evaluator.eval({
                    'y_true': data.y[split_idx['valid']['paper']],
                    'y_pred': y_pred[split_idx['valid']['paper']],
                })['acc']

                test_acc = evaluator.eval({
                    'y_true': data.y[split_idx['test']['paper']],
                    'y_pred': y_pred[split_idx['test']['paper']],
                })['acc']

                experiment.log_metric('train_accuracy', train_acc, step=epoch)
                experiment.log_metric('validation_accuracy', valid_acc, step=epoch)
                experiment.log_metric('test_accuracy', test_acc, step=epoch)

                valid_loss = F.nll_loss(out[valid_idx], data.y.squeeze(1)[valid_idx])
                experiment.log_metric('validation_loss', valid_loss.item(), step=epoch)

                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break


# ---------------------------------------------------
# Transfer Experiment: Arxiv -> MAG
# ---------------------------------------------------
elif args.type == 'transfer':
    print('Node Classification Experiment')
    print('MAG task with transferred Arxiv model')
    print('----------------------------------------------')
    print('Model: {}'.format(args.model))
    print('Number of runs: {}'.format(args.runs))
    print('Number of epochs: {}'.format(args.epochs))
    print('Learning rate: {}'.format(args.lr))
    print()

    model = networks[args.model](
        in_channels=128,
        hidden_channels=256,
        out_channels=40,
        num_layers=3
    )
    model.load_state_dict(torch.load( './saved_models/{}_arxiv.pth'.format(args.model) ))
    model.convs[-1] = layers[args.model](256, dataset.num_classes)
    model = model.to(device)

    print('Transferred Arxiv model for MAG...')
    print(model)

    for run in range(args.runs):
        print()
        print('Run #{}'.format(run + 1))

        experiment = Experiment(project_name='node-classification', display_summary_level=0)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : data.num_features,
            'num_classes' : dataset.num_classes,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        model.reset_parameters()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        early_stopping = EarlyStopping(patience=25, verbose=False)

        for epoch in tqdm(range(args.epochs)):
            #-----------------------------------
            #   TRAIN
            # ----------------------------------
            model.train()
            optimiser.zero_grad()

            out = model(data.x, data.adj_t)[train_idx]
            loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
            loss.backward()
            optimiser.step()

            experiment.log_metric('train_loss', loss.item(), step=epoch)

            #-----------------------------------
            #   EVAL
            # ----------------------------------
            with torch.no_grad():
                model.eval()

                out = model(data.x, data.adj_t)
                y_pred = out.argmax(dim=-1, keepdim=True)

                train_acc = evaluator.eval({
                    'y_true': data.y[split_idx['train']['paper']],
                    'y_pred': y_pred[split_idx['train']['paper']],
                })['acc']

                valid_acc = evaluator.eval({
                    'y_true': data.y[split_idx['valid']['paper']],
                    'y_pred': y_pred[split_idx['valid']['paper']],
                })['acc']

                test_acc = evaluator.eval({
                    'y_true': data.y[split_idx['test']['paper']],
                    'y_pred': y_pred[split_idx['test']['paper']],
                })['acc']

                experiment.log_metric('train_accuracy', train_acc, step=epoch)
                experiment.log_metric('validation_accuracy', valid_acc, step=epoch)
                experiment.log_metric('test_accuracy', test_acc, step=epoch)

                valid_loss = F.nll_loss(out[valid_idx], data.y.squeeze(1)[valid_idx])
                experiment.log_metric('validation_loss', valid_loss.item(), step=epoch)

                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break
