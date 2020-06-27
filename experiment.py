import argparse
import numpy as np
from comet_ml import Experiment
import copy
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.nn import VGAE
from torch_geometric.datasets import Planetoid, PPI, Reddit
from tqdm import tqdm
from sklearn.metrics import f1_score, auc, average_precision_score
from models import  GCN, GVAE_Encoder, GraphSAGE, GAT
from utils import reduce_dim_collect_dataset, gvae_split_dataset


"""
Parsing arguments
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=int, default=1)
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--type', type=str, default='base')
parser.add_argument('--num_exps', type=int, default=30)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()
assert args.exp in [1, 2, 3, 4]
assert args.model in ['gcn', 'gvae', 'graphsage', 'gat']
assert args.type in ['base', 'transfer', 'meta']
assert args.num_exps >= 1
assert args.epochs >= 1
assert args.lr > 0

"""
Networks & Datasets
"""
networks  = {
    'gcn': GCN,
    'graphsage': GraphSAGE,
    'gat': GAT,
}

if args.exp == 1:
    exp_target_dataset = [Planetoid('./data', 'Cora')]
    lowest_dim = 500 #PubMed

    if args.type == 'transfer':
        exp_transfer_dataset = [Planetoid('./data', 'Citeseer')]
    elif args.type == 'meta':
        exp_meta_datasets= [Planetoid('./data', 'Citeseer'),
                            Planetoid('./data', 'PubMed')]

    # dimensionality reduction
    n_classes = exp_target_dataset[0].num_classes
    exp_target_dataset = reduce_dim_collect_dataset(exp_target_dataset, lowest_dim)

    if args.model == 'gvae':
        exp_target_dataset = gvae_split_dataset(exp_target_dataset)

elif args.exp == 2:
    exp_target_dataset = [Planetoid('./data', 'Cora')]
    lowest_dim = 50 # PPI  - Too small?

    if args.type == 'transfer':
        exp_transfer_dataset = [PPI('./data')]
    elif args.type == 'meta':
        exp_meta_datasets= [Planetoid('./data', 'Citeseer'),
                            Planetoid('./data', 'PubMed'),
                            PPI('./data'),
                            Reddit('./data')]

    # dimensionality reduction
    n_classes = exp_target_dataset[0].num_classes
    exp_target_dataset = reduce_dim_collect_dataset(exp_target_dataset, lowest_dim)

    if args.model == 'gvae':
        exp_target_dataset = gvae_split_dataset(exp_target_dataset)

elif args.exp == 3:
    exp_target_datasets = {}
    exp_target_datasets['train'] = PPI('./data', split='train')
    exp_target_datasets['val'] = PPI('./data', split='val')

    n_classes = exp_target_datasets['train'].num_classes

    if args.model == 'gvae':
        exp_target_datasets['train']  = gvae_split_dataset(exp_target_datasets['train'])
        exp_target_datasets['val']  = gvae_split_dataset(exp_target_datasets['val'])

    exp_target_dataloaders = {}
    exp_target_dataloaders['train'] = DataLoader(exp_target_datasets['train'], batch_size=1, shuffle=True)
    exp_target_dataloaders['val'] = DataLoader(exp_target_datasets['val'], batch_size=1, shuffle=False)

    lowest_dim = 50

    if args.type == 'transfer':
        exp_transfer_dataset = [Reddit('./data')]
    elif args.type == 'meta':
        exp_meta_datasets= [Planetoid('./data', 'Cora'),
                            Planetoid('./data', 'Citeseer'),
                            Planetoid('./data', 'PubMed')]


# TODO: transfer and meta datasets

"""
Experiment setup
"""
exp_name = 'experiment-{}'.format(args.exp)         # Name of comet.ml project

exp_hyperparams = {
    'hidden_dim' : 256,
    'n_features' : lowest_dim,
    'target_classes' : n_classes,
    'learning_rate': args.lr,
    'num_epochs': args.epochs,
}

for exp_num in range(args.num_exps):
    """
    Base models
    """
    if args.type == 'base':
        experiment = Experiment(project_name=exp_name, display_summary_level=0)
        experiment.log_parameters(exp_hyperparams)
        experiment.add_tags([args.model, args.type])

        if args.model in ['gcn', 'graphsage', 'gat']:
            model = networks[args.model](
                        hidden_dim=exp_hyperparams['hidden_dim'],
                        n_features=exp_hyperparams['n_features'],
                        n_classes=exp_hyperparams['target_classes']
                    ).to(device)
        elif args.model == 'gvae':
            model = VGAE(GVAE_Encoder(
                        hidden_dim=exp_hyperparams['hidden_dim'],
                        n_features=exp_hyperparams['n_features'],
                        n_classes=exp_hyperparams['target_classes']
                    )).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=exp_hyperparams['learning_rate'])

        for epoch in range(args.epochs):
            if args.exp in [1, 2]:    # Single graph dataset
                """
                TRAINING
                """
                data = exp_target_dataset[0].to(device)
                model.train()

                optimizer.zero_grad()

                if args.model in ['gcn', 'graphsage', 'gat']:
                    out = model(data)
                    out = F.log_softmax(out, dim=1)
                    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                elif args.model == 'gvae':
                    z = model.encode(data)
                    loss = model.recon_loss(z, data.train_pos_edge_index) + (1 / data.num_nodes) * model.kl_loss()

                loss.backward()
                optimizer.step()

                experiment.log_metric('loss', loss.item(), step=epoch)

                """
                VALIDATION
                """
                model.eval()

                if args.model in ['gcn', 'graphsage', 'gat']:
                    out = F.log_softmax(model(data), dim=1)
                    _, pred = out.max(dim=1)
                    f1 = f1_score(
                        data.y[data.val_mask].cpu().numpy(),
                        pred[data.val_mask].cpu().numpy(),
                        average='weighted'
                    )

                    experiment.log_metric('test_F1_score', f1, step=epoch)
                elif args.model == 'gvae':
                    with torch.no_grad():
                        z = model.encode(data)
                    auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)

                    experiment.log_metric('auc', auc, step=epoch)
                    experiment.log_metric('ap', ap, step=epoch)


            elif args.exp == 3: # Multiple graphs in dataset
                """
                TRAINING
                """
                loss_op = torch.nn.BCEWithLogitsLoss()

                model.train()
                total_loss = 0

                for data in exp_target_dataloaders['train']:
                    data = data.to(device)
                    optimizer.zero_grad()

                    if args.model in ['gcn', 'graphsage', 'gat']:
                        loss = loss_op(model(data), data.y)
                    elif args.model == 'gvae':
                        z = model.encode(data)
                        loss = model.recon_loss(z, data.train_pos_edge_index) + (1 / data.num_nodes) * model.kl_loss()

                    total_loss += loss.item() * data.num_graphs

                    loss.backward()
                    optimizer.step()

                epoch_loss = total_loss / len(exp_target_dataloaders['train'].dataset)
                experiment.log_metric('loss', epoch_loss, step=epoch)

                """
                VALIDATION
                """
                model.eval()

                if args.model in ['gcn', 'graphsage', 'gat']:
                    ys, preds = [], []

                    for data in exp_target_dataloaders['val']:
                        ys.append(data.y)
                        with torch.no_grad():
                            out = model(data.to(device))
                        preds.append((out > 0).float().cpu())

                    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
                    epoch_f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

                    experiment.log_metric('test_F1_score', epoch_f1, step=epoch)

                elif args.model == 'gvae':
                    total_auc = 0
                    total_ap = 0

                    for data in exp_target_dataloaders['val']:
                        data = data.to(device)

                        with torch.no_grad():
                            z = model.encode(data)
                        auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)

                        total_auc += auc * data.num_graphs
                        total_ap += ap * data.num_graphs

                    epoch_auc = total_auc / len(exp_target_dataloaders['val'].dataset)
                    epoch_ap = total_ap / len(exp_target_dataloaders['val'].dataset)

                    experiment.log_metric('auc', epoch_auc, step=epoch)
                    experiment.log_metric('ap', epoch_ap, step=epoch)

        experiment.end()
        print('Trial {} complete'.format(exp_num))
