from comet_ml import Experiment
import argparse
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from models import *
from pytorchtools import EarlyStopping
from ogb.graphproppred import PygGraphPropPredDataset , Evaluator
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

networks  = {
    'gcn': GCN,
    'sage': SAGE,
    'gat': GAT,
    'gin': GIN,
}

layers  = {
    'gcn': GCNConv,
    'sage': SAGEConv,
    'gat': GAT,
    'gin': GINConv,
}

exp_description = {
    'base': 'Random seed initialisation',
    'transfer': 'Transferred from pretrained Mol-BBBP model',
    'self-transfer': 'Transferred from Mol-HIV source split',
    'meta': 'MAML'
}

# ---------------------------------------------------
# Data
# ---------------------------------------------------

dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

split_idx = len(dataset) // 2
source_dataset = dataset[:split_idx]
target_dataset = dataset[split_idx:]

BATCH_SIZE = 32

source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

evaluator = Evaluator('ogbg-molhiv')
cls_criterion = torch.nn.BCEWithLogitsLoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    num_batches = len(loader) / BATCH_SIZE
    total_loss = 0.0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            total_loss += loss

            loss.backward()
            optimizer.step()

    return total_loss / num_batches


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--type', type=str, default='base')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=5)

    args = parser.parse_args()
    assert args.model in ['gcn', 'sage', 'gat', 'gin']
    assert args.type in ['base', 'transfer', 'self-transfer', 'meta']
    assert args.runs >= 1
    assert args.epochs >= 1
    assert args.lr > 0
    assert args.hidden_dim > 0
    assert args.num_layers > 0

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    model = networks[args.model](
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_tasks,
        num_conv_layers=args.num_layers
    ).to(device)

    # ---------------------------------------------------
    #  EXPERIMENT DETAILS
    # ---------------------------------------------------
    print('Graph Classification Experiment')
    print('Mol_HIV task')
    print(exp_description[args.type])
    print('----------------------------------------------')
    print('Model: {}'.format(args.model))
    print('Number of runs: {}'.format(args.runs))
    print('Number of epochs: {}'.format(args.epochs))
    print('Learning rate: {}'.format(args.lr))
    print()
    print(model)

    # ---------------------------------------------------
    # EXPERIMENT LOOP
    # ---------------------------------------------------
    for run in range(args.runs):
        print()
        print('Run #{}'.format(run + 1))

        experiment = Experiment(project_name='graph-classification', display_summary_level=0, auto_metric_logging=False)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : dataset.num_features,
            'num_classes' : dataset.num_tasks,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        if args.type == 'base':
            model.reset_parameters()
        elif args.type == 'transfer':
            model.load_state_dict(
                torch.load( './saved_models/molbbbp/{}_molbbbp.pth'.format(args.model) )
            )
        elif args.type == 'self-transfer':
            model.load_state_dict(
                torch.load( './saved_models/source/{}_source_molhiv.pth'.format(args.model) )
            )

        optimizer = optim.Adam(model.parameters(), args.lr)

        for epoch in tqdm(range(args.epochs)):
            train_loss = train(model, device, target_loader, optimizer, dataset.task_type)
            train_performance = eval(model, device, source_loader, evaluator)

            experiment.log_metric('train_loss', train_loss.item(), step=epoch)
            experiment.log_metric('train_roc-auc', train_performance[dataset.eval_metric], step=epoch)

if __name__ == "__main__":
    main()
