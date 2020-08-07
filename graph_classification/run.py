from comet_ml import Experiment
import argparse
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
# from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset
from torch_geometric.data import Data
# from torch_geometric.utils import subgraph
from models import *
from pytorchtools import EarlyStopping
from ogb.graphproppred import PygGraphPropPredDataset , Evaluator
from tqdm import tqdm
# import learn2learn as l2l

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
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=5)

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
dataset = PygGraphPropPredDataset(name='ogbg-molpcba')
split_idx = dataset.get_idx_split()

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

evaluator = Evaluator('ogbg-molpcba')

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

"""
Helpers
"""
def train(model, device, loader, optimizer, task_type):
    model.train()

    num_batches = len(loader) / args.batch_size
    total_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            total_loss += loss

            loss.backward()
            optimizer.step()

    return total_loss / num_batches


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Evaluation", leave=False)):
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


# ---------------------------------------------------
# Base Experiment: Mol-PCBA - Random Seed
# ---------------------------------------------------
if  args.type == 'base':
    print('Graph Classification Experiment')
    print('Mol-PCBA task with random initialisation')
    print('----------------------------------------------')
    print('Model: {}'.format(args.model))
    print('Number of runs: {}'.format(args.runs))
    print('Number of epochs: {}'.format(args.epochs))
    print('Batch-size: {}'.format(args.batch_size))
    print('Learning rate: {}'.format(args.lr))
    print()

    model = networks[args.model](
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_tasks,
        num_conv_layers=args.num_layers
    ).to(device)

    print(model)

    model.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for run in range(args.runs):
        print()
        print('Run #{}'.format(run + 1))

        experiment = Experiment(project_name='graph-classification', display_summary_level=0, auto_metric_logging=False)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : dataset.num_features,
            'num_classes' : dataset.num_tasks,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        for epoch in range(args.epochs):
            print('Epoch #{}'.format(epoch))

            # ----------------------------------
            # TRAINING
            # ----------------------------------
            train_loss = train(model, device, train_loader, optimizer, dataset.task_type)
            experiment.log_metric('train_loss', train_loss.item(), step=epoch)

            # ----------------------------------
            # EVALUATION
            # ----------------------------------
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            experiment.log_metric('train_prcauc', train_perf[dataset.eval_metric], step=epoch)
            experiment.log_metric('validation_prcauc', valid_perf[dataset.eval_metric], step=epoch)
            experiment.log_metric('test_prcauc', test_perf[dataset.eval_metric], step=epoch)
