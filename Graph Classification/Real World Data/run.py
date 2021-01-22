from comet_ml import Experiment
import argparse
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from models import *
from ogb.graphproppred import PygGraphPropPredDataset , Evaluator
from tqdm import tqdm
from copy import deepcopy, copy
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

networks  = {
    'gcn': GCN,
    'sage': SAGE,
    'gin': GIN,
}

layers  = {
    'gcn': GCNConv,
    'sage': SAGEConv,
    'gin': GINConv,
}

exp_description = {
    'base': 'Random seed initialisation',
    'transfer': 'Transferred from pretrained Mol-BBBP model',
    'transfer-damaged': 'Transferred from pretrained Mol-BBBP model (Damaged features)',
    'self-transfer': 'Transferred from Mol-HIV source split',
    'self-transfer-damaged': 'Transferred from Mol-HIV source split (Damaged features)'
}

BATCH_SIZE = 64

# ---------------------------------------------------
# Data
# ---------------------------------------------------

# Mol-BBBP
bbbp_dataset = PygGraphPropPredDataset(name='ogbg-molbbbp')
bbbp_split_idx = bbbp_dataset.get_idx_split()

train_loader = DataLoader(bbbp_dataset[bbbp_split_idx["train"]], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(bbbp_dataset[bbbp_split_idx["valid"]], batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(bbbp_dataset[bbbp_split_idx["test"]], batch_size=BATCH_SIZE, shuffle=False)

bbbp_evaluator = Evaluator('ogbg-molbbbp')

# Mol-HIV
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

split_idx = len(dataset) // 2
source_dataset = dataset[:split_idx]
target_dataset = dataset[split_idx:]

source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

evaluator = Evaluator('ogbg-molhiv')
cls_criterion = torch.nn.BCEWithLogitsLoss()

# ---------------------------------------------------
# Methods
# ---------------------------------------------------

def damage_data(data):
    """
    Helper method for damaging a portion of the node attribute matrix.
    """
    N = data.num_nodes
    new_data = copy(data)
    new_data.x = torch.randn(N, data.num_features)

    return new_data


def train(model, device, loader, optimizer):
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


def pretrain_molbbbp(model, device, evaluator, optimizer, model_name, epochs=100, damage=False):
    best_val_perf = 0.0
    global bbbp_dataset

    if damage:
        print('Damaging data...')
        new_dataset = []
        for i, d in enumerate(bbbp_dataset):
            new_dataset.append(damage_data(d))

        train_loader = DataLoader( [new_dataset[idx] for idx in bbbp_split_idx["train"] ], batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader( [new_dataset[idx] for idx in bbbp_split_idx["valid"] ], batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader( [new_dataset[idx] for idx in bbbp_split_idx["test"] ], batch_size=BATCH_SIZE, shuffle=False)

    for epoch in tqdm(range(epochs)):
        train_loss = train(model, device, train_loader, optimizer)

        valid_perf = eval(model, device, valid_loader, evaluator)
        valid_rocauc = valid_perf[bbbp_dataset.eval_metric]

        if valid_rocauc > best_val_perf:
            best_val_perf = valid_rocauc
            torch.save(model.state_dict(), 'molbbbp_models/{}_molbbbp.pth'.format(model_name))

    return best_val_perf


def pretrain_source_molhiv(model, device, evaluator, optimizer, model_name, epochs=100, damage=False):
    best_perf = 0.0
    global source_dataset

    if damage:
        print('Damaging data...')
        new_dataset = []
        for i, d in enumerate(source_dataset):
            new_dataset.append(damage_data(d))

        source_loader = DataLoader(new_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in tqdm(range(epochs)):
        train_loss = train(model, device, source_loader, optimizer)
        perf = eval(model, device, source_loader, evaluator)
        rocauc = perf[source_dataset.eval_metric]

        if rocauc > best_perf:
            best_perf = rocauc
            torch.save(model.state_dict(), 'molhiv/{}_source_molhiv.pth'.format(model_name))

    return best_perf


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
    assert args.model in ['gcn', 'sage', 'gin']
    assert args.type in ['base', 'transfer', 'transfer-damaged', 'self-transfer', 'self-transfer-damaged']
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

        # Model initialisation
        if args.type == 'base':
            model.reset_parameters()

        elif args.type in ['transfer', 'transfer-damaged']:
            # Pretrain on Mol-BBBP
            model.reset_parameters()
            bbbp_optimiser = optim.Adam(model.parameters(), lr=0.001)
            to_damage = args.type == 'transfer-damaged'

            print('Pretraining model on Mol-BBBP...')
            best_val_acc = pretrain_molbbbp(model, device, bbbp_evaluator, bbbp_optimiser, args.model, damage=to_damage)
            print('Validation accuracy: {:.3}'.format(best_val_acc))

            model.load_state_dict(
                torch.load( 'molbbbp_models/{}_molbbbp.pth'.format(args.model) )
            )

        elif args.type in ['self-transfer', 'self-transfer-damaged']:
            # Pretrain on Mol-HIV Source Split
            model.reset_parameters()
            source_optimiser = optim.Adam(model.parameters(), lr=0.001)
            to_damage = args.type == 'self-transfer-damaged'

            print('Pretraining model on Mol-HIV Source Task...')
            best_val_acc = pretrain_source_molhiv(model, device, evaluator, source_optimiser, args.model, damage=to_damage)
            print('Validation accuracy: {:.3}'.format(best_val_acc))

            model.load_state_dict(
                torch.load( 'molhiv/{}_source_molhiv.pth'.format(args.model) )
            )

        # Comet Experiment
        experiment = Experiment(project_name='graph-classification', display_summary_level=0, auto_metric_logging=False)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : dataset.num_features,
            'num_classes' : dataset.num_tasks,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        # Mol-HIV Target Training
        print('Training on Mol-HIV')
        optimizer = optim.Adam(model.parameters(), args.lr)

        for epoch in tqdm(range(args.epochs)):
            train_loss = train(model, device, target_loader, optimizer)
            train_performance = eval(model, device, target_loader, evaluator)

            experiment.log_metric('train_loss', train_loss.item(), step=epoch)
            experiment.log_metric('train_roc-auc', train_performance[dataset.eval_metric], step=epoch)


        experiment.end()

if __name__ == "__main__":
    main()
