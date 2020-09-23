from comet_ml import Experiment
import argparse
import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from models import *
from parse import read_graph
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

networks  = {
    'gcn': GCN,
    'sage': SAGE,
    'gin': GIN,
}

exp_description = {
    'base': 'Random seed initialisation',
    'configuration_1': 'Transferred from pretrained model - Configuration 1',
    'configuration_2': 'Transferred from pretrained model - Configuration 2',
    'configuration_3': 'Transferred from pretrained model - Configuration 3',
    'meta': 'MAML'
}

# ---------------------------------------------------
# Data
# ---------------------------------------------------

target_data = read_graph('benchmark/configuration_4.graph')
target_data.to(device)

# ---------------------------------------------------
# Training Methods
# ---------------------------------------------------

def train(model, optimiser, data):
    # TRAIN
    model.train()
    optimiser.zero_grad()

    try:
        out = model(data.x, data.adj_t)
    except:
        out = model(data.x, data.edge_index)

    loss = F.nll_loss(out, data.y)

    loss.backward()
    optimiser.step()

    # EVAL
    y_pred = out.argmax(dim=-1, keepdim=True)
    acc = accuracy_score(
            y_true=data.y.cpu().numpy(),
            y_pred=y_pred.cpu().numpy()
        )

    return loss.item(), acc


def pretrain(model, optimiser, dataset_name, epochs=1000):
    source_data = read_graph('benchmark/{}.graph'.format(dataset_name))
    source_data.to(device)

    best_acc = 0.0

    for epoch in tqdm(range(epochs)):
        _, acc = train(model, optimiser, source_data)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'source_model.pth')

    return best_acc


# ---------------------------------------------------
# Main loop
# ---------------------------------------------------


def main():
    # ---------------------------------------------------
    # Parsing Arguments
    # ---------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--type', type=str, default='base')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)

    args = parser.parse_args()
    assert args.model in ['gcn', 'sage', 'gin']
    assert args.type in ['base', 'configuration_1', 'configuration_2', 'configuration_3', 'meta']
    assert args.runs >= 1
    assert args.epochs >= 1
    assert args.lr > 0
    assert args.hidden_dim > 0
    assert args.num_layers > 0


    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = networks[args.model](
        in_channels=target_data.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=len(target_data.y.unique()),
        num_layers=args.num_layers
    ).to(device)


    # ---------------------------------------------------
    #  Experiment details
    # ---------------------------------------------------
    print('Node Classification Experiment - Synthetic Data')
    print('Target task: Configuration 4')
    print(exp_description[args.type])
    print('---------------------------------------------------------------------')
    print('Model: {}'.format(args.model))
    print('Number of runs: {}'.format(args.runs))
    print('Number of epochs: {}'.format(args.epochs))
    print('Learning rate: {}'.format(args.lr))
    print()
    print(model)


    # ---------------------------------------------------
    # Experiment loop
    # ---------------------------------------------------
    for run in range(args.runs):
        print()
        print('---------------------------------------------------------------------')
        print('Run #{}'.format(run + 1))
        print()

        # Model initialisation
        if args.type == 'base':
            # Random init
            model.reset_parameters()

        elif args.type in ['configuration_1', 'configuration_2', 'configuration_3']:
            # Pretrain on other benchmark datasets
            model.reset_parameters()
            source_optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

            print('Pre-training model on ' + args.type)
            best_acc = pretrain(model, source_optimiser, args.type)
            print('Best accuracy: {:.3}'.format(best_acc))

            model.load_state_dict(
                torch.load( 'source_model.pth' )
            )

        # Comet Experiment
        experiment = Experiment(project_name='node-classification-synthetic', display_summary_level=0, auto_metric_logging=False)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : target_data.num_features,
            'num_classes' : len(target_data.y.unique()),
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        # Target training
        print('Training on Configuration 4')
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs)):
            train_loss, acc = train(model, optimiser, target_data)

            experiment.log_metric('train_loss', train_loss, step=epoch)
            experiment.log_metric('accuracy', acc, step=epoch)

        experiment.end()


if __name__ == "__main__":
    main()
