from comet_ml import Experiment
import argparse
import itertools
import torch
import networkx as nx
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader
from models import *
from generate import *
from tqdm import tqdm
from sklearn.metrics import f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

networks  = {
    'gcn': GCN,
    'sage': SAGE,
    'gin': GIN,
}

exp_description = {
    'base': 'Random seed initialisation',
    'conf1': 'Transferred from Configuration 1',
    'conf2': 'Transferred from Configuration 2',
    'conf3': 'Transferred from Configuration 3',
}

conf_params = {
    'conf1': {
        'percent_swap': 0.1,
        'percent_damage': 0
    },
    'conf2': {
        'percent_swap': 0.1,
        'percent_damage': 0.95
    },
    'conf3': {
        'percent_swap': 0.95,
        'percent_damage': 0
    },
    'base': 'N/A'
}

BATCH_SIZE=64


# ---------------------------------------------------
# Data
# ---------------------------------------------------

target_dataset = torch.load('target_dataset')
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------------------
# Training Methods
# ---------------------------------------------------

def train(model, device, loader, optimiser):
    num_batches = len(loader) / BATCH_SIZE
    total_loss = 0.0

    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # TRAIN
        model.train()
        out = model(batch)
        optimiser.zero_grad()

        loss = CrossEntropyLoss()(out, batch.y)
        total_loss += loss

        loss.backward()
        optimiser.step()

        # EVAL
        model.eval()
        with torch.no_grad():
            pred = model(batch).argmax(dim=-1)

        y_true.append(batch.y.detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    avg_loss = total_loss / num_batches

    acc = f1_score(
        y_true = y_true,
        y_pred = y_pred,
        average='micro'
    )

    return avg_loss.item(), acc


def pretrain(model, device, loader, optimiser, epochs=500):
    best_acc = 0.0

    for epoch in tqdm(range(epochs)):
        _, acc = train(model, device, loader, optimiser)

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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=5)

    args = parser.parse_args()
    assert args.model in ['gcn', 'sage', 'gin']
    assert args.type in ['base', 'conf1', 'conf2', 'conf3']
    assert args.runs >= 1
    assert args.epochs >= 1
    assert args.lr > 0
    assert args.hidden_dim > 0

    # ---------------------------------------------------
    # Comet Experiment Tags
    # ---------------------------------------------------
    experiment_tags = [args.model, args.type]

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = networks[args.model](
        in_channels=10,
        hidden_channels=args.hidden_dim,
        out_channels=20,
        num_conv_layers=args.num_layers
    ).to(device)


    # ---------------------------------------------------
    #  Experiment details
    # ---------------------------------------------------
    print('Graph Classification Experiment - Synthetic Data')
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

        elif args.type in ['conf1', 'conf2', 'conf3']:
            # Pretrain
            print('Generating source dataset')
            source_dataset = generate_dataset(
                n_classes=10,
                n_per_class=100,
                percent_swap=conf_params[args.type]['percent_swap'],
                percent_damage=conf_params[args.type]['percent_damage']
            )
            source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)
            print()

            model.reset_parameters()
            source_optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

            print('Pre-training model on generated dataset')
            best_acc = pretrain(model, device, source_loader, source_optimiser)
            print('Best accuracy: {:.3}'.format(best_acc))

            model.load_state_dict(
                torch.load( 'source_model.pth' )
            )

        # Comet Experiment
        experiment = Experiment(project_name='graph-classification-synthetic', display_summary_level=0, auto_metric_logging=False)
        experiment.add_tags(experiment_tags)
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : 10,
            'num_classes' : 10,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
            'generator_params': conf_params[args.type]
        })

        # Target training
        print('Training on Target task')
        target_optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs)):
            train_loss, acc = train(model, device, target_loader, target_optimiser)

            experiment.log_metric('train_loss', train_loss, step=epoch)
            experiment.log_metric('accuracy', acc, step=epoch)

        experiment.end()


if __name__ == "__main__":
    main()
