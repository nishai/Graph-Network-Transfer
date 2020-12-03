from comet_ml import Experiment
import argparse
import torch
from torch import optim, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from models import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
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
    'transfer': 'Transferred from pretrained Arxiv model',
    'transfer-damaged': 'Transferred from damaged Arxiv model',
    'self-transfer': 'Transferred from MAG source split',
    'self-transfer-damaged-new-layer': 'Transferred from damaged MAG source split with a new classification layer.',
    'self-transfer-new-layer': 'Transferred from MAG source split with a new classification layer.',
    'meta': 'MAML'
}


# ---------------------------------------------------
# Data
# ---------------------------------------------------

# LOAD Arxiv
arxiv_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
arxiv_data = arxiv_dataset[0]

arxiv_data = T.ToSparseTensor()(arxiv_data)
arxiv_data.adj_t = arxiv_data.adj_t.to_symmetric()
arxiv_data = arxiv_data.to(device)

arxiv_split_idx = arxiv_dataset.get_idx_split()
arxiv_evaluator = Evaluator(name='ogbn-arxiv')
# ---------------------------------------------------

# LOAD MAG
dataset = PygNodePropPredDataset(name="ogbn-mag")
rel_data = dataset[0]

data = Data(
    x=rel_data.x_dict['paper'],
    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
    y=rel_data.y_dict['paper']
).to(device)

# SPLIT INTO SOURCE & TARGET SET
years = rel_data.node_year_dict['paper'].unique()
source_years = years[:5]
target_years = years[5:]

source_nodes = torch.cat([
                    torch.where(rel_data.node_year_dict['paper'] == year)[0]
                    for year in source_years
                ])

target_nodes = torch.cat([
                    torch.where(rel_data.node_year_dict['paper'] == year)[0]
                    for year in target_years
                ])

source_nodes, _ = source_nodes.sort()
target_nodes, _ = target_nodes.sort()

source_edge_index, _ = subgraph(source_nodes, data.edge_index, relabel_nodes=True)
target_edge_index, _ = subgraph(target_nodes, data.edge_index, relabel_nodes=True)

source_data = Data(
                x=rel_data.x_dict['paper'][source_nodes],
                edge_index=source_edge_index,
                y=rel_data.y_dict['paper'][source_nodes]
            )

target_data = Data(
                x=rel_data.x_dict['paper'][target_nodes],
                edge_index=target_edge_index,
                y=rel_data.y_dict['paper'][target_nodes]
            )

data = target_data.to(device) # Train on Target split

# MAG EVALUATOR
evaluator = Evaluator(name="ogbn-mag")
# ---------------------------------------------------

def damage_data(data, percentage=1):
    print('Damaging data...')
    print()

    N = data.num_nodes
    N_damage = int(N * percentage)
    new_data = deepcopy(data)

    idx_damage = np.random.choice(N, N_damage)

    new_data.x[idx_damage] = torch.randn(N_damage, data.num_features)

    return new_data, idx_damage


def train(model, optimiser, data):
    # TRAIN
    model.train()
    optimiser.zero_grad()

    try:
        out = model(data.x, data.adj_t)
    except:
        out = model(data.x, data.edge_index)

    loss = F.nll_loss(out, data.y.squeeze(1))

    loss.backward()
    optimiser.step()

    # EVAL
    y_pred = out.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
            'y_true': data.y,
            'y_pred': y_pred,
        })['acc']

    return loss.item(), acc


def pretrain_arxiv(model, optimiser, data, split_idx, evaluator, model_name, epochs=1000):
    best_valid_acc = 0.0
    train_idx = split_idx['train']

    for epoch in tqdm(range(epochs)):
        # TRAIN
        model.train()
        optimiser.zero_grad()

        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])

        loss.backward()
        optimiser.step()

        # EVAL
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.adj_t)
            y_pred = out.argmax(dim=-1, keepdim=True)

            valid_acc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), 'arxiv_models/{}_arxiv.pth'.format(model_name))

    return best_valid_acc


def pretrain_mag_source(model, optimiser, data, model_name, epochs=1000):
    best_acc = 0.0

    for epoch in tqdm(range(epochs)):
        # TRAIN
        model.train()
        optimiser.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y.squeeze(1))

        loss.backward()
        optimiser.step()

        # EVAL
        y_pred = out.argmax(dim=-1, keepdim=True)
        acc = evaluator.eval({
                'y_true': data.y,
                'y_pred': y_pred,
            })['acc']

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'source/{}_mag_source.pth'.format(model_name))

    return best_acc


def main():
    # ---------------------------------------------------
    # Parsing arguments
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
    assert args.model in ['gcn', 'sage', 'gat', 'gin']
    assert args.type in ['base', 'transfer', 'transfer-damaged', 'self-transfer', 'self-transfer-damaged-new-layer', 'self-transfer-new-layer', 'meta']
    assert args.runs >= 1
    assert args.epochs >= 1
    assert args.lr > 0
    assert args.hidden_dim > 0
    assert args.num_layers > 0

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    # USE MAG MODEL
    model = networks[args.model](
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers
    ).to(device)


    # ---------------------------------------------------
    #  EXPERIMENT DETAILS
    # ---------------------------------------------------
    print('Node Classification Experiment')
    print('MAG task')
    print(exp_description[args.type])
    print('---------------------------------------------------------------------')
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
        print('---------------------------------------------------------------------')
        print('Run #{}'.format(run + 1))
        print()

        # Model initialisation
        if args.type == 'base':
            model.reset_parameters()

        elif args.type  == 'transfer' or args.type=='transfer-damaged':
            # PRETRAIN ON ARXIV
            model = networks[args.model](
                in_channels=128,
                hidden_channels=256,
                out_channels=40,
                num_layers=3
            ).to(device)
            arxiv_optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)


            print('Pretraining model on Arxiv...')

            if args.type=='transfer':
                best_val_acc = pretrain_arxiv(model, arxiv_optimiser, arxiv_data, arxiv_split_idx, arxiv_evaluator, args.model)
            elif args.type=='transfer-damaged':
                arxiv_damaged, _ =  damage_data(arxiv_data)
                best_val_acc = pretrain_arxiv(model, arxiv_optimiser, arxiv_damaged, arxiv_split_idx, arxiv_evaluator, args.model)

            print('Validation accuracy: {:.3}'.format(best_val_acc))

            # load best validation acc model
            model.load_state_dict(
                torch.load( 'arxiv_models/{}_arxiv.pth'.format(args.model) )
            )

            # replace final layer
            if args.model != 'gin':
                model.convs[-1] = layers[args.model](256, dataset.num_classes)
            else:
                model.convs[-1] = layers[args.model](Linear(256, dataset.num_classes))

            model = model.to(device)

        elif args.type == 'self-transfer':
            # PRETRAIN ON SOURCE SPLIT
            model.reset_parameters()
            source_optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

            print('Pretraining model on MAG Source split')
            best_acc = pretrain_mag_source(model, source_optimiser, source_data.to(device), args.model)
            print('Best accuracy: {:.3}'.format(best_acc))

            model.load_state_dict(
                torch.load( 'source/{}_mag_source.pth'.format(args.model) )
            )

        elif args.type == 'self-transfer-new-layer' or args.type == 'self-transfer-damaged-new-layer':
            # PRETRAIN ON SOURCE SPLIT - NEW CLASSIFICATION LAYER
            model.reset_parameters()
            source_optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

            print('Pretraining model on MAG Source split')

            if args.type == 'self-transfer-new-layer':
                best_acc = pretrain_mag_source(model, source_optimiser, source_data.to(device), args.model)
            elif args.type == 'self-transfer-damaged-new-layer':
                source_damaged, _ =  damage_data(source_data)
                best_acc = pretrain_mag_source(model, source_optimiser, source_damaged.to(device), args.model)

            print('Best accuracy: {:.3}'.format(best_acc))

            model.load_state_dict(
                torch.load( 'source/{}_mag_source.pth'.format(args.model) )
            )

            # replace final layer
            if args.model != 'gin':
                model.convs[-1] = layers[args.model](256, dataset.num_classes)
            else:
                model.convs[-1] = layers[args.model](Linear(256, dataset.num_classes))

            model = model.to(device)

        # COMET EXPERIMENT
        experiment = Experiment(project_name='node-classification', display_summary_level=0, auto_metric_logging=False)
        experiment.add_tags([args.model, args.type])
        experiment.log_parameters({
            'hidden_dim' : args.hidden_dim,
            'num_features' : data.num_features,
            'num_classes' : dataset.num_classes,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
        })

        # MAG TARGET TRAINING
        print('Training on MAG')
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs)):
            train_loss, acc = train(model, optimiser, data)

            experiment.log_metric('train_loss', train_loss, step=epoch)
            experiment.log_metric('accuracy', acc, step=epoch)

        experiment.end()

if __name__ == "__main__":
    main()
