import argparse
import numpy as np
from comet_ml import Experiment
import copy
import torch
import torch.nn.functional as F
import torch_geometric
# from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, PPI, Reddit
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, auc, average_precision_score
from .models import  GCN, GVAE_Encoder, GraphSAGE, GAT


"""
Parsing arguments & experiment setup
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=int, default=1)
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--type', type=str, default='base')
parser.add_argument('--num_exps', type=int, default=30)
parser.add_argument('--epochs', type=int, default=200)

args = parser.parse_args()
assert args.exp in [1, 2, 3, 4]
assert args.model in ['gcn', 'gvae', 'graphsage', 'gat']
assert args.type in ['base', 'transfer', 'meta']
assert args.num_exps >= 1
assert args.epochs >= 1

hyperparams = {}

if args.exp == 1:
    exp_target_dataset = Planetoid('./data', 'Cora')
    lowest_dim = 500 #PubMed

    if args.type == 'transfer':
        exp_transfer_dataset = Planetoid('./data', 'Citeseer')
    elif args.type == 'meta':
        exp_meta_datasets= [Planetoid('./data', 'Citeseer'),
                            Planetoid('./data', 'PubMed')]

elif args.exp == 2:
    exp_target_dataset = Planetoid('./data', 'Cora')
    lowest_dim = 50 # PPI  - Too small?

    if args.type == 'transfer':
        exp_transfer_dataset = PPI('./data')
    elif args.type == 'meta':
        exp_meta_datasets= [Planetoid('./data', 'Citeseer'),
                            Planetoid('./data', 'PubMed'),
                            PPI('./data'),
                            Reddit('./data')]

elif args.exp == 3:
    exp_target_dataset = PPI('./data')
    lowest_dim = 50 # PPI  - Too small?

    if args.type == 'transfer':
        exp_transfer_dataset = Reddit('./data')
    elif args.type == 'meta':
        exp_meta_datasets= [Planetoid('./data', 'Cora'),
                            Planetoid('./data', 'Citeseer'),
                            Planetoid('./data', 'PubMed')]

EXP_NAME = 'experiment-{}'.format(args.exp)         # Name of comet.ml project
