import torch
from torch import optim, nn
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from models import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from tqdm import tqdm
import matplotlib.pyplot as plt
import learn2learn as l2l

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
