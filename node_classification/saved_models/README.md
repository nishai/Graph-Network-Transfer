# Node Classification - MAG
# Pretrained Models

All pretrained models are 3 layer GNNs with a hidden dimensionalit of 256

## Arxiv pretrained models 

Input dim: 128

Hidden dim: 256

Output dim: 40

| File            | Model     | Train accuracy | Validation accuracy | Test accuracy |
|-----------------|-----------|----------------|---------------------|---------------|
| `gcn_arxiv.pth` | GCN       | 0.78           | 0.73                | 0.72          |
| `sage_arxiv.pth`| GraphSAGE | 0.76           | 0.72                | 0.71          |
| `gin_arxiv.pth` | GIN       | 0.78           | 0.69                | 0.66          |
| `gat_arxiv`     | GAT       | 0.78           | 0.73                | 0.72          |


## MAG Source split pretrained models 

Input dim: 128

Hidden dim: 256

Output dim: 349

| File            | Model     | Train accuracy |
|-----------------|-----------|----------------|
| `gcn_arxiv.pth` | GCN       | 0.274          |
| `sage_arxiv.pth`| GraphSAGE | 0.287          |
| `gin_arxiv.pth` | GIN       | 0.280          |
