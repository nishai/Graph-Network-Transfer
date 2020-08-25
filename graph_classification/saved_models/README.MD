# Graph Classification - Mol-HIV
# Pretrained Models

All pretrained models have 5 Graph Embedding layers  followed by a fully connected classification layer. ALl hidden dimensionalities are 300.

## Mol-BBBP pretrained models 

Input dim: 9

Hidden dim: 300

Output dim: 1

| File              | Model     | Train accuracy | Validation accuracy | Test accuracy |
|-------------------|-----------|----------------|---------------------|---------------|
| `gcn_molbbbp.pth` | GCN       | 0.90           | 0.91                | 0.66          |
| `sage_molbbbp.pth`| GraphSAGE | 0.92           | 0.93                | 0.68          |
| `gin_molbbbp.pth` | GIN       | 0.91           | 0.94                | 0.65          |


## Mol-HIV Source split pretrained models 

Input dim: 9

Hidden dim: 300

Output dim: 1

| File                    | Model     | Train accuracy |
|-------------------------|-----------|----------------|
| `gcn_source_molhiv.pth` | GCN       | 0.77           |
| `sage_source_molhiv.pth`| GraphSAGE | 0.77           |
| `gin_source_molhiv.pth` | GIN       | 0.78           |
