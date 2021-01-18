# Node Classification: Real-World Data 🎯
This directory contains all the code for the real-world node classification experiments.

![](results.jpg)

The experiment runs can be found [here](https://www.comet.ml/graph-net-experiments/node-classification).

## Datasets 🧩

We use two citation networks datasets from _Open Graph Benchmark_: [Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) and [MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag).

## Experiments 🔬
We perform six sets of transfer learning experiments:

| *#* | *Source Task*                            | *Runtime argument*                |
| --- | -----------------------------------------| --------------------------------- |
| *1* | None                                     | `base`                            |
| *2* | Arxiv                                    | `transfer`                        |
| *3* | Arxiv [Damaged features]                 | `transfer-damaged`                |
| *4* | MAG _(Source split)_ [No new layer]      | `self-transfer`                   |
| *5* | MAG _(Source split)_                     | `self-transfer-new-layer`         |
| *6* | MAG _(Source split)_ [Damaged features]  | `self-transfer-damaged-new-layer` |

## Running experiments🏃🏽‍♀️

Running the script `python.py` will run a batch of experiments.

The following parameters may be passed to the script when executed.
