# LRGB: Long Range Graph Benchmark

<img src="https://i.imgur.com/2LKoGbu.png" align="right" width="275"/>

In this repo, we provide the source code to train various GNN models on the proposed LRGB datasets. We also provide scripts to run baseline and exploratory experiments.


### Python environment setup with Conda

```bash
conda create -n lrgb python=3.9
conda activate lrgb

pip install torch torchvision torchaudio
pip install torch_geometric==2.0.2 performer-pytorch torchmetrics==0.7.2 ogb wandb pytorch_lightning yacs torch_scatter torch_sparse tensorboardX
```
