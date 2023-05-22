# LRGB: Long Range Graph Benchmark

<img src="https://i.imgur.com/2LKoGbu.png" align="right" width="275"/>

In this repo, we provide the source code to train various GNN models on the proposed LRGB datasets. We also provide scripts to run baseline and exploratory experiments.


### Python environment setup with Conda

```bash
conda create -n lrgb python=3.9
conda activate lrgb

pip install torch torchvision torchaudio
pip install torch_geometric==2.0.2 performer-pytorch torchmetrics==0.7.2
pip install ogb wandb pytorch_lightning yacs 
pip install torch_scatter torch_sparse 
pip install tensorboardX
pip install numba
```

### Training the Graph models

The configuration for training graph model on a particular LRGB dataset is provided in src/configs directory. The training process can be started by running the following script,

#### For GCN and Transformers+LapPE

```bash
python src/main.py --cfg src/configs/GCN/vocsuperpixels-GCN.yaml device cuda:0 wandb.use False

python src/main.py --cfg src/configs/GT/vocsuperpixels-Transformer+LapPE.yaml device cuda:0 wandb.use False
```

#### For E(n)-Invariant (EGNN) and E(n)-Equivariant (EGNN2)

```bash
python src/main_egnn.py --cfg src/configs/EGNN/vocsuperpixels-EGNN.yaml device cuda:0 wandb.use False

python src/main_egnn.py --cfg src/configs/EGNN/vocsuperpixels-EGNN2.yaml device cuda:0 wandb.use False
```

### W&B logging
To use W&B logging, set `wandb.use True` and have a `gtransformers` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`).

### Calculate Influence Scores
After training, the model checkpoints could be loaded up directly by providing the checkpoint path in the configuration at `cfg.train.finetune` and set `cfg.train.freeze_pretrained` to True. To compute influence scores, run src/model_inference.py and provide the relevant configuration file,

```bash
python src/model_inference.py --cfg src/configs/EGNN/vocsuperpixels-EGNN2.yaml device cuda:0 wandb.use False
```

### Calculate cheeger constant, diameter and normalized shortest path

```bash
python src/metrices_main.py --cfg src/configs/GCN/vocsuperpixels-GCN.yaml device cuda:0 wandb.use False
```
