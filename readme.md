## ASCADiff: Adaptive Sparse Conditional Attention Diffusion for Structure-Based 3D Molecular Design

Official implementation of ASCADiff, a conditional diffusion framework
for protein-pocket guided three-dimensional molecular generation.

ASCADiff introduces adaptive sparse conditional attention to capture
informative protein-ligand interactions and enables controllable
molecular generation under structural and property constraints.

------------------------------------------------------------------------

## Data

The data preparation follows previous structure-based molecular generation  methods. The CrossDocked2020 dataset can be downloaded from its official  source. After downloading the dataset, please run the provided preprocessing  scripts to generate the required data format for ASCADiff.
The required preprocessing scripts are provided in this archive.

```txt
scripts/data_preparation
```



------------------------------------------------------------------------

## Requirements

The experiments are conducted with:

-   Python 3.9
-   PyTorch 2.1.0
-   CUDA 11.8
-    PyTorch Geometric 2.6.1

Install dependencies:

``` bash
conda env create -f environment.yml
conda activate ascadiff
```

Main dependencies:

``` text
numpy
scipy
pandas
matplotlib
rdkit
torch
torch-geometric
tqdm
networkx
```

------------------------------------------------------------------------

## Training

Modify the dataset path and training parameters in:

``` text
configs/training.yml
```

Train ASCADiff:

``` bash
python scripts/train_diffusion.py \
--config configs/training.yml
```

The trained checkpoints will be saved in:

``` text
logs_diffusion
```

------------------------------------------------------------------------

## Molecular Generation

Generate molecules conditioned on a protein pocket:

``` bash
python scripts/sample.py \
--config configs/sampling.yml 
```

The generated molecules will be saved as:

``` text
results/
└── results.pt
```

------------------------------------------------------------------------

## Evaluation

ASCADiff supports evaluation with:

-   Molecular validity
-   Uniqueness
-   Novelty
-   QED
-   SA
-   Vina score
-   Molecular diversity

Run evaluation:

``` bash
python scripts/evaluate.py \
--result_dir results/
```

For docking evaluation, AutoDock Vina or QVina is required.

------------------------------------------------------------------------

## Pretrained Model

Pretrained ASCADiff checkpoints will be provided for molecular
generation and evaluation.

Example:

``` text
checkpoints/
└── 168000.pt
```





