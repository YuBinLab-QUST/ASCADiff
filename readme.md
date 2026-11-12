## ASCADiff: Adaptive Sparse Conditional Attention Diffusion for Structure-based 3D Molecular Design

Official implementation of ASCADiff, a conditional diffusion framework
for protein-pocket guided three-dimensional molecular generation.

ASCADiff introduces adaptive sparse conditional attention to capture
informative protein-ligand interactions and enables controllable
molecular generation under structural and property constraints.

------------------------------------------------------------------------

## Data

The data used for training / evaluating the model are organized in the [data](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM?dmr=1&ec=wgc-drive-[module]-goto) Google Drive folder.

To train the model from scratch, you need to download the preprocessed lmdb file and split file:

* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, you need to download _and_ unzip the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), save it into `data/CrossDocked2020`, and run the scripts in `scripts/data_preparation`:

* [clean_crossdocked.py](scripts/data_preparation/clean_crossdocked.py) will filter the original dataset and keep the ones with RMSD < 1A.
  It will generate a `index.pkl` file and create a new directory containing the original filtered data (corresponds to `crossdocked_v1.1_rmsd1.0.tar.gz` in the drive). *You don't need these files if you have downloaded .lmdb file.*

    ```bash
  python scripts/data_preparation/clean_crossdocked.py --source data/CrossDocked2020 --dest data/crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
    ```

* [extract_pockets.py](scripts/data_preparation/extract_pockets.py) will clip the original protein file to a 10A region around the binding molecule. E.g.

  ```bash
  python scripts/data_preparation/extract_pockets.py --source data/crossdocked_v1.1_rmsd1.0 --dest data/crossdocked_v1.1_rmsd1.0_pocket10
  ```

* [split_pl_dataset.py](scripts/data_preparation/split_pl_dataset.py) will split the training and test set. We use the same split `split_by_name.pt` as 
  [AR](https://arxiv.org/abs/2203.10446) and [Pocket2Mol](https://arxiv.org/abs/2205.07249), which can also be downloaded in the Google Drive - data folder.

    ```bash
  python scripts/data_preparation/split_pl_dataset.py --path data/crossdocked_v1.1_rmsd1.0_pocket10 --dest data/crossdocked_pocket10_pose_split.pt --fixed_split data/split_by_name.pt
    ```

## Training

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





