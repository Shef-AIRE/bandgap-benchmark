# Benchmarking Band Gap Prediction For Semiconductor Materials Using Multimodal And Multi-fidelity Data

This repository contains the PyTorch Lightning implementation of the benchmark described in our paper:

"Benchmarking Band Gap Prediction for Semiconductor Materials Using Multimodal and Multi-fidelity Data."

The benchmark evaluates machine learning models for semiconductor band gap prediction under more realistic deployment scenarios, including experimental data prediction, computational pretraining, and domain-based out-of-distribution evaluation.

## Dataset

We compiled a new multimodal, multi-fidelity dataset by combining data from:

- Materials Project (MP) – computational band gaps
- BandgapDatabase1; DS2; Matbench-expt – experimentally measured band gaps

The resulting dataset contains:

- 60,218 low-fidelity computational band gaps
- 1,705 high-fidelity experimental band gaps

Each experimental sample is aligned with a crystal structure through the Materials Project ID (MPID). Crystal structures can be retrieved directly from the Materials Project database.

## Models

We evaluated eight machine learning models:

- Classical machine learning models
  - Linear Regression (LR)
  - Random Forest Regression (RFR)
  - Support Vector Regression (SVR)
- Graph neural networks
  - CGCNN
  - CartNet
  - ALIGNN
  - CHGNet
  - LEFTNet

For classical machine learning models, we used structure-derived atomic features, including the atomic encoding originally introduced in CGCNN.

## Repository Structure

`cif_file.zip` - Contains `.cif` files and the atomic encoding file used in the benchmark.

`data/` - Directory containing MPIDs and corresponding band gap values:

* `pretrain_data.json` - 60,218 PBE band gap values.
* `fine_tune/train_data.json` - 1,534 experimental band gap values.
* `fine_tune/test_data.json` - 171 experimental band gap values.
* `fine_tune/` total - 1,705 experimental band gap values.
* `data_by_type/` - Data used for "leave-one-material-out" splits, categorized by material type.

`configs/` - Configuration files for training models.

`realmat_bag/pipeline/models/` - Implementations of baseline models.

`loaddata/` - Data preparation, splitting, and processing.

`leave_one_material_out/` - Scripts and data for running leave-one-material-out experiments.

`saved_models` - Pretrained models.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training

To train a model, use the following command (add `--pretrain` to perform pretraining only once instead of k-fold training):

```bash
python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml
```

After training, predictions can be generated using:

```bash
python test_model.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml --checkpoint saved_models/PATH_TO_YOUR_MODEL.ckpt --cif_folder cif_file --test_data data/fine_tune/test_data.json
```

## Download CIF

Downloading CIF data requires a Materials Project API key: https://next-gen.materialsproject.org/api

Option 1: download explicitly before training

```bash
# Optional: avoid entering key every time
export MP_API_KEY=YOUR_MP_API_KEY

# Download CIFs for data/pretrain_data.json
python3 -m realmat_bag.utils.cif_downloader --stage pretrain

# Download CIFs for data/fine_tune/train_data.json and data/fine_tune/test_data.json
python3 -m realmat_bag.utils.cif_downloader --stage finetune
```

Option 2: download automatically during training

```bash
python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml
```

When running any config, missing CIF files will be downloaded automatically.
