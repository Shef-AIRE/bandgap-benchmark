# Benchmarking Bandgap Prediction for Semiconductor Materials Using Multimodal and Multi-Fidelity Data

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the PyTorch Lightning implementation of the benchmark described in our paper:

"Benchmarking bandgap prediction in semiconductors under experimental and realistic evaluation settings."

The benchmark evaluates machine learning models for semiconductor bandgap prediction under more realistic deployment scenarios, including prediction on experimental data, computational pretraining, and domain-based out-of-distribution evaluation.

## Dataset

We compiled a new multimodal, multi-fidelity dataset by combining data from:

- Materials Project (MP) – computational bandgaps
- BandgapDatabase1, DS2, and Matbench-expt – experimentally measured bandgaps

The resulting dataset contains:

- 60,218 low-fidelity computational bandgaps
- 1,705 high-fidelity experimental bandgaps

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

`cif_file.zip` - `.cif` files and the atomic encoding file used in the benchmark.

`data/` - MPIDs and corresponding bandgap values:

* `pretrain_data.json` - 60,218 PBE bandgap values.
* `fine_tune/` - 1,705 experimental bandgap values in total.
  * `fine_tune/train_data.json` - 1,534 experimental bandgap values.
  * `fine_tune/test_data.json` - 171 experimental bandgap values.
* `data_by_type/` - Data used for "leave-one-material-out" splits, categorized by material type.

`configs/` - Configuration files for training models.

`realmat_bag/` - Implementations of models and analysis.

`loaddata/` - Data preparation, splitting, and processing.

`leave_one_material_out/` - Scripts and data for running leave-one-material-out experiments.

`saved_models` - Pretrained models.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Shef-AIRE/bandgap-benchmark.git
cd bandgap-benchmark
```

Install dependencies with:

```bash
pip install -r requirements.txt
```
The whole installation process should take ~10 minutes.


## Training

To train a model, use the following command (add `--pretrain` to perform pretraining only once instead of k-fold training):

```bash
python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml
```

The model output should be the predicted band gap.

After training, predictions can be generated using:

```bash
python test_model.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml --checkpoint saved_models/PATH_TO_YOUR_MODEL.ckpt --cif_folder cif_file --test_data data/fine_tune/test_data.json
```

A demo is provided in [`example.ipynb`](example.ipynb), including data loading, model training and predicted bandgap. The demo is expected to run in ~3 minutes.

To reproduce the paper results, use the relevant YAML configuration files from [`configs/`](configs/). Please note that different random seeds or devices may affect the results.

## Data Download

Crystal structures are taken from the Materials Project and are available under a CC BY 4.0 license. The structures are included in this repository as `cif_file.zip`. To use the packaged structures locally, unzip the archive before training:

```bash
unzip cif_file.zip
```

To download structures by MPID directly from the Materials Project, an API key is required. You can obtain one from: https://next-gen.materialsproject.org/api

Option 1: Download automatically during training

```bash
python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml
```

When running any configuration, missing CIF files will be downloaded automatically.

Option 2: Download explicitly before training

```bash
# Optional: avoid entering key every time
export MP_API_KEY=YOUR_MP_API_KEY

# Download CIFs for data/pretrain_data.json
python3 -m realmat_bag.utils.cif_downloader --stage pretrain

# Download CIFs for data/fine_tune/train_data.json and data/fine_tune/test_data.json
python3 -m realmat_bag.utils.cif_downloader --stage finetune
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
