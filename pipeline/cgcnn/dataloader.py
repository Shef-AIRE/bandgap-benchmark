import argparse
import sys
import os
import shutil
import time
import warnings
from random import sample

import numpy as np
from sklearn import metrics
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import os
import csv
import re
import json
import functools
import random
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure


def get_train_val_test_loader(dataset, split_ratios=[0.7, 0.2, 0.1], collate_fn=default_collate,
                              batch_size=64, num_workers=1, pin_memory=False):
    """
    Utility function for dividing a dataset to train, val, test datasets using split ratios.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    split_ratios: list
      Ratios for splitting the dataset (e.g., [0.7, 0.2, 0.1] for train, val, test).
    batch_size: int
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    test_loader: torch.utils.data.DataLoader or None
      DataLoader that random samples the test data, or None if the test ratio is None.
    """

    # Adjust split ratios if None is provided for the test set
    if split_ratios[-1] is None:
        split_ratios = split_ratios[:-1]

    # Split dataset using split_by_ratios
    datasets = split_by_ratios(dataset, split_ratios)

    # Create DataLoaders
    train_loader = DataLoader(datasets[0], batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(datasets[1], batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    test_loader = None

    if len(datasets) == 3:
        test_loader = DataLoader(datasets[2], batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


# split by ratio code from Pykale
def split_by_ratios(dataset, split_ratios):
    """Randomly split a dataset into non-overlapping new datasets of given ratios.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be split.
        split_ratios (list): Ratios of splits to be produced, where 0 < sum(split_ratios) <= 1.

    Returns:
        [List]: A list of subsets.

    Examples:
        >>> import torch
        >>> from kale.loaddata.dataset_access import split_by_ratios
        >>> subset1, subset2 = split_by_ratios(range(10), [0.3, 0.7])
        >>> len(subset1)
        3
        >>> len(subset2)
        7
        >>> subset1, subset2 = split_by_ratios(range(10), [0.3])
        >>> len(subset1)
        3
        >>> len(subset2)
        7
        >>> subset1, subset2, subset3 = split_by_ratios(range(10), [0.3, 0.3])
        >>> len(subset1)
        3
        >>> len(subset2)
        3
        >>> len(subset3)
        4
    """
    n_total = len(dataset)
    ratio_sum = sum(split_ratios)
    if ratio_sum > 1 or ratio_sum <= 0:
        raise ValueError("The sum of ratios should be in range(0, 1]")
    elif ratio_sum == 1:
        split_ratios_ = split_ratios[:-1]
    else:
        split_ratios_ = split_ratios.copy()
    split_sizes = [int(n_total * ratio_) for ratio_ in split_ratios_]
    split_sizes.append(n_total - sum(split_sizes))

    return torch.utils.data.random_split(dataset, split_sizes)




def extract_features(dataset):
    features, targets = [], []
    for data in dataset:
        atom_fea = data.atom_fea.cpu().numpy()  # (N_atoms, atom_fea_len)
        nbr_fea = data.nbr_fea.cpu().numpy()  # (N_atoms, N_neighbors, nbr_fea_len)
        nbr_fea_idx = data.nbr_fea_idx.cpu().numpy()  # (N_atoms, N_neighbors)
        # Get neighbor atom features
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_fea[nbr_fea_idx]  # (N_atoms, N_neighbors, atom_fea_len)
        # Expand atom_fea to match dimensions
        atom_fea = np.expand_dims(atom_fea, axis=1)  # (N_atoms, 1, feature_dim)
        atom_fea = np.tile(atom_fea, (1, M, 1))  # (N_atoms, N_neighbors, feature_dim)

        # Concatenate atom and neighbor features
        total_fea = np.concatenate([atom_fea, atom_nbr_fea, nbr_fea], axis=2)  # (N_atoms, N_neighbors, input_dim)

        # Aggregate over neighbors
        total_fea = np.mean(total_fea, axis=1)  # (N_atoms, input_dim)
        # total_fea = np.concatenate([atom_fea, total_fea], axis=-1)  # (N_atoms, input_dim + atom_fea_len)
        total_fea = np.mean(total_fea, axis=0)



        # Store features and target (bandgap value)
        features.append(total_fea)
        targets.append(data.target.cpu().numpy())  # Bandgap target

    features = np.vstack(features)  # (Total samples, Feature size)
    targets = np.concatenate(targets)  # (Total samples,)
    return features, targets