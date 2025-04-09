import numpy as np
import torch
from torch.utils.data import default_collate


def collate_pool_leftnet(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties, handling variable sizes.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_positions, batch_atom_num = [], [], [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    batch_atom_indices = []
    base_idx = 0

    for i, data_item in enumerate(dataset_list):
        n_i = data_item.atom_fea.shape[0]  # number of atoms for this crystal

        # Append features and neighbor info
        batch_atom_fea.append(data_item.atom_fea)
        batch_nbr_fea.append(data_item.nbr_fea)
        batch_nbr_fea_idx.append(data_item.nbr_fea_idx + base_idx)
        batch_positions.append(data_item.positions)
        batch_atom_num.append(data_item.atom_num)

        # Create index mappings
        crystal_atom_idx.append(torch.arange(n_i) + base_idx)
        batch_target.append(data_item.target)
        batch_cif_ids.append(data_item.cif_id)
        batch_atom_indices.append(torch.full((n_i,), i, dtype=torch.long))

        # Update base index for the next crystal
        base_idx += n_i

    # Concatenate all the features along the appropriate dimension
    return BatchData(
        atom_fea=torch.cat(batch_atom_fea, dim=0),  # Concatenate all atom features
        nbr_fea=torch.cat(batch_nbr_fea, dim=0),  # Concatenate all neighbor features
        nbr_fea_idx=torch.cat(batch_nbr_fea_idx, dim=0),  # Concatenate neighbor indices
        positions=torch.cat(batch_positions, dim=0),  # Concatenate positions
        atom_num=torch.cat(batch_atom_num, dim=0),  # Concatenate atom numbers
        crystal_atom_idx=crystal_atom_idx,  # List of tensors for crystal-to-atom mapping
        target=torch.stack(batch_target, dim=0),  # Stack targets (assuming they are uniform in shape)
        cif_ids=batch_cif_ids,  # List of cif_ids
        batch_idx=torch.cat(batch_atom_indices, dim=0)  # Concatenate batch indices
    )


class BatchData:
    def __init__(self, atom_fea, nbr_fea, nbr_fea_idx, positions, atom_num, crystal_atom_idx, target, cif_ids, batch_idx):
        self.atom_fea = atom_fea
        self.nbr_fea = nbr_fea
        self.nbr_fea_idx = nbr_fea_idx
        self.positions = positions
        self.atom_num = atom_num
        self.crystal_atom_idx = crystal_atom_idx
        self.target = target
        self.cif_ids = cif_ids
        self.batch_idx = batch_idx
        self.batch_size = len(cif_ids)