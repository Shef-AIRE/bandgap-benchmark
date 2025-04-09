import json
import os
import warnings

import torch
import numpy as np
from pymatgen.core import Structure
from torch.utils.data import Dataset

# from cgcnn_train_bg import AtomCustomJSONInitializer, GaussianDistance


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):

    def __init__(self, mpids_bg, cif_folder, init_file, max_nbrs, radius, randomize, dmin=0, step=0.2):

        self.max_num_nbr, self.radius = max_nbrs, radius

        if randomize:
            self.mpids_bg_dataset = mpids_bg.sample(frac=1).reset_index(
                drop=True).values  # Shuffling data and converting df to array
        else:
            self.mpids_bg_dataset = mpids_bg.reset_index(drop=True).values

        atom_init_file = init_file
        self.cif_folder = cif_folder
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.mpids_bg_dataset)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):

        cif_id, target_np = self.mpids_bg_dataset[idx]
        crystal = Structure.from_file(os.path.join(self.cif_folder,
                                                   cif_id + '.cif'))
        atom_fea_np = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                 for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea_np)
        target = torch.Tensor([float(target_np)])
        positions = torch.Tensor(crystal.cart_coords)
        atom_num = torch.Tensor(crystal.atomic_numbers).long()

        all_nbrs = crystal.get_all_neighbors(self.radius)  # include_index is depreciated. Index is always included now.
        all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance) for nbrs in
                    all_nbrs]  # Sorts nbrs based on the value of key as applied to each element of the list.
        nbr_fea_idx, nbr_fea_np = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea_np.append(list(map(lambda x: x[1], nbr)) +
                                  [self.radius + 1.] * (self.max_num_nbr -
                                                        len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x.index,
                                            nbr[:self.max_num_nbr])))
                nbr_fea_np.append(list(map(lambda x: x.nn_distance,
                                           nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea_np = np.array(nbr_fea_idx), np.array(nbr_fea_np)
        nbr_fea_gp = self.gdf.expand(nbr_fea_np)

        nbr_fea = torch.Tensor(nbr_fea_gp)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return CIFDataItem(atom_fea, nbr_fea, nbr_fea_idx, positions, atom_num, target, cif_id)


class CIFDataItem:
    def __init__(self, atom_fea, nbr_fea, nbr_fea_idx, positions, atom_num, target, cif_id):
        self.atom_fea = atom_fea
        self.nbr_fea = nbr_fea
        self.nbr_fea_idx = nbr_fea_idx
        self.positions = positions
        self.atom_num = atom_num
        self.target = target
        self.cif_id = cif_id

    def to_dict(self):
        return {
            "atom_fea": self.atom_fea,
            "nbr_fea": self.nbr_fea,
            "nbr_fea_idx": self.nbr_fea_idx,
            "positions": self.positions,
            "atom_num": self.atom_num,
            "target": self.target,
            "cif_id": self.cif_id
        }