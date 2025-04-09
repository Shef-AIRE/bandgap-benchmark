from torch.utils.data import Dataset

from cgcnn.cifdata import CIFData


# from cgcnn_train_bg import CIFData

class CIFDataAccess(Dataset):
    """
    Common API for CIF dataset access
    Args:
        data (list): List of CIF data entries
        cif_folder (str): Path to the folder containing CIF files
        init_file (str): Initialization file path
        max_nbrs (int): Maximum number of neighbors
        radius (float): Radius for neighbor search
        randomize (bool): Whether to randomize the dataset
    """
    def __init__(self, data, cif_folder, init_file, max_nbrs, radius, randomize):
        self.data = data
        self.cif_folder = cif_folder
        self.init_file = init_file
        self.max_nbrs = max_nbrs
        self.radius = radius
        self.randomize = randomize

    def get_data(self):
        return CIFData(self.data, self.cif_folder, self.init_file, self.max_nbrs, self.radius, self.randomize)

