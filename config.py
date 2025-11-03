from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Configuration definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = "data/ds1.json"
_C.DATASET.VAL = "data/ds3.json"
_C.DATASET.RATIO = 1.0 # Ratio of the dataset to use
_C.DATASET.PREDEFINED_SPLIT = False
_C.DATASET.SPLIT_GLOB = ""


# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.EPOCHS = 1
_C.SOLVER.LR = 0.01
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_RUNS = 1
_C.SOLVER.NUM_FOLDS = 10
_C.SOLVER.RANDOMIZE = False
_C.SOLVER.TASK = "regression"  # Choices: ['regression', 'classification']
_C.SOLVER.DISABLE_CUDA = False
_C.SOLVER.WORKERS = 0
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.LR_MILESTONES = [100, 200]

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.RESUME = ""

_C.SOLVER.OPTIM = "SGD"  # Choices: ['SGD', 'Adam']

# -----------------------------------------------------------------------------
# Model paths and parameters
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "cgcnn"  # Choices: ['cgcnn', 'leftnet', 'alignn', 'cartnet', 'chgnet', 'random_forest', 'linear_regression', 'svm']
_C.MODEL.PRETRAINED_MODEL_PATH = ""
_C.MODEL.CIF_FOLDER = "./cifs"
_C.MODEL.INIT_FILE = "./init.json"
_C.MODEL.MAX_NBRS = 12
_C.MODEL.RADIUS = 7.0
_C.MODEL.HYPERPARAMS = CN(new_allowed=True)


# -----------------------------------------------------------------------------
# Common model hyperparameters
# -----------------------------------------------------------------------------
_C.MODEL_COMMON = CN()
_C.MODEL_COMMON.CUTOFF = 6.0
_C.MODEL_COMMON.NUM_RADIAL = 32
_C.MODEL_COMMON.OUTPUT_DIM = 1


# -----------------------------------------------------------------------------
# CGCNN parameters
# -----------------------------------------------------------------------------

_C.CGCNN = CN()
_C.CGCNN.ATOM_FEA_LEN = 64
_C.CGCNN.H_FEA_LEN = 128
_C.CGCNN.N_CONV = 3
_C.CGCNN.N_H = 1
_C.CGCNN.NUM_REPEAT = 1

_C.CGCNN.LAYER_FREEZE = "none"  # Choices: ['all', 'embedding', 'none']
_C.CGCNN.FEATURE_FUSION = "none"  # Choices: ['none', 'data level', 'fc level', 'feature level']
_C.CGCNN.ORIG_ATOM_FEA_LEN = 92
_C.CGCNN.NBR_FEA_LEN = 41
_C.CGCNN.POS_FEA_LEN = 3



# -----------------------------------------------------------------------------
# LeftNet configs
# -----------------------------------------------------------------------------
_C.LEFTNET = CN()
_C.LEFTNET.HIDDEN_CHANNELS = 128
_C.LEFTNET.NUM_LAYERS = 4

_C.LEFTNET.REGRESS_FORCES = False
_C.LEFTNET.USE_PBC = True
_C.LEFTNET.OTF_GRAPH = False
_C.LEFTNET.LAYER_FREEZE = "none"  # Choices: ['all', 'embedding', 'none']
_C.LEFTNET.ENCODING = "none" # Choices: ['one-hot', 'none'], none for LEFTNet-Z, one-hot for LEFTNet-Prop

# -----------------------------------------------------------------------------
# ALIGNN specific parameters
# -----------------------------------------------------------------------------
_C.ALIGNN = CN()
_C.ALIGNN.ATOM_FEA_LEN = 92
_C.ALIGNN.HIDDEN_DIM = 128
_C.ALIGNN.NUM_LAYERS = 4
_C.ALIGNN.NUM_RBF = None
_C.ALIGNN.CUTOFF = 6.0
_C.ALIGNN.DROPOUT = 0.0
_C.ALIGNN.READOUT = "mean"  # Choices: ['mean', 'sum', 'max']
_C.ALIGNN.MAX_NEIGHBORS = 1000
_C.ALIGNN.LAYER_FREEZE = "none"  # Choices: ['all', 'embedding', 'none']
_C.ALIGNN.ACTIVATION = "silu"
_C.ALIGNN.RBF_TRAINABLE = False
_C.ALIGNN.ENCODING = "prop"  # Choices: ['prop', 'z']
_C.ALIGNN.MAX_NUM_ELEMENTS = 94

# -----------------------------------------------------------------------------
# CARTNET specific parameters
# -----------------------------------------------------------------------------
_C.CARTNET = CN()
_C.CARTNET.DIM_IN = 256  # Add appropriate default value
_C.CARTNET.DIM_RBF = 64  # Add appropriate default value
_C.CARTNET.NUM_LAYERS = 4  # Add appropriate default value
_C.CARTNET.INVARIANT = False  # Add appropriate default value
_C.CARTNET.TEMPERATURE = False  # Add appropriate default value
_C.CARTNET.USE_ENVELOPE = True  # Add appropriate default value
_C.CARTNET.ATOM_TYPES = True  # Add appropriate default value


# -----------------------------------------------------------------------------
# CHGNet specific parameters
# -----------------------------------------------------------------------------
_C.CHGNET = CN()
_C.CHGNET.ATOM_FEA_DIM = 64
_C.CHGNET.BOND_FEA_DIM = 64
_C.CHGNET.ANGLE_FEA_DIM = 64
_C.CHGNET.NUM_RADIAL = 31
_C.CHGNET.NUM_ANGULAR = 31
_C.CHGNET.N_CONV = 4
_C.CHGNET.ATOM_CONV_HIDDEN_DIM = 64
_C.CHGNET.BOND_CONV_HIDDEN_DIM = 64
_C.CHGNET.ANGLE_LAYER_HIDDEN_DIM = 0
_C.CHGNET.CONV_DROPOUT = 0.0
_C.CHGNET.READ_OUT = "ave"
_C.CHGNET.MLP_HIDDEN_DIMS = (64, 64, 64)
_C.CHGNET.MLP_DROPOUT = 0.0
_C.CHGNET.MLP_FIRST = True
_C.CHGNET.IS_INTENSIVE = True
_C.CHGNET.NON_LINEARITY = "silu"
_C.CHGNET.ATOM_GRAPH_CUTOFF = 6.0
_C.CHGNET.BOND_GRAPH_CUTOFF = 3.0
_C.CHGNET.GRAPH_CONVERTER_ALGORITHM = "fast"
_C.CHGNET.CUTOFF_COEFF = 8
_C.CHGNET.LEARNABLE_RBF = True
_C.CHGNET.GMLP_NORM = "layer"
_C.CHGNET.READOUT_NORM = "layer"
_C.CHGNET.ENCODING = "z"  # Choices: ['z', 'prop']
_C.CHGNET.ATOM_INPUT_DIM = 92
_C.CHGNET.MAX_NUM_ELEMENTS = 94


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIR = "results"
_C.OUTPUT.LOOP_RESULTS = "loop_50epochs.csv"
_C.OUTPUT.PREDICTIONS = "predictions_reduced_ds2.csv"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_C.LOGGING = CN()
_C.LOGGING.LOG_DIR = "./logs"

_C.LOGGING.LOG_DIR_NAME = None

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the configuration."""
    return _C.clone()
