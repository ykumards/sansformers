"""Configuration file (powered by YACS)."""

import os
from datetime import datetime

import torch
from yacs.config import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type
_C.MODEL.TYPE = ""

# Topcap for the count regression prediction
_C.MODEL.TOP_CAP = 24

_C.MODEL.EMBED_SIZE = 128

_C.MODEL.DROPOUT_P = 0.0

_C.MODEL.VOCAB_SIZE = 1000
_C.MODEL.DIAG_INPUT_VOCAB_SIZE = 1000
_C.MODEL.PROC_INPUT_VOCAB_SIZE = 1000
_C.MODEL.ICPC_INPUT_VOCAB_SIZE = 1000

_C.MODEL.SPECIALITY_INPUT_VOCAB_SIZE = 10
_C.MODEL.SERVICE_PROVIDER_INPUT_VOCAB_SIZE = 10
_C.MODEL.SERVICE_SECTOR_INPUT_VOCAB_SIZE = 10
_C.MODEL.NOTIFICATION_TYPE_INPUT_VOCAB_SIZE = 10
_C.MODEL.REGISTER_TYPE_INPUT_VOCAB_SIZE = 10
_C.MODEL.PALVELUMUOTO_INPUT_VOCAB_SIZE = 10
_C.MODEL.YHTEYSTAPA_INPUT_VOCAB_SIZE = 10
_C.MODEL.PROFESSIONAL_INPUT_VOCAB_SIZE = 10

_C.MODEL.VOCAB_SIZE = 258
_C.MODEL.MAX_SEQ_LENGTH = 200

# RNN params
_C.MODEL.RNN_HIDDEN_SIZE = 64
_C.MODEL.RNN_DEPTH = 2

# Transformer Params
_C.MODEL.NUM_HEADS = 6
_C.MODEL.TRANS_DEPTH = 2

# Accumulate Gradient Steps
_C.MODEL.ACCU_GRAD_STEPS = 1

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps', '1cycle'}
_C.OPTIM.LR_POLICY = "cos"

# number of iters for lr warmup
_C.OPTIM.LR_WARMUP = 0

# factor for noam optimizer
_C.OPTIM.NOAM_FACTOR = 2.0

# Exponential decay factor
_C.OPTIM.LR_GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Number of steps per epoch
_C.OPTIM.STEPS_PER_EPOCH = 1

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCHS = 2

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Grad clipping value for exploding gradients
_C.OPTIM.GRAD_CLIP_T = 0.5

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Dataset filename
_C.TRAIN.FILENAME = ""

# Dataset and split
_C.TRAIN.DATASET = ""


# column names of icd and proc in the dataframe
_C.TRAIN.ICD_COLNAME = ""
_C.TRAIN.PROC_COLNAME = ""

# Size of validation set
_C.TRAIN.VALIDATION_SPLIT = 0.01

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 2

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

# Patience for early stopping
_C.TRAIN.ES_PATIENCE = 0

# Loss threshold for early stopping
_C.TRAIN.ES_THRESHOLD = 0.0

# After training, load the model with best val loss for evaluation?
_C.TRAIN.LOAD_BEST_CKPT = True

# Boolean flag for if test run is needed
_C.TRAIN.IS_TEST_RUN_NEEDED = True

# Boolean flag for if binary prediction is needed
_C.TRAIN.IS_BINARY_PRED_NEEDED = True

# Boolean flag for if seq regression prediction is needed
_C.TRAIN.IS_DIST_PRED_NEEDED = False

# Set this flag to true when we are finetuning the GPT model
_C.TRAIN.IS_SUPERVISED = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Dataset filename
_C.TEST.FILENAME = ""

_C.TEST.FILENAME2 = ""

# Dataset filename
_C.TEST.VAL_FILENAME = ""

# Dataset and split
_C.TEST.DATASET = _C.TRAIN.DATASET

# Total mini-batch size
_C.TEST.BATCH_SIZE = 2

# Weights to use for testing
_C.TEST.WEIGHTS = ""

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()

# Number of data loader workers per training process
_C.DATALOADER.NUM_WORKERS = 0

# Load data to pinned host memory
_C.DATALOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True

# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True

# ---------------------------------------------------------------------------- #
# Precise timing options
# ---------------------------------------------------------------------------- #
_C.PREC_TIME = CN()

# Perform precise timing at the start of training
_C.PREC_TIME.ENABLED = False

# Total mini-batch size
_C.PREC_TIME.BATCH_SIZE = 128

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------------------------------------------------- #
# Path options
# ---------------------------------------------------------------------------- #
_C.PATHS = CN()
# Data Path
_C.PATHS.DATAPATH = ""
# Data file
_C.PATHS.DATAFILE = ""
# Output directory parent folder
_C.PATHS.OUT_DIR = ""
# Results log file name
_C.PATHS.RESULTS_LOG_FILENAME = "results.csv"
# Experiment name
_C.PATHS.EXPERIMENT_NAME = ""
# Experiment description
_C.PATHS.EXPERIMENT_DESC = ""
# Get current timestamp
_C.PATHS.TIMESTAMP = "at_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# Outdirectory for TB logging
_C.PATHS.TB_OUT_DIR = ""
# Outdirectory for model checkpoints
_C.PATHS.MODEL_OUT_DIR = ""
# path to pretrained transformer file (if any)
_C.PATHS.PRETRAINED_TRANSFORMER_FILE = ""
# path to vectorizer (optional)
_C.PATHS.VECTORIZER_PATH = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# try fitting the model on a small dataset ~ 20 examples
_C.OVERFIT_ON_BATCH = False
# overfit on batch percentage
_C.OVERFIT_ON_BATCH_PCT = 1.0

# Build vecotorizer from scratch
_C.BUILD_VEC_FROM_SCRATCH = False

# tune params if set to true, tunes model between a range of param values by using run_optuna method
_C.TUNE_PARAMS = False

# Number of trials while tuning params
_C.NUM_TRIALS = 1

# Choose device type
_C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of GPUs to use (applies to both training and testing)
_C.GPU_NUM = 1

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# create a git tag based on experiment name and description
_C.CREATE_GIT_TAG = False

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 100

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Tensorboard logging flag
_C.IS_TB_LOG = True

# Verbose flag for print logs
_C.VERBOSE = True

# Frequency for logging gradient histograms in TB
_C.TB_LOG_GRAD_INTV = 500  # iteration

# use AMP
_C.USE_AMP = False

# Multiple GPUS
_C.MULTI_GPU = False


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    pass


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.PATHS.OUT_DIR, cfg.CFG_DEST)
    with open(cfg_file, "w") as f:
        cfg.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
