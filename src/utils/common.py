import argparse
import collections
import copy
import csv
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Line2D
from prettytable import PrettyTable

from core.config import assert_and_infer_cfg, dump_cfg, get_cfg_defaults


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See src/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def how_many_nas(df):
    ctr = collections.Counter()
    for col in df.columns:
        ctr.update({col: df[col].isnull().sum()})
    print(f"Total columns: {df.shape[0]}")
    for el in ctr.most_common():
        print(el)


def threshed_sigmoid(logits, threshold=0.5):
    return torch.where(
        torch.sigmoid(logits) > threshold,
        torch.ones_like(logits),
        torch.zeros_like(logits),
    )


def fetch_best_model_filename(model_save_path):
    checkpoint_files = os.listdir(model_save_path)
    best_checkpoint_files = [f for f in checkpoint_files if "best_" in f]
    best_checkpoint_val_loss = [
        float(".".join(x.split("=")[1].split(".")[0:2])) for x in best_checkpoint_files
    ]
    best_idx = np.array(best_checkpoint_val_loss).argmax()
    return os.path.join(model_save_path, best_checkpoint_files[best_idx])


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    # ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def handle_config_and_log_paths(args):
    # Load default config options
    cfg = get_cfg_defaults()
    # merge config modifications from config file
    cfg.merge_from_file(args.cfg_file)
    # merge config modifications from command line arguments
    cfg.merge_from_list(args.opts)
    # checks and assertions on config
    assert_and_infer_cfg()
    cfg.PATHS.OUT_DIR = os.path.join(
        cfg.PATHS.OUT_DIR, cfg.PATHS.EXPERIMENT_NAME, cfg.PATHS.TIMESTAMP
    )
    model_save_path = os.path.join(cfg.PATHS.DATAPATH, "experiments/")
    cfg.PATHS.MODEL_OUT_DIR = os.path.join(
        model_save_path, cfg.PATHS.EXPERIMENT_NAME, cfg.PATHS.TIMESTAMP, "saved_models"
    )
    cfg.PATHS.TB_OUT_DIR = os.path.join(cfg.PATHS.OUT_DIR, "tb_logs")

    # freeze config before running experiments
    cfg.freeze()

    # Ensure that the output dir exists
    try:
        os.makedirs(cfg.PATHS.OUT_DIR, exist_ok=True)
        os.makedirs(cfg.PATHS.MODEL_OUT_DIR, exist_ok=True)
        os.makedirs(cfg.PATHS.TB_OUT_DIR, exist_ok=False)
    except FileExistsError:
        print("Wait for a minute and try again :)")
        exit()

    dump_cfg(cfg)

    return cfg


def log_test_results_to_csv(cfg, file_path, test_metrics_l):
    for i, test_metrics_d in enumerate(test_metrics_l):
        test_metrics_d.update(
            {
                "base_lr": cfg.OPTIM.BASE_LR,
                "lr_policy": cfg.OPTIM.LR_POLICY,
                "batch_size": cfg.TRAIN.BATCH_SIZE,
                "optim_momentum": cfg.OPTIM.MOMENTUM,
                "max_seq_length": cfg.MODEL.MAX_SEQ_LENGTH,
                "embed_size": cfg.MODEL.EMBED_SIZE,
                "max_epochs": cfg.OPTIM.MAX_EPOCHS,
                "exp_dir": cfg.PATHS.OUT_DIR,
                "exp_name": cfg.PATHS.EXPERIMENT_NAME,
                "test_id": i,
            }
        )
        log_file = Path(file_path)
        if not log_file.is_file():
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=test_metrics_d.keys())
                writer.writeheader()
                writer.writerow(test_metrics_d)
        else:
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=test_metrics_d.keys())
                writer.writerow(test_metrics_d)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
