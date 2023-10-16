"""
Base trainer class
ref: https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import Line2D
from tqdm import tqdm

sns.set_style("white")

import scipy.stats as stats
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast

from core.config import dump_cfg
from utils import common

PLOT_STYLE = "seaborn-muted"


class BaseTrainer:
    def __init__(self, cfg, model, train_dataloader, val_dataloader, test_dataloaders):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloaders = test_dataloaders
        self.cfg = cfg
        self.grad_scaler = GradScaler(enabled=cfg.USE_AMP)
        self.best_epoch = 0

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.device("cuda", cfg.GPU_NUM)
            self.model = self.model.to(self.device)

        # names of the variables that we will keep track of
        # during training and validation
        self.track_metric_names = [
            "loss",
            "count_loss",
            "bin_loss",
            "dist_loss",
            "los_loss",
            "diag_loss",
            "proc_loss",
            "icpc_loss",
            "accuracy",
            "precision",
            "recall",
            "F1_gpt",
            "acc_gpt",
            "F1_dist",
            "F1_los",
            "auc_roc",
            "r2_count",
            "F1_count",
            "spearman_count",
            "spearman_los",
            "spearman_dist",
            "pval_count",
            "AUC_dist",
            "auc_bin",
            "mae_count",
            "mae_dist",
            "poisson_dev_count",
            "poisson_dev_dist",
            "lr",
        ]
        self.track_metric_names.extend([f"F1_dist_sep_{i}" for i in range(6)])
        self.track_metric_names.extend([f"spearman_dist_sep_{i}" for i in range(6)])
        self.track_metric_names.extend([f"mae_dist_sep_{i}" for i in range(6)])

        self.training_metrics = {k: [] for k in self.track_metric_names}
        self.validation_metrics = {k: [] for k in self.track_metric_names}

        self.epoch_log_variables = {
            "y_outcomes": [],
            "y_los": [],
            "y_dist_preds": [],
            "y_count_preds": [],
            "y_los_preds": [],
        }
        # count the number of trainable parameters in the model
        common.count_parameters(model)

    def _init_training_metrics(self):
        for k, _ in self.training_metrics.items():
            self.training_metrics[k] = []

    def _init_validation_metrics(self):
        for k, _ in self.validation_metrics.items():
            self.validation_metrics[k] = []

    def get_score_checkpoint(self, epoch, val_loss):
        """Retrieves the path to a checkpoint file."""
        # name = f"model_{epoch}_score={val_loss:4f}.pth"
        name = "model.pth"
        return os.path.join(self.cfg.PATHS.MODEL_OUT_DIR, name)

    def save_checkpoint(self, epoch, val_loss):
        """Saves a checkpoint."""
        sd = self.model.state_dict()
        # Record the state
        checkpoint = {
            "epoch": epoch,
            "model_state": sd,
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        # Write the checkpoint
        checkpoint_file = self.get_score_checkpoint(epoch, val_loss)
        # print(f"saving to {checkpoint_file}")
        torch.save(checkpoint, checkpoint_file)
        return checkpoint_file

    def get_best_score_checkpoint_path(self):
        """
        Retrieves the checkpoint with lowest loss score.
        Note: this method is sensitive to the model filename format
        """
        checkpoint_dir = self.cfg.PATHS.MODEL_OUT_DIR
        # Checkpoint file names are in lexicographic order
        checkpoints = [f for f in os.listdir(checkpoint_dir) if ".pth" in f]
        # best_checkpoint_val_loss = [
        #     float(".".join(x.split("=")[1].split(".")[0:2])) for x in checkpoints
        # ]
        # best_idx = np.array(best_checkpoint_val_loss).argmin()
        # name = checkpoints[best_idx]
        name = checkpoints[0]
        return os.path.join(checkpoint_dir, name)

    def load_checkpoint_from_path(self, model_path, load_optimizer=False):
        """Loads the checkpoint from the given file."""
        # Load the checkpoint on CPU to avoid GPU mem spike
        checkpoint = torch.load(model_path, map_location="cpu")
        # Account for the DDP wrapper in the multi-gpu setting
        self.model.load_state_dict(checkpoint["model_state"])
        # Load the optimizer state (commonly not done when fine-tuning)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["epoch"]

    def load_best_score_checkpoint(self, load_optimizer=False):
        """Loads the checkpoint from the given file."""
        checkpoint_file_path = self.get_best_score_checkpoint_path()
        return self.load_checkpoint_from_path(checkpoint_file_path)

    def _prepare_batch(self, batch):
        "places the input tensors in the appropriate gpu device"
        device_batch = {}
        for k, v in batch.items():
            device_batch[k] = v.to(self.device, non_blocking=True)
        return device_batch

    def run_epoch(self, split, dataloader, epoch_count=0):
        "to be overloaded in the child classes"
        return {}

    def plot_prediction_diagnostics(self, prefix="val"):
        "Plots the histogram and scatter plots of the predicted vs true count values"

        from scipy.stats import poisson

        if not os.path.isdir(os.path.join(self.cfg.PATHS.OUT_DIR, "predictions")):
            os.mkdir(os.path.join(self.cfg.PATHS.OUT_DIR, "predictions"))

        pred_fig_save_path = os.path.join(
            self.cfg.PATHS.OUT_DIR,
            "predictions",
            f"predictions_{prefix}.png",
        )
        box_fig_save_path = os.path.join(
            self.cfg.PATHS.OUT_DIR,
            "predictions",
            f"box_plot_{prefix}.png",
        )
        pred_save_path = os.path.join(
            self.cfg.PATHS.OUT_DIR,
            "predictions",
            f"predictions_{prefix}.feather",
        )
        pred_df = pd.DataFrame(self.epoch_log_variables)
        pred_df.to_feather(pred_save_path)

        y_count_pred = np.array(self.epoch_log_variables["y_count_preds"])
        y_count = np.array(self.epoch_log_variables["y_outcomes"])[:, 0]

        sample_indices = np.random.choice(y_count.shape[0], size=1000)
        x_poiss = np.arange(0, self.cfg.MODEL.TOP_CAP, 1)
        with plt.style.context(PLOT_STYLE):
            # plot training curves
            fig, ax = plt.subplots(
                nrows=1, ncols=2, figsize=(14, 4), constrained_layout=True
            )

            for idx in sample_indices:
                lamb = y_count_pred[idx]
                y = poisson.pmf(x_poiss, mu=lamb)
                ax.flat[0].plot(x_poiss, y, alpha=0.05, color="r")
            # ax.flat[0].hist(y_count_pred, label="Predicted Counts", alpha=0.6)
            ax.flat[0].hist(
                y_count,
                label="True Counts",
                alpha=0.6,
                color="b",
                density=True,
                bins=self.cfg.MODEL.TOP_CAP,
            )
            ax.flat[0].set_xlabel("No. of Visits")
            ax.flat[0].set_ylabel("Count")
            ax.flat[0].set_title("Histogram of Predicted and True Visits")
            ax.flat[0].legend()

            ax.flat[1].scatter(
                x=y_count_pred[sample_indices],
                y=y_count[sample_indices],
                alpha=0.7,
                marker="o",
            )
            ax.flat[1].set_xlabel("Predicted Visits")
            ax.flat[1].set_ylabel("True Visits")
            ax.flat[1].set_title("True vs Predicted No. of Visits")

            fig.savefig(pred_fig_save_path)

        ### box plot to visualize the predictions according to risk groups
        # break predictions into 10 groups
        y_count_pred_bin = pd.cut(
            y_count_pred, bins=np.linspace(0, self.cfg.MODEL.TOP_CAP, num=10)
        )
        box_cats = []
        box_labels = []

        for i in y_count_pred_bin.categories:
            idxs = y_count_pred_bin == i
            box_cats.append(y_count[idxs])
            box_labels.append(str(i))

        with plt.style.context(PLOT_STYLE):
            fig1, ax = plt.subplots(
                nrows=1, ncols=1, figsize=(8, 5), constrained_layout=True
            )

            medianprops = dict(linestyle="-.", linewidth=2.5, color="firebrick")
            bplot = ax.boxplot(
                box_cats, labels=box_labels, patch_artist=True, medianprops=medianprops
            )
            # fill with colors
            for patch in bplot["boxes"]:
                patch.set_facecolor("pink")

            ax.set_xlabel("Predicted Category")
            ax.set_ylabel("True Visits")
            ax.set_title("Prediction Distribution in Categories")
            xtickNames = plt.setp(ax, xticklabels=box_labels)
            plt.setp(xtickNames, rotation=45, fontsize=8)
            fig1.savefig(box_fig_save_path)

    def plot_grad_flow(self, epoch_number=0):
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
        # ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

        if not os.path.isdir(os.path.join(self.cfg.PATHS.OUT_DIR, "gradient_flow")):
            os.mkdir(os.path.join(self.cfg.PATHS.OUT_DIR, "gradient_flow"))

        gradflow_fig_save_path = os.path.join(
            self.cfg.PATHS.OUT_DIR,
            "gradient_flow",
            f"gradient_flow_ep{epoch_number}.png",
        )
        named_parameters = self.model.named_parameters()
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and (p.grad is not None) and ("bias" not in n):
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
        plt.savefig(gradflow_fig_save_path, bbox_inches="tight")

    def plot_training_curves(self):
        loss_fig_save_path = os.path.join(self.cfg.PATHS.OUT_DIR, "training_losses.png")

        with plt.style.context(PLOT_STYLE):
            # plot training curves
            fig, ax = plt.subplots(
                nrows=2, ncols=2, figsize=(14, 8), constrained_layout=True
            )
            steps_grid = np.arange(len(self.training_metrics["lr"])) + 1

            x_grid = np.arange(len(self.training_metrics["loss"])) + 1
            min_loss_at = np.argmin(np.array(self.validation_metrics["loss"]))

            ax.flat[0].plot(
                x_grid, self.validation_metrics["loss"], label="Val Loss", marker="o"
            )
            ax.flat[0].plot(
                x_grid, self.training_metrics["loss"], label="Train Loss", marker="o"
            )
            ax.flat[0].axvline(min_loss_at + 1, linestyle="--", label="Min Loss")
            ax.flat[0].set_xlabel("Epochs")
            ax.flat[0].set_ylabel("Loss")
            ax.flat[0].set_title("Total Loss Curve")
            ax.flat[0].legend()

            min_dist_loss_at = np.argmin(np.array(self.validation_metrics["los_loss"]))
            ax.flat[1].plot(
                x_grid,
                self.validation_metrics["los_loss"],
                label="Val. LOS Loss",
                marker="o",
            )
            ax.flat[1].plot(
                x_grid,
                self.training_metrics["los_loss"],
                label="Train LOS Loss",
                marker="o",
            )
            ax.flat[1].axvline(min_dist_loss_at + 1, linestyle="--", label="Min Loss")
            ax.flat[1].set_xlabel("Epochs")
            ax.flat[1].set_ylabel("Loss")
            ax.flat[1].set_title("LOS Loss Curve")
            ax.flat[1].legend()

            min_dist_loss_at = np.argmin(np.array(self.validation_metrics["dist_loss"]))
            ax.flat[2].plot(
                x_grid,
                self.validation_metrics["dist_loss"],
                label="Val Dist. Loss",
                marker="o",
            )
            ax.flat[2].plot(
                x_grid,
                self.training_metrics["dist_loss"],
                label="Train Dist. Loss",
                marker="o",
            )
            ax.flat[2].axvline(min_dist_loss_at + 1, linestyle="--", label="Min Loss")
            ax.flat[2].set_xlabel("Epochs")
            ax.flat[2].set_ylabel("Loss")
            ax.flat[2].set_title("Dist Loss Curve")
            ax.flat[2].legend()

            min_count_loss_at = np.argmin(
                np.array(self.validation_metrics["count_loss"])
            )
            ax.flat[3].plot(
                x_grid,
                self.validation_metrics["count_loss"],
                label="Val Count Loss",
                marker="o",
            )
            ax.flat[3].plot(
                x_grid,
                self.training_metrics["count_loss"],
                label="Train Count Loss",
                marker="o",
            )
            ax.flat[3].axvline(min_count_loss_at + 1, linestyle="--", label="Min Loss")
            ax.flat[3].set_xlabel("Epochs")
            ax.flat[3].set_ylabel("Loss")
            ax.flat[3].set_title("Count Loss Curve")
            ax.flat[3].legend()

            fig.savefig(loss_fig_save_path)

    def compute_dist_metrics(self, log_variables: dict):
        "Note: log_variables is available from state, passing in dummy arg for clarity"

        return_dict = {}
        if (
            self.epoch_log_variables["y_dist_preds"]
            and self.epoch_log_variables["y_outcomes"]
        ):
            y_count_pred = np.array(self.epoch_log_variables["y_count_preds"])
            y_count_pred_thresh = (y_count_pred > 6).astype(np.int16)
            y_dist_pred_cnt = np.array(self.epoch_log_variables["y_dist_preds"])
            y_dist_pred_thresh = (y_dist_pred_cnt > 0).astype(np.int16)
            y_los_pred = np.array(self.epoch_log_variables["y_los_preds"]).ravel()
            y_los_pred_thresh = (y_los_pred > 1).astype(np.int16)

            y_count = np.array(self.epoch_log_variables["y_outcomes"])[:, 0]
            y_dist = np.array(self.epoch_log_variables["y_outcomes"])[:, 1:]
            y_los = np.array(self.epoch_log_variables["y_los"])
            y_dist_thresh = (y_dist > 0).astype(np.int16)
            y_count_thresh = (y_count > 6).astype(np.int16)
            y_los_thresh = (y_los > 1).astype(np.int16)

            # count metrics (regression)
            print(y_count_pred.max())
            r2_score_count = r2_score(y_count, y_count_pred)
            f1_score_count = f1_score(y_count_thresh, y_count_pred_thresh)

            spearman_count, pval = stats.spearmanr(y_count, y_count_pred)
            mae_count = mean_absolute_error(y_count, y_count_pred)

            # dist metrics (binary classification)
            mae_sep = []
            spearman_dist_sep = []
            for i in range(6):
                mae_sep.append(mean_absolute_error(y_dist[:, i], y_dist_pred_cnt[:, i]))
                spearman_dist, _ = stats.spearmanr(y_dist[:, i], y_dist_pred_cnt[:, i])
                spearman_dist_sep.append(
                    # f1_score(y_dist_thresh[:, i], y_dist_pred_thresh[:, i])
                    spearman_dist
                )

            spearman_dist = np.array(spearman_dist_sep).mean()
            mae_dist = np.array(mae_sep).mean()

            # los classification
            # print(y_los)
            # print(y_los_pred)
            # f1_los = f1_score(y_los_thresh, y_los_pred_thresh)
            spearman_los, _ = stats.spearmanr(y_los, y_los_pred)

            return_dict = {
                "r2_count": r2_score_count,
                "F1_count": f1_score_count,
                "mae_count": mae_count,
                "spearman_count": spearman_count,
                "pval_count": pval,
                "spearman_dist": spearman_dist,
                "mae_dist": mae_dist,
                "spearman_los": spearman_los,
            }

            return_dict.update({f"mae_dist_sep_{i}": mae_sep[i] for i in range(6)})
            return_dict.update(
                {f"spearman_dist_sep_{i}": spearman_dist_sep[i] for i in range(6)}
            )

        return return_dict

    def fit(self):
        model, cfg = self.model, self.cfg
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer, self.scheduler = raw_model.configure_optimizers(cfg)

        best_loss = float("inf")
        self._init_training_metrics()
        self._init_validation_metrics()
        for epoch_count in range(cfg.OPTIM.MAX_EPOCHS):
            train_epoch_metrics = self.run_epoch(
                "train", self.train_dataloader, epoch_count
            )
            # self.plot_grad_flow(epoch_count)

            # scheduler step for 1cycle is done within run_epoch
            if self.cfg.OPTIM.LR_POLICY not in ["1cycle"]:
                self.scheduler.step()

            val_epoch_metrics = self.run_epoch(
                "validation", self.val_dataloader, epoch_count
            )
            # append the epoch metric into
            for k, v in train_epoch_metrics.items():
                self.training_metrics[k].append(v)
            for k, v in val_epoch_metrics.items():
                self.validation_metrics[k].append(v)

            val_loss = val_epoch_metrics["loss"]
            # supports early stopping based on the test loss, or just save always if no test set is provided
            is_good_model = val_loss < best_loss
            if is_good_model:
                best_loss = val_loss
                self.best_epoch = epoch_count
                self.save_checkpoint(epoch_count, val_loss)

        self.plot_training_curves()
        print(f"Experiment logs stored at: {cfg.PATHS.OUT_DIR}")

        return best_loss

    def predict(self):
        print("=" * 100)
        best_epoch = self.load_best_score_checkpoint()
        print(f"loaded model from epoch: {best_epoch}")

        _ = self.run_epoch("validation", self.val_dataloader)
        self.plot_prediction_diagnostics(prefix="val")

        # get test_loss
        test_epoch_metrics_l = []
        for i, test_loader in enumerate(self.test_dataloaders):
            test_epoch_metrics = self.run_epoch("test", test_loader)
            test_epoch_metrics_l.append(test_epoch_metrics)
            self.plot_prediction_diagnostics(prefix=f"test_{i}")

        return test_epoch_metrics_l
