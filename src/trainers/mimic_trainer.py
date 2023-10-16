import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from matplotlib.pyplot import Line2D
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
from tqdm import tqdm

sns.set_style("white")

from trainers.base_trainer import BaseTrainer


class Trainer_MIMIC(BaseTrainer):
    def __init__(self, cfg, model, train_dataloader, val_dataloader, test_dataloaders):
        super().__init__(cfg, model, train_dataloader, val_dataloader, test_dataloaders)

        self.epoch_log_variables = {
            "y_outcomes": [],
            "y_los": [],
            "y_bin_preds": [],
            "y_los_preds": [],
        }

    def _init_log_variables(self):
        for k, _ in self.epoch_log_variables.items():
            self.epoch_log_variables[k] = []

    def plot_training_curves(self):
        loss_fig_save_path = os.path.join(self.cfg.PATHS.OUT_DIR, "training_losses.png")

        with plt.style.context("seaborn-muted"):
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

            min_loss_at = np.argmin(np.array(self.validation_metrics["los_loss"]))

            ax.flat[2].plot(
                x_grid,
                self.validation_metrics["los_loss"],
                label="Val Loss",
                marker="o",
            )
            ax.flat[2].plot(
                x_grid,
                self.training_metrics["los_loss"],
                label="Train Loss",
                marker="o",
            )
            ax.flat[2].axvline(min_loss_at + 1, linestyle="--", label="Min Loss")
            ax.flat[2].set_xlabel("Epochs")
            ax.flat[2].set_ylabel("Loss")
            ax.flat[2].set_title("LoS Loss Curve")
            ax.flat[2].legend()

            min_loss_at = np.argmin(np.array(self.validation_metrics["bin_loss"]))

            ax.flat[3].plot(
                x_grid,
                self.validation_metrics["bin_loss"],
                label="Val Loss",
                marker="o",
            )
            ax.flat[3].plot(
                x_grid,
                self.training_metrics["bin_loss"],
                label="Train Loss",
                marker="o",
            )
            ax.flat[3].axvline(min_loss_at + 1, linestyle="--", label="Min Loss")
            ax.flat[3].set_xlabel("Epochs")
            ax.flat[3].set_ylabel("Loss")
            ax.flat[3].set_title("Bin. Loss Curve")
            ax.flat[3].legend()

            fig.savefig(loss_fig_save_path)

        return

    def compute_dist_metrics(self, log_variables: dict):
        "Note: log_variables is available from state, passing in dummy arg for clarity"
        # override base trainer

        return_dict = {}

        y_bin_pred = np.array(self.epoch_log_variables["y_bin_preds"])
        y_bin_pred_thresh = (y_bin_pred > 0.5).astype(np.int8)
        y_bin_true = np.array(self.epoch_log_variables["y_outcomes"])

        y_los_pred = np.array(self.epoch_log_variables["y_los_preds"])
        y_los_true = np.array(self.epoch_log_variables["y_los"])

        # metrics
        auc_bin = roc_auc_score(y_bin_true, y_bin_pred)

        spearman_los, _ = stats.spearmanr(y_los_true, y_los_pred)
        # r2_los = r2_score(y_los_true, y_los_pred)

        return_dict = {
            "auc_bin": auc_bin,
            "spearman_los": spearman_los,
        }

        return return_dict

    def run_epoch(self, split, dataloader, epoch_count=0):
        assert split.lower() in ("train", "validation", "test")

        self._init_log_variables()

        is_train = True if split.lower() == "train" else False
        self.model.train(is_train)

        lr = 0.0
        losses = []
        bin_losses = []
        los_losses = []
        pbar = (
            tqdm(enumerate(dataloader), total=len(dataloader))
            if is_train
            else enumerate(dataloader)
        )
        self.model.zero_grad()
        acc_steps = self.cfg.MODEL.ACCU_GRAD_STEPS
        for it, batch in pbar:
            # place data on the correct device
            batch_dict = self._prepare_batch(batch)
            # forward the model
            with torch.set_grad_enabled(is_train):
                with autocast(enabled=self.cfg.USE_AMP):
                    (
                        loss,
                        bin_loss,
                        los_loss,
                        y_pred_bin,
                        y_los_pred,
                        y_los_true,
                        patient_vec,
                    ) = self.model(**batch_dict)
            report_loss = loss
            if is_train:
                loss = loss / acc_steps
                # backprop and update the parameters
                self.grad_scaler.scale(loss).backward()
                if (it + 1) % self.cfg.MODEL.ACCU_GRAD_STEPS == 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.cfg.OPTIM.GRAD_CLIP_T
                    )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.model.zero_grad()

                    lr = 0.0
                    for param_group in self.optimizer.param_groups:
                        lr = param_group["lr"]

                    if self.cfg.OPTIM.LR_POLICY in ["1cycle"]:
                        self.scheduler.step()
                    self.training_metrics["lr"].append(lr)

                # report progress
                pbar.set_description(
                    f"epoch {epoch_count + 1} iter {it}: train loss {loss.item() * acc_steps:.5f} lr {lr:e}"
                )

            # store metrics for logging
            losses.append(report_loss.item())
            bin_losses.append(bin_loss.item())
            los_losses.append(los_loss.item())

            self.epoch_log_variables["y_outcomes"].extend(
                batch_dict["y_outcome"].cpu().detach().numpy().tolist()
            )
            self.epoch_log_variables["y_los"].extend(
                y_los_true.cpu().detach().numpy().tolist()
            )
            self.epoch_log_variables["y_bin_preds"].extend(
                y_pred_bin.cpu().detach().numpy().tolist()
            )
            self.epoch_log_variables["y_los_preds"].extend(
                y_los_pred.cpu().detach().numpy().tolist()
            )

        metrics_d = {
            "loss": float(np.mean(losses)),
            "bin_loss": float(np.mean(bin_losses)),
            "los_loss": float(np.mean(los_losses)),
        }

        if not is_train:
            pass
            metrics_d.update(self.compute_dist_metrics(self.epoch_log_variables))

        # compute metrics
        print_str = f"{split} epoch: {epoch_count+1} "
        for k, v in metrics_d.items():
            print_key = str(k)
            print_str += f" | {print_key}: {v:.5f}"

        print(print_str)
        print("=" * 100)
        return metrics_d

    def predict(self):
        print("=" * 100)
        best_epoch = self.load_best_score_checkpoint()
        print(f"loaded model from epoch: {best_epoch}")

        _ = self.run_epoch("validation", self.val_dataloader)

        # get test_loss
        test_epoch_metrics_l = []
        for i, test_loader in enumerate(self.test_dataloaders):
            test_epoch_metrics = self.run_epoch("test", test_loader)
            test_epoch_metrics_l.append(test_epoch_metrics)

        return test_epoch_metrics_l
