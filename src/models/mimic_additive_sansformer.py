import sys

sys.path.append("../")

import warnings

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as t_optim

import utils.lr_policy as lr_policy
from models.attention_blocks import SansformerLayer
from models.embeddings import MIMICVisitwiseAxialEmbedding

warnings.simplefilter("ignore")


class MimicAdditiveSansformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        embed_size = self.cfg.MODEL.EMBED_SIZE
        depth = self.cfg.MODEL.TRANS_DEPTH
        self.dropout_p = self.cfg.MODEL.DROPOUT_P

        self.pummel_embed = MIMICVisitwiseAxialEmbedding(cfg)
        self.ethn_embed = nn.Embedding(8, 4)

        self.sansformer = nn.ModuleList(
            [
                SansformerLayer(
                    d_model=embed_size,
                    causal=True,
                    ff_hidden_mult=4,
                    p_do=self.dropout_p,
                )
                for _ in range(depth)
            ]
        )

        self.bin_fc = nn.Sequential(
            nn.LayerNorm([embed_size]),
            nn.Linear(embed_size, 1),
        )

        if cfg.PATHS.PRETRAINED_TRANSFORMER_FILE:
            checkpoint = torch.load(
                cfg.PATHS.PRETRAINED_TRANSFORMER_FILE, map_location="cpu"
            )
            self.load_state_dict(checkpoint["model_state"])
            print("loaded pretrained model.")

    def forward(
        self,
        diag_seq,
        proc_seq,
        drg_seq,
        seq_length,
        delta_t,
        age,
        gender,
        language,
        ethnicity,
        service,
        admtype,
        insurance,
        marit,
        y_outcome,
        y_los,
        y_next_v,
        x_los,
        x_next_v,
        **kwargs
    ):
        seq_lengths = seq_length.long()
        ethnicity_embed = self.ethn_embed(ethnicity.long())

        indices = torch.arange(seq_length.max(), device=seq_length.device)
        len_mask = indices.view(1, -1) < seq_length.view(-1, 1)

        x_num = torch.cat(
            [
                gender[:, None],
                age[:, None],
                language[:, None],
                ethnicity_embed,
            ],
            dim=1,
        )

        visit_embed = self.pummel_embed(
            diag_seq,
            proc_seq,
            drg_seq,
            delta_t,
            service,
            admtype,
            insurance,
            marit,
            seq_lengths,
        )

        add_embed = einops.reduce(visit_embed, "b t v e -> b t e", "sum")

        b, t, e = add_embed.shape

        x_num_ext = einops.repeat(x_num, "b h -> b t h", t=t)
        x_trans = torch.cat([add_embed, x_num_ext, x_los[:, :, None]], dim=-1)

        b, t, e = x_trans.shape
        for layer in self.sansformer:
            shortcut = x_trans  # b t e
            x_trans = layer(F.layer_norm(x_trans, [t, e])) + shortcut

        x_trans_max = einops.reduce(x_trans, "b t e -> b e", "max")
        x_trans_avg = einops.reduce(x_trans, "b t e -> b e", "mean")
        patient_vec = x_trans_avg + F.gelu(x_trans_max)

        # bin
        logit_pred = self.bin_fc(patient_vec)
        # probs are needed for AUC score
        y_pred_bin = torch.sigmoid(logit_pred)

        bin_loss = F.binary_cross_entropy_with_logits(
            logit_pred.view(-1),
            y_outcome.float(),
            reduction="mean",
            # add small positive weight to handle class imbalance
            pos_weight=torch.tensor(3.0, device=y_pred_bin.device),
        )

        total_loss = bin_loss

        #TODO returning some empty tensors for API conformity, needs cleanup
        return (
            total_loss,
            bin_loss,
            bin_loss * 0,
            y_pred_bin,
            torch.zeros_like(y_pred_bin),
            torch.zeros_like(y_pred_bin),
            patient_vec,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def configure_optimizers(self, cfg):
        optimizer = t_optim.RAdam(
            self.parameters(),
            lr=cfg.OPTIM.BASE_LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            betas=(0.9, 0.99),
            eps=1e-9,
        )
        scheduler = lr_policy.get_lr_sched(cfg, optimizer)

        return optimizer, scheduler
