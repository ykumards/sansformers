import math
from functools import reduce

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMICVisitwiseAxialEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        embed_size = self.cfg.MODEL.EMBED_SIZE
        vocab_size = self.cfg.MODEL.VOCAB_SIZE

        self.token_embed = nn.Embedding(vocab_size, embed_size - 8)
        self.embed_sum = EmbeddingAdder()

        self.delta_temb = DeltTEncoding(embed_size - 8, dropout=0.1, max_len=100000)
        self.pos_embedding = FixedAxialPositionalEncoding(embed_size - 8, dropout=0.1)

    def forward(
        self,
        diag_seq,
        proc_seq,
        drug_seq,
        delta_t,
        service,
        admtype,
        insur,
        marit,
        seq_length,
    ):
        # taking intervals of every 15 days instead of everyday
        # could improve perf
        delta_t.div_(15.0)

        indices = torch.arange(seq_length.max(), device=diag_seq.device)
        len_mask = indices.view(1, -1) < seq_length.view(-1, 1)

        # delta_t.clamp_(0, 999)
        delta_t = delta_t.round().cumsum(dim=1)
        delta_t = delta_t * len_mask.int()
        doy_embedded = self.delta_temb(delta_t.long())[:, :, None, :]

        icd_embed_sum = self.token_embed(diag_seq)
        proc_embed_sum = self.token_embed(proc_seq)
        drug_embed_sum = self.token_embed(drug_seq)
        service_embed = self.token_embed(service)
        admtype_embed = self.token_embed(admtype)

        # position embedding
        b, t, v, e = icd_embed_sum.size()

        insur_embed = self.token_embed(insur)[:, :, None, :]
        marit_embed = self.token_embed(marit)[:, :, None, :]

        visit_embed = torch.cat(
            [
                icd_embed_sum,
                proc_embed_sum,
                drug_embed_sum,
                service_embed,
                admtype_embed,
                insur_embed,
                marit_embed,
                doy_embedded,
            ],
            dim=2,
        )
        visit_embed = self.pos_embedding(visit_embed)

        b, t, v, e = visit_embed.shape

        return F.layer_norm(visit_embed, [t, v, e])


## Transformer Helper Embeddings
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class FixedPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[None, :, :]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].detach()
        return self.dropout(x)


class FixedAxialPositionalEncoding(FixedPositionalEncoding):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__(d_model, dropout, max_len)

    def forward(self, x):
        b, t, v, e = x.shape
        penc = self.pe[:, : x.size(1)].detach()
        x = x + einops.repeat(penc, "b t e -> b t v e", v=v)
        return self.dropout(x)


class DeltTEncoding(nn.Module):
    "Implement the PE function specifically for delta_t."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, delta_t):
        # delta_t will be a cummulative number so we will just pick
        # the corresponding indices from the positional encoding tensor
        b, t = delta_t.shape
        e = self.pe.shape[-1]
        res_enc = torch.zeros((b, t, e)).to(delta_t.device)
        # TODO fix this embarrasing for-loop
        for i in range(b):
            res_enc[i, :, :] = self.pe[:, delta_t[i], :]

        return res_enc


class EmbeddingAdder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=2):
        return x.sum(dim=dim)
