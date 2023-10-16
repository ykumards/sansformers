import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class SmallAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        return self.to_out(out)


class FullMLPLayer(nn.Module):
    def __init__(
        self, d_model, max_seq_len, causal, act=nn.GELU(), ff_hidden_mult=4, p_do=0.8
    ):
        super().__init__()

        dim = int(d_model * ff_hidden_mult)
        dim_out = dim // 2

        self.d_model = d_model
        self.causal = causal
        self.do = nn.Dropout(p_do)

        self.proj_ch1 = nn.Sequential(nn.Linear(d_model, dim), nn.GELU())
        self.conv_proj = nn.Conv1d(max_seq_len, max_seq_len, 1)

        self.proj_out = nn.Linear(dim // 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(dim_out)
        self.norm3 = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len
        self.act = act
        # optional small attention improves accuracy
        # self.attn = SmallAttention(d_model, dim // 2, 64, causal)

        init_eps = 1e-3
        init_eps /= max_seq_len
        nn.init.uniform_(self.conv_proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.conv_proj.bias, 1.0)

    def forward(self, x):
        visit_len = x.size(1)
        shortcut = x
        x = self.norm1(x)
        x = self.act(self.proj_ch1(x))

        x_attn = None
        # x_attn = self.attn(shortcut)

        # spatial proj unit
        res, gate = x.chunk(2, dim=-1)
        gate = self.norm2(gate)

        weight, bias = self.conv_proj.weight, self.conv_proj.bias
        weight, bias = weight[:visit_len, :visit_len], bias[:visit_len]
        if self.causal:
            mask = torch.ones(weight.shape[:2], device=x.device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.0)

        gate = F.conv1d(gate, weight, bias)

        # x = gate * res
        if x_attn is not None:
            x = (gate + x_attn) * res
        else:
            x = gate * res

        x = self.proj_out(x)

        return x


class SansformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        causal,
        ff_hidden_mult=4,
        p_do=0.0,
    ):
        super().__init__()

        self.sans_attention = FullMLPLayer(
            d_model=d_model,
            max_seq_len=200,
            causal=causal,
            ff_hidden_mult=ff_hidden_mult,
            p_do=p_do,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # GEGLU improves perf but is param heavy
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_mult * d_model),
            nn.GELU(),
            # GEGLU(ff_hidden_mult * d_model, ff_hidden_mult * d_model),
            nn.Dropout(p_do),
            nn.Linear(ff_hidden_mult * d_model, d_model),
            nn.Dropout(p_do),
        )
        self.do = nn.Dropout(p_do)

    def forward(self, x, length_mask=None):
        sans_attended = self.sans_attention(x)
        y = x = self.norm1(sans_attended)
        y = self.ff(y)
        return self.norm2(x + y)


###########
# Helpers
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        breakpoint()
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class GEGLUNarow(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Rezero(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, fn, x):
        return fn(x) * self.g


## helpers
def exists(val):
    return val is not None
