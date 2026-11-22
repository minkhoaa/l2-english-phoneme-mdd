"""Inspect one-query, batch-first MultiheadAttention without changing trained weights."""

import math

import torch
from torch.nn import functional as F


def check(module, q, k, v, invalid):
    if (
        module.training
        or not module.batch_first
        or not module._qkv_same_embed_dim
        or module.bias_k is not None
        or module.bias_v is not None
        or module.add_zero_attn
        or q.shape[1] != 1
        or k.shape != v.shape
        or invalid.dtype != torch.bool
        or invalid.shape != v.shape[:2]
        or invalid.all(1).any()
    ):
        raise ValueError(
            "expected an eval-mode, batched one-query attention with at least one valid key"
        )


def project(module, x, part):
    d = module.embed_dim
    sl = slice(part * d, (part + 1) * d)
    bias = module.in_proj_bias
    return F.linear(x, module.in_proj_weight[sl], None if bias is None else bias[sl])


def reconstruct(module, q, k, v, invalid):
    """Return independent output and actual weights [slot,head,frame]."""
    check(module, q, k, v, invalid)
    n, w, d = k.shape
    h = module.num_heads
    dh = d // h
    qh = project(module, q, 0).reshape(n, h, dh)
    kh = project(module, k, 1).reshape(n, w, h, dh).transpose(1, 2)
    s = torch.einsum("nhd,nhwd->nhw", qh / math.sqrt(dh), kh)
    a = s.masked_fill(invalid[:, None, :], -torch.inf).softmax(-1)
    vh = project(module, v, 2).reshape(n, w, h, dh)
    z = torch.einsum("nhw,nwhd->nhd", a, vh).reshape(n, 1, d)
    return module.out_proj(z), a


def intervene(mode):
    def hook(module, args, kwargs, original):
        q, k, v = args
        invalid = kwargs["key_padding_mask"]
        check(module, q, k, v, invalid)
        if mode == "uniform":
            w = (~invalid).to(v.dtype)
            w = w / w.sum(-1, keepdim=True)
        elif mode == "centre":
            i = v.shape[1] // 2
            if invalid[:, i].any():
                raise ValueError("invalid target peak")
            w = torch.zeros_like(invalid, dtype=v.dtype)
            w[:, i] = 1
        else:
            raise ValueError(mode)
        # All heads use the same distribution, so averaging before V is equivalent.
        z = module.out_proj(project(module, torch.bmm(w.unsqueeze(1), v), 2))
        return z, (None if original[1] is None else w.unsqueeze(1))

    return hook
