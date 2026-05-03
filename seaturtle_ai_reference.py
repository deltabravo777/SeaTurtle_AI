#!/usr/bin/env python3
# SeaTurtle AI: compact reference architecture for GitHub.
# token stream -> causal lane fields -> final GQA polish -> tied LM head

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SeaTurtleConfig:
    vocab_size: int = 50257
    seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_lanes: int = 16
    d_state: int = 32
    field_kernel: int = 6
    dilations: Tuple[int, ...] = (1, 2, 4)
    n_q_heads: int = 8
    n_kv_heads: int = 4
    gqa_last_layers: int = 1
    ffn_hidden: int = 3072
    dropout: float = 0.05
    residual_scale: float = 0.91

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class LaneField(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.width = cfg.n_lanes * cfg.d_state
        self.in_proj = nn.Linear(cfg.d_model, self.width)
        self.write_gate = nn.Linear(cfg.d_model, self.width)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, cfg.field_kernel,
                      groups=cfg.n_lanes, dilation=d, bias=False)
            for d in cfg.dilations
        ])
        self.lane_decay = nn.Parameter(torch.full((cfg.n_lanes, 1), 1.25))
        self.out_proj = nn.Linear(self.width, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        u = self.in_proj(x)
        h = (u * torch.sigmoid(self.write_gate(x))).transpose(1, 2)
        y = 0
        for conv, d in zip(self.convs, self.cfg.dilations):
            y = y + conv(F.pad(h, ((self.cfg.field_kernel - 1) * d, 0)))
        y = (y / len(self.convs)).transpose(1, 2)
        decay = torch.sigmoid(self.lane_decay).repeat_interleave(self.cfg.d_state, 0)
        y = y * decay.view(1, 1, -1) + u * (1 - decay.view(1, 1, -1))
        return self.out_proj(self.drop(torch.tanh(y)))

class GQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_q_heads == 0 and cfg.n_q_heads % cfg.n_kv_heads == 0
        self.nq, self.nkv = cfg.n_q_heads, cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_q_heads
        self.q = nn.Linear(cfg.d_model, self.nq * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout
    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.nq, self.hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        rep = self.nq // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, -1))

class SeaTurtleBlock(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__(); self.scale = cfg.residual_scale
        self.field_norm, self.field = RMSNorm(cfg.d_model), LaneField(cfg)
        self.attn = GQA(cfg) if layer_id >= cfg.n_layers - cfg.gqa_last_layers else None
        self.attn_norm = RMSNorm(cfg.d_model) if self.attn else None
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.up, self.down = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden), nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        x = x + self.scale * self.field(self.field_norm(x))
        if self.attn is not None: x = x + self.scale * self.attn(self.attn_norm(x))
        a, b = self.up(self.ffn_norm(x)).chunk(2, -1)
        return x + self.scale * self.down(self.drop(F.silu(a) * b))

class SeaTurtleLM(nn.Module):
    def __init__(self, cfg=SeaTurtleConfig()):
        super().__init__(); self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([SeaTurtleBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm, self.head = RMSNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for block in self.blocks: x = block(x)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    cfg = SeaTurtleConfig(d_model=512, n_layers=6, ffn_hidden=1536)
    model = SeaTurtleLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    _, loss = model(x, x)
    print(f"SeaTurtle AI | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M | loss={loss.item():.3f}")
