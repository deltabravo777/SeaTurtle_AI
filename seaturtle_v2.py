#!/usr/bin/env python3
"""SeaTurtle AI 6+1+9: compact single-file reference.

Architecture:
    token stream
    -> 6 SeaTurtleSpectral causal mixer blocks
    -> 1 SeaTurtleMegaConv refresh block
    -> 9 GPT/GQA blocks
    -> tied language-model head

This file is meant to be a small GitHub-friendly architecture + pipeline demo,
not a full production trainer.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeaTurtleConfig:
    vocab_size: int = 50_257
    seq_len: int = 1_024
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int = 2_304
    dropout: float = 0.08
    emb_dropout: float = 0.08
    bias: bool = False
    spectral_layers: int = 6
    megaconv_layers: int = 1
    gpt_layers: int = 9
    residual_scale: float = 1.0

    @property
    def n_layers(self) -> int:
        return self.spectral_layers + self.megaconv_layers + self.gpt_layers

    @property
    def schedule(self) -> str:
        return "S" * self.spectral_layers + "M" * self.megaconv_layers + "G" * self.gpt_layers

    def validate(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide n_kv_heads"
        assert self.spectral_layers == 6 and self.megaconv_layers == 1 and self.gpt_layers == 9, (
            "this reference file is locked to the 6+1+9 layout"
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x.transpose(1, 2), (self.left_pad, 0))
        return self.conv(y).transpose(1, 2)


class SwiGLU(nn.Module):
    def __init__(self, cfg: SeaTurtleConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=cfg.bias)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(a) * b))


class SeaTurtleSpectralMixer(nn.Module):
    """Causal multi-band mixer for the first 6 layers.

    The three bands act like short/mid/long causal filters. The mixer never reads
    future tokens, then merges the bands through a gated projection.
    """

    def __init__(self, cfg: SeaTurtleConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.band3 = CausalDepthwiseConv1d(cfg.d_model, kernel_size=3, dilation=1, bias=cfg.bias)
        self.band7 = CausalDepthwiseConv1d(cfg.d_model, kernel_size=7, dilation=1, bias=cfg.bias)
        self.band15 = CausalDepthwiseConv1d(cfg.d_model, kernel_size=5, dilation=4, bias=cfg.bias)
        self.band_weight = nn.Parameter(torch.zeros(3))
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(x)
        gate = torch.sigmoid(self.gate(x))
        weights = F.softmax(self.band_weight, dim=0)
        y = weights[0] * self.band3(u) + weights[1] * self.band7(u) + weights[2] * self.band15(u)
        return self.out_proj(self.drop(torch.tanh(y) * gate))


class SeaTurtleSpectralBlock(nn.Module):
    def __init__(self, cfg: SeaTurtleConfig):
        super().__init__()
        self.scale = cfg.residual_scale
        self.mix_norm = RMSNorm(cfg.d_model)
        self.mix = SeaTurtleSpectralMixer(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale * self.mix(self.mix_norm(x))
        x = x + self.scale * self.ffn(self.ffn_norm(x))
        return x


class SeaTurtleMegaConvBlock(nn.Module):
    """Single causal refresh block between the spectral stack and GPT stack."""

    def __init__(self, cfg: SeaTurtleConfig):
        super().__init__()
        self.scale = cfg.residual_scale
        self.mix_norm = RMSNorm(cfg.d_model)
        self.expand = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=cfg.bias)
        self.conv = CausalDepthwiseConv1d(2 * cfg.d_model, kernel_size=9, dilation=2, bias=cfg.bias)
        self.squeeze = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.expand(self.mix_norm(x)).chunk(2, dim=-1)
        y = self.conv(torch.cat([a, b], dim=-1))
        a2, b2 = y.chunk(2, dim=-1)
        x = x + self.scale * self.squeeze(self.drop(F.silu(a2) * b2))
        x = x + self.scale * self.ffn(self.ffn_norm(x))
        return x


class GQA(nn.Module):
    def __init__(self, cfg: SeaTurtleConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o(y.transpose(1, 2).contiguous().view(bsz, seq_len, -1))


class GPTBlock(nn.Module):
    def __init__(self, cfg: SeaTurtleConfig):
        super().__init__()
        self.scale = cfg.residual_scale
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GQA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale * self.attn(self.attn_norm(x))
        x = x + self.scale * self.ffn(self.ffn_norm(x))
        return x


class SeaTurtleLM(nn.Module):
    def __init__(self, cfg: Optional[SeaTurtleConfig] = None):
        super().__init__()
        self.cfg = cfg or SeaTurtleConfig()
        self.cfg.validate()

        self.tok = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos = nn.Embedding(self.cfg.seq_len, self.cfg.d_model)
        self.emb_drop = nn.Dropout(self.cfg.emb_dropout)

        blocks: list[nn.Module] = []
        for item in self.cfg.schedule:
            if item == "S":
                blocks.append(SeaTurtleSpectralBlock(self.cfg))
            elif item == "M":
                blocks.append(SeaTurtleMegaConvBlock(self.cfg))
            elif item == "G":
                blocks.append(GPTBlock(self.cfg))
            else:
                raise ValueError(f"unknown schedule item: {item}")

        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, seq_len = idx.shape
        if seq_len > self.cfg.seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds configured limit {self.cfg.seq_len}")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.emb_drop(self.tok(idx) + self.pos(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss


DEFAULT_TEXT = """
SeaTurtle AI is a compact causal language model reference.
It mixes tokens with six spectral causal blocks, refreshes them with one
mega-conv block, and polishes the stream with nine GPT/GQA blocks.
This tiny demo repeats a short corpus only to prove the pipeline runs.
""".strip()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_bytes(path: Optional[str]) -> torch.Tensor:
    if path:
        data = Path(path).read_bytes()
    else:
        data = (DEFAULT_TEXT + "\n").encode("utf-8") * 2048
    return torch.tensor(list(data), dtype=torch.long)


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    if tokens.numel() <= seq_len + 1:
        raise ValueError("text corpus must contain more bytes than seq_len + 1")
    starts = torch.randint(0, tokens.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([tokens[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([tokens[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y


def build_demo_model(args: argparse.Namespace, vocab_size: int) -> SeaTurtleLM:
    cfg = SeaTurtleConfig(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_hidden=args.ffn_hidden,
        dropout=args.dropout,
        emb_dropout=args.dropout,
    )
    return SeaTurtleLM(cfg)


def forward_demo(args: argparse.Namespace) -> None:
    cfg = SeaTurtleConfig(
        vocab_size=256,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_hidden=args.ffn_hidden,
        dropout=args.dropout,
        emb_dropout=args.dropout,
    )
    model = SeaTurtleLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = -100
    _, loss = model(x, y)
    print(
        f"SeaTurtle 6+1+9 | schedule={cfg.schedule} | "
        f"params={count_parameters(model) / 1e6:.2f}M | loss={loss.item():.3f}"
    )


def train_tiny(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    tokens = load_bytes(args.text_file)
    model = build_demo_model(args, vocab_size=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    print(f"SeaTurtle 6+1+9 tiny training demo | device={device} amp_bf16={use_amp}")
    print(
        f"schedule={model.cfg.schedule} params={count_parameters(model) / 1e6:.2f}M "
        f"tokens={tokens.numel():,} dropout={model.cfg.dropout}"
    )

    model.train()
    start = time.perf_counter()
    seen = 0
    for step in range(1, args.steps + 1):
        x, y = sample_batch(tokens, args.batch_size, args.seq_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        seen += args.batch_size * args.seq_len
        if step == 1 or step % args.log_every == 0:
            elapsed = max(time.perf_counter() - start, 1e-9)
            print(f"step={step:04d} loss={loss.item():.4f} tok/s={seen / elapsed:,.0f}")

    if args.save:
        ckpt = {"model": model.state_dict(), "config": model.cfg.__dict__}
        torch.save(ckpt, args.save)
        print(f"saved {args.save}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file SeaTurtle 6+1+9 architecture + tiny training demo")
    parser.add_argument("--train", action="store_true", help="run the tiny byte-level training loop instead of a forward pass")
    parser.add_argument("--text-file", type=str, default=None, help="optional text file for tiny byte-level training")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save", type=str, default="seaturtle_6_1_9_demo.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        train_tiny(args)
    else:
        forward_demo(args)
        print("Run with --train to launch the tiny byte-level training pipeline.")


if __name__ == "__main__":
    main()
