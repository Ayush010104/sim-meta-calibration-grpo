"""Learned causal self-attention over the last K steps -> next-step 1-bit phase logits.
NOT trained or evaluated for rate gain; stage 1 results gated this off (see notes)."""
import math
import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, explicit W_Q / W_K / W_V, causal mask."""
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)

    def forward(self, x):                                   # x: (B, K, d)
        B, K, d = x.shape
        split = lambda t: t.view(B, K, self.h, self.dk).transpose(1, 2)   # (B, h, K, dk)
        Q, Kt, V = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = Q @ Kt.transpose(-2, -1) / math.sqrt(self.dk)             # (B, h, K, K)
        causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(causal, float("-inf"))                 # step t sees only steps <= t
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, K, d)
        return self.Wo(out), attn


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = SelfAttention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        a, attn = self.att(self.ln1(x))
        x = x + a
        return x + self.ff(self.ln2(x)), attn


class ContextPhasePredictor(nn.Module):
    """tokens (B, K, feat_dim) -> logits (B, n_phase): P(bit=1) per element of the NEXT step's solution."""
    def __init__(self, feat_dim, n_phase=144, d=64, heads=4, layers=2, K=8):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d)
        self.pos = nn.Parameter(torch.zeros(K, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, n_phase)

    def forward(self, tokens):
        x = self.inp(tokens) + self.pos[: tokens.shape[1]]
        for b in self.blocks:
            x, attn = b(x)
        return self.head(x[:, -1]), attn


def canonicalize(bits, L=4, M=36):
    """bits: (..., L*M) in {0,1}. Layer-flip is a reward symmetry (verified: 16 equivalent
    solutions), so flip each layer so its first element is 0 -> unique label."""
    b = np.array(bits).copy().reshape(*np.shape(bits)[:-1], L, M)
    flip = b[..., :1] == 1
    b = np.where(flip, 1 - b, b)
    return b.reshape(*np.shape(bits)[:-1], L * M)


def step_features(H, prev_bits, prev_minr):
    """One token: per-user gain (K) + |normalized Gram| upper-tri (K(K+1)/2) + prev canonical bits (144) + min-rate."""
    g = np.linalg.norm(H, axis=0)
    G = np.abs(H.conj().T @ H) / (np.outer(g, g) + 1e-30)
    return np.concatenate([g / (g.max() + 1e-30), G[np.triu_indices(H.shape[1])], 2.0 * prev_bits - 1.0, [prev_minr]])


def loss_fn(logits, target_bits):
    return nn.functional.binary_cross_entropy_with_logits(logits, target_bits)
