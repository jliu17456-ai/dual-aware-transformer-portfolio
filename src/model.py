"""Dual-aware Transformer policy with an information bottleneck.

  temporal-aware layer  : per-asset Transformer encoder over the lookback window
  VIB bottleneck        : compress each asset's temporal state to a minimal sufficient z
  asset-aware layer     : cross-asset attention biased by a transfer-entropy prior
  allocation head       : softmax over assets -> long-only, fully-invested weights (sum = 1)

Deliberately small (anti-overfit on a tiny financial dataset). The VIB and TE prior are
toggled by flags so the ablations Base / +VIB / +Prior / Full share one code path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, n_feat, d_model, n_heads, lookback, layers, dropout):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(lookback, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):                       # x [B, N, T, Ffeat]
        B, N, T, Ff = x.shape
        h = self.proj(x.reshape(B * N, T, Ff)) + self.pos
        M, lim = h.shape[0], 16384              # chunk: cap activation memory (and SDPA's 65535 limit)
        if M <= lim:
            h = self.enc(h)
        else:
            h = torch.cat([self.enc(h[i:i + lim]) for i in range(0, M, lim)], dim=0)
        return h[:, -1].reshape(B, N, -1)       # last step -> [B, N, d]


class CrossAssetAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads; self.hd = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x, bias, alpha):          # x [B,N,d]; bias [N,N] or None
        B, N, d = x.shape
        q = self.q(x).view(B, N, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.hd).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)   # [B,H,query,key]
        if bias is not None:
            # bias[i,j] = flow i->j ; key=i attends into query=j  -> index [query=j, key=i]
            logits = logits + alpha * bias.t().unsqueeze(0).unsqueeze(0)
        att = F.softmax(logits, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, d)
        return self.o(out)


class DualAwareNet(nn.Module):
    def __init__(self, n_feat, n_assets, lookback, d_model=48, n_heads=4, z_dim=16,
                 t_layers=1, dropout=0.1, use_vib=True, use_prior=True):
        super().__init__()
        self.use_vib = use_vib; self.use_prior = use_prior
        self.temporal = TemporalEncoder(n_feat, d_model, n_heads, lookback, t_layers, dropout)
        self.to_stats = nn.Linear(d_model, 2 * z_dim)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.cross = CrossAssetAttention(d_model, n_heads)
        self.alpha = nn.Parameter(torch.zeros(1))       # learnable prior strength
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, te_bias=None):
        h = self.temporal(x)                            # [B,N,d]
        mu, logvar = self.to_stats(h).chunk(2, dim=-1)
        if self.use_vib and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        else:
            z = mu
            kl = x.new_zeros(())
        if not self.use_vib:
            kl = x.new_zeros(())
        zc = self.z_proj(z)                             # [B,N,d]
        bias = te_bias if self.use_prior else None
        zc = zc + self.drop(self.cross(zc, bias, self.alpha))
        score = self.head(zc).squeeze(-1)               # [B,N]
        w = F.softmax(score, dim=1)                      # long-only, fully invested (sum = 1)
        return w, kl
