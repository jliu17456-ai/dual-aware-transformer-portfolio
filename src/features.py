"""Per-asset time-series features (Qlib Alpha158-style), OHLCV-only, no cross-section.

All features are computed independently per asset from its own OHLCV series, then
z-scored using statistics from the training slice only (no look-ahead). Cross-sectional
rank factors are deliberately avoided (unstable with a small universe).
"""
import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 60]
EPS = 1e-8


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + EPS)
    return 100 - 100 / (1 + rs)


def _beta(close, w):
    x = np.arange(w)
    xm = x.mean(); xd = x - xm; denom = (xd ** 2).sum()
    def slope(arr):
        return np.dot(xd, arr - arr.mean()) / denom
    return close.rolling(w).apply(slope, raw=True) / (close + EPS)


def asset_features(o, h, l, c, v):
    """o,h,l,c,v: 1-D Series for one asset. Returns DataFrame [T, F]."""
    f = {}
    f["kmid"] = (c - o) / (o + EPS)
    f["klen"] = (h - l) / (o + EPS)
    f["kmid2"] = (c - o) / (h - l + EPS)
    f["kup"] = (h - np.maximum(o, c)) / (o + EPS)
    f["klow"] = (np.minimum(o, c) - l) / (o + EPS)
    f["ksft"] = (2 * c - h - l) / (o + EPS)
    for w in WINDOWS:
        f[f"roc{w}"] = c / c.shift(w) - 1
        f[f"ma{w}"] = c.rolling(w).mean() / c - 1
        f[f"std{w}"] = c.rolling(w).std() / (c + EPS)
        f[f"max{w}"] = h.rolling(w).max() / c - 1
        f[f"min{w}"] = l.rolling(w).min() / c - 1
        f[f"rsv{w}"] = (c - l.rolling(w).min()) / (h.rolling(w).max() - l.rolling(w).min() + EPS)
        f[f"vma{w}"] = v.rolling(w).mean() / (v + EPS)
    f["rsi14"] = _rsi(c, 14) / 100.0
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    f["macd"] = (ema12 - ema26) / (c + EPS)
    f["beta20"] = _beta(c, 20)
    return pd.DataFrame(f)


def build_features(o, h, l, c, v, tickers, split):
    """Returns feats [T, N, F] z-scored on train stats (first `split` rows), and names."""
    sample = asset_features(o[tickers[0]], h[tickers[0]], l[tickers[0]], c[tickers[0]], v[tickers[0]])
    names = list(sample.columns)
    T = len(c); N = len(tickers); F = len(names)
    feats = np.zeros((T, N, F), dtype=np.float32)
    for j, t in enumerate(tickers):
        df = asset_features(o[t], h[t], l[t], c[t], v[t])
        feats[:, j, :] = df[names].values
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    mu = feats[:split].mean(axis=0, keepdims=True)
    sd = feats[:split].std(axis=0, keepdims=True) + EPS
    feats = np.clip((feats - mu) / sd, -3, 3).astype(np.float32)
    return feats, names
