"""A focused subset of the WorldQuant 101 Formulaic Alphas (Kakushadze, 2015).

Ten alphas chosen for (a) volume / price interaction signal that the Alpha158-style
per-asset features lack, and (b) cross-sectional rank operators that the dual-aware
architecture's cross-asset attention can actually exploit. Each returns a [T, N] DataFrame.
"""
import numpy as np
import pandas as pd

EPS = 1e-8


def cs_rank(df):
    """Per-row cross-sectional rank in (0, 1]."""
    return df.rank(axis=1, pct=True)


def ts_rank(df, w):
    """Per-column rolling rank of the last value within the window, normalized to (0, 1]."""
    return df.rolling(w).apply(lambda x: x.argsort().argsort()[-1] / max(len(x) - 1, 1), raw=True)


def delta(df, p):
    return df - df.shift(p)


def delay(df, p):
    return df.shift(p)


def stddev(df, w):
    return df.rolling(w).std()


def rolling_corr(a, b, w):
    return a.rolling(w).corr(b)


def rolling_cov(a, b, w):
    return a.rolling(w).cov(b)


def build_alpha101(o, h, l, c, v):
    """o,h,l,c,v: DataFrames [T, N]. Returns DataFrame-like [T, N, K] and list of names."""
    rets = c.pct_change()
    vwap = (h + l + c) / 3.0                                     # rough VWAP proxy

    alphas = {}

    # 2  : -corr(rank(delta(log(vol),2)), rank((close-open)/open), 6)
    a2 = -rolling_corr(cs_rank(delta(np.log(v + 1), 2)),
                       cs_rank((c - o) / (o + EPS)), 6)
    alphas["a2"] = a2

    # 3  : -corr(rank(open), rank(volume), 10)
    alphas["a3"] = -rolling_corr(cs_rank(o), cs_rank(v), 10)

    # 6  : -corr(open, volume, 10)
    alphas["a6"] = -rolling_corr(o, v, 10)

    # 12 : sign(delta(vol,1)) * (-delta(close,1))
    alphas["a12"] = np.sign(delta(v, 1)) * (-delta(c, 1))

    # 13 : -rank(cov(rank(close), rank(volume), 5))
    alphas["a13"] = -cs_rank(rolling_cov(cs_rank(c), cs_rank(v), 5))

    # 18 : -rank(stddev(|close-open|,5) + (close-open) + corr(close,open,10))
    alphas["a18"] = -cs_rank(stddev((c - o).abs(), 5) + (c - o)
                             + rolling_corr(c, o, 10))

    # 22 : -delta(corr(high,vol,5), 5) * rank(stddev(close,20))
    alphas["a22"] = -delta(rolling_corr(h, v, 5), 5) * cs_rank(stddev(c, 20))

    # 41 : sqrt(high*low) - vwap_proxy
    alphas["a41"] = np.sqrt(h * l) - vwap

    # 54 : -((low-close)*open**5) / ((low-high)*close**5)        ; clipped
    num = -((l - c) * (o ** 5))
    den = (l - h) * (c ** 5)
    alphas["a54"] = (num / (den - EPS)).clip(-10, 10)

    # 101: (close - open) / (high - low + .001)
    alphas["a101"] = (c - o) / (h - l + 1e-3)

    names = ["alpha_" + k for k in alphas.keys()]
    # stack -> [T, N, K]
    arr = np.stack([alphas[k].values for k in alphas.keys()], axis=2).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, names


def build_alpha101_normalized(o, h, l, c, v, split):
    """As above but z-scored using the first `split` rows only (train-stat z-score)."""
    arr, names = build_alpha101(o, h, l, c, v)
    mu = arr[:split].mean(axis=0, keepdims=True)
    sd = arr[:split].std(axis=0, keepdims=True) + EPS
    return np.clip((arr - mu) / sd, -3, 3).astype(np.float32), names
