"""Cross-sectional rank features + macro regime features.

The per-asset Alpha158-style features capture each name's *own* time series, but they carry
no information about how that name compares to the rest of the universe -- which is exactly
what a cross-asset attention layer + transfer-entropy prior is supposed to exploit. We add:

  cross-sectional ranks   :  per-day rank of return / volatility / momentum / dollar-volume
                             / trend across the 46-name universe (scaled to [-1, 1])
  macro regime            :  VIX level and 21-day change, 10Y Treasury yield level and change
                             (broadcast across assets, z-scored on train statistics only)

Both are point-in-time clean: rank features use only same-day cross-sectional data; macro is
fetched aligned to the equity calendar and z-scored on the earliest 50% of the sample.
"""
import numpy as np
import pandas as pd
import yfinance as yf

EPS = 1e-8


def _rank_pct(df):
    """df [T, N] -> per-row rank in (-1, 1], NaNs -> 0."""
    return ((df.rank(axis=1, pct=True) * 2 - 1).fillna(0).values).astype(np.float32)


def _macro(c_index, start, end, split):
    raw = yf.download(["^VIX", "^TNX"], start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.reindex(c_index).ffill().bfill()
    out = np.stack([
        raw["^VIX"].values,
        (raw["^VIX"] - raw["^VIX"].shift(21)).bfill().values,
        raw["^TNX"].values,
        (raw["^TNX"] - raw["^TNX"].shift(21)).bfill().values,
    ], axis=1).astype(np.float32)              # [T, 4]
    mu = out[:split].mean(0); sd = out[:split].std(0) + EPS
    return np.clip((out - mu) / sd, -3, 3)


def build_extra(c, v, c_index, start, end, split):
    """Returns extra features [T, N, F_extra] and their names."""
    T, N = c.shape
    rets1 = c.pct_change(1)
    feats_cs = {
        "csrank_ret1":   rets1,
        "csrank_ret5":   c.pct_change(5),
        "csrank_ret21":  c.pct_change(21),
        "csrank_ret63":  c.pct_change(63),
        "csrank_ret252": c.pct_change(252),
        "csrank_vol20":  rets1.rolling(20).std(),
        "csrank_vol60":  rets1.rolling(60).std(),
        "csrank_dvol20": (c * v).rolling(20).mean(),
        "csrank_trend200": c / c.rolling(200).mean() - 1,
    }
    cs_names = list(feats_cs.keys())
    cs = np.stack([_rank_pct(df) for df in feats_cs.values()], axis=2)   # [T, N, F_cs]
    macro = _macro(c_index, start, end, split)                            # [T, 4]
    macro_b = np.broadcast_to(macro[:, None, :], (T, N, macro.shape[1])).copy()
    extra = np.concatenate([cs, macro_b], axis=2).astype(np.float32)
    return extra, cs_names + ["macro_vix", "macro_vix_d21", "macro_tnx", "macro_tnx_d21"]
