"""Fetch a ~45-name large-cap universe (daily, 2010-2022) and build per-asset features.

Individual stocks (not sector ETFs) are used on purpose: their real cross-sectional
dispersion is what a cross-asset-attention + transfer-entropy model can actually exploit,
whereas correlated sector ETFs leave equal-weight near-optimal. All names were listed well
before 2010 so they have full history; any with gaps are dropped by the coverage filter.
This is a fixed, current-membership list, so it carries a survivorship bias (noted in README).
Features are z-scored on the earliest 50% of the sample only, so no walk-forward fold ever
sees future statistics.
"""
import sys
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features
from cs_macro import build_extra
from alpha101 import build_alpha101_normalized

SECTORS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN",
    "JPM", "BAC", "WFC", "C", "GS", "AXP",
    "JNJ", "PFE", "MRK", "ABT", "AMGN", "GILD", "LLY", "BMY", "UNH",
    "XOM", "CVX", "COP", "SLB",
    "PG", "KO", "PEP", "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS",
    "CAT", "BA", "HON", "MMM", "UPS", "UNP",
]
START, END = "2010-01-01", "2022-12-31"
OUT = Path(__file__).resolve().parents[1] / "data" / "market.npz"
WARMUP = 65


def main():
    raw = yf.download(SECTORS, start=START, end=END, auto_adjust=True, progress=False)
    o, h, l, c, v = (raw[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    good = [t for t in SECTORS if t in c.columns and c[t].notna().mean() > 0.99]
    o, h, l, c, v = [x[good].ffill().bfill() for x in (o, h, l, c, v)]
    idx = c.index; N = len(good); T = len(idx)
    norm_split = int(0.5 * T)                      # feature z-score uses earliest 50% only
    feats, names = build_features(o, h, l, c, v, good, norm_split)
    extra, extra_names = build_extra(c[good], v[good], idx, START, END, norm_split)
    a101, a101_names = build_alpha101_normalized(o[good], h[good], l[good], c[good], v[good], norm_split)
    feats = np.concatenate([feats, extra, a101], axis=2).astype(np.float32)
    names = names + extra_names + a101_names
    rets = c[good].pct_change().fillna(0.0).values.astype(np.float32)
    print(f"DAILY: tickers {N}  bars={T}  F={len(names)}  norm_split={norm_split}")
    print(f"range {idx.min().date()} .. {idx.max().date()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, feats=feats, rets=rets,
                        dates=np.array([str(x.date()) for x in idx]),
                        tickers=np.array(good), norm_split=norm_split,
                        warmup=WARMUP, factor_names=np.array(names))
    print("saved", OUT, "feats", feats.shape)


if __name__ == "__main__":
    main()
