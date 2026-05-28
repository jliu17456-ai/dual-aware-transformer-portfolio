"""Aggregate walk-forward cells -> ablation table, baselines, equity curve, TE heatmap."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load, fold_ranges, metrics, COST, ROOT
from te import compute_bias

CELLS = ROOT / "results" / "cells"
RES = ROOT / "results"
HEADLINE_BETA = 1e-3
LOOKBACK = 40
N_FOLDS = 5


def baseline(rets, train_t, test_t, kind):
    nxt = rets[test_t + 1]                       # [n_test, N]
    N = nxt.shape[1]
    if kind == "ew":                             # daily-rebalanced equal weight (long-only)
        return nxt.mean(1)
    if kind == "bah":
        eq = np.cumprod(1 + nxt, axis=0)         # buy & hold equal weight
        val = eq.mean(1)
        return np.concatenate([[val[0] - 1], val[1:] / val[:-1] - 1])
    rtr = rets[train_t[0]:train_t[-1] + 1]
    if kind == "iv":                             # inverse-vol (risk-parity proxy)
        sig = rtr.std(0) + 1e-8
        w = (1 / sig); w /= w.sum()
    elif kind == "mv":                           # long-only min-variance (diag-regularized)
        cov = np.cov(rtr.T) + 1e-4 * np.eye(N)
        w = np.linalg.solve(cov, np.ones(N))
        w = np.clip(w, 0, None); w /= w.sum() + 1e-12
    else:
        raise ValueError(kind)
    return (nxt * w).sum(1)


def main():
    feats, rets, warmup = load()
    T = feats.shape[0]
    folds = fold_ranges(T, LOOKBACK, warmup, N_FOLDS)

    cells = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    variants = ["base", "vib", "prior", "full"]
    table = {}
    for v in variants:
        sh = [c["metrics"]["Sharpe"] for c in cells
              if c["variant"] == v and (v in ("base", "prior") or c["beta"] == HEADLINE_BETA)]
        if sh:
            table[v] = (float(np.mean(sh)), float(np.std(sh)), len(sh))

    print("\n=== Ablation: OOS Sharpe across folds x seeds (mean +/- std) ===")
    for v in variants:
        if v in table:
            m, s, n = table[v]
            print(f"  {v:6s}  {m:+.3f} +/- {s:.3f}   (n={n})")

    # equity curve: full model (headline beta, seed 0) concatenated over folds vs baselines
    def concat_full():
        rr = []
        for f in range(N_FOLDS):
            cand = [c for c in cells if c["variant"] == "full" and c["fold"] == f
                    and c["beta"] == HEADLINE_BETA and c["seed"] == 0]
            if cand:
                rr += cand[0]["port_rets"]
        return np.array(rr)

    full_r = concat_full()
    ew_r = np.concatenate([baseline(rets, folds[f][0], folds[f][1], "ew") for f in range(N_FOLDS)])
    bah_r = np.concatenate([baseline(rets, folds[f][0], folds[f][1], "bah") for f in range(N_FOLDS)])
    iv_r = np.concatenate([baseline(rets, folds[f][0], folds[f][1], "iv") for f in range(N_FOLDS)])
    mv_r = np.concatenate([baseline(rets, folds[f][0], folds[f][1], "mv") for f in range(N_FOLDS)])

    print("\n=== OOS performance (concatenated test folds) ===")
    for name, r in [("Dual-Aware (Full)", full_r), ("Equal-Weight", ew_r),
                    ("Buy & Hold", bah_r), ("Inverse-Vol", iv_r), ("Min-Variance", mv_r)]:
        if len(r):
            m = metrics(r)
            print(f"  {name:20s} Sharpe {m['Sharpe']:+.3f}  CR {m['CR']*100:6.2f}%  "
                  f"Sortino {m['Sortino']:+.3f}  MDD {m['MDD']*100:6.2f}%")

    RES.mkdir(parents=True, exist_ok=True)
    json.dump({"ablation": table}, open(RES / "comparison.json", "w"), indent=2)

    if len(full_r):
        plt.figure(figsize=(9, 4.5))
        for name, r, st in [("Dual-Aware (Full)", full_r, "-"),
                            ("Min-Variance", mv_r, "-."),
                            ("Inverse-Vol", iv_r, "--"),
                            ("Equal-Weight", ew_r, ":")]:
            plt.plot(np.cumprod(1 + r), st, label=name, lw=1.8)
        plt.legend(); plt.title("Out-of-sample equity (walk-forward, net of 5bps cost)")
        plt.ylabel("growth of $1"); plt.xlabel("test days"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(RES / "equity_curves.png", dpi=130); plt.close()

    # TE heatmap on fold-0 training window
    d = np.load(ROOT / "data" / "market.npz", allow_pickle=True)
    tickers = list(d["tickers"]); tr = folds[0][0]
    B = compute_bias(rets[tr[0]:tr[-1] + 1], method="te")
    plt.figure(figsize=(5.2, 4.4))
    plt.imshow(B, cmap="viridis")
    plt.xticks(range(len(tickers)), tickers, rotation=90, fontsize=7)
    plt.yticks(range(len(tickers)), tickers, fontsize=7)
    plt.colorbar(label="standardized TE  (row i -> col j)")
    plt.title("Transfer-entropy prior (fold-0 train)")
    plt.tight_layout(); plt.savefig(RES / "te_heatmap.png", dpi=130); plt.close()
    print("\nsaved plots + comparison.json to", RES)


if __name__ == "__main__":
    main()
