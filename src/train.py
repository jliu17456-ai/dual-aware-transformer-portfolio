"""End-to-end differentiable Sharpe optimization with walk-forward evaluation.

The network outputs long/short weights; we roll them over the holding sequence, compute the
realized portfolio return net of turnover cost, and back-propagate the negative annualized
Sharpe directly (plus the VIB KL term). No labels, no RL.

Model selection is done by early stopping on a validation slice carved from the END of each
fold's training window -- the test block is never seen during training or selection.

One CLI run = one (variant, fold, seed, beta) cell. A launcher fans these out across the GPU.
"""
import argparse, json, sys, copy, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DualAwareNet
from te import compute_bias

ROOT = Path(__file__).resolve().parents[1]
ANN = 252.0
COST = 0.0005          # 5 bps per unit turnover
VARIANTS = {           # (use_vib, use_prior)
    "base":  (False, False),
    "vib":   (True,  False),
    "prior": (False, True),
    "full":  (True,  True),
}


def load():
    d = np.load(ROOT / "data" / "market.npz", allow_pickle=True)
    return d["feats"].astype(np.float32), d["rets"].astype(np.float32), int(d["warmup"])


def fold_ranges(T, lookback, warmup, n_folds, test_frac=0.5):
    """Expanding-window walk-forward. Returns list of (train_times, test_times)."""
    t0 = warmup + lookback
    all_t = np.arange(t0, T - 1)                  # need rets[t+1]
    n = len(all_t)
    test_region = all_t[int((1 - test_frac) * n):]
    blocks = np.array_split(test_region, n_folds)
    folds = []
    for b in blocks:
        train = all_t[all_t < b[0]]
        folds.append((train, b))
    return folds


def make_windows(feats_t, times, lookback):
    """feats_t [T,N,F] on device; times 1-D -> [B, N, lookback, F] on device."""
    idx = torch.as_tensor(times, device=feats_t.device, dtype=torch.long)
    off = torch.arange(-lookback + 1, 1, device=feats_t.device)
    gather = idx[:, None] + off[None, :]          # [B, L]
    w = feats_t[gather]                           # [B, L, N, F]
    return w.permute(0, 2, 1, 3).contiguous()


def metrics(rets):
    rets = np.asarray(rets)
    mu, sd = rets.mean(), rets.std() + 1e-12
    neg = rets[rets < 0]
    dn = (neg.std() if len(neg) else 0.0) + 1e-12
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    return {"CR": float(eq[-1] - 1), "Sharpe": float(mu / sd * np.sqrt(ANN)),
            "Sortino": float(mu / dn * np.sqrt(ANN)), "MDD": float(((eq - peak) / peak).min())}


def roll(model, feats_t, rets_t, times, lookback, te_bias):
    """Net portfolio returns over an ordered set of decision times (eval mode)."""
    model.eval()
    with torch.no_grad():
        W, _ = model(make_windows(feats_t, times, lookback), te_bias)   # [B,N]
        nxt = rets_t[torch.as_tensor(times + 1, device=feats_t.device, dtype=torch.long)]
        gross = (W * nxt).sum(1)
        turn = (W[1:] - W[:-1]).abs().sum(1)
        port = gross.clone(); port[1:] = port[1:] - COST * turn
    return port.cpu().numpy()


def train_cell(args):
    feats, rets, warmup = load()
    T, N, Ff = feats.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats_t = torch.from_numpy(feats).to(device)
    rets_t = torch.from_numpy(rets).to(device)
    folds = fold_ranges(T, args.lookback, warmup, args.n_folds)
    train_t, test_t = folds[args.fold]

    # validation = last val_frac of the (ordered) training window
    cut = int((1 - args.val_frac) * len(train_t))
    core_t, val_t = train_t[:cut], train_t[cut:]

    use_vib, use_prior = VARIANTS[args.variant]
    te_bias = None
    if use_prior:
        B = compute_bias(rets[core_t[0]:core_t[-1] + 1], method=args.method)
        te_bias = torch.from_numpy(B).to(device)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = DualAwareNet(Ff, N, args.lookback, d_model=args.d_model,
                         use_vib=use_vib, use_prior=use_prior).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    S = args.seq_len
    lo, hi = core_t[0], core_t[-1] - S + 1
    best_val, best_state, since = -1e9, None, 0
    for step in range(1, args.steps + 1):
        model.train()
        starts = np.random.randint(lo, hi + 1, size=args.batch)
        times = np.concatenate([np.arange(s, s + S) for s in starts])
        W, kl = model(make_windows(feats_t, times, args.lookback), te_bias)
        W = W.view(args.batch, S, N)
        nxt = rets_t[torch.as_tensor(times + 1, device=device, dtype=torch.long)].view(args.batch, S, N)
        gross = (W * nxt).sum(-1)
        turn = (W[:, 1:] - W[:, :-1]).abs().sum(-1)
        port = gross.clone(); port[:, 1:] = port[:, 1:] - COST * turn
        sharpe = (port.mean(1) / (port.std(1) + 1e-6) * math.sqrt(ANN)).mean()
        loss = -sharpe + args.beta * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0:
            v = metrics(roll(model, feats_t, rets_t, val_t, args.lookback, te_bias))["Sharpe"]
            if v > best_val:
                best_val, best_state, since = v, copy.deepcopy(model.state_dict()), 0
            else:
                since += 1
                if since >= args.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    port = roll(model, feats_t, rets_t, test_t, args.lookback, te_bias)
    m = metrics(port)
    out = {"variant": args.variant, "fold": args.fold, "seed": args.seed,
           "beta": args.beta, "method": args.method, "val_sharpe": best_val,
           "metrics": m, "n_test": int(len(test_t)), "port_rets": port.tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"{args.variant} f{args.fold} s{args.seed} b{args.beta}: "
          f"val {best_val:+.2f} | test Sharpe {m['Sharpe']:+.3f}  CR {m['CR']*100:.2f}%  MDD {m['MDD']*100:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="full")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--method", choices=["te", "mi", "corr"], default="te")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--seq_len", type=int, default=63)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=48)
    p.add_argument("--out", type=Path, required=True)
    train_cell(p.parse_args())


if __name__ == "__main__":
    main()
