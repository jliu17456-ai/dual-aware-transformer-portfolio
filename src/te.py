"""Transfer-entropy / mutual-information prior over the asset graph.

Estimated from a returns window (training data of a fold only -> no look-ahead).
Returns a standardized [N, N] matrix B used as an additive bias on the cross-asset
attention logits. Three methods, in decreasing fidelity / increasing robustness:

  te   : transfer entropy  TE(i->j) = I(j_{t+1}; i_t | j_t)   -- directed information flow
  mi   : mutual information I(i_t; j_t)                       -- undirected, easier to estimate
  corr : |Pearson correlation|                                -- linear fallback, most stable
"""
import numpy as np

EPS = 1e-12


def _discretize(x, bins):
    qs = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.clip(np.digitize(x, qs), 0, bins - 1).astype(int)


def _te_pair(jf, jp, ip, bins):
    M = len(jf)
    idx = (jf * bins + jp) * bins + ip
    p = np.bincount(idx, minlength=bins ** 3).reshape(bins, bins, bins).astype(float) / M
    p_jp = p.sum(axis=(0, 2))      # P(j_t)
    p_jf_jp = p.sum(axis=2)        # P(j_{t+1}, j_t)
    p_jp_ip = p.sum(axis=0)        # P(j_t, i_t)
    te = 0.0
    for a in range(bins):
        for b in range(bins):
            for cc in range(bins):
                pv = p[a, b, cc]
                if pv > 0:
                    den = p_jf_jp[a, b] * p_jp_ip[b, cc]
                    if den > 0:
                        te += pv * np.log(pv * p_jp[b] / den)
    return max(te, 0.0)


def _mi_pair(xi, xj, bins):
    M = len(xi)
    p = np.bincount(xi * bins + xj, minlength=bins ** 2).reshape(bins, bins).astype(float) / M
    pi = p.sum(1); pj = p.sum(0)
    mi = 0.0
    for a in range(bins):
        for b in range(bins):
            if p[a, b] > 0:
                mi += p[a, b] * np.log(p[a, b] / (pi[a] * pj[b] + EPS))
    return max(mi, 0.0)


def _standardize(B):
    off = ~np.eye(B.shape[0], dtype=bool)
    vals = B[off]
    B = (B - vals.mean()) / (vals.std() + EPS)
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)


def compute_bias(rets_window, method="te", bins=3):
    """rets_window: [W, N] returns. Returns standardized [N, N] bias matrix."""
    W, N = rets_window.shape
    if method == "corr":
        B = np.abs(np.corrcoef(rets_window.T))
        return _standardize(np.nan_to_num(B))
    codes = np.stack([_discretize(rets_window[:, j], bins) for j in range(N)], axis=1)  # [W,N]
    B = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if method == "te":
                B[i, j] = _te_pair(codes[1:, j], codes[:-1, j], codes[:-1, i], bins)
            elif method == "mi":
                B[i, j] = _mi_pair(codes[:, i], codes[:, j], bins)
            else:
                raise ValueError(method)
    return _standardize(B)
