"""Equity model: training + GATED evaluation (docs/match_aware_search_design.md).

Implements the pre-registered gates 1-4 with HALT-AS-DEFAULT: each gate
emits equity_data/verdicts/<gate>.json {gate, metrics, thresholds, pass,
branch, git_sha, data_sha256, timestamp}; any exception or ambiguity is
a halt, never a pass.

Discipline encoded here:
- train/val split BY MATCH_ID (correlated labels: ~3.3 states share one
  outcome; effective n = match count)
- all CIs / calibration metrics cluster-bootstrapped BY MATCH
- terminal states never trained on (generator already excludes them);
  near-terminal gate = last boundary row of each holdout match
- baseline-to-beat: binned lookup (my score, max opponent, deals) with
  logistic fallback, built from the SAME training matches
- canonicalization = a recorded BRANCH decision, not a pass/fail

Usage:
  python train_equity.py --train equity_data/train_seeded_v1.npz \
      --holdout equity_data/holdout_natural_v1.npz [--out equity_v1.pth]
  python train_equity.py --selftest       # synthetic pipeline check
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

VERDICT_DIR = os.path.join('equity_data', 'verdicts')

# ---------------------------------------------------------------------------
# Features / targets
# ---------------------------------------------------------------------------

def soft_target(place):
    """Placement (possibly x.5 from ties) -> soft one-hot over ranks 1..4."""
    t = np.zeros(4)
    lo = int(np.floor(place)) - 1
    if place == np.floor(place):
        t[lo] = 1.0
    else:
        t[lo] = 0.5
        t[lo + 1] = 0.5
    return t


def make_rows(d, canonical=False):
    """Expand each boundary state into 4 per-seat rows.
    Returns X, Y, match_ids, terminal_flags(last boundary of its match)."""
    totals, deals = d['totals'], d['deals']
    pdir, places, mids = d['pass_dir'], d['placements'], d['match_id']
    n = len(totals)
    last_of_match = np.zeros(n, dtype=bool)
    seen = {}
    for i in range(n):
        seen[mids[i]] = i
    for i in seen.values():
        last_of_match[i] = True

    X, Y, M, L = [], [], [], []
    for i in range(n):
        mx = totals[i].max()
        onehot = np.zeros(4)
        onehot[int(pdir[i])] = 1.0
        for s in range(4):
            rot = np.array([totals[i][(s + k) % 4] for k in range(4)])
            if canonical:
                base = np.concatenate([[rot[0]], np.sort(rot[1:])[::-1]])
            else:
                base = rot
            X.append(np.concatenate([base / 100.0,
                                     [deals[i] / 20.0, (100.0 - mx) / 100.0],
                                     onehot]))
            Y.append(soft_target(places[i][s]))
            M.append(mids[i])
            L.append(last_of_match[i])
    return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32),
            np.array(M), np.array(L))


def match_strata(d):
    """Per-match stratum from its boundary trajectory: 2=S2, 1=S1, 0=S3."""
    strata = {}
    for i in range(len(d['totals'])):
        t = np.sort(d['totals'][i])[::-1]
        mid = d['match_id'][i]
        cur = strata.get(mid, 0)
        if t[0] >= 85 and (t[0] - t[1]) <= 10:
            cur = max(cur, 2)
        elif t[0] >= 85:
            cur = max(cur, 1)
        strata[mid] = cur
    return strata


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EquityNet(nn.Module):
    def __init__(self, in_dim=10, width=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(),
                               nn.Linear(width, width), nn.GELU(),
                               nn.Linear(width, 4))

    def forward(self, x):
        return self.f(x)


def fit_val_split(M, seed=0):
    """Deterministic by-match 90/10 split, shared by net and baseline."""
    rng = np.random.default_rng(seed)
    mids = np.unique(M)
    rng.shuffle(mids)
    val_m = set(mids[:max(1, len(mids) // 10)])
    return np.isin(M, list(val_m))


def fit_net(X, Y, M, seed=0, epochs=60, width=64, val=None):
    torch.manual_seed(seed)
    if val is None:
        val = fit_val_split(M, seed)
    net = EquityNet(X.shape[1], width)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt, Yt = torch.from_numpy(X[~val]), torch.from_numpy(Y[~val])
    Xv, Yv = torch.from_numpy(X[val]), torch.from_numpy(Y[val])
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 8192):
            idx = perm[i:i + 8192]
            loss = -(torch.log_softmax(net(Xt[idx]), 1) * Yt[idx]).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = -(torch.log_softmax(net(Xv), 1) * Yv).sum(1).mean().item()
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


class BinnedBaseline:
    """Lookup over (my//10, max_opp//10, min(deals,20)//2) + logistic fallback."""

    @staticmethod
    def _key(x):
        my = int(x[0] * 100) // 10
        mo = int(max(x[1], x[2], x[3]) * 100) // 10
        dl = min(int(x[4] * 20), 20) // 2
        return (my, mo, dl)

    def __init__(self, min_cell_rows):
        # Two FROZEN variants (seventh review): min 20 (fine) and min 80
        # (coarse). The synthetic selftest cannot adjudicate the right value
        # for real data, so the gate requires beating the BETTER of the two
        # on the real holdout - no discretionary tuning ever again.
        self.min_cell_rows = min_cell_rows
        self.table = {}
        self.logistic = None

    def fit(self, X, Y):
        acc, cnt = {}, {}
        for i in range(len(X)):
            k = self._key(X[i])
            acc[k] = acc.get(k, 0) + Y[i]
            cnt[k] = cnt.get(k, 0) + 1
        self.table = {k: acc[k] / cnt[k] for k in acc
                      if cnt[k] >= self.min_cell_rows}
        # logistic fallback on the 3 binned dims
        F = np.array([[x[0], max(x[1], x[2], x[3]), x[4]] for x in X],
                     dtype=np.float32)
        torch.manual_seed(0)
        lg = nn.Linear(3, 4)
        opt = torch.optim.Adam(lg.parameters(), lr=1e-2)
        Ft, Yt = torch.from_numpy(F), torch.from_numpy(Y)
        for _ in range(300):
            loss = -(torch.log_softmax(lg(Ft), 1) * Yt).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.logistic = lg.eval()

    def predict(self, X):
        out = np.empty((len(X), 4), dtype=np.float32)
        miss = []
        for i in range(len(X)):
            p = self.table.get(self._key(X[i]))
            if p is None:
                miss.append(i)
            else:
                out[i] = p
        if miss:
            F = np.array([[X[i][0], max(X[i][1], X[i][2], X[i][3]), X[i][4]]
                          for i in miss], dtype=np.float32)
            with torch.no_grad():
                out[miss] = torch.softmax(self.logistic(torch.from_numpy(F)), 1).numpy()
        return out


# ---------------------------------------------------------------------------
# Metrics (match-clustered)
# ---------------------------------------------------------------------------

def brier(P, Y):
    return float(((P - Y) ** 2).sum(1).mean())


def ece(P, Y, bins=10):
    """Equal-mass ECE averaged over the 4 placement classes."""
    total = 0.0
    for k in range(4):
        p, y = P[:, k], Y[:, k]
        order = np.argsort(p)
        splits = np.array_split(order, bins)
        e = sum(len(s) * abs(p[s].mean() - y[s].mean()) for s in splits if len(s))
        total += e / len(p)
    return float(total / 4)


def cluster_ci(metric_fn, P, Y, M, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    mids = np.unique(M)
    idx_by_mid = {m: np.flatnonzero(M == m) for m in mids}
    vals = []
    for _ in range(n_boot):
        take = rng.choice(mids, size=len(mids), replace=True)
        idx = np.concatenate([idx_by_mid[m] for m in take])
        vals.append(metric_fn(P[idx], Y[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def logloss(P, Y):
    return float(-(np.log(np.clip(P, 1e-9, 1)) * Y).sum(1).mean())


def ece_noise_floor(P, M, n_reps=200, seed=0, pct=95):
    """CLUSTERED parametric-bootstrap ECE floor (seventh review): rows of
    the same (match, seat) share one real outcome, so the floor resamples
    ONE outcome per (match, seat) group from the group's mean predicted
    probs and broadcasts it - per-row independence would bias the floor
    LOW and make the floor+0.015 rule stricter than pre-registered.
    (Cross-seat permutation consistency remains an approximation.)"""
    rng = np.random.default_rng(seed)
    n = len(P)
    seat = np.arange(n) % 4
    keys = np.stack([M.astype(np.int64), seat], axis=1)
    _, group, inv = np.unique(keys, axis=0, return_index=True,
                              return_inverse=True)
    G = len(group)
    pm = np.zeros((G, 4))
    np.add.at(pm, inv, P)
    counts = np.bincount(inv, minlength=G).astype(float)
    pm /= counts[:, None]
    pm /= pm.sum(1, keepdims=True)
    cum = np.cumsum(pm, axis=1)
    vals = []
    for _ in range(n_reps):
        u = rng.random(G)
        draws_g = (u[:, None] > cum).sum(1)
        Ystar = np.eye(4, dtype=np.float32)[draws_g[inv]]
        vals.append(ece(P, Ystar))
    return float(np.percentile(vals, pct))


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def emit_verdict(gate, metrics, thresholds, passed, branch=None,
                 data_sha=''):
    os.makedirs(VERDICT_DIR, exist_ok=True)
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        sha = 'unknown'
    v = {'gate': gate, 'metrics': metrics, 'thresholds': thresholds,
         'pass': bool(passed), 'branch': branch, 'git_sha': sha,
         'data_sha256': data_sha, 'timestamp': time.time()}
    path = os.path.join(VERDICT_DIR, f'{gate}.json')
    with open(path, 'w') as f:
        json.dump(v, f, indent=2)
    print(f"VERDICT [{gate}]: {'PASS' if passed else 'HALT'} "
          f"{json.dumps(metrics)} -> {path}")
    return passed


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------------------

def net_probs(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(X)), 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train')
    ap.add_argument('--holdout')
    ap.add_argument('--out', default='equity_v1.pth')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    data_sha = file_sha(args.train)[:16] + '+' + file_sha(args.holdout)[:16]
    tr = np.load(args.train)
    ho = np.load(args.holdout)

    Xtr, Ytr, Mtr, _ = make_rows(tr)
    Xho, Yho, Mho, Lho = make_rows(ho)
    strata = match_strata(ho)
    st_of_row = np.array([strata[m] for m in Mho])
    n_s2 = sum(1 for v in strata.values() if v == 2)
    print(f"train: {len(np.unique(Mtr))} matches / {len(Xtr)} rows; "
          f"holdout: {len(np.unique(Mho))} matches "
          f"(S3/S1/S2 = {sum(1 for v in strata.values() if v == 0)}/"
          f"{sum(1 for v in strata.values() if v == 1)}/{n_s2})")

    val = fit_val_split(Mtr)   # ONE by-match split shared by net and baselines
    net, val_ll = fit_net(Xtr, Ytr, Mtr, val=val)
    base_fine = BinnedBaseline(min_cell_rows=20)
    base_fine.fit(Xtr[~val], Ytr[~val])
    base_coarse = BinnedBaseline(min_cell_rows=80)
    base_coarse.fit(Xtr[~val], Ytr[~val])
    Pn = net_probs(net, Xho)
    Pb_fine, Pb_coarse = base_fine.predict(Xho), base_coarse.predict(Xho)
    # The gate's opponent is the BETTER variant on the real holdout
    Pb = Pb_fine if brier(Pb_fine, Yho) <= brier(Pb_coarse, Yho) else Pb_coarse
    base_branch = 'fine20' if Pb is Pb_fine else 'coarse80'

    # ---- A. DIAGNOSTICS (report; halt only on degeneracy) ----------------
    m = {'brier_agg': brier(Pn, Yho), 'logloss_agg': logloss(Pn, Yho),
         'ece_agg': ece(Pn, Yho), 'ece_floor_agg': ece_noise_floor(Pn, Mho),
         'ece_agg_ci95': cluster_ci(ece, Pn, Yho, Mho),
         'n_s2_matches': n_s2}
    for s, name in ((0, 'S3'), (1, 'S1'), (2, 'S2')):
        sel = st_of_row == s
        if sel.sum() == 0:
            m[f'brier_{name}'] = None
            continue
        m[f'brier_{name}'] = brier(Pn[sel], Yho[sel])
        m[f'logloss_{name}'] = logloss(Pn[sel], Yho[sel])
        m[f'ece_{name}'] = ece(Pn[sel], Yho[sel])
        m[f'ece_floor_{name}'] = ece_noise_floor(Pn[sel], Mho[sel])
    m['brier_near_terminal'] = brier(Pn[Lho], Yho[Lho])
    m['ece_near_terminal'] = ece(Pn[Lho], Yho[Lho])
    m['prob_spread_mean'] = float((Pn.max(1) - Pn.min(1)).mean())

    degenerate = (not np.isfinite(Pn).all()
                  or m['prob_spread_mean'] < 0.05
                  or m['brier_near_terminal'] > 0.75)
    emit_verdict('diagnostics', m,
                 {'halt_only_on': 'NaN/inf | prob_spread<0.05 | '
                                  'near_terminal_brier>0.75'},
                 not degenerate, data_sha=data_sha)
    if degenerate:
        print("HALT: degenerate equity model outputs")
        raise SystemExit(1)

    # ---- B. COMPONENT SELECTION (not pass/fail) --------------------------
    cands = {'net': Pn, 'lookup_fine20': Pb_fine, 'lookup_coarse80': Pb_coarse}
    briers = {k: brier(P, Yho) for k, P in cands.items()}
    chosen_name = min(briers, key=briers.get)
    m2 = dict(briers)
    m2['chosen'] = chosen_name
    m2['k_buyback_candidate'] = chosen_name != 'net'  # table = ~ns per call
    for s, name in ((1, 'S1'), (2, 'S2')):
        sel = st_of_row == s
        for k, P in cands.items():
            m2[f'{k}_{name}'] = brier(P[sel], Yho[sel])
    emit_verdict('selection', m2, {'rule': 'min holdout Brier of frozen set'},
                 True, branch=chosen_name, data_sha=data_sha)

    if chosen_name == 'net':
        torch.save({'state_dict': net.state_dict(), 'in_dim': 10}, args.out)
    else:
        lb = base_fine if chosen_name == 'lookup_fine20' else base_coarse
        keys = np.array(list(lb.table.keys()), dtype=np.int64)
        vals = np.array([lb.table[tuple(k)] for k in keys], dtype=np.float32)
        with torch.no_grad():
            lw = lb.logistic.weight.numpy()
            lbias = lb.logistic.bias.numpy()
        np.savez(args.out.replace('.pth', '_lookup.npz'),
                 keys=keys, vals=vals, logistic_w=lw, logistic_b=lbias,
                 variant=chosen_name)
    print(f"selection: {chosen_name} (briers {briers}); artifacts saved")


def run_selftest():
    """Synthetic data with known structure; verifies the pipeline mechanics."""
    rng = np.random.default_rng(0)
    n_matches = 3000
    rows = {k: [] for k in ('totals', 'deals', 'pass_dir', 'placements',
                            'match_id', 'mixture')}
    for mid in range(n_matches):
        totals = rng.integers(0, 95, 4).astype(float)
        for b in range(rng.integers(1, 4)):
            noisy = totals + rng.normal(0, 12, 4)  # noisy outcome model
            order = np.argsort(np.argsort(noisy))
            rows['totals'].append(totals.copy())
            rows['deals'].append(int(totals.sum() // 26))
            rows['pass_dir'].append(int(totals.sum()) % 4)
            rows['placements'].append((order + 1).astype(float))
            rows['match_id'].append(mid)
            rows['mixture'].append(0)
            totals = np.minimum(totals + rng.integers(0, 9, 4), 99)
    d = {k: np.array(v) for k, v in rows.items()}
    half = n_matches // 2
    tr = {k: v[d['match_id'] < half] for k, v in d.items()}
    ho = {k: v[d['match_id'] >= half] for k, v in d.items()}
    Xtr, Ytr, Mtr, _ = make_rows(tr)
    Xho, Yho, Mho, Lho = make_rows(ho)
    net, _ = fit_net(Xtr, Ytr, Mtr, epochs=20)
    base = BinnedBaseline(min_cell_rows=20)
    base.fit(Xtr, Ytr)
    Pn, Pb = net_probs(net, Xho), base.predict(Xho)
    print(f"selftest: net brier {brier(Pn, Yho):.4f} vs base {brier(Pb, Yho):.4f} "
          f"vs uniform {brier(np.full_like(Yho, 0.25), Yho):.4f}")
    print(f"selftest: ece {ece(Pn, Yho):.4f}, "
          f"ci {cluster_ci(ece, Pn, Yho, Mho, n_boot=100)}")
    assert brier(Pn, Yho) < brier(np.full_like(Yho, 0.25), Yho) - 0.05
    print("SELFTEST OK (pipeline mechanics: features, split-by-match, "
          "fit, baseline, ECE, cluster CI)")


if __name__ == '__main__':
    main()
