"""League round 5, T0 information audit (docs/exploiter_league_r5_prereg.md §3.1).

Question: does the frozen champion (8a89da90) ALREADY decode the
seat-attributed threat information that obs v2 makes explicit? If a
LINEAR probe on its hidden states recovers it, the r5 adapters add
little; if not, the information is genuinely absent from the net.

Data: the v6 bank's frozen full holdout (80,220 records / 123 matches;
obs v2 rows carry the ground truth). Champion features: HeartsNetV5
tokens on the 556 prefix - the global token (320-d) and the 52 card
tokens (320-d each). Baseline features: the raw 556 inputs (what a
linear reader of the observation itself can get).

Targets (all RELATIVE frame, the frame the net acts in):
  A. per-seat deal points taken so far (4 values /26)     - ridge R^2
  B. moon-alive per seat (4 binaries; ext dims 316-319)     - logistic AUC
  C. Q-spade status (5-way one-hot; ext dims 321-325)      - softmax acc
  D. per-card taken-by (5-way: unseen + 4 seats)            - softmax acc,
     from the CARD token vs from the card's own raw channels
Split BY MATCH (90 train / 33 holdout matches). Closed-form ridge for A;
full-batch L-BFGS logistic/softmax for B-D (torch, no sklearn).

Registered halt band (prereg §3.1): moon-alive AUC >= 0.97 AND per-seat
points R^2 >= 0.95 from the champion's activations -> HALT-DEFAULT.
Also reports the search-vs-raw DISAGREEMENT bins (informs only).

Usage: python audit_t0_info.py [--json out.json]
"""
import argparse
import json

import numpy as np
import torch

from hearts_net import net_from_checkpoint
from v6_probe_eval import walk_holdout

CHAMPION = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
PEN = np.zeros(52, dtype=np.float32); PEN[39:52] = 1.0; PEN[36] = 13.0   # hearts, QS


def r2(y, yhat):
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return float(1 - (((y - yhat) ** 2).sum(0) / np.maximum(ss, 1e-9)).mean())


def ridge_fit(X, Y, lam=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xb.T @ Y)
    return lambda Z: np.concatenate([Z, np.ones((len(Z), 1), np.float32)], 1) @ W


def auc(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_fit(X, y, device, classes=None, iters=200):
    """Full-batch L-BFGS softmax/logistic; returns predict-scores fn."""
    Xt = torch.from_numpy(X).to(device); yt = torch.from_numpy(y).to(device)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
    Xn = (Xt - mu) / sd
    k = classes or 1
    W = torch.zeros(X.shape[1], k, device=device, requires_grad=True)
    b = torch.zeros(k, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        z = Xn @ W + b
        if classes:
            loss = torch.nn.functional.cross_entropy(z, yt.long())
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z.squeeze(1), yt.float())
        loss = loss + 1e-4 * (W ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    def predict(Z):
        with torch.no_grad():
            Zt = (torch.from_numpy(Z).to(device) - mu) / sd
            out = Zt @ W + b
            return out.cpu().numpy()
    return predict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--json'); ap.add_argument('--max', type=int, default=0)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hold, keys = walk_holdout()
    if a.max: hold, keys = hold[:a.max], keys[:a.max]
    n = len(hold)
    obs = np.concatenate([np.ascontiguousarray(hold['obs']), np.ascontiguousarray(hold['ext'])], 1).astype(np.float32) / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(hold['mask'])).bool()
    ext = obs[:, 556:]
    # ---- targets (relative frame) ----
    taken = ext[:, 104:312].reshape(n, 4, 52)               # rel seat x card
    pts = (taken * PEN[None, None, :]).sum(2) / 26.0        # A: 4 values
    moon = ext[:, 316:320]                                  # B: 4 binaries
    qs = ext[:, 321:326].argmax(1)                          # C: 5-way
    taken_by = np.full((n, 52), 0, np.int64)                 # D: per card 0=unseen, 1..4 = rel seat
    for s in range(4): taken_by[taken[:, s, :] > 0.5] = s + 1
    play = ((hold['flags'] & 1) == 0)
    # ---- champion features ----
    champ = net_from_checkpoint(CHAMPION).to(device).eval()
    G, C = [], []
    with torch.no_grad():
        for s in range(0, n, 1024):
            o = torch.from_numpy(obs[s:s + 1024, :556]).to(device)
            x = champ._tokens(o)                          # (b, 53, d)
            G.append(x[:, 0, :].cpu().numpy()); C.append(x[:, 1:, :].cpu().numpy().astype(np.float32))
    G = np.concatenate(G); Cards = np.concatenate(C)          # (n,320), (n,52,320)
    # champion argmax for disagreement bins
    with torch.no_grad():
        am = []
        for s in range(0, n, 2048):
            lg, _ = champ(torch.from_numpy(obs[s:s + 2048, :556]).to(device), mask[s:s + 2048].to(device))
            am.append(lg.argmax(1).cpu().numpy())
        argmax = np.concatenate(am)
    # ---- by-match split ----
    matches = sorted(set(keys)); rng = np.random.default_rng(0); rng.shuffle(matches)
    test_m = set(matches[:max(1, len(matches) * 27 // 100)])   # ~27% of matches held out (33 of 123)
    te = np.array([k in test_m for k in keys]); tr = ~te
    raw = obs[:, :556]
    res = {'n': int(n), 'train_records': int(tr.sum()), 'test_records': int(te.sum())}
    # A. per-seat points
    for name, X in (('champion_global', G), ('raw_556', raw)):
        f = ridge_fit(X[tr], pts[tr]); res[f'A_points_r2_{name}'] = r2(pts[te], f(X[te]))
    # B. moon-alive per seat (pool the 4 seats' AUCs)
    for name, X in (('champion_global', G), ('raw_556', raw)):
        aucs = []
        for s in range(4):
            y = (moon[:, s] > 0.5).astype(np.int64)
            p = logistic_fit(X[tr], y[tr], device); aucs.append(auc(y[te], p(X[te]).squeeze(1)))
        res[f'B_moonalive_auc_{name}'] = [float(x) for x in aucs]; res[f'B_moonalive_auc_mean_{name}'] = float(np.nanmean(aucs))
    # B2 (informs): moon-alive decodability CONDITIONAL on points already taken
    # (the unconditional AUC is inflated by trivial early-deal states where every
    # seat is alive - raw inputs alone reach ~0.985 there).
    taken_any = pts.sum(1) > 0
    for name, X in (('champion_global', G), ('raw_556', raw)):
        aucs = []
        for s in range(4):
            y = (moon[:, s] > 0.5).astype(np.int64)
            m_tr = tr & taken_any; m_te = te & taken_any
            p = logistic_fit(X[m_tr], y[m_tr], device); aucs.append(auc(y[m_te], p(X[m_te]).squeeze(1)))
        res[f'B2_moonalive_auc_given_points_{name}'] = [float(x) for x in aucs]
        res[f'B2_moonalive_auc_given_points_mean_{name}'] = float(np.nanmean(aucs))
    res['B2_n_test_given_points'] = int((te & taken_any).sum())
    # C. QS status
    for name, X in (('champion_global', G), ('raw_556', raw)):
        p = logistic_fit(X[tr], qs, device, classes=5) if False else logistic_fit(X[tr], qs[tr], device, classes=5)
        res[f'C_qs_acc_{name}'] = float((p(X[te]).argmax(1) == qs[te]).mean())
    res['C_qs_majority_baseline'] = float(np.bincount(qs[te]).max() / te.sum())
    # D. per-card taken-by from the card token vs the card's raw channels (subsample cards for size)
    rng2 = np.random.default_rng(1); ci = rng2.integers(0, 52, size=n)
    Xtok = Cards[np.arange(n), ci]                                       # (n,320)
    blocks = [0, 52, 104, 186, 238, 290, 342, 394, 446, 498]
    Xraw = np.stack([obs[np.arange(n), b + ci] for b in blocks], 1)       # (n,10) the card's own v5 channels
    Xraw = np.concatenate([Xraw, obs[:, 156:186]], 1)                    # + ctx 30
    ytb = taken_by[np.arange(n), ci]
    for name, X in (('champion_cardtoken', Xtok), ('raw_cardchannels+ctx', Xraw)):
        p = logistic_fit(X[tr], ytb[tr], device, classes=5)
        res[f'D_takenby_acc_{name}'] = float((p(X[te]).argmax(1) == ytb[te]).mean())
    res['D_takenby_majority_baseline'] = float(np.bincount(ytb[te]).max() / te.sum())
    # ---- disagreement bins (informs) ----
    dis = (argmax != hold['action'].astype(np.int64))
    tricks_played = ext[:, 312:316].sum(1) * 13.0
    bins = {'all_play': play, 'pass': ~play,
            'moon_alive_opponent': play & (moon[:, 1:].max(1) > 0.5) & (pts[:, 1:].max(1) > 0),
            'moon_alive_none': play & (moon.max(1) < 0.5),
            'endgame_tricks>=9': play & (tricks_played >= 9),
            'early_tricks<4': play & (tricks_played < 4)}
    res['disagreement_by_bin'] = {k: {'n': int(v.sum()), 'rate': float(dis[v].mean()) if v.sum() else None} for k, v in bins.items()}
    # ---- registered halt ----
    res['registered_halt'] = bool(res['B_moonalive_auc_mean_champion_global'] >= 0.97 and res['A_points_r2_champion_global'] >= 0.95)
    for k, v in res.items():
        if k != 'disagreement_by_bin': print(f'{k}: {v}')
    print('disagreement_by_bin:', json.dumps(res['disagreement_by_bin']))
    print('REGISTERED HALT:', res['registered_halt'])
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
