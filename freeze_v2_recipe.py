"""Expert-iter v2 recipe freeze: anchor-coefficient diagnostics + report.

Per the prereg amendment freeze procedure (2026-08-03): trains the
b_even50 mix at each candidate anchor coefficient {0.25, 1.0} x two
freeze-only seeds {909, 910} (binary recipe), then measures on the
bank's holdout tail:

- confident-slice teacher match   (signal absorbed)
- non-confident KL(cand||base)    (knowledge disturbed - what the anchor
                                   exists to protect; measured directly)
- non-confident entropy ratio     (hard constraint: <= 2x baseline, the
                                   v1 un-sharpening signature)
- value EV / belief BCE           (substrate health)

Emits a structured JSON + a human-readable report with a RECOMMENDATION
(dominance -> dominator; genuine trade-off -> protective default 1.0).
THE FINAL CHOICE IS THE USER'S: v1 proved imitation-style metrics can
look healthy while played strength collapses, so this report informs a
human decision - it does not gate one (rules #5 spirit).

Usage: python freeze_v2_recipe.py  (defaults suit the real pipeline)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill import MATCH_RECORD_V2, V2_MAGIC  # noqa: E402
from hearts_net import net_from_checkpoint  # noqa: E402

BASELINE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'  # 8a89da90


def load_holdout(bank, frac):
    with open(bank, 'rb') as fh:
        assert fh.read(4) == V2_MAGIC, f'{bank}: not an HMR2 bank'
    n = (os.path.getsize(bank) - 32) // MATCH_RECORD_V2.itemsize
    a = np.fromfile(bank, dtype=MATCH_RECORD_V2, offset=32, count=n)
    hold = a[len(a) - int(len(a) * frac):]  # same tail rule as distill
    play = (hold['flags'] & 1) == 0
    gap = hold['eq_best'].astype(np.float32) - hold['eq_second']
    conf = play & (hold['second_action'] != 0xFFFF) & (gap > 2.0 * hold['gap_se'])
    return hold[conf], hold[play & ~conf]


def forward_batches(net, recs, device, batch=1024):
    outs = []
    with torch.no_grad():
        for i in range(0, len(recs), batch):
            b = recs[i:i + batch]
            obs = torch.from_numpy(np.ascontiguousarray(b['obs'])).to(device).float() / 255.0
            mask = torch.from_numpy(np.ascontiguousarray(b['mask'])).to(device).bool()
            logits, value, belief = net.forward_all(obs, mask)
            outs.append((logits.cpu(), value.cpu(), belief.cpu(), mask.cpu()))
    return [torch.cat(x) for x in zip(*outs)]


def diagnose(cand_path, base_path, conf, nonconf, device, cap=4096):
    rng = np.random.default_rng(7)
    conf_s = conf[rng.choice(len(conf), min(cap, len(conf)), replace=False)]
    non_s = nonconf[rng.choice(len(nonconf), min(cap, len(nonconf)), replace=False)]
    cand = net_from_checkpoint(cand_path).to(device).eval()
    base = net_from_checkpoint(base_path).to(device).eval()

    cl, cv, cb, cm = forward_batches(cand, conf_s, device)
    match = float((cl.argmax(1).numpy() ==
                   conf_s['action'].astype(np.int64)).mean())
    rew = torch.from_numpy(conf_s['reward'].astype(np.float32))
    ev = float(1.0 - F.mse_loss(cv.squeeze(-1), rew) /
               (rew.var() + 1e-8))
    labels = torch.from_numpy(np.ascontiguousarray(conf_s['labels'])).float()
    bce = float(F.binary_cross_entropy_with_logits(cb, labels))

    nl, _, _, nm = forward_batches(cand, non_s, device)
    bl, _, _, _ = forward_batches(base, non_s, device)

    def ent(logits, mask):
        lp = F.log_softmax(logits, 1).masked_fill(~mask, 0.0)
        p = F.softmax(logits, 1)
        return float(-(p * lp).sum(1).mean())

    e_c, e_b = ent(nl, nm), ent(bl, nm)
    clp = F.log_softmax(nl, 1).masked_fill(~nm, 0.0)
    blp = F.log_softmax(bl, 1).masked_fill(~nm, 0.0)
    kl = float((F.softmax(nl, 1) * (clp - blp)).sum(1).mean())
    return {'conf_match': round(match, 4), 'value_ev': round(ev, 4),
            'belief_bce': round(bce, 4), 'nonconf_entropy': round(e_c, 4),
            'baseline_entropy': round(e_b, 4),
            'entropy_ratio': round(e_c / max(e_b, 1e-6), 3),
            'nonconf_kl_to_baseline': round(kl, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='expert_data/mixes/b_even50.bin')
    ap.add_argument('--baseline', default=BASELINE)
    ap.add_argument('--coefs', nargs='+', type=float, default=[0.25, 1.0])
    ap.add_argument('--seeds', nargs='+', type=int, default=[909, 910])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--holdout', type=float, default=0.10)
    ap.add_argument('--cand-dir', default='.')
    ap.add_argument('--out-report', default='docs/expert_iter_v2_freeze_report.md')
    ap.add_argument('--out-json', default='equity_data/verdicts/expert_iter_v2_freeze.json')
    args = ap.parse_args()

    conf, nonconf = load_holdout(args.bank, args.holdout)
    print(f"holdout: {len(conf)} confident + {len(nonconf)} non-confident play records")

    runs = []
    for coef in args.coefs:
        for seed in args.seeds:
            cand = os.path.join(args.cand_dir, f'cand_v2_freeze_c{coef}_s{seed}.pth')
            if not os.path.exists(cand):
                print(f"training coef={coef} seed={seed} ...")
                r = subprocess.run([sys.executable, '-u', 'distill.py',
                                    '--data', args.bank, '--match',
                                    '--init', args.baseline, '--out', cand,
                                    '--epochs', str(args.epochs),
                                    '--holdout', str(args.holdout),
                                    '--min-confidence', '2.0',
                                    '--anchor-coef', str(coef),
                                    '--anchor-model', args.baseline,
                                    '--train-seed', str(seed),
                                    '--device', args.device],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f'freeze training failed (coef {coef} '
                                     f'seed {seed}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
            d = diagnose(cand, args.baseline, conf, nonconf, args.device)
            d.update({'coef': coef, 'seed': seed, 'cand': cand})
            runs.append(d)
            print(f"  coef {coef} seed {seed}: {d}")

    # Per-coefficient means + the pre-specified recommendation logic.
    means = {}
    for coef in args.coefs:
        rs = [r for r in runs if r['coef'] == coef]
        means[coef] = {k: round(float(np.mean([r[k] for r in rs])), 4)
                       for k in ('conf_match', 'value_ev', 'belief_bce',
                                 'entropy_ratio', 'nonconf_kl_to_baseline')}
    eligible = [c for c in args.coefs if means[c]['entropy_ratio'] <= 2.0]
    lo, hi = min(args.coefs), max(args.coefs)
    if not eligible:
        rec, why = None, ('BOTH coefficients violate the entropy constraint '
                         '(ratio > 2x baseline) - the recipe itself looks '
                         'broken; HALT and investigate before any mix runs.')
    elif len(eligible) == 1:
        rec = eligible[0]
        why = f'only coef {rec} satisfies the entropy constraint.'
    else:
        m_lo, m_hi = means[lo], means[hi]
        lo_dominates = (m_lo['conf_match'] >= m_hi['conf_match']
                        and m_lo['nonconf_kl_to_baseline'] <= m_hi['nonconf_kl_to_baseline'])
        hi_dominates = (m_hi['conf_match'] >= m_lo['conf_match']
                        and m_hi['nonconf_kl_to_baseline'] <= m_lo['nonconf_kl_to_baseline'])
        if hi_dominates:
            rec, why = hi, (f'coef {hi} DOMINATES: absorbs at least as much '
                            'teacher signal with no more flat-state drift.')
        elif lo_dominates:
            rec, why = lo, (f'coef {lo} DOMINATES: absorbs at least as much '
                            'teacher signal with no more flat-state drift.')
        else:
            rec = hi
            why = (f'GENUINE TRADE-OFF (coef {lo} absorbs more signal, '
                   f'coef {hi} preserves more knowledge): protective default '
                   f'-> {hi}. Rationale: the catastrophic historical failure '
                   '(v1, -17 SE) was under-protection; over-anchoring merely '
                   'underperforms. YOUR CALL - the metrics below cannot '
                   'settle this (v1 proved they can look healthy while '
                   'strength collapses).')

    out = {'runs': runs, 'means_by_coef': {str(k): v for k, v in means.items()},
           'entropy_constraint': 'ratio <= 2.0',
           'eligible': eligible, 'recommendation': rec, 'rationale': why,
           'bank': args.bank, 'baseline': args.baseline,
           'note': 'holdout-only diagnostics; final choice is the user\'s'}
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=1)

    os.makedirs(os.path.dirname(args.out_report) or '.', exist_ok=True)
    with open(args.out_report, 'w') as f:
        f.write("# Expert-iter v2 recipe freeze report - anchor coefficient\n\n")
        f.write("The anchor coefficient weights KL(candidate || champion) on\n"
                "non-confident (flat) states: LOW = absorb more teacher signal,\n"
                "risk overwriting real knowledge (the v1 failure); HIGH =\n"
                "protective, may damp learning through the shared trunk.\n\n")
        f.write("**Caveat that makes this a human decision:** v1's disaster\n"
                "had GOOD imitation metrics. Nothing below measures played\n"
                "strength - only the comparative stage does that.\n\n")
        f.write("| coef | seed | conf match | value EV | belief BCE | "
                "entropy ratio | non-conf KL |\n|---|---|---|---|---|---|---|\n")
        for r in runs:
            f.write(f"| {r['coef']} | {r['seed']} | {r['conf_match']:.4f} | "
                    f"{r['value_ev']:.3f} | {r['belief_bce']:.4f} | "
                    f"{r['entropy_ratio']:.3f} | "
                    f"{r['nonconf_kl_to_baseline']:.5f} |\n")
        f.write("\n**Means:**\n\n| coef | conf match | entropy ratio | "
                "non-conf KL |\n|---|---|---|---|\n")
        for c in args.coefs:
            f.write(f"| {c} | {means[c]['conf_match']:.4f} | "
                    f"{means[c]['entropy_ratio']:.3f} | "
                    f"{means[c]['nonconf_kl_to_baseline']:.5f} |\n")
        f.write("\n- conf match: holdout teacher agreement on confident "
                "states (signal absorbed)\n- non-conf KL: policy drift from "
                "the champion in flat states (knowledge disturbed; the thing "
                "the anchor protects)\n- entropy ratio: candidate/baseline "
                "on flat states; hard constraint <= 2.0 (v1's un-sharpening "
                "signature)\n\n")
        f.write(f"## Recommendation: **{rec}**\n\n{why}\n\n")
        f.write("To continue the pipeline with your chosen coefficient:\n\n"
                "    nohup bash ops/auto_v2_continue.sh <0.25|1.0> &\n")
    print(f"\nreport -> {args.out_report}\njson -> {args.out_json}\n"
          f"RECOMMENDATION: {rec} ({why})")
    if rec is None:
        sys.exit(2)


if __name__ == '__main__':
    main()
