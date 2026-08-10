"""Phase 2 Stage B-2 — sharper validity instrument analysis
(docs/phase2_visitcount_prereg.md, Stage B-2 addendum, signed
2026-08-10 BEFORE data).

Inputs: expert_data/p2/stageb2_200.csv   (seed 212000200, K=256 ref,
                                          row-paired with stageb_200.csv)
        expert_data/p2/stageb2_200b.csv  (seed 212001200, K=256 ref)
        expert_data/p2/stageb_200.csv    (original K=64 reference)

Order of operations is binding: the pairing validity check runs FIRST;
if it fails, no band is read (determinism broken -> HALT analysis).

Verdict JSON: equity_data/verdicts/p2_stageb2.json
"""
import json
import sys

import numpy as np
from scipy import stats

TREE_COLS = ['match', 'deal', 'seat', 'legal_n', 'tree_top1', 'tree_top2',
             'pi1', 'pi2']

old = np.genfromtxt('expert_data/p2/stageb_200.csv', delimiter=',',
                    names=True)
new = np.genfromtxt('expert_data/p2/stageb2_200.csv', delimiter=',',
                    names=True)
newb = np.genfromtxt('expert_data/p2/stageb2_200b.csv', delimiter=',',
                     names=True)

out = {'addendum': 'stage B-2 2026-08-10'}

# --- pairing validity check (binding precondition) ---
if len(old) != len(new) or any(
        not np.allclose(old[c], new[c], atol=1e-6) for c in TREE_COLS):
    out['pairing'] = 'FAILED - determinism broken, no unblinding'
    json.dump(out, open('equity_data/verdicts/p2_stageb2.json', 'w'),
              indent=1)
    print('PAIRING CHECK FAILED: tree columns differ from stageb_200.csv - '
          'HALT analysis, bands not read')
    sys.exit(2)
out['pairing'] = 'exact'
print(f'pairing check: exact on {len(new)} rows')


def gaps(r):
    vg = r['pi1'] - r['pi2']
    fg = r['flat_v_tree1'] - r['flat_v_tree2']
    ok = np.isfinite(vg) & np.isfinite(fg)
    return vg[ok], fg[ok]


# --- PRIMARY: tree vs K256, pooled over both matches ---
vg_a, fg_a = gaps(new)
vg_b, fg_b = gaps(newb)
vg = np.concatenate([vg_a, vg_b])
fg = np.concatenate([fg_a, fg_b])
primary, p_primary = stats.spearmanr(vg, fg)
out['primary'] = {'spearman_tree_vs_k256': round(float(primary), 3),
                  'p': float(p_primary), 'n': int(len(vg))}

# --- RELIABILITY DATUM: K64 vs K256 on the row-paired match ---
f64 = old['flat_v_tree1'] - old['flat_v_tree2']
f256 = new['flat_v_tree1'] - new['flat_v_tree2']
ok = np.isfinite(f64) & np.isfinite(f256)
rel, p_rel = stats.spearmanr(f64[ok], f256[ok])
out['reliability'] = {'spearman_k64_vs_k256': round(float(rel), 3),
                      'p': float(p_rel), 'n': int(ok.sum())}

# --- SECONDARY: sign agreement, confident third, vs K256 ---
thr = np.quantile(vg, 2 / 3)
conf = vg >= thr
secondary = float((fg[conf] > 0).mean())
out['secondary'] = {'sign_agree_confident_third': round(secondary, 3),
                    'n_confident': int(conf.sum())}

print(f'PRIMARY  Spearman(tree, K256) = {primary:.3f} '
      f'(p={p_primary:.2e}, n={len(vg)})')
print(f'RELIAB.  Spearman(K64, K256)  = {rel:.3f} '
      f'(p={p_rel:.2e}, n={int(ok.sum())})')
print(f'SECOND.  sign-agree conf-3rd  = {secondary:.1%}')

# --- registered bands ---
if primary >= 0.30:
    band = 'PROCEED'
    msg = ('validity DEMONSTRATED vs cleaner reference - amendment signed '
           'as construction fix; Stage C budget 200 local pre-authorized')
elif primary >= 0.20:
    ceiling = 0.8 * np.sqrt(max(rel, 0.0))
    att = primary >= ceiling
    band = 'JUDGMENT ZONE'
    msg = (f'primary in [0.20,0.30); attenuation test: primary '
           f'{primary:.3f} vs 0.8*sqrt(rel) {ceiling:.3f} -> '
           f'{"attenuation CONFIRMED, recommend proceed" if att else "attenuation NOT confirmed"}'
           '; user decides')
    out['attenuation_confirmed'] = bool(att)
else:
    band = 'HALT-RECOMMEND'
    msg = 'signal weak even against clean reference; user confirms halt'
if secondary < 0.55:
    out['secondary_flag'] = 'sign-agree < 0.55 - flags the confident-decision story'
    print('FLAG: secondary sign agreement below 0.55')

out['band'] = band
out['meaning'] = msg
json.dump(out, open('equity_data/verdicts/p2_stageb2.json', 'w'), indent=1)
print(f'STAGE B-2 BAND: {band} - {msg}')
