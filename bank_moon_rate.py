"""League r9 §3.1 — the SEARCH shooter's moon rate against a given defender,
from SearchEval shooter-mode decision CSVs (one row per deal; `moon_success`
= 1 when the shooter completed a moon that deal). Prints moons/deal with a
binomial 95% CI; the number feeds verify_shooter.py --search-rate as the
certification reference (bar = 50% of it).

Usage: python bank_moon_rate.py equity_data/exploiter_r9/bank/selv2
"""
import glob
import os
import sys

import pandas as pd

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, 'm*.csv')))
done = [f for f in files if os.path.exists(f[:-4] + '.done')]
df = pd.concat([pd.read_csv(f) for f in done], ignore_index=True) if done else pd.DataFrame()
if df.empty:
    print('no complete units yet'); sys.exit(1)
deals = len(df)
moons = int(df['moon_success'].sum())
rate = moons / deals
se = (rate * (1 - rate) / deals) ** 0.5
matches = df.groupby(['seed', 'match']).ngroups
print(f'{len(done)} complete units, {matches} matches, {deals} deals: search-SEL moons/deal = {rate:.4f} '
      f'(95% CI [{rate - 1.96 * se:.4f}, {rate + 1.96 * se:.4f}]); moons/match {moons / matches:.3f}; '
      f'defender_moon/deal {df["defender_moon"].mean():.4f}')
print(f'certification bar (50%): {0.5 * rate:.4f} moons/deal')
