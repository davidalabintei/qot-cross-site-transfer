"""Why transport gives no gain on the left rail.

Reports the point-biserial correlation between each own-rail ballast index and
the per-rail profile defect label, for every site and both rails. The centre
channel is excluded because both rails share it, so only the four own-rail
features are informative about a rail-specific difference.

Requires the cleaned per-site input files, which are not distributed with this
repository. See the README for the columns they must contain.

    python3 leftrail.py --data /path/to/cleaned/csvs

Segment construction here is identical to run_everything.py, same run selection,
same minimum record count, same threshold, so the labels match the main results.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr
import warnings; warnings.filterwarnings('ignore')

DATA = '.'
KEEP = {'HTL': 2022100717, 'PTT': 2022100607, 'RTT': 2022100710, 'WRM': 2022100714}
SIDE_CFG = {
    'Left':  ('LProf62', ['BFI_L_Cat', 'BFI_C_Cat', 'BTI_L', 'BTI_C',
                          'MLI_L', 'MLI_C', 'LRI_L', 'LRI_C']),
    'Right': ('RProf62', ['BFI_R_Cat', 'BFI_C_Cat', 'BTI_R', 'BTI_C',
                          'MLI_R', 'MLI_C', 'LRI_R', 'LRI_C'])}
THRESH, MIN_ROWS = 0.4, 10
SITES = list(KEEP)


def load(site, side):
    prof, feats = SIDE_CFG[side]
    d = pd.read_csv(f'{DATA}/cleaned_{site}.csv',
                    usecols=lambda c: c in feats + [prof, 'BMP', 'EMP', 'run_id'],
                    low_memory=False)
    d = d[d.run_id == KEEP[site]].dropna(subset=feats + [prof, 'BMP', 'EMP'])
    g = d.groupby(['BMP', 'EMP'], sort=True)
    keep = g.size() >= MIN_ROWS
    Xf = g[feats].first()[keep]
    y = (g[prof].apply(lambda v: v.abs().max())[keep] > THRESH).astype(int).values
    mid = np.array([(b + e) / 2 for b, e in Xf.index])
    o = np.argsort(mid, kind='stable')
    return Xf.values.astype(float)[o], y[o], feats


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=DATA,
                    help='folder holding cleaned_<SITE>.csv (default: current folder)')
    DATA = ap.parse_args().data
    missing = [s for s in SITES
               if not os.path.exists(os.path.join(DATA, f'cleaned_{s}.csv'))]
    if missing:
        raise SystemExit(
            'ERROR: ' + ', '.join(f'cleaned_{s}.csv' for s in missing)
            + f' not found in {DATA!r}.\n'
            'These inputs are proprietary and are not distributed with this\n'
            'repository. Point --data at a folder holding them. The README\n'
            'lists the columns they must contain.')
    print('point-biserial correlation of each feature with the per-rail label')
    print('own-rail features only (the center channel is shared by both rails)\n')
    rows = []
    for site in SITES:
        for side in ('Left', 'Right'):
            X, y, feats = load(site, side)
            own = [i for i, f in enumerate(feats) if not f.split('_')[1] == 'C']
            r = {}
            for i in own:
                v = X[:, i]
                r[feats[i]] = np.nan if v.std() == 0 else pointbiserialr(y, v)[0]
            rows.append(dict(site=site, side=side, n=len(y), pos=int(y.sum()),
                             rate=y.mean(), **r))
    d = pd.DataFrame(rows)
    d.columns = [c.replace('_L', '').replace('_R', '') for c in d.columns]
    pd.set_option('display.width', 200)
    print(d.round(3).to_string(index=False))

    print('\nmax |correlation| over the four own-rail features')
    for site in SITES:
        s = d[d.site == site]
        fc = [c for c in d.columns if c not in ('site', 'side', 'n', 'pos', 'rate')]
        L = s[s.side == 'Left'][fc].abs().max(axis=1).iloc[0]
        R = s[s.side == 'Right'][fc].abs().max(axis=1).iloc[0]
        print(f'  {site}  left {L:.3f}   right {R:.3f}')
