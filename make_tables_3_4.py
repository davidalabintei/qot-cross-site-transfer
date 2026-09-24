"""Tables 3 and 4 of the revised manuscript, regenerated from the result CSVs.

Table 3  tab:withinsite   within-site performance, target only at 50%,
                          both rails, BAC, precision, recall, F1.
Table 4  tab:transfer     cross-site transfer, right rail, BAC, 20/30/50%.

Run from the repository root:   python3 make_tables_3_4.py
Reads   qot_results/main_{Left,Right}_contig.csv, qot_results/uncond_classical.csv
        qot_angle_results/main_{Left,Right}_contig_Ang.csv,
        qot_angle_results/uncond_classical_Ang.csv
Writes  tables/table3_withinsite_rows.tex and tables/table4_transfer_rows.tex
        (table body rows only, to paste between \\midrule and \\bottomrule)

HOW EACH NUMBER IS MADE
  A fold is one (source, target, starting position, labeled percentage).
  Every CSV row is one fold and one method. The evaluation half is started at
  eight positions spaced evenly along the target site, giving eight folds.

  Table 3.  Target only never reads the source site, so its value for a target
  site is the same whatever the source. For each site and fold the value is
  taken once. Cell = mean over the eight folds (SD over folds).
  Mean row = mean of the four site means [SD across the four sites].
  F1 is computed per fold as 2PR/(P+R), then averaged like the others.

  Table 4.  Step 1, average each directed pair over its eight folds, giving
  twelve pair values. Cell = mean of the twelve (SD of the twelve).
  Step 2, average each target site over its three source sites, giving four
  values. Square brackets = SD of those four. All SDs are sample SDs (n - 1).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.join(HERE, 'qot_results')
ANG = os.path.join(HERE, 'qot_angle_results')


def load(enc, side):
    if enc == 'Amp':
        m = pd.read_csv(f'{AMP}/main_{side}_contig.csv')
        u = pd.read_csv(f'{AMP}/uncond_classical.csv')
    else:
        m = pd.read_csv(f'{ANG}/main_{side}_contig_Ang.csv')
        u = pd.read_csv(f'{ANG}/uncond_classical_Ang.csv')
    u = u[(u.Side == side) & (u.Arc == 'contig')]
    d = pd.concat([m, u], ignore_index=True)
    d['Method'] = d.Method.str.replace('-Ang-', '-', regex=False)
    den = (d.Precision + d.Recall).replace(0, np.nan)
    d['F1'] = (2 * d.Precision * d.Recall / den).fillna(0)
    return d


# ---------------------------------------------------------------- Table 3
def table3():
    out = []
    for side, lab in [('Left', 'Left'), ('Right', 'Right')]:
        d = load('Amp', side)
        t = d[(d.Method == 'TargetOnly') & np.isclose(d.AdaptFrac, 0.5)]
        # one value per (target, rotation); identical across sources
        t = t.groupby(['Target', 'Rot'])[['BAC', 'Precision', 'Recall', 'F1']].first()
        site_mean = t.groupby(level=0).mean()
        site_sd = t.groupby(level=0).std()
        for i, s in enumerate(['HTL', 'PTT', 'RTT', 'WRM']):
            cells = ' & '.join(f'{site_mean.loc[s, m]:.3f} ({site_sd.loc[s, m]:.3f})'
                               for m in ['BAC', 'Precision', 'Recall', 'F1'])
            out.append(f'    {lab if i == 0 else "":5s} & {s}  & {cells} \\\\')
        out.append(r'    \cmidrule(l){2-6}')
        cells = ' & '.join(f'{site_mean[m].mean():.3f} [{site_mean[m].std():.3f}]'
                           for m in ['BAC', 'Precision', 'Recall', 'F1'])
        out.append(f'          & Mean & {cells} \\\\')
        out.append(r'    \midrule')
    out.pop()
    return out


# ---------------------------------------------------------------- Table 4
FR = [0.20, 0.30, 0.50]


def cell(d, meth, f):
    s = d[(d.Method == meth) & np.isclose(d.AdaptFrac, f)]
    p = s.groupby(['Source', 'Target']).BAC.mean()          # 12 pair values
    tg = p.groupby(level=1).mean()                           # 4 target means
    return f'{p.mean():.3f} ({p.std():.3f}) [{tg.std():.3f}]'


def table4():
    out = []
    rows = [('QOTu', 'QOT-u'), ('QOTc', 'QOT-c'), ('SDMu', r'COT-$\rho$-u'),
            ('SDM', r'COT-$\rho$-c'), ('none', 'no transport')]
    for enc, title in [('Amp', 'Amplitude encoding'), ('Ang', 'Angle encoding')]:
        d = load(enc, 'Right')
        out.append(r'    \multicolumn{4}{l}{\textbf{' + title + r'}} \\')
        for ro, rl in [('HSsrc', 'HS-2'), ('HS4', 'HS-4'), ('PCA', 'PCA')]:
            out.append(r'    \addlinespace[2pt]')
            out.append(r'    \multicolumn{4}{l}{\textit{' + rl + r' readout}} \\')
            for key, name in rows:
                m = f'{key}-{ro}'
                out.append(f'    \\quad {name} & ' + ' & '.join(cell(d, m, f) for f in FR) + r' \\')
        out.append(r'    \midrule')
    d = load('Amp', 'Right')
    out.append(r'    \multicolumn{4}{l}{\textbf{Eight features, no encoding}} \\')
    out.append(r'    \addlinespace[2pt]')
    for m, name in [('COT-u', 'COT-u'), ('COT', 'COT-c'), ('NoAdapt', 'no adaptation'),
                    ('TargetOnly', 'target only'), ('Source+Target', 'source + target')]:
        out.append(f'    \\quad {name} & ' + ' & '.join(cell(d, m, f) for f in FR) + r' \\')
    return out


if __name__ == '__main__':
    t3, t4 = table3(), table4()
    out = os.path.join(HERE, 'tables')
    os.makedirs(out, exist_ok=True)
    for fn, rows in [('table3_withinsite_rows.tex', t3), ('table4_transfer_rows.tex', t4)]:
        open(os.path.join(out, fn), 'w').write('\n'.join(rows) + '\n')
        print(f'wrote tables/{fn}')
