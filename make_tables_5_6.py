"""Tables 5 and 6 of the manuscript, regenerated from the result CSVs.

Table 5  tab:gain     Paired contrasts, right rail. Five blocks under one
                      header. QOT-u minus its no-transport control in balanced
                      accuracy, recall and precision, then QOT-u minus COT-rho-u
                      and QOT-c minus COT-rho-c in balanced accuracy.
Table 6  tab:perpair  Transfer by directed site pair, right rail, 50% labeled,
                      balanced accuracy.

Run from the repository root.
    python3 make_tables_5_6.py
Reads the same CSVs as make_tables_3_4.py, whose load() is reused.
Writes tables/table5_contrasts_rows.tex and tables/table6_perpair_rows.tex,
the table body rows only, to paste between \midrule and \bottomrule.

HOW EACH NUMBER IS MADE

  Every CSV row is one fold, that is one (source, target, starting position,
  labeled percentage), for one method.

  Table 5, the contrasts. For each directed pair, average method A and method B
  over the eight folds, then take A minus B. That gives twelve paired
  differences. The cell is

      mean of the twelve (SD of the twelve) pairs above zero/12, Wilcoxon p

  where p is the two-sided Wilcoxon signed-rank test on the twelve differences,
  scipy.stats.wilcoxon with default settings. Both methods in a contrast share
  the encoding, the readout and the information they take from the target site,
  so only the transport differs.

  The SD is of the paired differences, not of the two methods separately. It is
  much smaller than either, because the pairs are strongly correlated, which is
  the reason for the paired design.

  Table 6. One directed pair per row, 50% labeled, HS-2 readout for the
  density-matrix methods. The cell is the mean over the eight folds with the SD
  over those folds in parentheses. Bold marks the largest value in the row. The
  mean row is the mean of the twelve pair values, with the SD of the four
  target-site means in square brackets, each target first averaged over its
  three sources. That is the same bracket convention as Table 4.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from make_tables_3_4 import load, HERE

FR = [0.20, 0.30, 0.50]


def pairs(d, meth, f, metric='BAC'):
    s = d[(d.Method == meth) & np.isclose(d.AdaptFrac, f)]
    return s.groupby(['Source', 'Target'])[metric].mean()


def num(x, dp=3, sign=True):
    s = f'{x:+.{dp}f}' if sign else f'{x:.{dp}f}'
    return s.replace('-', '$-$')


def pfmt(p):
    # p rounded to three decimals, "<.001" only when it rounds to .000
    r = round(p, 3)
    return '$<$.001' if r < 0.001 else ('1.000' if r >= 1 else f'{r:.3f}'.lstrip('0'))


def contrast(d, a, b, f, metric='BAC'):
    j = pd.concat([pairs(d, a, f, metric), pairs(d, b, f, metric)], axis=1).dropna()
    x = j.iloc[:, 0] - j.iloc[:, 1]
    p = wilcoxon(x).pvalue
    return f'{num(x.mean())} ({x.std():.3f}) {int((x > 0).sum())}/{len(x)}, {pfmt(p)}'


RO = [('HSsrc', 'HS-2'), ('HS4', 'HS-4'), ('PCA', 'PCA')]


def contrast_block(D, a, b, metric):
    out = []
    for ro, rl in RO:
        for i, enc in enumerate(['Amp', 'Ang']):
            cells = ' & '.join(contrast(D[enc], f'{a}-{ro}', f'{b}-{ro}', f, metric) for f in FR)
            out.append(f'    \\quad {rl if i == 0 else ""} & {enc} & {cells} \\\\')
    return out


def table5(D):
    """The merged contrast table. Three metric blocks against no transport,
    then the two blocks against Sinkhorn on the same density matrix."""
    out = []
    for metric, title in [('BAC', 'balanced accuracy'), ('Recall', 'recall'),
                          ('Precision', 'precision')]:
        out.append(r'    \addlinespace[2pt]')
        out.append(r'    \multicolumn{5}{l}{\textit{QOT-u minus no transport, ' + title + r'}} \\')
        out += contrast_block(D, 'QOTu', 'none', metric)
    for a, b, title in [('QOTu', 'SDMu', r'QOT-u minus COT-$\rho$-u, balanced accuracy'),
                        ('QOTc', 'SDM', r'QOT-c minus COT-$\rho$-c, balanced accuracy')]:
        out.append(r'    \addlinespace[2pt]')
        out.append(r'    \multicolumn{5}{l}{\textit{' + title + r'}} \\')
        out += contrast_block(D, a, b, 'BAC')
    return out


def table6(D):
    cols = [('Amp', 'none-HSsrc'), ('Amp', 'QOTu-HSsrc'), ('Ang', 'QOTu-HSsrc'),
            ('Amp', 'SDMu-HSsrc'), ('Amp', 'COT-u'), ('Amp', 'TargetOnly')]
    mean, sd, pm = [], [], []
    for enc, m in cols:
        s = D[enc][(D[enc].Method == m) & np.isclose(D[enc].AdaptFrac, 0.5)]
        g = s.groupby(['Source', 'Target']).BAC
        mean.append(g.mean()); sd.append(g.std())
    M = pd.concat(mean, axis=1); S = pd.concat(sd, axis=1)
    out = []
    for (src, tgt), row in M.iterrows():
        best = int(np.argmax(row.round(3).values))
        cells = []
        for k in range(len(cols)):
            c = f'{row.iloc[k]:.3f} ({S.loc[(src, tgt)].iloc[k]:.2f})'
            cells.append(r'\textbf{' + c + '}' if k == best else c)
        out.append(f'    {src} & {tgt} & ' + ' & '.join(cells) + r' \\')
    out.append(r'    \midrule')
    tg = M.groupby(level=1).mean()
    out.append(r'    \multicolumn{2}{l}{\textit{Mean}} & ' +
               ' & '.join(f'{M.iloc[:, k].mean():.3f} [{tg.iloc[:, k].std():.3f}]'
                          for k in range(len(cols))) + r' \\')
    return out


if __name__ == '__main__':
    D = {'Amp': load('Amp', 'Right'), 'Ang': load('Ang', 'Right')}
    out = os.path.join(HERE, 'tables')
    os.makedirs(out, exist_ok=True)
    for fn, rows in [('table5_contrasts_rows.tex', table5(D)),
                     ('table6_perpair_rows.tex', table6(D))]:
        open(os.path.join(out, fn), 'w').write('\n'.join(rows) + '\n')
        print(f'wrote tables/{fn}')
