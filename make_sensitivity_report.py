"""Every number in the Sensitivity subsection, recomputed from the eight-fold
result files, written to qot_results/SENSITIVITY_REPORT.txt.

Run from qot_run:   python3 make_sensitivity_report.py

WHY THIS EXISTS
    run_everything.py and run_angle.py ran the permutation and sweep stages at
    four folds, and their REPORT.txt files still carry those four-fold
    numbers. The paper uses the eight-fold reruns (rerun_perm8.py and
    rerun_sweeps8.py), which print to the terminal only and never saved a
    report. This script is that report.

    Each claim in the paper is printed next to the value recomputed here, marked
    MATCH when the two agree at the precision the paper uses and DIFFERS when not.

INPUTS, all right rail unless stated, contiguous arc, eight folds
    Left rail         qot_results/main_Left_contig.csv, qot_angle_results/main_Left_contig_Ang.csv,
                      cleaned_<SITE>.csv for the correlations
    Bures delta       qot_results/sweeps8_Amp.csv, qot_angle_results/sweeps8_Ang.csv   (knob EPS)
    Sinkhorn eps      the same files                                                  (knob OTREG)
    Angle alpha       qot_angle_results/alpha8_Ang.csv
    PCA components    qot_results/main_Right_contig.csv, qot_angle_results/main_Right_contig_Ang.csv
    Arc construction  main_Right_contig against main_Right_random, both encodings
    Block bootstrap   qot_results/bootstrap.csv  (written by run_everything.py stage 4)
    Label permutation qot_results/perm8_Amp.csv, qot_angle_results/perm8_Ang.csv

CONVENTION
    "QOT-u minus no transport" and every other contrast is formed per directed
    pair after averaging that pair over its folds, then averaged over the
    twelve pairs, the same as Tables 4 to 7. HS-2 is HSsrc in the files, COT-rho-u
    is SDMu, COT-rho-c is SDM, COT-c is COT.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Folder holding the cleaned per-site input files. Those are not distributed
# with this repository, so the one check that needs them is skipped when they
# are absent and every other check still runs. Point DATA at them, or set
# QOT_DATA, to get the full report.
DATA = os.environ.get('QOT_DATA', HERE)
A = os.path.join(HERE, 'qot_results')
G = os.path.join(HERE, 'qot_angle_results')
RO = ['HSsrc', 'HS4', 'PCA']
RN = {'HSsrc': 'HS-2', 'HS4': 'HS-4', 'PCA': 'PCA'}
OUT = []


def P(s=''):
    print(s); OUT.append(s)


def H(t):
    P(); P('=' * 88); P(t); P('=' * 88)


def claim(text, paper, value, dp=3):
    ok = round(float(paper), dp) == round(float(value), dp)
    P(f'  [{"MATCH " if ok else "DIFFERS"}] {text:58s} paper {paper:>8}   computed {value:+.{dp + 1}f}'
      if isinstance(paper, str) and paper.startswith(('+', '-')) else
      f'  [{"MATCH " if ok else "DIFFERS"}] {text:58s} paper {paper:>8}   computed {value:.{dp + 1}f}')


def rd(path):
    d = pd.read_csv(path)
    d['Method'] = d.Method.str.replace('-Ang-', '-', regex=False)
    return d


def pairmean(d, meth, frac, extra=None):
    s = d[(d.Method == meth) & np.isclose(d.AdaptFrac, frac)]
    if extra is not None:
        s = s[extra(s)]
    return s.groupby(['Source', 'Target']).BAC.mean()


def diff(d, a, b, frac, extra=None):
    return (pairmean(d, a, frac, extra) - pairmean(d, b, frac, extra)).mean()


# ------------------------------------------------------------------ left rail
def left_rail():
    H('LEFT RAIL')
    cells = []
    for enc, fn in [('Amp', f'{A}/main_Left_contig.csv'), ('Ang', f'{G}/main_Left_contig_Ang.csv')]:
        d = rd(fn)
        for ro in RO:
            for f in [0.05, 0.10, 0.20, 0.30, 0.50]:
                cells.append((enc, ro, f, diff(d, f'QOTu-{ro}', f'none-{ro}', f)))
    c = pd.DataFrame(cells, columns=['enc', 'ro', 'frac', 'd'])
    P('  QOT-u minus no transport, left rail, 30 cells')
    P(c.pivot_table(index=['enc', 'ro'], columns='frac', values='d').round(4).to_string())
    claim('minimum over the 30 cells', '-0.016', c.d.min())
    claim('maximum over the 30 cells', '+0.013', c.d.max())

    # strongest own-rail correlation with the per-rail label (point-biserial = Pearson r)
    KEEP = {'HTL': 2022100717, 'PTT': 2022100607, 'RTT': 2022100710, 'WRM': 2022100714}
    missing = [s for s in KEEP if not os.path.exists(os.path.join(DATA, f'cleaned_{s}.csv'))]
    if missing:
        P('\n  SKIPPED, the correlation check needs the cleaned input files, which are')
        P('  not distributed with this repository. Missing: '
          + ', '.join(f'cleaned_{s}.csv' for s in missing))
        P('  Set QOT_DATA to the folder holding them to run this check.')
        return
    CFG = {'Left': ('LProf62', ['BFI_L_Cat', 'BTI_L', 'MLI_L', 'LRI_L']),
           'Right': ('RProf62', ['BFI_R_Cat', 'BTI_R', 'MLI_R', 'LRI_R'])}
    best = {'Left': [], 'Right': []}
    P('\n  strongest |r| between an own-rail index and the per-rail label, by site')
    for s in KEEP:
        row = []
        for side, (prof, feats) in CFG.items():
            d = pd.read_csv(os.path.join(DATA, f'cleaned_{s}.csv'),
                            usecols=lambda c: c in feats + [prof, 'BMP', 'EMP', 'run_id'],
                            low_memory=False)
            d = d[d.run_id == KEEP[s]].dropna(subset=feats + [prof, 'BMP', 'EMP'])
            g = d.groupby(['BMP', 'EMP']); keep = g.size() >= 10
            X = g[feats].first()[keep]
            y = (g[prof].apply(lambda v: v.abs().max())[keep] > 0.4).astype(float)
            r = {f: abs(np.corrcoef(y, X[f])[0, 1]) for f in feats if X[f].std() > 0}
            k = max(r, key=r.get); best[side].append(r[k])
            row.append(f'{side} {r[k]:.3f} ({k.split("_")[0]})')
        P(f'    {s}   ' + '   '.join(row))
    claim('left rail, lowest site', '0.052', min(best['Left']))
    claim('left rail, highest site', '0.209', max(best['Left']))
    claim('right rail, lowest site', '0.193', min(best['Right']))
    claim('right rail, highest site', '0.365', max(best['Right']))


# ------------------------------------------------------------------ Bures delta
def bures():
    H('BURES REGULARIZATION  delta, 5% and 50%')
    worst = 0
    for enc, fn in [('Amp', f'{A}/sweeps8_Amp.csv'), ('Ang', f'{G}/sweeps8_Ang.csv')]:
        d = rd(fn); d = d[(d.Side == 'Right') & (d.knob == 'EPS')]
        vals = sorted(d.value.unique())
        P(f'\n  {enc}  QOT-u minus no transport at each delta')
        P('  ' + ' ' * 12 + ''.join(f'{v:>10.0e}' for v in vals) + '     range')
        for ro in RO:
            for f in [0.05, 0.50]:
                x = [diff(d[d.value == v], f'QOTu-{ro}', f'none-{ro}', f) for v in vals]
                worst = max(worst, max(x) - min(x))
                P(f'  {RN[ro]:5s} {f:>4.0%}  ' + ''.join(f'{v:+10.4f}' for v in x)
                  + f'   {max(x) - min(x):.4f}')
    claim('largest movement across the seven values', '0.007', worst)


# ------------------------------------------------------------------ Sinkhorn eps
def sinkhorn():
    H('SINKHORN REGULARIZATION  epsilon, 5% and 50%')
    R = {}
    for enc, fn in [('Amp', f'{A}/sweeps8_Amp.csv'), ('Ang', f'{G}/sweeps8_Ang.csv')]:
        d = rd(fn); d = d[d.Side == 'Right']
        q = d[(d.knob == 'EPS') & np.isclose(d.value, 1e-9)]      # QOT-u does not depend on eps
        o = d[d.knob == 'OTREG']; grid = sorted(o.value.unique())
        P(f'\n  {enc}  levels, mean BAC over the twelve pairs')
        P('  ' + ' ' * 18 + ''.join(f'{g:>9g}' for g in grid))
        for m in ['COT-u', 'COT-c'] + [f'SDMu-{r}' for r in RO] + [f'SDM-{r}' for r in RO]:
            for f in [0.05, 0.50]:
                P(f'  {m:12s} {f:>4.0%}  ' + ''.join(
                    f'{pairmean(o[np.isclose(o.value, g)], m, f).mean():9.3f}' for g in grid))
        P(f'\n  {enc}  QOT-u minus COT-rho-u at each epsilon')
        for ro in RO:
            for f in [0.05, 0.50]:
                qu = pairmean(q, f'QOTu-{ro}', f)
                row = []
                for g in grid:
                    v = (qu - pairmean(o[np.isclose(o.value, g)], f'SDMu-{ro}', f)).mean()
                    R[(enc, ro, f, g)] = v; row.append(v)
                R[(enc, ro, f, 'qot')] = qu.mean()
                P(f'  {RN[ro]:5s} {f:>4.0%}  ' + ''.join(f'{v:+9.3f}' for v in row))
        if enc == 'Ang':
            allpos = all(R[('Ang', ro, f, g)] > 0 for ro in RO for f in [0.05, 0.5] for g in grid)
            P(f'\n  angle, QOT-u minus COT-rho-u above zero at every epsilon, readout, share: {allpos}')
    lvl = lambda f: R[('Amp', 'HSsrc', f, 'qot')] - R[('Amp', 'HSsrc', f, 0.001)]
    claim('Amp, COT-rho-u HS-2 level at eps .001, 5%', '0.617', lvl(0.05))
    claim('Amp, COT-rho-u HS-2 level at eps .001, 50%', '0.633', lvl(0.50))
    claim('Amp, QOT-u HS-2 level, 5% ("level with")', '0.617', R[('Amp', 'HSsrc', 0.05, 'qot')])
    claim('Amp, QOT-u HS-2 level, 50% ("level with")', '0.633', R[('Amp', 'HSsrc', 0.50, 'qot')])
    claim('Amp, HS-4 difference at eps .001, 5%', '+0.009', R[('Amp', 'HS4', 0.05, 0.001)])
    claim('Amp, HS-4 difference at eps .001, 50%', '-0.005', R[('Amp', 'HS4', 0.50, 0.001)])
    claim('Amp, PCA difference at eps .001, 5%', '+0.006', R[('Amp', 'PCA', 0.05, 0.001)])
    claim('Amp, PCA difference at eps .001, 50%', '+0.019', R[('Amp', 'PCA', 0.50, 0.001)])
    claim('Amp, PCA difference at eps .05, 5%', '+0.044', R[('Amp', 'PCA', 0.05, 0.05)])
    claim('Amp, PCA difference at eps .05, 50%', '+0.038', R[('Amp', 'PCA', 0.50, 0.05)])
    claim('Ang, HS-2 difference at eps .001, 5%', '+0.059', R[('Ang', 'HSsrc', 0.05, 0.001)])
    claim('Ang, HS-2 difference at eps .001, 50%', '+0.037', R[('Ang', 'HSsrc', 0.50, 0.001)])
    claim('Ang, HS-2 difference at eps .05, 5%', '+0.042', R[('Ang', 'HSsrc', 0.05, 0.05)])
    claim('Ang, HS-2 difference at eps .05, 50%', '+0.018', R[('Ang', 'HSsrc', 0.50, 0.05)])


# ------------------------------------------------------------------ angle alpha
def alpha():
    H('ANGLE SCALING  alpha, HS-2, QOT-u minus no transport')
    d = rd(f'{G}/alpha8_Ang.csv'); d = d[d.Side == 'Right']
    paper = {0.05: ['+0.051', '+0.050', '+0.049', '+0.045'], 0.50: ['+0.039', '+0.038', '+0.040', '+0.040']}
    for f in [0.05, 0.50]:
        for al, pv in zip([0.01, 0.025, 0.05, 0.1], paper[f]):
            claim(f'alpha {al}, {f:.0%}', pv, diff(d[np.isclose(d.alpha, al)], 'QOTu-HSsrc', 'none-HSsrc', f))


# ------------------------------------------------------------------ PCA components
def pca_k():
    H('PCA COMPONENTS  share of folds choosing each K, QOT-u, all labeled shares')
    for enc, fn in [('Amp', f'{A}/main_Right_contig.csv'), ('Ang', f'{G}/main_Right_contig_Ang.csv')]:
        d = rd(fn); k = d[d.Method == 'QOTu-PCA'].K.dropna()
        P(f'  {enc}  ' + '   '.join(f'K={int(v)} {s:.0%}' for v, s in k.value_counts(normalize=True).sort_index().items()))
        if enc == 'Amp':
            claim('Amp, share with eight or fewer', '0.54', (k <= 8).mean(), dp=2)
        else:
            claim('Ang, share with 16 or more', '0.81', (k >= 16).mean(), dp=2)


# ------------------------------------------------------------------ arc construction
def arc():
    H('ARC CONSTRUCTION  contiguous against random')
    worst, mv = 0, []
    for enc, c, r in [('Amp', f'{A}/main_Right_contig.csv', f'{A}/main_Right_random.csv'),
                      ('Ang', f'{G}/main_Right_contig_Ang.csv', f'{G}/main_Right_random_Ang.csv')]:
        dc, dr = rd(c), rd(r)
        for ro in RO:
            for f in [0.05, 0.10, 0.20, 0.30, 0.50]:
                worst = max(worst, abs(diff(dc, f'QOTu-{ro}', f'none-{ro}', f)
                                       - diff(dr, f'QOTu-{ro}', f'none-{ro}', f)))
        for m in sorted(set(dc.Method)):
            v = abs(pairmean(dc, m, 0.05).mean() - pairmean(dr, m, 0.05).mean())
            if not np.isnan(v):
                mv.append((v, enc, m))
    claim('largest change in QOT-u minus no transport, 30 cells', '0.017', worst)
    mv.sort(reverse=True)
    P('  largest single-method movements at 5%: ' + ', '.join(f'{m} {e} {v:.3f}' for v, e, m in mv[:4]))
    claim(f'largest movement at 5% ({mv[0][2]})', '0.056', mv[0][0])


# ------------------------------------------------------------------ block bootstrap
def boot():
    H('BLOCK BOOTSTRAP  amplitude, QOT-u minus no transport at HS-2')
    b = pd.read_csv(f'{A}/bootstrap.csv')
    bl = b.groupby('Target').Block.first()
    P('  block length by target site: ' + '   '.join(f'{s} {v}' for s, v in bl.items()))
    for s, v in [('HTL', 10), ('PTT', 21), ('RTT', 30), ('WRM', 14)]:
        claim(f'block length at {s}', str(v), bl[s], dp=0)
    c = 'QOTu-HSsrc|none-HSsrc'
    for f, pos, exc in [(0.05, 9, 6), (0.50, 9, 7)]:
        q = b[np.isclose(b.AdaptFrac, f)]
        claim(f'{f:.0%}, pairs with a positive difference', str(pos), (q[c] > 0).sum(), dp=0)
        claim(f'{f:.0%}, pairs whose interval excludes zero', str(exc), ((q[c + '_lo'] > 0) | (q[c + '_hi'] < 0)).sum(), dp=0)
        P(f'    folds used per pair at {f:.0%}: ' + ' '.join(str(int(n)) for n in q.nRot))
    P('  NOTE  the bootstrap keeps only folds whose labeled stretch holds at least two')
    P('        of each class, so at 5% some pairs rest on fewer than eight folds.')


# ------------------------------------------------------------------ label permutation
def perm():
    H('LABEL PERMUTATION  5, 20 and 50%')
    L = {}
    for enc, fn in [('Amp', f'{A}/perm8_Amp.csv'), ('Ang', f'{G}/perm8_Ang.csv')]:
        d = rd(fn); d = d[d.Side == 'Right']
        L[enc] = d.groupby(['kind', 'Method', 'AdaptFrac', 'Source', 'Target']).BAC.mean() \
                  .groupby(['kind', 'Method', 'AdaptFrac']).mean()
    fr = [0.05, 0.20, 0.50]
    unc = [f'{p}-{r}' for p in ('QOTu', 'SDMu', 'none') for r in RO] + ['COT-u', 'NoAdapt']
    v = [L[e][('srcperm', m, f)] for e in L for m in unc for f in fr]
    claim('source shuffle, lowest of the unconditional methods', '0.487', min(v))
    claim('source shuffle, highest of the unconditional methods', '0.527', max(v))
    claim('QOT-u HS-2 Amp 50%, real', '0.633', L['Amp'][('real', 'QOTu-HSsrc', 0.5)])
    claim('QOT-u HS-2 Amp 50%, source shuffle', '0.503', L['Amp'][('srcperm', 'QOTu-HSsrc', 0.5)])
    t = max(abs(L[e][('real', 'TargetOnly', f)] - L[e][('srcperm', 'TargetOnly', f)]) for e in L for f in fr)
    claim('target only, change under source shuffle (4 dp)', '0.0000', t, dp=4)
    claim('target only, target shuffle, 50%', '0.498', L['Amp'][('arcperm', 'TargetOnly', 0.5)])
    same = [f'{p}-{r}' for p in ('QOTu', 'SDMu', 'none') for r in ('HSsrc', 'PCA')] + ['COT-u']
    t = max(abs(L[e][('real', m, f)] - L[e][('arcperm', m, f)]) for e in L for m in same for f in fr)
    claim('unconditional, HS-2 and PCA, change under target shuffle', '0.0000', t, dp=4)
    h4 = [f'{p}-HS4' for p in ('QOTu', 'SDMu', 'none')]
    t = max(abs(L[e][('real', m, f)] - L[e][('arcperm', m, f)]) for e in L for m in h4 for f in fr)
    claim('unconditional, HS-4, largest change under target shuffle', '0.006', t)
    r = L['Amp'][('real', 'QOTc-HSsrc', 0.5)]
    claim('QOT-c HS-2 Amp 50%, loss to target shuffle', '0.063', r - L['Amp'][('arcperm', 'QOTc-HSsrc', 0.5)])
    claim('QOT-c HS-2 Amp 50%, loss to source shuffle', '0.019', r - L['Amp'][('srcperm', 'QOTc-HSsrc', 0.5)])


if __name__ == '__main__':
    P('SENSITIVITY REPORT   eight folds, right rail unless stated, contiguous arc')
    P('Supersedes the four-fold permutation and sweep stages in REPORT.txt.')
    for f in (left_rail, bures, sinkhorn, alpha, pca_k, arc, boot, perm):
        f()
    n = sum('[DIFFERS]' in s for s in OUT)
    P(); P(f'{n} claim(s) differ from the paper.')
    open(os.path.join(A, 'SENSITIVITY_REPORT.txt'), 'w').write('\n'.join(OUT) + '\n')
    print(f'\nwrote {os.path.join(A, "SENSITIVITY_REPORT.txt")}')
