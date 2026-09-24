"""Unconditional Sinkhorn transport, the matched classical comparator.

WHAT IT ADDS
    On the density-matrix side both an unconditional and a class-conditional
    Bures map are run. This script supplies the matching unconditional Sinkhorn
    transports, so that the quantum and classical arms differ in one thing only.

        COT-u        unconditional Sinkhorn on the 8 raw features
        SDMu-*       unconditional Sinkhorn on the vectorized density matrix,
                     read out through HSsrc (HS-2), HS4 (HS-4) and PCA

    Both use the target adaptation SEGMENTS and not their labels, which is what
    the unconditional Bures map uses. The isolated contrast is therefore
    QOTu-<readout> minus SDMu-<readout>. Same information from the target, same
    readout, one difference, the geometry the transport acts in.

    Without these rows, QOT-u could only be compared against the
    class-conditional COT, which differs from it in two things at once, the
    transport geometry and whether target labels are read.

HOW IT STAYS COMPARABLE
    It imports run_everything.py and reuses load, split, clf, the encodings and
    the readouts unchanged, so the folds, seeds and features are identical by
    construction rather than by reimplementation. Output merges with main_*.csv
    on (Side, Arc, Source, Target, Rot, AdaptFrac).

RUN
    The cleaned_<SITE>.csv input files are not distributed with this repository.
    See the README for the columns they must contain. Put this file beside
    run_everything.py and those inputs, then

        OMP_NUM_THREADS=1 python3 run_uncond_classical.py --data . --enc Amp
        OMP_NUM_THREADS=1 python3 run_uncond_classical.py --data . --enc Ang

    Roughly a quarter of the main stage, so on the order of 5 to 10 minutes.
    Output: uncond_classical_<Enc>.csv in that encoding's results folder.
"""
import os, sys, argparse, itertools
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import warnings; warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--enc', default='Amp', choices=['Amp', 'Ang'],
                help='Amp imports run_everything.py, Ang imports run_angle.py')
a = ap.parse_args()
os.environ['QOT_DATA'] = a.data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if a.enc == 'Ang':
    import run_angle as M                 # angle encoding, 16x16
else:
    import run_everything as M            # amplitude encoding, 8x8
M.DATA = a.data
PREP = (lambda side: M.prep(side, 'Ang')) if a.enc == 'Ang' else M.prep


def uncond_sinkhorn(Xs, Xt, reg=None):
    """Sinkhorn with NO class loop. Target labels are never touched."""
    reg = M.OTREG if reg is None else reg
    if not len(Xs) or not len(Xt):
        return Xs.copy()
    return M.sinkhorn(Xs, Xt, reg)


def main():
    rots = 3 if a.quick else M.N_ROT
    fracs = [.05, .50] if a.quick else M.FRACS
    rows = []
    for side in ('Right', 'Left'):
        D = PREP(side)
        for arc_mode in ('contig', 'random'):
            for src, tgt in itertools.permutations(M.SITES, 2):
                S, T = D[src], D[tgt]; ys = S['y']
                for rot in range(rots):
                    for frac in fracs:
                        arc, ev = M.split(T['mid'], rot, arc_mode, frac, rots)
                        if not len(arc): continue
                        if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                        ya, ye = T['y'][arc], T['y'][ev]
                        Rs, Rta = S['R'], T['R'][arc]
                        base = dict(Enc=a.enc, Side=side, Arc=arc_mode,
                                    Source=src, Target=tgt,
                                    Rot=rot, AdaptFrac=frac, nEval=len(ev),
                                    nEvalPos=int(ye.sum()), nArc=len(arc))
                        p = {}
                        # raw features, unconditional
                        p['COT-u'] = M.clf(rot).fit(
                            uncond_sinkhorn(S['X'], T['X'][arc]), ys).predict(T['X'][ev])
                        # vectorized density matrix, unconditional, three readouts
                        Vu = uncond_sinkhorn(S['V'], T['V'][arc])
                        Ru = M.unvec_rho(Vu, Rs.shape[1])
                        pb, km = M.pca_basis(S['V'], T['V'][arc], rot)
                        for ro in M.READOUTS:
                            if ro == 'HS4' and min(np.bincount(ya, minlength=2)) < 2:
                                continue          # HS4 needs both arc classes
                            if ro == 'PCA':
                                A_, B_, _ = M.pca_apply(pb, km, Vu, T['V'][ev], ys, rot)
                            else:
                                A_, B_ = M.hs(Ru, ys, Rs, Rta, ya, T['R'][ev],
                                              ro == 'HSsrc')
                            p[f'SDMu-{ro}'] = M.clf(rot).fit(A_, ys).predict(B_)
                        for nm, pred in p.items():
                            m_ = M.metrics(ye, pred)
                            rows.append({**base, 'Method': nm,
                                         'Uses': 'arc labels' if nm.endswith('HS4')
                                                 else 'arc segments',
                                         **m_})
                print(f'  {side} {arc_mode} {src}->{tgt} done', flush=True)

    d = pd.DataFrame(rows)
    os.makedirs(M.OUT, exist_ok=True)
    fn = f'{M.OUT}/uncond_classical_{a.enc}.csv'
    d.to_csv(fn, index=False)

    # ---- the contrast this was written for
    for side in ('Right', 'Left'):
        for arc_mode in ('contig', 'random'):
            q = d[(d.Side == side) & (d.Arc == arc_mode)]
            if not len(q): continue
            print(f'\n{"="*90}\n{side} rail, {arc_mode} arc   '
                  f'unconditional classical transport\n{"="*90}')
            print(f'  {"method":14s} ' + ' '.join(f'{f:.0%}'.rjust(14) for f in fracs))
            for m_ in ['COT-u'] + [f'SDMu-{r}' for r in M.READOUTS]:
                cells = []
                for f in fracs:
                    s_ = q[(q.Method == m_) & np.isclose(q.AdaptFrac, f)]
                    pp = s_.groupby(['Source', 'Target']).BAC.mean()
                    cells.append((f'{pp.mean():.3f}({pp.std():.3f})'
                                  if len(pp) else '-').rjust(14))
                print(f'  {m_:14s} ' + ' '.join(cells))
    print(f'\nwrote {fn}')
    print('\nMerge with main_<side>_<arc>.csv on Side/Arc/Source/Target/Rot/AdaptFrac.')
    print('The contrast to form is QOTu-<readout> minus SDMu-<readout>, which holds')
    print('the readout and the label usage fixed and varies only the transport geometry.')


if __name__ == '__main__':
    main()
