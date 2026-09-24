"""Every sensitivity sweep at eight starting positions, with the unconditional
classical methods included.

WHY IT IS SEPARATE
    The sweep stages inside run_everything.py and run_angle.py run at four
    starting positions, and they sweep the Sinkhorn regularization for the
    class-conditional methods only. The main results use eight positions, and
    the classical comparator the quantum-against-classical result rests on is
    the unconditional COT-rho-u, which those stages never sweep.

    A sensitivity check at four positions cannot be compared with a main table
    at eight. This script runs everything at eight and adds the unconditional
    methods, so the sweeps and the main tables are like for like. The reported
    sensitivity results come from here, not from the sweep stages.

WHAT IT RUNS, per encoding, right and left rail, contiguous label stretch
    EPS     1e-12 1e-10 1e-9 1e-8 1e-6 1e-4 1e-2   on QOTu / QOTc / none,
                                                   3 readouts. This is the floor
                                                   added to the density matrices
                                                   before the matrix square root.
    OTREG   0.001 0.005 0.01 0.05 0.1 0.5          the Sinkhorn regularization,
                                                   on COT-c, COT-u,
                                                   SDM-* (class-conditional) and
                                                   SDMu-* (unconditional)
    ALPHA   0.01 0.025 0.05 0.1                    angle encoding only. It
                                                   re-encodes the data at each
                                                   value, so it is a second pass.

    It imports run_everything.py or run_angle.py and reuses load, split, clf,
    the encodings and the readouts unchanged, so the folds and seeds match the
    main run by construction. Passing n_rot = 8 to split() is what makes the
    evaluation halves identical to the main tables.

RUN
    The cleaned_<SITE>.csv input files are not distributed with this repository.
    See the README for the columns they must contain.

        OMP_NUM_THREADS=1 python3 rerun_sweeps8.py --data . --enc Amp
        OMP_NUM_THREADS=1 python3 rerun_sweeps8.py --data . --enc Ang

    Around 30 to 50 minutes each. The angle run does the alpha pass as well and
    takes longer.
    Output: sweeps8_<Enc>.csv and, for Ang, alpha8_Ang.csv
"""
import os, sys, argparse, itertools
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import warnings; warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--enc', default='Amp', choices=['Amp', 'Ang'])
a = ap.parse_args()
os.environ['QOT_DATA'] = a.data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if a.enc == 'Ang':
    import run_angle as M
else:
    import run_everything as M
M.DATA = a.data
PREP = (lambda side: M.prep(side, 'Ang')) if a.enc == 'Ang' else M.prep

EPS_GRID = [1e-12, 1e-9, 1e-2] if a.quick else \
           [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2]
REG_GRID = [0.001, 0.05, 0.5] if a.quick else [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
ALPHA_GRID = [0.01, 0.025, 0.05, 0.1]
FRACS = [0.05, 0.50]
N_ROT = 8


def main():
    rots = 2 if a.quick else N_ROT
    rows = []
    for side in ('Right', 'Left'):
        D = PREP(side)
        for src, tgt in itertools.permutations(M.SITES, 2):
            S, T = D[src], D[tgt]; ys = S['y']
            for rot in range(rots):
                for frac in FRACS:
                    arc, ev = M.split(T['mid'], rot, 'contig', frac, rots)
                    if not len(arc): continue
                    if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                    ya, ye = T['y'][arc], T['y'][ev]
                    arc_ok = min(np.bincount(ya, minlength=2)) >= 2
                    Rs, Rta = S['R'], T['R'][arc]
                    pb, km = M.pca_basis(S['V'], T['V'][arc], rot)
                    base = dict(Enc=a.enc, Side=side, Source=src, Target=tgt,
                                Rot=rot, AdaptFrac=frac, ArcOK=int(arc_ok))

                    def fit(A_, B_):
                        return balanced_accuracy_score(
                            ye, M.clf(rot).fit(A_, ys).predict(B_))

                    def readouts(Rtr, V, tag, knob, val):
                        for ro in M.READOUTS:
                            if ro == 'HS4' and not arc_ok:
                                continue
                            if ro == 'PCA':
                                A_, B_, _ = M.pca_apply(pb, km, V, T['V'][ev], ys, rot)
                            else:
                                A_, B_ = M.hs(Rtr, ys, Rs, Rta, ya, T['R'][ev],
                                              ro == 'HSsrc')
                            rows.append({**base, 'knob': knob, 'value': val,
                                         'Method': f'{tag}-{ro}', 'BAC': fit(A_, B_)})

                    # ---- Bures floor
                    for e in EPS_GRID:
                        readouts(M.qot_uncond(Rs, Rta, e),
                                 M.vec_rho(M.qot_uncond(Rs, Rta, e)),
                                 'QOTu', 'EPS', e)
                        if arc_ok:
                            Rc = M.qot_cond(Rs, ys, Rta, ya, e)
                            readouts(Rc, M.vec_rho(Rc), 'QOTc', 'EPS', e)
                        readouts(Rs, M.vec_rho(Rs), 'none', 'EPS', e)

                    # ---- Sinkhorn regularization, BOTH forms
                    for g in REG_GRID:
                        Vu = M.sinkhorn(S['V'], T['V'][arc], g)
                        rows.append({**base, 'knob': 'OTREG', 'value': g,
                                     'Method': 'COT-u',
                                     'BAC': fit(M.sinkhorn(S['X'], T['X'][arc], g),
                                                T['X'][ev])})
                        readouts(M.unvec_rho(Vu, Rs.shape[1]), Vu,
                                 'SDMu', 'OTREG', g)
                        if arc_ok:
                            rows.append({**base, 'knob': 'OTREG', 'value': g,
                                         'Method': 'COT-c',
                                         'BAC': fit(M.cond_sinkhorn(
                                             S['X'], ys, T['X'][arc], ya, g),
                                             T['X'][ev])})
                            Vc = M.cond_sinkhorn(S['V'], ys, T['V'][arc], ya, g)
                            readouts(M.unvec_rho(Vc, Rs.shape[1]), Vc,
                                     'SDM', 'OTREG', g)
            print(f'  {side} {src}->{tgt} done', flush=True)

    # ---- alpha, angle encoding only. Changing alpha changes the density
    #      matrices themselves, so the data has to be re-encoded per value.
    arows = []
    if a.enc == 'Ang':
        for al in ALPHA_GRID:
            for side in ('Right', 'Left'):
                Da = M.prep(side, 'Ang', al)
                for src, tgt in itertools.permutations(M.SITES, 2):
                    S, T = Da[src], Da[tgt]; ys = S['y']
                    for rot in range(rots):
                        for frac in FRACS:
                            arc, ev = M.split(T['mid'], rot, 'contig', frac, rots)
                            if not len(arc): continue
                            if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                            ya, ye = T['y'][arc], T['y'][ev]
                            Rs, Rta = S['R'], T['R'][arc]
                            for tag, Rtr in (('QOTu', M.qot_uncond(Rs, Rta)),
                                             ('none', Rs)):
                                A_, B_ = M.hs(Rtr, ys, Rs, Rta, ya, T['R'][ev], True)
                                arows.append(dict(
                                    Side=side, alpha=al, Source=src, Target=tgt,
                                    Rot=rot, AdaptFrac=frac, Method=f'{tag}-HSsrc',
                                    BAC=balanced_accuracy_score(
                                        ye, M.clf(rot).fit(A_, ys).predict(B_))))
            print(f'  alpha {al} done', flush=True)
        pd.DataFrame(arows).to_csv(f'{M.OUT}/alpha8_Ang.csv', index=False)
        print(f'wrote {M.OUT}/alpha8_Ang.csv')

    d = pd.DataFrame(rows)
    os.makedirs(M.OUT, exist_ok=True)
    fn = f'{M.OUT}/sweeps8_{a.enc}.csv'
    d.to_csv(fn, index=False)
    print(f'\nwrote {fn}')

    for knob, grid in (('EPS', EPS_GRID), ('OTREG', REG_GRID)):
        for frac in FRACS:
            q = d[(d.knob == knob) & (d.Side == 'Right') &
                  np.isclose(d.AdaptFrac, frac)]
            if not len(q): continue
            print(f'\n{knob}, right rail, {frac:.0%}, 8 rotations')
            print(f'  {"method":14s} ' + ' '.join(f'{g:g}'.rjust(11) for g in grid))
            for m_ in sorted(q.Method.unique()):
                cells = []
                for g in grid:
                    s_ = q[(q.Method == m_) & np.isclose(q.value, g)]
                    pp = s_.groupby(['Source', 'Target']).BAC.mean()
                    cells.append((f'{pp.mean():.3f}' if len(pp) else '-').rjust(11))
                print(f'  {m_:14s} ' + ' '.join(cells))


if __name__ == '__main__':
    main()
