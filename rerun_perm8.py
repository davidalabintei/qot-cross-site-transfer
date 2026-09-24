"""Label permutation test at eight starting positions on the contiguous label
stretch, with the unconditional classical methods included.

WHAT IT CHECKS
    Every method is described as using either the source labels alone or the
    source and target labels together. This test confirms that each one uses
    what it is described as using, by rerunning the full protocol with labels
    shuffled.

WHY IT IS SEPARATE
    The permutation stage inside run_everything.py runs at four starting
    positions and on the randomly drawn label stretch. Every reported number
    uses eight positions and the contiguous stretch, so that stage cannot be
    reported beside the main tables. That stage also covers only the METHODS
    list, which has no COT-u and no COT-rho-u, and those are the unconditional
    classical comparators where the test is most informative.

WHAT IT RUNS, per encoding, right and left rail, contiguous label stretch
    8 starting positions, 5, 20 and 50 percent.
    kinds   real        source labels and target labels as they are
            srcperm     source labels shuffled, target labels intact
            arcperm     target labels in the labeled stretch shuffled,
                        source labels intact
    Two permutation replicates per fold.

    A fold is kept only when the labeled stretch holds at least two of each
    class, so that every method is scored on the same folds. That filter bites
    hardest at 5 percent. The script prints how many folds survive at each
    percentage.

HOW TO READ THE OUTPUT
    loss(src) = real minus srcperm. A method that genuinely transfers from the
                source site has to lose accuracy here.
    loss(arc) = real minus arcperm. Exactly zero means the method never reads a
                target label.

    TargetOnly is the control for the test itself. It must show loss(src) of
    exactly 0.0000 on every fold, since it never reads a source label. Anything
    else means the test is broken rather than the method.

RUN
    The cleaned_<SITE>.csv input files are not distributed with this repository.
    See the README for the columns they must contain.

        OMP_NUM_THREADS=1 python3 rerun_perm8.py --data . --enc Amp
        OMP_NUM_THREADS=1 python3 rerun_perm8.py --data . --enc Ang

    Three percentages, three kinds, two replicates, eight positions. Around 30
    to 50 minutes each.
    Output: perm8_<Enc>.csv in that encoding's results folder. Nothing written
    by the main run is touched.
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

FRACS = [0.50] if a.quick else [0.05, 0.20, 0.50]
N_ROT = 8
REPS = 1 if a.quick else 2


def extra_uncond(S, T, ys, ya, arc, ev, rot):
    """COT-u and COT-rho-u, which fold_preds does not produce.

    Neither reads a target label, so both must show loss(arc) of exactly zero.
    ya is passed in rather than read from T, so that under arcperm these see the
    shuffled labels exactly as fold_preds does. Only the HS-4 readout uses it.
    """
    p = {}
    p['COT-u'] = M.clf(rot).fit(
        M.sinkhorn(S['X'], T['X'][arc], M.OTREG), ys).predict(T['X'][ev])
    Vu = M.sinkhorn(S['V'], T['V'][arc], M.OTREG)
    Ru = M.unvec_rho(Vu, S['R'].shape[1])
    pb, km = M.pca_basis(S['V'], T['V'][arc], rot)
    for ro in M.READOUTS:
        if ro == 'PCA':
            A_, B_, _ = M.pca_apply(pb, km, Vu, T['V'][ev], ys, rot)
        else:
            A_, B_ = M.hs(Ru, ys, S['R'], T['R'][arc], ya,
                          T['R'][ev], ro == 'HSsrc')
        p[f'SDMu-{ro}'] = M.clf(rot).fit(A_, ys).predict(B_)
    return p


def main():
    rots = 2 if a.quick else N_ROT
    rows, kept = [], []
    for side in ('Right', 'Left'):
        D = PREP(side)
        for src, tgt in itertools.permutations(M.SITES, 2):
            S, T = D[src], D[tgt]
            for rot in range(rots):
                for frac in FRACS:
                    arc, ev = M.split(T['mid'], rot, 'contig', frac, rots)
                    if not len(arc): continue
                    if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                    if min(np.bincount(T['y'][arc], minlength=2)) < 2:
                        kept.append((side, frac, 0)); continue
                    kept.append((side, frac, 1))
                    ye = T['y'][ev]

                    draws = [('real', S['y'], None)]
                    for k in range(REPS):
                        g = np.random.default_rng(1000 * k + rot)
                        draws += [('srcperm', g.permutation(S['y']), None),
                                  ('arcperm', S['y'], g)]

                    for kind, ys, g in draws:
                        yt = T['y'].copy()
                        if kind == 'arcperm':
                            yt[arc] = g.permutation(yt[arc])
                        if len(np.unique(ys)) < 2: continue
                        p, ok, _ = M.fold_preds(S, T, ys, yt, arc, ev, rot)
                        p.update(extra_uncond(S, T, ys, yt[arc], arc, ev, rot))
                        for m_, pred in p.items():
                            rows.append(dict(
                                Enc=a.enc, Side=side, Source=src, Target=tgt,
                                Rot=rot, AdaptFrac=frac, kind=kind, Method=m_,
                                BAC=balanced_accuracy_score(ye, pred)))
            print(f'  {side} {src}->{tgt} done', flush=True)

    d = pd.DataFrame(rows)
    os.makedirs(M.OUT, exist_ok=True)
    fn = f'{M.OUT}/perm8_{a.enc}.csv'
    d.to_csv(fn, index=False)
    print(f'\nwrote {fn}')

    k = pd.DataFrame(kept, columns=['Side', 'AdaptFrac', 'ok'])
    print('\nfolds with at least two of each class in the labeled stretch')
    print(k.groupby(['Side', 'AdaptFrac']).ok.agg(['sum', 'count']).to_string())

    for side in ('Right', 'Left'):
        for frac in FRACS:
            q = d[(d.Side == side) & np.isclose(d.AdaptFrac, frac)]
            if not len(q): continue
            w = q.pivot_table(index=['Source', 'Target', 'Rot', 'Method'],
                              columns='kind', values='BAC')
            if not {'real', 'srcperm', 'arcperm'} <= set(w.columns): continue
            print(f'\n{side} rail, {a.enc}, {frac:.0%}, 8 rotations, contiguous arc')
            print(f'  {"method":14s} {"real":>7s} {"srcperm":>8s} {"arcperm":>8s} '
                  f'{"loss(src)":>10s} {"loss(arc)":>10s}')
            for m_ in sorted(w.index.get_level_values('Method').unique()):
                x = w.xs(m_, level='Method')
                print(f'  {m_:14s} {x["real"].mean():7.4f} '
                      f'{x["srcperm"].mean():8.4f} {x["arcperm"].mean():8.4f} '
                      f'{(x["real"] - x["srcperm"]).mean():+10.4f} '
                      f'{(x["real"] - x["arcperm"]).mean():+10.4f}')


if __name__ == '__main__':
    main()
