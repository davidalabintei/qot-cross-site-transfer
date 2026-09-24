"""Does discarding absolute severity help cross-site transfer?

THE QUESTION
    Amplitude encoding divides each segment's eight-value vector by its own
    length, so a segment reading BFI 4, BTI 4, MLI 2, LRI 2 and one reading
    BFI 2, BTI 2, MLI 1, LRI 1 become the identical quantum state. Absolute
    severity is discarded, yet amplitude encoding gives the higher balanced
    accuracy at every readout.

    One explanation is that the four sites differ mainly in the LEVEL of their
    ballast indices rather than in the pattern between them, so normalizing
    deletes much of the domain shift before transport begins. That is a story
    rather than a measurement, and this script measures it.

THE DESIGN, three arms, so that magnitude is isolated from dimension
    Amp8    the encoding used throughout. Eight features, normalized, 8 x 8 rho,
            three qubits.
    Amp16z  the same eight features placed in the first eight of sixteen
            amplitudes with zeros in the rest, then normalized. 16 x 16 rho,
            four qubits. NO magnitude. This is the control for the dimension
            change alone.
    Amp16m  the same eight features in the first eight slots and the vector
            length ||x||, rescaled to the feature range, in the ninth, then
            normalized. 16 x 16 rho. Magnitude IS carried, inside the state.

    Amp16z against Amp16m isolates magnitude with dimension held fixed.
    Amp8 against Amp16z confirms that padding on its own changes nothing much.

HOW TO READ THE OUTPUT
    The quantity is QOT-u minus no transport, the same contrast as the main
    transfer table. If the explanation holds, that contrast is SMALLER under
    Amp16m than under Amp16z. If it is the same or larger, the explanation is
    wrong.

    A difference below about 0.01 is not evidence either way. The contrast
    itself moves by that much across readouts.

WHAT THE FULL RUN FOUND
    The explanation does not hold. The padding control passed, Amp8 and Amp16z
    agree throughout. Restoring magnitude did not shrink the contrast, and at
    the HS-4 and PCA readouts it widened it, by up to 0.028. Levels show the
    untransported model, rather than the transported one, as the one affected
    by carrying severity.

RUN
    The cleaned_<SITE>.csv input files are not distributed with this repository.
    See the README for the columns they must contain. Put this beside
    run_everything.py and those inputs, then

        OMP_NUM_THREADS=1 python3 test_magnitude.py --data .

    Right rail only, contiguous label stretch, eight starting positions, 5, 20
    and 50 percent, three readouts. Roughly 20 to 40 minutes.
    Output: magnitude_test.csv in qot_results, plus a table to the terminal.
    Nothing existing is touched.
"""
import os, sys, argparse, itertools
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import wilcoxon
import warnings; warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.')
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
os.environ['QOT_DATA'] = a.data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_everything as M
M.DATA = a.data

FRACS = [0.50] if a.quick else [0.05, 0.20, 0.50]
N_ROT = 8
SIDE = 'Right'


def rho_from(V):
    """Unit-trace rank-one density matrices from a stack of unit vectors."""
    return np.einsum('ni,nj->nij', V, V)


def encode(X, arm):
    """X is (n, 8) raw feature values on their own ordinal scales."""
    if arm == 'Amp8':
        V = X / np.linalg.norm(X, axis=1, keepdims=True)
    else:
        n = len(X)
        P = np.zeros((n, 16))
        P[:, :8] = X
        if arm == 'Amp16m':
            # vector length, rescaled so it sits on the same numeric range as
            # the features themselves and cannot dominate the normalization
            g = np.linalg.norm(X, axis=1)
            P[:, 8] = (g - g.min()) / (np.ptp(g) + 1e-12) * X.max()
        V = P / np.linalg.norm(P, axis=1, keepdims=True)
    return V, rho_from(V)


def main():
    rots = 2 if a.quick else N_ROT
    raw = {s: M.load(s, SIDE) for s in M.SITES}
    rows = []
    for arm in ('Amp8', 'Amp16z', 'Amp16m'):
        D = {}
        for s in M.SITES:
            X, y, mid = raw[s]
            V, R = encode(X, arm)
            D[s] = dict(X=X, y=y, mid=mid, R=R, V=M.vec_rho(R))
        for src, tgt in itertools.permutations(M.SITES, 2):
            S, T = D[src], D[tgt]; ys = S['y']
            for rot in range(rots):
                for frac in FRACS:
                    arc, ev = M.split(T['mid'], rot, 'contig', frac, rots)
                    if not len(arc): continue
                    if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                    ya, ye = T['y'][arc], T['y'][ev]
                    if min(np.bincount(ya, minlength=2)) < 2: continue
                    Rs, Rta = S['R'], T['R'][arc]
                    pb, km = M.pca_basis(S['V'], T['V'][arc], rot)
                    base = dict(Arm=arm, Source=src, Target=tgt,
                                Rot=rot, AdaptFrac=frac)
                    for tag, Rtr in (('QOTu', M.qot_uncond(Rs, Rta)),
                                     ('none', Rs)):
                        Vtr = M.vec_rho(Rtr)
                        for ro in M.READOUTS:
                            if ro == 'PCA':
                                A_, B_, _ = M.pca_apply(pb, km, Vtr,
                                                        T['V'][ev], ys, rot)
                            else:
                                A_, B_ = M.hs(Rtr, ys, Rs, Rta, ya,
                                              T['R'][ev], ro == 'HSsrc')
                            p = M.clf(rot).fit(A_, ys).predict(B_)
                            rows.append({**base, 'Method': f'{tag}-{ro}',
                                         'BAC': balanced_accuracy_score(ye, p)})
            print(f'  {arm} {src}->{tgt} done', flush=True)

    d = pd.DataFrame(rows)
    os.makedirs(M.OUT, exist_ok=True)
    fn = f'{M.OUT}/magnitude_test.csv'
    d.to_csv(fn, index=False)
    print(f'\nwrote {fn}\n')

    print('QOT-u minus no transport, right rail, eight folds')
    print('cell = mean difference (pairs improved, Wilcoxon p)\n')
    print(f'  {"arm":8s} {"readout":8s} ' +
          ' '.join(f'{f:.0%}'.rjust(22) for f in FRACS))
    for arm in ('Amp8', 'Amp16z', 'Amp16m'):
        for ro in M.READOUTS:
            cells = []
            for frac in FRACS:
                q = d[(d.Arm == arm) & np.isclose(d.AdaptFrac, frac)]
                A = q[q.Method == f'QOTu-{ro}'].groupby(['Source', 'Target']).BAC.mean()
                B = q[q.Method == f'none-{ro}'].groupby(['Source', 'Target']).BAC.mean()
                j = A.index.intersection(B.index)
                x = (A[j] - B[j]).values
                if len(x) < 3:
                    cells.append('-'.rjust(22)); continue
                p = wilcoxon(x).pvalue if np.ptp(x) > 0 else 1.0
                cells.append(f'{x.mean():+.3f} ({(x>0).sum()}/{len(x)}, {p:.3f})'.rjust(22))
            print(f'  {arm:8s} {ro:8s} ' + ' '.join(cells))

    print('\nLEVELS, mean BAC over the twelve pairs, so that a wider contrast')
    print('is not mistaken for a better model.\n')
    print(f'  {"arm":8s} {"readout":8s} ' +
          ' '.join((f'{f:.0%} QOT-u' + ' ' + f'{f:.0%} none').rjust(24) for f in FRACS))
    for arm in ('Amp8', 'Amp16z', 'Amp16m'):
        for ro in M.READOUTS:
            cells = []
            for frac in FRACS:
                q = d[(d.Arm == arm) & np.isclose(d.AdaptFrac, frac)]
                A = q[q.Method == f'QOTu-{ro}'].groupby(['Source', 'Target']).BAC.mean().mean()
                B = q[q.Method == f'none-{ro}'].groupby(['Source', 'Target']).BAC.mean().mean()
                cells.append(f'{A:.3f} / {B:.3f}'.rjust(24))
            print(f'  {arm:8s} {ro:8s} ' + ' '.join(cells))

    print('\nREAD IT LIKE THIS. Compare Amp16m against Amp16z, same readout and')
    print('same percentage. A smaller contrast under Amp16m means carrying')
    print('magnitude hurts transfer, which supports the explanation. The same or')
    print('larger means it does not. Amp8 against Amp16z shows whether the')
    print('padding alone did anything, and it should not have.')


if __name__ == '__main__':
    main()
