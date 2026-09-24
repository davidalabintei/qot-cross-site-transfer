#!/usr/bin/env python3
"""
================================================================================
 QOT CROSS-SITE TRANSFER  -  COMPLETE EXPERIMENT, ONE FILE
================================================================================

Runs the main grid of the experiment and prints all of it. This file imports
nothing from the repository. The other scripts import it, so the folds, the
encodings and the readouts are shared rather than reimplemented.

Not everything is here. The unconditional Sinkhorn comparators are produced by
run_uncond_classical.py. The sweep and permutation stages below run at four
starting positions and are superseded by rerun_sweeps8.py and rerun_perm8.py,
which run at the eight used everywhere else.

WHAT IT DOES, IN ORDER

  0  VERIFY      correctness checks (splits, isometry, Bures identity, no
                 evaluation data reaching any fit) AND a sensitivity control that
                 asks whether each readout can see a transport effect at all.
                 A null from a blind readout is not evidence, so this gates
                 everything after it.
  1  MAIN        16 methods x 2 rails x 2 arc constructions x 5 label budgets,
                 12 directed site pairs, 8 rotations.
  2  PERM        every method rerun with the SOURCE labels scrambled, and again
                 with the ARC labels scrambled. Separates methods that transfer
                 from methods that quietly do target-only learning.
  3  SWEEP       sensitivity to the two regularization constants, EPS in the
                 Bures map and the Sinkhorn epsilon in the classical baselines.
  4  BOOT        moving-block bootstrap 95% intervals for every method and every
                 paired contrast, block length measured from the data.
  5  DIAG        why the left rail behaves differently from the right.

TASK
  Per-rail profile exceedance. A BMP-EMP segment is defective when the maximum
  absolute 62 ft profile on that rail exceeds 0.4 in. Features are 8 ballast
  channels, own rail and center, one ballast index per pair.

METHODS   4 on raw features, then a 4 x 3 grid.
    transports  none | QOTu unconditional Bures | QOTc conditional Bures
                | SDM class-conditional Sinkhorn on vec(rho)
    readouts    HS4 source + arc class prototypes | HSsrc source prototypes only
                | PCA on vec(rho), K by cross-validation

  The `none` row is each transport's own control in the same readout, so
  QOTu-HSsrc minus none-HSsrc is the map and nothing else.

  "Uses" records what the WHOLE pipeline touches on the target, since no
  transport is not the same as no adaptation. HS4 uses the arc segments and their
  labels through its prototypes, PCA uses the arc segments alone through its
  basis, HSsrc uses neither.
  Only none-HSsrc and raw NoAdapt are true no-adaptation baselines.

--------------------------------------------------------------------------------
 HOW TO RUN
--------------------------------------------------------------------------------

  Requirements
      python 3.9+, numpy, pandas, scipy, scikit-learn
      pip install numpy pandas scipy scikit-learn

  Data
      Needs cleaned_HTL.csv, cleaned_PTT.csv, cleaned_RTT.csv, cleaned_WRM.csv
      in one folder. Point --data at that folder (or set QOT_DATA).

  Smoke test first, about 10 minutes, confirms it runs end to end
      python run_everything.py --data /path/to/csvs --quick

  Full run, about 2 to 3 hours on 2 cores
      python run_everything.py --data /path/to/csvs

  One stage at a time
      python run_everything.py --data /path/to/csvs --stage verify
      python run_everything.py --data /path/to/csvs --stage main
      python run_everything.py --data /path/to/csvs --stage perm
      python run_everything.py --data /path/to/csvs --stage sweep
      python run_everything.py --data /path/to/csvs --stage boot
      python run_everything.py --data /path/to/csvs --stage diag

  Faster on a multi-core machine
      OMP_NUM_THREADS=1 python run_everything.py --data /path/to/csvs

  Outputs
      ./qot_results/REPORT.txt     everything printed, in one file
      ./qot_results/*.csv          raw per-fold rows for independent checking

================================================================================
"""
import os, sys, argparse, itertools, time, textwrap
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────── configuration
KEEP = {'HTL': 2022100717, 'PTT': 2022100607, 'RTT': 2022100710, 'WRM': 2022100714}
SITES = list(KEEP)
SIDE_CFG = {
    'Left':  ('LProf62', ['BFI_L_Cat', 'BFI_C_Cat', 'BTI_L', 'BTI_C',
                          'MLI_L', 'MLI_C', 'LRI_L', 'LRI_C']),
    'Right': ('RProf62', ['BFI_R_Cat', 'BFI_C_Cat', 'BTI_R', 'BTI_C',
                          'MLI_R', 'MLI_C', 'LRI_R', 'LRI_C'])}
THRESH = 0.4            # inches, profile exceedance threshold
MIN_ROWS = 10           # geometry records required in a segment
BUFFER_MI = 200.0/5280  # 200 ft buffer either side of the evaluation block
EPS = 1e-9              # ridge and eigenvalue floor inside the Bures map
OTREG = 0.05            # Sinkhorn entropic regularization
K_GRID = (4, 8, 16, 32, 64)
FRACS = [.05, .10, .20, .30, .50]
N_ROT = 8
READOUTS = ['HS4', 'HSsrc', 'PCA']
TRANSPORTS = ['none', 'QOTu', 'QOTc', 'SDM']
METHODS = ['NoAdapt', 'TargetOnly', 'Source+Target', 'COT'] + \
          [f'{t}-{r}' for t in TRANSPORTS for r in READOUTS]
USES = {('none', 'HSsrc'): 'nothing', ('none', 'PCA'): 'arc segments',
        ('none', 'HS4'): 'arc labels', ('QOTu', 'HSsrc'): 'arc segments',
        ('QOTu', 'PCA'): 'arc segments', ('QOTu', 'HS4'): 'arc labels'}
for _r in READOUTS:
    USES[('QOTc', _r)] = 'arc labels'
    USES[('SDM', _r)] = 'arc labels'
RAW_USES = {'NoAdapt': 'nothing', 'TargetOnly': 'arc labels',
            'Source+Target': 'arc labels', 'COT': 'arc labels'}
def uses(m):
    return RAW_USES.get(m) or USES[tuple(m.split('-'))]

CONTRASTS = [(f'{t}-{r}', f'none-{r}') for t in ('QOTu', 'QOTc', 'SDM') for r in READOUTS] \
          + [(f'QOTu-{r}', f'SDM-{r}') for r in READOUTS] \
          + [('QOTu-HSsrc', 'TargetOnly'), ('TargetOnly', 'NoAdapt'), ('COT', 'NoAdapt')]

DATA = os.environ.get('QOT_DATA', '.')
OUT = './qot_results'
_LOG = None
def L(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    if _LOG:
        _LOG.write(s + '\n'); _LOG.flush()
def H(title, ch='='):
    L('\n' + ch*94); L(title); L(ch*94)


# ──────────────────────────────────────────────────────────────────────── data
def load(site, side):
    """(X, y, mid) for one site, in milepost order. Reads the CLEANED file."""
    prof, feats = SIDE_CFG[side]
    p = os.path.join(DATA, f'cleaned_{site}.csv')
    if not os.path.exists(p):
        sys.exit(f'ERROR: {p} not found. Point --data at the folder holding the '
                 f'four cleaned_<SITE>.csv files.')
    d = pd.read_csv(p, usecols=lambda c: c in feats+[prof, 'BMP', 'EMP', 'run_id'],
                    low_memory=False)
    d = d[d.run_id == KEEP[site]].dropna(subset=feats+[prof, 'BMP', 'EMP'])
    g = d.groupby(['BMP', 'EMP'], sort=True)
    keep = g.size() >= MIN_ROWS
    Xf = g[feats].first()[keep]
    y = (g[prof].apply(lambda v: v.abs().max())[keep] > THRESH).astype(int).values
    assert (Xf.index == g.size()[keep].index).all()
    mid = np.array([(b+e)/2 for b, e in Xf.index])
    o = np.argsort(mid, kind='stable')
    return Xf.values.astype(float)[o], y[o], mid[o]


def amp_rho(X):
    """Amplitude encoding. 8 features -> unit vector -> rank-one 8x8 density matrix."""
    n = np.linalg.norm(X, axis=1); P = X.copy().astype(float)
    g = n > 1e-12; P[g] = X[g]/n[g, None]; P[~g] = 0; P[~g, 0] = 1.0
    return np.einsum('ni,nj->nij', P, P)


def vec_rho(R):
    """Hilbert-Schmidt isometry, so Euclidean distance here IS the HS distance."""
    d = R.shape[1]; iu = np.triu_indices(d, 1); off = R[:, iu[0], iu[1]]
    return np.hstack([np.diagonal(R, axis1=1, axis2=2).real,
                      np.sqrt(2)*off.real, np.sqrt(2)*off.imag])


def unvec_rho(V, d):
    """Inverse of vec_rho. A Sinkhorn barycenter is a convex combination of target
    density matrices, so unvecking it gives a valid mixed state the HS readout
    can consume exactly like a transported one."""
    iu = np.triu_indices(d, 1); m = len(iu[0])
    R = np.zeros((len(V), d, d), complex)
    R[:, np.arange(d), np.arange(d)] = V[:, :d]
    off = (V[:, d:d+m] + 1j*V[:, d+m:d+2*m])/np.sqrt(2)
    R[:, iu[0], iu[1]] = off; R[:, iu[1], iu[0]] = off.conj()
    return R


# ────────────────────────────────────────────────────────────────────── splits
def split(mid, rot, arc_mode, frac, n_rot=N_ROT):
    """Evaluation block is a contiguous half of the site. Its start sweeps from 0
    to n-n_eval across the rotations, so the union of evaluation blocks covers the
    WHOLE site. A 200 ft buffer removes pool segments beside the block boundaries."""
    n = len(mid); ne = n//2
    start = int(round(rot/max(n_rot-1, 1) * (n-ne)))
    ev = np.arange(start, start+ne)
    pool = np.r_[np.arange(0, start), np.arange(start+ne, n)]
    lo, hi = mid[ev[0]], mid[ev[-1]]
    pool = pool[(mid[pool] < lo-BUFFER_MI) | (mid[pool] > hi+BUFFER_MI)]
    k = min(int(round(frac*n)), len(pool))
    if k <= 0:
        return np.array([], int), ev
    if arc_mode == 'random':
        arc = np.random.default_rng(1000*rot+7).permutation(pool)[:k]
    else:
        runs = np.split(pool, np.flatnonzero(np.diff(pool) != 1)+1)
        arc = []
        for i, rn in enumerate(runs):
            ki = k-sum(len(x) for x in arc) if i == len(runs)-1 \
                 else int(round(k*len(rn)/len(pool)))
            ki = min(max(ki, 0), len(rn)); s = (len(rn)-ki)//2
            arc.append(rn[s:s+ki])
        arc = np.concatenate(arc) if arc else np.array([], int)
    return np.sort(arc), ev


# ────────────────────────────────────────────────────────────────── transports
def msqrt(A):
    A = (A+A.conj().T)/2; e, v = np.linalg.eigh(A)
    return v@np.diag(np.sqrt(np.clip(e.real, 0, None)))@v.conj().T

def minvsqrt(A, eps):
    A = (A+A.conj().T)/2; e, v = np.linalg.eigh(A)
    return v@np.diag(1/np.sqrt(np.clip(e.real, eps, None)))@v.conj().T

def bures_map(A, B, d, eps=EPS):
    Ar = A+eps*np.eye(d); Br = B+eps*np.eye(d)
    sq = msqrt(Ar); inv = minvsqrt(Ar, eps)
    return inv@msqrt(sq@Br@sq)@inv

def apply_map(M_, R):
    Z = np.einsum('ij,njk,lk->nil', M_, R, M_.conj())
    tr = np.trace(Z, axis1=1, axis2=2).real
    g = tr > 1e-12; Z[g] /= tr[g, None, None]
    return Z

def qot_uncond(Rs, Rta, eps=EPS):
    """Pooled source mean onto pooled arc mean. Uses the arc segments alone."""
    if not len(Rs) or not len(Rta): return Rs.copy()
    return apply_map(bures_map(Rs.mean(0), Rta.mean(0), Rs.shape[1], eps), Rs)

def qot_cond(Rs, ys, Rta, ya, eps=EPS):
    """One Bures map per class. Uses the arc segments and their labels."""
    out = Rs.copy(); d = Rs.shape[1]
    for c in (0, 1):
        si = np.flatnonzero(ys == c); ti = np.flatnonzero(ya == c)
        if not len(si) or not len(ti): continue
        out[si] = apply_map(bures_map(Rs[si].mean(0), Rta[ti].mean(0), d, eps), Rs[si])
    return out

def sinkhorn(A, B, reg, it=300):
    C = cdist(A, B, 'sqeuclidean'); C /= C.max()+1e-12
    a = np.ones(len(A))/len(A); b = np.ones(len(B))/len(B)
    K = np.maximum(np.exp(-C/reg), 1e-300); u = np.ones(len(A)); v = np.ones(len(B))
    for _ in range(it):
        u0 = u; u = a/(K@v+1e-300); v = b/(K.T@u+1e-300)
        if np.max(np.abs(u-u0)) < 1e-7: break
    T = (u[:, None]*K)*v[None, :]
    return T@B/(T.sum(1, keepdims=True)+1e-12)

def cond_sinkhorn(Xs, ys, Xt, ya, reg=OTREG):
    out = Xs.copy()
    for c in (0, 1):
        si = np.flatnonzero(ys == c); ti = np.flatnonzero(ya == c)
        if not len(si) or not len(ti): continue
        out[si] = sinkhorn(Xs[si], Xt[ti], reg)
    return out


# ──────────────────────────────────────────────────────────────────── readouts
def meanclass(R, y, c):
    i = np.flatnonzero(y == c); return R[i].mean(0) if len(i) else None

def hs(Rtr, ys, Rs, Rta, ya, Rte, src_only):
    """Hilbert-Schmidt prototype readout, Tr(rho P) against class-mean states."""
    sm = Rs.mean(0); tm = Rta.mean(0) if len(Rta) else sm
    spec = [(Rs, ys, 0, sm), (Rs, ys, 1, sm)]
    if not src_only:
        spec += [(Rta, ya, 0, tm), (Rta, ya, 1, tm)]
    P = [(meanclass(R, y, c) if meanclass(R, y, c) is not None else fb)
         for R, y, c, fb in spec]
    f = lambda R: np.stack([np.einsum('nij,ji->n', R, p).real for p in P], 1)
    return f(Rtr), f(Rte)

def clf(seed=0):
    return Pipeline([('sc', StandardScaler()),
                     ('lr', LogisticRegression(max_iter=3000, class_weight='balanced',
                                               random_state=seed))])

def pca_basis(V_src_untransported, V_arc, seed=0):
    """Fitted ONCE on the untransported source plus the arc, then reused for every
    transport, so a QOT-minus-none contrast changes the coordinates and not the basis."""
    Vb = np.vstack([V_src_untransported, V_arc]) if len(V_arc) else V_src_untransported
    kmax = min(max(K_GRID), Vb.shape[1], len(Vb))
    return PCA(n_components=kmax, random_state=seed).fit(Vb), kmax

def pca_apply(p, kmax, V_tr, V_te, ys, seed=0):
    As, Ate = p.transform(V_tr), p.transform(V_te)
    grid = [k for k in K_GRID if k <= kmax] or [kmax]
    if min(np.bincount(ys, minlength=2)) < 3:
        k = min(8, kmax); return As[:, :k], Ate[:, :k], k
    best, bk = -np.inf, grid[0]
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    for k in grid:
        s = [balanced_accuracy_score(ys[te], clf(seed).fit(As[tr, :k], ys[tr])
                                     .predict(As[te, :k])) for tr, te in cv.split(As, ys)]
        if np.mean(s) > best: best, bk = float(np.mean(s)), k
    return As[:, :bk], Ate[:, :bk], bk

def metrics(ye, yp):
    return dict(BAC=balanced_accuracy_score(ye, yp),
                Precision=precision_score(ye, yp, zero_division=0),
                Recall=recall_score(ye, yp, zero_division=0))
NANM = dict(BAC=np.nan, Precision=np.nan, Recall=np.nan)


def fold_preds(S, T, ys, yt, arc, ev, rot, need_all=True):
    """Predictions for all 16 methods on one fold. Returns (dict, arc_ok)."""
    ya, ye = yt[arc], yt[ev]
    arc_ok = len(arc) > 0 and min(np.bincount(ya, minlength=2)) >= 2
    Rs, Rta = S['R'], T['R'][arc]
    p = {}
    p['NoAdapt'] = clf(rot).fit(S['X'], ys).predict(T['X'][ev])
    if arc_ok:
        p['TargetOnly'] = clf(rot).fit(T['X'][arc], ya).predict(T['X'][ev])
        p['Source+Target'] = clf(rot).fit(np.vstack([S['X'], T['X'][arc]]),
                                          np.r_[ys, ya]).predict(T['X'][ev])
        p['COT'] = clf(rot).fit(cond_sinkhorn(S['X'], ys, T['X'][arc], ya),
                                ys).predict(T['X'][ev])
    pb, km = pca_basis(S['V'], T['V'][arc], rot)
    Vsdm = cond_sinkhorn(S['V'], ys, T['V'][arc], ya) if arc_ok else None
    tr = {'none': Rs, 'QOTu': qot_uncond(Rs, Rta),
          'QOTc': qot_cond(Rs, ys, Rta, ya) if arc_ok else None,
          'SDM': unvec_rho(Vsdm, Rs.shape[1]) if arc_ok else None}
    ks = {}
    for t in TRANSPORTS:
        for ro in READOUTS:
            nm = f'{t}-{ro}'
            if ((t in ('QOTc', 'SDM')) or ro == 'HS4') and not arc_ok:
                continue
            if ro == 'PCA':
                V = Vsdm if t == 'SDM' else vec_rho(tr[t])
                A_, B_, k = pca_apply(pb, km, V, T['V'][ev], ys, rot)
                ks[nm] = k
            else:
                A_, B_ = hs(tr[t], ys, Rs, Rta, ya, T['R'][ev], ro == 'HSsrc')
            p[nm] = clf(rot).fit(A_, ys).predict(B_)
    return p, arc_ok, ks


def prep(side):
    D = {}
    for s in SITES:
        X, y, mid = load(s, side)
        R = amp_rho(X)
        D[s] = dict(X=X, y=y, mid=mid, R=R, V=vec_rho(R))
    return D


# ═══════════════════════════════════════════════════════════ 0. VERIFY
def stage_verify(quick):
    H('STAGE 0  VERIFY   correctness, then whether the readouts can see anything')
    FAIL = []
    def chk(name, ok, det=''):
        L(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {det}' if det else ''))
        if not ok: FAIL.append(name)
    rng = np.random.default_rng(0)

    L('\n--- splits ---')
    mid = np.sort(rng.uniform(0, 40, 900))
    cover = np.zeros(len(mid), bool); disj = True; half = True; worst = np.inf
    for rot in range(N_ROT):
        for mode in ('random', 'contig'):
            for f in FRACS:
                arc, ev = split(mid, rot, mode, f)
                disj &= len(np.intersect1d(arc, ev)) == 0
                half &= len(ev) == len(mid)//2 and bool(np.all(np.diff(ev) == 1))
                if len(arc):
                    lo, hi = mid[ev[0]], mid[ev[-1]]
                    out = (mid[arc] < lo) | (mid[arc] > hi)
                    disj &= bool(out.all())
                    worst = min(worst, np.minimum(np.abs(mid[arc]-lo),
                                                  np.abs(mid[arc]-hi))[out].min())
        cover[split(mid, rot, 'random', .5)[1]] = True
    chk('arc and eval never overlap; arc always outside the eval block', disj)
    chk('eval block is exactly half the site and contiguous', half)
    chk('EVAL BLOCKS COVER THE WHOLE SITE across rotations', cover.all(),
        f'{cover.mean():.1%} of segments appear in some eval block')
    chk('200 ft buffer respected', worst >= BUFFER_MI-1e-12,
        f'closest arc segment {worst*5280:.0f} ft from the block')
    rr = split(mid, 4, 'random', .3)[0]; cc = split(mid, 4, 'contig', .3)[0]
    nr = lambda a: len(np.split(a, np.flatnonzero(np.diff(a) != 1)+1))
    chk('the two arc modes really differ', nr(cc) < nr(rr),
        f'contig {nr(cc)} runs vs random {nr(rr)} runs, same size {len(cc)}')

    L('\n--- encoding and vectorisation ---')
    X = rng.uniform(1, 5, (50, 8)); R = amp_rho(X); V = vec_rho(R)
    chk('rho has unit trace', np.allclose(np.trace(R, axis1=1, axis2=2), 1.0))
    chk('rho is PSD', np.linalg.eigvalsh(R).min() > -1e-12)
    chk('unvec(vec(rho)) == rho', np.allclose(unvec_rho(V, 8), R))
    hsn = np.linalg.norm(R[0]-R[1], 'fro'); eu = np.linalg.norm(V[0]-V[1])
    chk('vec_rho is a Hilbert-Schmidt isometry', abs(hsn-eu) < 1e-10,
        f'HS {hsn:.6f} vs Euclidean {eu:.6f}')
    iu = np.triu_indices(8, 1)
    Vb = np.hstack([np.diagonal(R, axis1=1, axis2=2).real, R[:, iu[0], iu[1]].real,
                    np.zeros((len(R), 28))])
    chk('and dropping the sqrt(2) would break it (check is not vacuous)',
        abs(hsn-np.linalg.norm(Vb[0]-Vb[1])) > 1e-6)

    L('\n--- the Bures map ---')
    A = amp_rho(rng.uniform(1, 5, (200, 8))).mean(0)
    B = amp_rho(rng.uniform(1, 5, (200, 8))).mean(0)
    Mm = bures_map(A, B, 8)
    chk('M rho_A M^dag == rho_B for the class means',
        np.allclose(Mm@A@Mm.conj().T, B, atol=1e-7))
    o = apply_map(Mm, amp_rho(rng.uniform(1, 5, (30, 8))))
    chk('transported segments keep unit trace and stay PSD',
        np.allclose(np.trace(o, axis1=1, axis2=2).real, 1.0)
        and np.linalg.eigvalsh(o).min() > -1e-9)

    L('\n--- Sinkhorn output is still a density matrix ---')
    Rs = amp_rho(rng.uniform(1, 5, (60, 8))); Rt = amp_rho(rng.uniform(2, 6, (40, 8)))
    ys = np.r_[np.zeros(30, int), np.ones(30, int)]
    ya = np.r_[np.zeros(20, int), np.ones(20, int)]
    Rsd = unvec_rho(cond_sinkhorn(vec_rho(Rs), ys, vec_rho(Rt), ya), 8)
    chk('unvec(Sinkhorn barycenter) has unit trace and is PSD',
        np.allclose(np.trace(Rsd, axis1=1, axis2=2).real, 1.0, atol=1e-8)
        and np.linalg.eigvalsh(Rsd).min() > -1e-9)

    L('\n--- nothing in the evaluation half reaches any fit ---')
    ev = np.arange(20, 40); arc = np.arange(0, 20)
    ya20 = np.r_[np.zeros(10, int), np.ones(10, int)]
    Rt2 = Rt.copy(); Rt2[ev] = amp_rho(rng.uniform(50, 90, (len(ev), 8)))
    chk('unconditional Bures map unchanged when eval is replaced',
        np.allclose(qot_uncond(Rs, Rt[arc]), qot_uncond(Rs, Rt2[arc])))
    chk('conditional Bures map unchanged when eval is replaced',
        np.allclose(qot_cond(Rs, ys, Rt[arc], ya20), qot_cond(Rs, ys, Rt2[arc], ya20)))
    chk('conditional Sinkhorn unchanged when eval is replaced',
        np.allclose(cond_sinkhorn(vec_rho(Rs), ys, vec_rho(Rt)[arc], ya20),
                    cond_sinkhorn(vec_rho(Rs), ys, vec_rho(Rt2)[arc], ya20)))
    p1, k1 = pca_basis(vec_rho(Rs), vec_rho(Rt)[arc])
    p2, k2 = pca_basis(vec_rho(Rs), vec_rho(Rt2)[arc])
    chk('PCA basis unchanged when eval is replaced',
        np.allclose(np.abs(p1.components_), np.abs(p2.components_)) and k1 == k2)
    chk('HS training features unchanged when eval is replaced',
        np.allclose(hs(Rs, ys, Rs, Rt[arc], ya20, Rt[ev], False)[0],
                    hs(Rs, ys, Rs, Rt2[arc], ya20, Rt2[ev], False)[0]))

    L('\n--- the readouts touch what the Uses column claims ---')
    ya2 = 1-ya20
    chk('HSsrc ignores arc labels',
        np.allclose(hs(Rs, ys, Rs, Rt[arc], ya20, Rt[ev], True)[0],
                    hs(Rs, ys, Rs, Rt[arc], ya2, Rt[ev], True)[0]))
    chk('HS4 does NOT ignore arc labels',
        not np.allclose(hs(Rs, ys, Rs, Rt[arc], ya20, Rt[ev], False)[0],
                        hs(Rs, ys, Rs, Rt[arc], ya2, Rt[ev], False)[0]))
    chk('the unconditional map ignores arc labels',
        np.allclose(qot_uncond(Rs, Rt[arc]), qot_uncond(Rs, Rt[arc])))
    chk('the conditional map does NOT',
        not np.allclose(qot_cond(Rs, ys, Rt[arc], ya20),
                        qot_cond(Rs, ys, Rt[arc], ya2)))

    H('SENSITIVITY CONTROL   can each readout SEE a transport effect?', '-')
    L(textwrap.dedent('''
      A known gap A is applied to the target. Under amplitude encoding x -> Ax sends
      rho to A rho A'/tr, which is exactly the form of the Bures map, so a correct
      transport can undo it. Recovery is measured against the readout's own ceiling
      and floor, both with NO transport, gap off and gap on.

          recovery = (transport - floor) / (ceiling - floor)

      A readout scoring near zero here cannot be used to report a null, since its
      null would say nothing about the method.
      '''))
    draws = 1 if quick else 3
    acc = {(ro, g, t): [] for ro in READOUTS for g in ('off', 'on')
           for t in ('none', 'QOTu', 'QOTc')}
    for site in SITES:
        Xa, ya_, _ = load(site, 'Right'); h = len(Xa)//2
        Xs_, ys_ = Xa[:h], ya_[:h]; Xt0, yt_ = Xa[h:], ya_[h:]
        if min(np.bincount(ys_, minlength=2)) < 2: continue
        for dr in range(draws):
            g_ = np.random.default_rng(100+dr)
            pm = g_.permutation(len(Xt0)); na = int(0.3*len(Xt0))
            arc_, ev_ = pm[:na], pm[na:]
            if min(np.bincount(yt_[arc_], minlength=2)) < 2: continue
            if min(np.bincount(yt_[ev_], minlength=2)) < 2: continue
            A_ = np.diag(np.exp(g_.uniform(np.log(0.55), np.log(1.8), 8)))
            for gap, Xt in (('off', Xt0), ('on', Xt0@A_.T)):
                Rs_ = amp_rho(Xs_); Rt_ = amp_rho(Xt); Vt_ = vec_rho(Rt_)
                pb, km = pca_basis(vec_rho(Rs_), Vt_[arc_])
                for t in ('none', 'QOTu', 'QOTc'):
                    Rtr = (Rs_ if t == 'none' else qot_uncond(Rs_, Rt_[arc_])
                           if t == 'QOTu' else qot_cond(Rs_, ys_, Rt_[arc_], yt_[arc_]))
                    for ro in READOUTS:
                        if ro == 'PCA':
                            a1, b1, _ = pca_apply(pb, km, vec_rho(Rtr), Vt_[ev_], ys_)
                        else:
                            a1, b1 = hs(Rtr, ys_, Rs_, Rt_[arc_], yt_[arc_],
                                        Rt_[ev_], ro == 'HSsrc')
                        acc[(ro, gap, t)].append(balanced_accuracy_score(
                            yt_[ev_], clf().fit(a1, ys_).predict(b1)))
    mn = lambda k: float(np.mean(acc[k])) if acc[k] else np.nan
    L(f'  averaged over {len(acc[("PCA","off","none")])} (site, draw) cells\n')
    L(f'  {"readout":8s} {"ceiling":>8s} {"floor":>8s} {"QOTu":>8s} {"QOTc":>8s}'
      f' {"rec(QOTu)":>10s} {"rec(QOTc)":>10s}')
    for ro in READOUTS:
        c_, f_ = mn((ro, 'off', 'none')), mn((ro, 'on', 'none')); den = c_-f_
        ru = (mn((ro, 'on', 'QOTu'))-f_)/den if den > 0.01 else np.nan
        rc = (mn((ro, 'on', 'QOTc'))-f_)/den if den > 0.01 else np.nan
        best = np.nanmax([ru, rc]) if den > 0.01 else np.nan
        L(f'  {ro:8s} {c_:8.3f} {f_:8.3f} {mn((ro,"on","QOTu")):8.3f} '
          f'{mn((ro,"on","QOTc")):8.3f} {ru:10.2f} {rc:10.2f}')
        chk(f'{ro} can see a transport effect (recovery >= 0.15)',
            (not np.isnan(best)) and best >= 0.15, f'best recovery {best:.2f}')
    L('')
    chk('the injected gap actually degrades performance (control is live)',
        all(mn((ro, 'off', 'none'))-mn((ro, 'on', 'none')) > 0.01 for ro in READOUTS))
    H(f'{len(FAIL)} failed check(s)' + (': '+'; '.join(FAIL) if FAIL else '  — CLEAR'), '-')
    return len(FAIL) == 0


# ═══════════════════════════════════════════════════════════ 1. MAIN
def stage_main(quick):
    H('STAGE 1  MAIN   16 methods, 2 rails, 2 arc constructions, 5 budgets')
    rots = 3 if quick else N_ROT
    fracs = [.05, .50] if quick else FRACS
    allr = []
    for side in ('Right', 'Left'):
        D = prep(side)
        L(f'\n{side} rail segments:  ' +
          '   '.join(f'{s} n={len(D[s]["y"])} pos={int(D[s]["y"].sum())} '
                     f'({D[s]["y"].mean():.1%})' for s in SITES))
        for arc_mode in ('contig', 'random'):
            rows = []
            for src, tgt in itertools.permutations(SITES, 2):
                S, T = D[src], D[tgt]
                for rot in range(rots):
                    for frac in fracs:
                        arc, ev = split(T['mid'], rot, arc_mode, frac, rots)
                        if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                        p, arc_ok, ks = fold_preds(S, T, S['y'], T['y'], arc, ev, rot)
                        ye = T['y'][ev]
                        base = dict(Side=side, Arc=arc_mode, Source=src, Target=tgt,
                                    Rot=rot, AdaptFrac=frac, ArcOK=int(arc_ok),
                                    nEval=len(ev), nEvalPos=int(ye.sum()),
                                    nArc=len(arc),
                                    nArcPos=int(T['y'][arc].sum()) if len(arc) else 0)
                        for m in METHODS:
                            met = metrics(ye, p[m]) if m in p else NANM
                            rows.append({**base, 'Method': m, 'Uses': uses(m),
                                         'K': ks.get(m, np.nan), **met})
            d = pd.DataFrame(rows)
            d.to_csv(f'{OUT}/main_{side}_{arc_mode}.csv', index=False)
            allr.append(d)
            H(f'{side} rail, {arc_mode} arc   mean BAC over 12 pairs '
              f'(sd across pairs) [sd across the 4 target sites]', '-')
            L(f'  {"method":16s} {"uses":>14s} ' +
              ' '.join(f'{f:.0%}'.rjust(23) for f in fracs) + f'{"n":>7s}')
            for m in METHODS:
                q = d[d.Method == m]
                if q.BAC.notna().sum() == 0: continue
                cells = []
                for f in fracs:
                    s_ = q[np.isclose(q.AdaptFrac, f)]
                    pp = s_.groupby(['Source', 'Target']).BAC.mean().dropna()
                    st = s_.groupby('Target').BAC.mean().dropna()
                    cells.append((f'{pp.mean():.3f} ({pp.std():.3f}) [{st.std():.3f}]'
                                  if len(pp) else '-').rjust(23))
                L(f'  {m:16s} {uses(m):>14s} ' + ' '.join(cells)
                  + f'{int(q.BAC.notna().sum()):7d}')
            _contrasts(d, fracs)
    return pd.concat(allr, ignore_index=True)


def _contrasts(d, fracs):
    pm = d.groupby(['Source', 'Target', 'AdaptFrac', 'Method']).BAC.mean().reset_index()
    L(f'\n  ISOLATED CONTRASTS   mean  wins/12  p(12 pairs)  p(4 sites, floor 0.125)')
    L(f'  {"contrast":34s} ' + ' '.join(f'{f:.0%}'.rjust(23) for f in fracs))
    for a, b in CONTRASTS:
        cells = []
        for f in fracs:
            w = pm[np.isclose(pm.AdaptFrac, f)].pivot_table(
                index=['Source', 'Target'], columns='Method', values='BAC')
            if a not in w or b not in w: cells.append('-'.rjust(23)); continue
            x = (w[a]-w[b]).dropna()
            if len(x) < 3: cells.append('-'.rjust(23)); continue
            st = x.groupby(level='Target').mean()
            p12 = wilcoxon(x).pvalue if x.abs().sum() > 0 else 1.0
            p4 = wilcoxon(st).pvalue if len(st) >= 3 and st.abs().sum() > 0 else np.nan
            cells.append(f'{x.mean():+.3f} {int((x>0).sum())}/{len(x)} '
                         f'{p12:.3f} {p4:.2f}'.rjust(23))
        L(f'  {a+" - "+b:34s} ' + ' '.join(cells))
    L('\n  PER TARGET SITE   QOTu-HSsrc minus none-HSsrc  (n = 4 independent units)')
    for f in fracs:
        w = pm[np.isclose(pm.AdaptFrac, f)].pivot_table(
            index=['Source', 'Target'], columns='Method', values='BAC')
        if 'QOTu-HSsrc' not in w or 'none-HSsrc' not in w: continue
        st = (w['QOTu-HSsrc']-w['none-HSsrc']).groupby(level='Target').mean()
        L(f'    {f:5.0%}  ' + '  '.join(f'{k} {v:+.3f}' for k, v in st.items())
          + f'   |  {int((st>0).sum())}/4 positive')


# ═══════════════════════════════════════════════════════════ 2. PERM
def stage_perm(quick):
    H('STAGE 2  PERMUTATION   does the method need the SOURCE, or the ARC labels?')
    L(textwrap.dedent('''
      loss(src)  real minus source-labels-scrambled. A method that genuinely
                 transfers from the source site must lose accuracy here.
      loss(arc)  real minus arc-labels-scrambled. Exactly zero means the method
                 never touches a target label.
      TargetOnly must show loss(src) of exactly 0.0000 on every fold, which is the
      built-in control confirming the test itself is sound.
      '''))
    rots = 2 if quick else 4
    reps = 1 if quick else 2
    fracs = [.50] if quick else [.05, .20, .50]
    out = []
    for side in ('Right', 'Left'):
        D = prep(side); rows = []
        for src, tgt in itertools.permutations(SITES, 2):
            S, T = D[src], D[tgt]
            for rot in range(rots):
                for frac in fracs:
                    arc, ev = split(T['mid'], rot, 'random', frac, rots)
                    if not len(arc): continue
                    if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                    if min(np.bincount(T['y'][arc], minlength=2)) < 2: continue
                    ye = T['y'][ev]
                    draws = [('real', S['y'], None)]
                    for k in range(reps):
                        g = np.random.default_rng(1000*k+rot)
                        draws += [('srcperm', g.permutation(S['y']), None),
                                  ('arcperm', S['y'], g)]
                    for kind, ys, g in draws:
                        yt = T['y'].copy()
                        if kind == 'arcperm':
                            yt[arc] = g.permutation(yt[arc])
                        if len(np.unique(ys)) < 2: continue
                        p, ok, _ = fold_preds(S, T, ys, yt, arc, ev, rot)
                        for m in METHODS:
                            if m not in p: continue
                            rows.append(dict(Side=side, Source=src, Target=tgt, Rot=rot,
                                             AdaptFrac=frac, kind=kind, Method=m,
                                             BAC=balanced_accuracy_score(ye, p[m])))
        d = pd.DataFrame(rows); d.to_csv(f'{OUT}/perm_{side}.csv', index=False)
        out.append(d)
        piv = d.pivot_table(index='Method', columns='kind', values='BAC')
        w = d.pivot_table(index=['Source', 'Target', 'Rot', 'AdaptFrac', 'Method'],
                          columns='kind', values='BAC')
        H(f'{side} rail', '-')
        L(f'  {"method":16s} {"uses":>14s} {"real":>7s} {"srcperm":>8s} {"arcperm":>8s} '
          f'{"loss(src)":>10s} {"p":>8s} {"loss(arc)":>10s}   verdict')
        for m in METHODS:
            if m not in piv.index: continue
            r = piv.loc[m]; x = w.xs(m, level='Method').dropna()
            if not len(x): continue
            ls = x['real']-x['srcperm']
            pv = wilcoxon(ls).pvalue if ls.abs().sum() > 0 else 1.0
            la = (x['real']-x['arcperm']).mean()
            v = ('uses the source' if ls.mean() > 0.03 else
                 'SOURCE IRRELEVANT' if ls.mean() < 0.01 else 'weak')
            L(f'  {m:16s} {uses(m):>14s} {r["real"]:7.4f} {r["srcperm"]:8.4f} '
              f'{r["arcperm"]:8.4f} {ls.mean():+10.4f} {pv:8.4f} {la:+10.4f}   {v}')
    return out


# ═══════════════════════════════════════════════════════════ 3. SWEEP
def stage_sweep(quick):
    H('STAGE 3  SWEEPS   sensitivity to the two regularization constants')
    EPS_GRID = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2] if quick else \
               [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2]
    REG_GRID = [0.001, 0.01, 0.05, 0.5] if quick else [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    rots = 2 if quick else 4
    fracs = [.50] if quick else [.05, .50]
    D = prep('Right')
    L('\n  How close to singular are the class-mean density matrices?')
    L('  (EPS can only matter if some eigenvalue is comparable to it)\n')
    L(f'  {"site":6s} {"class":>6s} {"n":>6s} ' + ' '.join(f'e{i}'.rjust(9) for i in range(8)))
    worst = np.inf
    for s in SITES:
        for c in (0, 1):
            i = np.flatnonzero(D[s]['y'] == c)
            if len(i) < 2: continue
            e = np.linalg.eigvalsh(D[s]['R'][i].mean(0))[::-1]
            worst = min(worst, e.min())
            L(f'  {s:6s} {c:6d} {len(i):6d} ' + ' '.join(f'{v:9.2e}' for v in e))
    L(f'\n  smallest eigenvalue anywhere = {worst:.2e}, default EPS = {EPS:.0e}, '
      f'ratio {worst/EPS:.0f}x')

    rows = []
    for src, tgt in itertools.permutations(SITES, 2):
        S, T = D[src], D[tgt]
        for rot in range(rots):
            for frac in fracs:
                arc, ev = split(T['mid'], rot, 'contig', frac, rots)
                if not len(arc): continue
                if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                ya, ye = T['y'][arc], T['y'][ev]
                if min(np.bincount(ya, minlength=2)) < 2: continue
                ys = S['y']; Rs, Rta = S['R'], T['R'][arc]
                base = dict(Source=src, Target=tgt, Rot=rot, AdaptFrac=frac)
                pb, km = pca_basis(S['V'], T['V'][arc], rot)
                fit = lambda A_, B_: balanced_accuracy_score(
                    ye, clf(rot).fit(A_, ys).predict(B_))
                for e in EPS_GRID:
                    for t, Rtr in (('QOTu', qot_uncond(Rs, Rta, e)),
                                   ('QOTc', qot_cond(Rs, ys, Rta, ya, e)),
                                   ('none', Rs)):
                        for ro in READOUTS:
                            if ro == 'PCA':
                                A_, B_, _ = pca_apply(pb, km, vec_rho(Rtr),
                                                      T['V'][ev], ys, rot)
                            else:
                                A_, B_ = hs(Rtr, ys, Rs, Rta, ya, T['R'][ev],
                                            ro == 'HSsrc')
                            rows.append({**base, 'knob': 'EPS', 'value': e,
                                         'Method': f'{t}-{ro}', 'BAC': fit(A_, B_)})
                for g in REG_GRID:
                    rows.append({**base, 'knob': 'OTREG', 'value': g, 'Method': 'COT',
                                 'BAC': balanced_accuracy_score(ye, clf(rot).fit(
                                     cond_sinkhorn(S['X'], ys, T['X'][arc], ya, g),
                                     ys).predict(T['X'][ev]))})
                    Vsd = cond_sinkhorn(S['V'], ys, T['V'][arc], ya, g)
                    Rsd = unvec_rho(Vsd, Rs.shape[1])
                    for ro in READOUTS:
                        if ro == 'PCA':
                            A_, B_, _ = pca_apply(pb, km, Vsd, T['V'][ev], ys, rot)
                        else:
                            A_, B_ = hs(Rsd, ys, Rs, Rta, ya, T['R'][ev], ro == 'HSsrc')
                        rows.append({**base, 'knob': 'OTREG', 'value': g,
                                     'Method': f'SDM-{ro}', 'BAC': fit(A_, B_)})
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/sweeps.csv', index=False)
    for knob, grid in (('EPS', EPS_GRID), ('OTREG', REG_GRID)):
        q = d[d.knob == knob]
        H(f'{knob} SWEEP   mean BAC over the 12 pairs (sd across pairs)', '-')
        for frac in fracs:
            L(f'\n  budget {frac:.0%}')
            L(f'    {"method":16s} ' + ' '.join(f'{v:g}'.rjust(14) for v in grid))
            for m in sorted(set(q.Method)):
                cells = []
                for v in grid:
                    s_ = q[(q.Method == m) & np.isclose(q.value, v)
                           & np.isclose(q.AdaptFrac, frac)]
                    pp = s_.groupby(['Source', 'Target']).BAC.mean()
                    cells.append(('-' if not len(pp) else
                                  f'{pp.mean():.3f}({pp.std():.3f})').rjust(14))
                L(f'    {m:16s} ' + ' '.join(cells))
        if knob == 'EPS':
            L('\n  headline contrast at each EPS   QOTu-HSsrc minus none-HSsrc')
        else:
            L('\n  QOTu-HSsrc (no epsilon to tune) minus the swept classical method')
        L(f'    {"budget":8s} ' + ' '.join(f'{v:g}'.rjust(16) for v in grid))
        for frac in fracs:
            cells = []
            for v in grid:
                s_ = q[np.isclose(q.value, v) & np.isclose(q.AdaptFrac, frac)]
                w = s_.groupby(['Source', 'Target', 'Method']).BAC.mean().unstack()
                if knob == 'EPS':
                    a, b = 'QOTu-HSsrc', 'none-HSsrc'
                else:
                    ee = d[(d.knob == 'EPS') & np.isclose(d.value, EPS)
                           & np.isclose(d.AdaptFrac, frac)]
                    w2 = ee.groupby(['Source', 'Target', 'Method']).BAC.mean().unstack()
                    if 'SDM-HSsrc' not in w or 'QOTu-HSsrc' not in w2:
                        cells.append('-'.rjust(16)); continue
                    x = (w2['QOTu-HSsrc']-w['SDM-HSsrc']).dropna()
                    cells.append(f'{x.mean():+.3f} {int((x>0).sum())}/{len(x)}'.rjust(16))
                    continue
                if a not in w or b not in w: cells.append('-'.rjust(16)); continue
                x = (w[a]-w[b]).dropna()
                cells.append(f'{x.mean():+.3f} {int((x>0).sum())}/{len(x)}'.rjust(16))
            L(f'    {frac:8.0%} ' + ' '.join(cells))
    return d


# ═══════════════════════════════════════════════════════════ 4. BOOT
def acf_block_length(y, max_lag=200):
    y = y - y.mean(); v = float(y @ y)
    if v <= 0: return 2
    for Lg in range(1, max_lag+1):
        if float(y[:-Lg] @ y[Lg:]) / v < 0.05:
            return max(Lg, 2)
    return max_lag

def mbb(n, Lb, rng):
    nb = int(np.ceil(n/Lb))
    st = rng.integers(0, max(n-Lb+1, 1), nb)
    return np.concatenate([np.arange(s, min(s+Lb, n)) for s in st])[:n]

def stage_boot(quick):
    H('STAGE 4  BOOTSTRAP   moving-block 95% intervals on individual cells')
    L(textwrap.dedent('''
      The fitted model is held FIXED and only the evaluation sample is resampled,
      which is the quantity in question: how precisely does this model's balanced
      accuracy on this site's track pin down. Segments are resampled in contiguous
      BLOCKS, since adjacent track is correlated and independent draws would give
      intervals far too narrow. Block length is measured from the label
      autocorrelation, not chosen. Contrasts are PAIRED, both methods scored on the
      same resampled blocks.

      Only folds whose arc holds both classes are used, so all 16 methods sit on
      identical folds and every paired contrast is legitimate.
      '''))
    B = 400 if quick else 2000
    rots = 3 if quick else N_ROT
    fracs = [.50] if quick else [.05, .50]
    D = prep('Right')
    BL = {s: acf_block_length(D[s]['y'].astype(float)) for s in SITES}
    L('  block length from the label autocorrelation:  ' +
      '   '.join(f'{s} {BL[s]} segs' for s in SITES) + f'\n  B = {B}\n')
    rows = []
    for src, tgt in itertools.permutations(SITES, 2):
        S, T = D[src], D[tgt]; Lb = BL[tgt]
        for frac in fracs:
            packs = []
            for rot in range(rots):
                arc, ev = split(T['mid'], rot, 'contig', frac, rots)
                if not len(arc): continue
                if min(np.bincount(T['y'][ev], minlength=2)) < 2: continue
                if min(np.bincount(T['y'][arc], minlength=2)) < 2: continue
                p, ok, _ = fold_preds(S, T, S['y'], T['y'], arc, ev, rot)
                if not all(m in p for m in METHODS): continue
                packs.append((T['y'][ev].astype(np.int64),
                              np.stack([p[m] for m in METHODS]).astype(np.int64)))
            if not packs: continue
            def bac_all(idxs):
                o = np.zeros(len(METHODS))
                for (ye, P), idx in zip(packs, idxs):
                    yb = ye[idx]; Pb = P[:, idx]
                    npz, nn = yb.sum(), len(yb)-yb.sum()
                    if npz == 0 or nn == 0: return None
                    o += 0.5*((Pb*yb).sum(1)/npz + ((1-Pb)*(1-yb)).sum(1)/nn)
                return o/len(packs)
            point = bac_all([np.arange(len(ye)) for ye, _ in packs])
            rng = np.random.default_rng(0)
            bo = np.full((B, len(METHODS)), np.nan)
            for b in range(B):
                v = bac_all([mbb(len(ye), Lb, rng) for ye, _ in packs])
                if v is not None: bo[b] = v
            bo = bo[~np.isnan(bo[:, 0])]
            r = dict(Source=src, Target=tgt, AdaptFrac=frac, nRot=len(packs), Block=Lb,
                     nEval=len(packs[0][0]), nEvalPos=int(packs[0][0].sum()))
            for i, m in enumerate(METHODS):
                r[m] = point[i]
                r[m+'_lo'], r[m+'_hi'] = np.percentile(bo[:, i], [2.5, 97.5])
            for x, y_ in CONTRASTS:
                i, j = METHODS.index(x), METHODS.index(y_)
                dd = bo[:, i]-bo[:, j]
                r[f'{x}|{y_}'] = point[i]-point[j]
                r[f'{x}|{y_}_lo'], r[f'{x}|{y_}_hi'] = np.percentile(dd, [2.5, 97.5])
            rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/bootstrap.csv', index=False)
    for frac in fracs:
        q = d[np.isclose(d.AdaptFrac, frac)]
        if not len(q): continue
        H(f'PER-CELL INTERVALS, budget {frac:.0%}, right rail, contiguous arc', '-')
        L(f'  {"pair":12s} {"eval":>6s} {"pos":>5s} | {"none-HSsrc":>21s} '
          f'{"QOTu-HSsrc":>21s} | {"difference":>24s}')
        for _, r in q.iterrows():
            k = 'QOTu-HSsrc|none-HSsrc'
            sg = '*' if r[k+'_lo'] > 0 else ('-' if r[k+'_hi'] < 0 else ' ')
            L(f'  {r.Source}->{r.Target:6s} {int(r.nEval):6d} {int(r.nEvalPos):5d} | '
              + f'{r["none-HSsrc"]:.3f} [{r["none-HSsrc_lo"]:.3f},{r["none-HSsrc_hi"]:.3f}]'.rjust(21)
              + f'{r["QOTu-HSsrc"]:.3f} [{r["QOTu-HSsrc_lo"]:.3f},{r["QOTu-HSsrc_hi"]:.3f}]'.rjust(22)
              + f' | {r[k]:+.3f} [{r[k+"_lo"]:+.3f},{r[k+"_hi"]:+.3f}] {sg}')
        L(f'\n  ALL 16 METHODS, mean and median interval width')
        L(f'  {"method":16s} {"uses":>14s} {"mean":>8s} {"median CI width":>16s}')
        for m in METHODS:
            L(f'  {m:16s} {uses(m):>14s} {q[m].mean():8.3f} '
              f'{(q[m+"_hi"]-q[m+"_lo"]).median():16.3f}')
        L(f'\n  PAIRED CONTRASTS   (intervals excluding zero, out of {len(q)} pairs)')
        L(f'  {"contrast":34s} {"mean":>8s} {"+ve":>5s} {"-ve":>5s} {"spans 0":>8s} '
          f'{"median width":>13s}')
        for x, y_ in CONTRASTS:
            k = f'{x}|{y_}'
            npos = int((q[k+'_lo'] > 0).sum()); nneg = int((q[k+'_hi'] < 0).sum())
            L(f'  {x+" - "+y_:34s} {q[k].mean():+8.3f} {npos:5d} {nneg:5d} '
              f'{len(q)-npos-nneg:8d} {(q[k+"_hi"]-q[k+"_lo"]).median():13.3f}')
    return d


# ═══════════════════════════════════════════════════════════ 5. DIAG
def stage_diag(quick):
    H('STAGE 5  LEFT RAIL DIAGNOSIS   why does one rail work and the other not?')
    L('\n--- 1. is the LABEL different between rails? ---')
    L(f'  {"site":6s} {"n":>6s} {"L pos":>7s} {"R pos":>7s} | {"med|LProf|":>10s} '
      f'{"med|RProf|":>10s} {"L near tau":>11s} {"R near tau":>11s}')
    for s in SITES:
        p = os.path.join(DATA, f'cleaned_{s}.csv')
        d = pd.read_csv(p, usecols=lambda c: c in ['LProf62', 'RProf62', 'BMP', 'EMP',
                                                   'run_id'], low_memory=False)
        d = d[d.run_id == KEEP[s]].dropna(subset=['LProf62', 'RProf62', 'BMP', 'EMP'])
        g = d.groupby(['BMP', 'EMP']); keep = g.size() >= MIN_ROWS
        ml = g['LProf62'].apply(lambda v: v.abs().max())[keep]
        mr = g['RProf62'].apply(lambda v: v.abs().max())[keep]
        nr = lambda v: ((v > 0.30) & (v < 0.50)).mean()
        L(f'  {s:6s} {len(ml):6d} {int((ml>THRESH).sum()):7d} {int((mr>THRESH).sum()):7d} | '
          f'{ml.median():10.3f} {mr.median():10.3f} {nr(ml):11.1%} {nr(mr):11.1%}')

    L('\n--- 2. can each rail be predicted WITHIN a site at all? (5-fold) ---')
    L('  if left is unpredictable within site, the transfer null is a feature problem')
    L(f'  {"site":6s} {"left BAC":>10s} {"right BAC":>10s}')
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for s in SITES:
        XL, yL, _ = load(s, 'Left'); XR, yR, _ = load(s, 'Right')
        a = cross_val_score(clf(), XL, yL, cv=cv, scoring='balanced_accuracy').mean()
        b = cross_val_score(clf(), XR, yR, cv=cv, scoring='balanced_accuracy').mean()
        L(f'  {s:6s} {a:10.3f} {b:10.3f}')

    L('\n--- 3. how strongly does each channel relate to its own-rail label? ---')
    nms = ['BFI_own', 'BFI_C', 'BTI_own', 'BTI_C', 'MLI_own', 'MLI_C', 'LRI_own', 'LRI_C']
    for s in SITES:
        XL, yL, _ = load(s, 'Left'); XR, yR, _ = load(s, 'Right')
        cl = [np.corrcoef(XL[:, k], yL)[0, 1] for k in range(8)]
        cr = [np.corrcoef(XR[:, k], yR)[0, 1] for k in range(8)]
        L(f'  {s}   ' + '  '.join(n.rjust(8) for n in nms))
        L(f'    L  ' + '  '.join(f'{v:+.3f}'.rjust(8) for v in cl))
        L(f'    R  ' + '  '.join(f'{v:+.3f}'.rjust(8) for v in cr))

    L('\n--- 4. repeat passes: does the label flip more on one rail? ---')
    for s in SITES:
        p = os.path.join(DATA, f'cleaned_{s}.csv')
        d = pd.read_csv(p, usecols=lambda c: c in ['LProf62', 'RProf62', 'BMP', 'EMP',
                                                   'run_id'], low_memory=False)
        runs = sorted(d.run_id.unique())
        if len(runs) < 2:
            L(f'  {s}: only {len(runs)} run, no repeat pass available'); continue
        lab = {}
        for r in runs:
            q = d[d.run_id == r].dropna(subset=['LProf62', 'RProf62', 'BMP', 'EMP'])
            g = q.groupby(['BMP', 'EMP']); keep = g.size() >= MIN_ROWS
            lab[r] = pd.DataFrame({
                'L': (g['LProf62'].apply(lambda v: v.abs().max())[keep] > THRESH).astype(int),
                'R': (g['RProf62'].apply(lambda v: v.abs().max())[keep] > THRESH).astype(int)})
        base = KEEP[s]
        L(f'  {s}  (reference run {base})')
        for r in runs:
            if r == base: continue
            i = lab[base].index.intersection(lab[r].index)
            if len(i) < 50: continue
            L(f'    vs {r}   shared {len(i):6d}   L flip {(lab[base].L[i]!=lab[r].L[i]).mean():6.2%}'
              f'   R flip {(lab[base].R[i]!=lab[r].R[i]).mean():6.2%}')

    L('\n--- 5. survey runs present per site (a §4 fact) ---')
    for s in SITES:
        d = pd.read_csv(os.path.join(DATA, f'cleaned_{s}.csv'), usecols=['run_id'],
                        low_memory=False)
        L(f'  {s}: {d.run_id.nunique()} run(s)  {sorted(d.run_id.unique())}   '
          f'used {KEEP[s]}')


# ═══════════════════════════════════════════════════════════════════════ main
def main():
    global DATA, _LOG
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Run every QOT cross-site transfer experiment and print all results.')
    ap.add_argument('--data', default=DATA,
                    help='folder holding cleaned_HTL.csv, cleaned_PTT.csv, '
                         'cleaned_RTT.csv, cleaned_WRM.csv')
    ap.add_argument('--out', default=OUT, help='output folder (default ./qot_results)')
    ap.add_argument('--stage', default='all',
                    choices=['all', 'verify', 'main', 'perm', 'sweep', 'boot', 'diag'])
    ap.add_argument('--quick', action='store_true',
                    help='reduced rotations/resamples, about 10 minutes, for a smoke test')
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(a.out, exist_ok=True)
    globals()['OUT'] = a.out
    _LOG = open(os.path.join(a.out, 'REPORT.txt'), 'w')

    t0 = time.time()
    H('QOT CROSS-SITE TRANSFER   COMPLETE EXPERIMENT')
    L(f'  data folder : {os.path.abspath(DATA)}')
    L(f'  output      : {os.path.abspath(a.out)}')
    L(f'  stage       : {a.stage}' + ('   [QUICK MODE, reduced settings]' if a.quick else ''))
    L(f'  started     : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    L(textwrap.dedent('''
      Task    per-rail profile exceedance, max |Prof62| in a BMP-EMP segment > 0.4 in
      Data    cleaned_<SITE>.csv, 4 TTC sites, 12 directed pairs
      Feats   8 ballast channels, own rail and center, one index per pair
      Encode  amplitude, 8 features -> 3 qubits -> rank-one 8x8 density matrix
      '''))

    ok = True
    if a.stage in ('all', 'verify'):
        ok = stage_verify(a.quick)
        if not ok and a.stage == 'all':
            L('\n*** VERIFICATION FAILED. Stopping. Fix before trusting any result. ***')
            _LOG.close(); sys.exit(1)
    if a.stage in ('all', 'main'):  stage_main(a.quick)
    if a.stage in ('all', 'perm'):  stage_perm(a.quick)
    if a.stage in ('all', 'sweep'): stage_sweep(a.quick)
    if a.stage in ('all', 'boot'):  stage_boot(a.quick)
    if a.stage in ('all', 'diag'):  stage_diag(a.quick)

    H(f'DONE in {(time.time()-t0)/60:.1f} minutes.  '
      f'Everything above is also in {os.path.join(a.out, "REPORT.txt")}')
    _LOG.close()


if __name__ == '__main__':
    main()
