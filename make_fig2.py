"""Fig. 2  The two pipelines, feature vector (left) and density matrix (right).

No "classical" or "quantum" path labels. Sinkhorn (COT-rho) also runs on the
density matrix path, so neither label is true of a whole path.

WHY REDRAWN. The submitted figure showed only the class-conditional
transports, had no no-transport control, and did not show Sinkhorn applied to
the vectorized density matrix. All three are now part of the method, and the
no-transport control is the main comparison.

Greyscale, serif, fonts embedded, to match Fig. 1.
Output: tex/fig_pipeline.pdf and tex/fig_pipeline.png
Run from qot_run:  python3 make_fig2.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FILL_DATA = '0.80'   # data and scoring
FILL_REP = '0.92'    # representation
FILL_TR = '1.00'     # transport choices
FILL_RO = '0.92'     # readout
FILL_CLF = '0.80'    # classifier
EDGE = '0.15'

fig, ax = plt.subplots(figsize=(7.0, 7.6))
ax.set_xlim(0, 11)
ax.set_ylim(-0.2, 12.4)
ax.axis('off')


def box(x0, y0, w, h, text, fill, bold=False, fs=None, ls='-', lw=0.8, pad=0.06):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f'round,pad={pad},rounding_size=0.12',
                                fc=fill, ec=EDGE, lw=lw, ls=ls))
    if text:
        ax.text(x0 + w / 2, y0 + h / 2, text, ha='center', va='center',
                fontsize=fs or matplotlib.rcParams['font.size'],
                fontweight='bold' if bold else 'normal', linespacing=1.25)


def container(x0, y0, w, h, title):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle='round,pad=0.04,rounding_size=0.15',
                                fc='none', ec='0.35', lw=0.8, ls=(0, (4, 3))))
    ax.text(x0 + 0.12, y0 + h - 0.12, title, ha='left', va='top',
            fontsize=8, style='italic', color='0.2')


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=EDGE,
                                shrinkA=0, shrinkB=0, mutation_scale=9))


# columns
LX0, LX1 = 0.25, 4.25          # feature-vector path
RX0, RX1 = 4.75, 10.75         # density-matrix path
LC = (LX0 + LX1) / 2
RC = (RX0 + RX1) / 2

# --- data --------------------------------------------------------------------
box(0.25, 11.05, 5.05, 0.95,
    'Source site\nall segments and their labels', FILL_DATA)
box(5.70, 11.05, 5.05, 0.95,
    'Target site\nadaptation segments, labels on a share of them,\n'
    'and a held-out evaluation half', FILL_DATA)
ax.text(5.5, 12.25, 'Eight GPR ballast indices per segment, twelve directed site pairs',
        ha='center', va='center', fontsize=8.5, style='italic')

# single merge bar, then one arrow into each path
ax.plot([2.8, 2.8], [11.0, 10.72], color=EDGE, lw=0.8)
ax.plot([8.2, 8.2], [11.0, 10.72], color=EDGE, lw=0.8)
ax.plot([LC, 8.2], [10.72, 10.72], color=EDGE, lw=0.8)
arrow(LC, 10.72, LC, 9.02)
arrow(RC, 10.72, RC, 10.21)

# --- representation ------------------------------------------------------------
box(LX0, 8.05, LX1 - LX0, 0.9,
    'Segment feature vector\nthe eight indices', FILL_REP)
box(RX0, 9.25, RX1 - RX0, 0.9,
    'Encode the segment as a quantum state\namplitude (3 qubits) or angle (4 qubits)', FILL_REP)
arrow(RC, 9.19, RC, 9.02)
box(RX0, 8.05, RX1 - RX0, 0.9,
    'Segment density matrix\n8 \u00d7 8 (amplitude) or 16 \u00d7 16 (angle)', FILL_REP)

arrow(LC, 7.99, LC, 7.62)
arrow(RC, 7.99, RC, 7.62)

# --- transport of source segments ----------------------------------------------
TY0, TY1 = 5.35, 7.55
container(LX0, TY0, LX1 - LX0, TY1 - TY0, 'Transport of source segments, one of')
container(RX0, TY0, RX1 - RX0, TY1 - TY0, 'Transport of source segments, one of')
bh, by = 1.5, 5.55
# classical: two choices
w2 = (LX1 - LX0 - 0.45) / 2
box(LX0 + 0.15, by, w2, bh, 'Sinkhorn OT\n\nCOT-u\nCOT-c', FILL_TR)
box(LX0 + 0.30 + w2, by, w2, bh,
    'None\n\nno adaptation\ntarget only\nsource + target', FILL_TR)
# quantum: three choices
w3 = (RX1 - RX0 - 0.60) / 3
box(RX0 + 0.15, by, w3, bh, 'Bures map\n\nQOT-u\nQOT-c', FILL_TR)
box(RX0 + 0.30 + w3, by, w3, bh,
    'Sinkhorn OT on\nvectorized matrix\n\nCOT-ρ-u\nCOT-ρ-c', FILL_TR)
box(RX0 + 0.45 + 2 * w3, by, w3, bh, 'None\n\nno transport\ncontrol', FILL_TR)

# --- readout ------------------------------------------------------------------------
arrow(RC, TY0 - 0.06, RC, 4.62)
container(RX0, 3.05, RX1 - RX0, 1.5, 'Readout, one of')
w2r = (RX1 - RX0 - 0.45) / 2
box(RX0 + 0.15, 3.2, w2r, 0.95,
    'Hilbert-Schmidt\nHS-2 (source prototypes)\nHS-4 (adds target prototypes)', FILL_RO, fs=8)
box(RX0 + 0.30 + w2r, 3.2, w2r, 0.95, 'Vectorized PCA', FILL_RO, fs=8)

# --- classifier and scoring --------------------------------------------------------
CY0 = 1.75
box(LX0, CY0, RX1 - LX0, 0.7, 'Logistic regression classifier', FILL_CLF, bold=True)
arrow(LC, TY0 - 0.06, LC, CY0 + 0.76)
arrow(RC, 3.0, RC, CY0 + 0.76)
arrow(5.5, CY0 - 0.06, 5.5, 1.22)
box(LX0, 0.45, RX1 - LX0, 0.72,
    'Scored on the target evaluation half\nbalanced accuracy, precision, recall, F1', FILL_DATA, fs=8)

# --- key -------------------------------------------------------------------------------
ax.text(5.5, 0.05,
        '-u  unconditional, reads no target label          -c  class-conditional, reads target labels',
        ha='center', va='center', fontsize=8, color='0.15')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'fig_pipeline.pdf'), bbox_inches='tight', pad_inches=0.03)
fig.savefig(os.path.join(out, 'fig_pipeline.png'), dpi=300, bbox_inches='tight', pad_inches=0.03)
print('wrote figures/fig_pipeline.pdf and .png')
