"""Label-efficiency figure. Right rail, contiguous label stretch.

Two panels, one per encoding. The x axis is the fraction of the target site made
available for adaptation, the y axis is mean balanced accuracy over the twelve
directed site pairs.

Six series, every method in the main transfer table that is not a readout
variant. The Bures map, the two Sinkhorn transports, and the three references.
The density-matrix methods use the HS-2 readout.

No error band is drawn. The spread across the twelve pairs is 0.04 to 0.10, and
almost all of it is shared between the methods, so bands overlap completely and
suggest that nothing is distinguishable.

Run from the repository root, after the result CSVs are in place.
    python3 make_fig_labeleff.py
Writes figures/fig_labeleff.pdf and figures/fig_labeleff.png.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from make_tables_3_4 import load


def pv(d, meth, frac):
    """Mean balanced accuracy per directed site pair, averaged over the folds."""
    s = d[(d.Method == meth) & np.isclose(d.AdaptFrac, frac)]
    return s.groupby(['Source', 'Target']).BAC.mean()

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FR = [0.05, 0.10, 0.20, 0.30, 0.50]
X = np.arange(len(FR))

#        label              key            color      marker ls    ms   lw
SERIES = [
    ('QOT-u',            'QOT',          '#111111', 'o',  '-',  3.0, 1.6),
    ('COT-$\\rho$-u',    'COT-rho-u',    '#5a5a5a', 's',  '-',  2.4, 1.0),
    ('COT-u',            'COT-u',        '#8e8e8e', '^',  '-',  2.6, 1.0),
    ('target only',      'TargetOnly',   '#111111', 'D',  '--', 2.4, 1.1),
    ('source + target',  'Source+Target', '#5a5a5a', 'v', ':',  2.6, 1.1),
    ('no adaptation',    'NoAdapt',      '#a9a9a9', None, '-.', 0,   1.1),
]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85), sharey=True)

for ax, enc, title in zip(axes, ['Amp', 'Ang'],
                          ['Amplitude encoding', 'Angle encoding']):
    d = load(enc, 'Right')
    for lab, key, c, mk, ls, ms, lw in SERIES:
        k = {'QOT': 'QOTu-HSsrc', 'COT-rho-u': 'SDMu-HSsrc'}.get(key, key)
        m = [pv(d, k, f).mean() for f in FR]
        ax.plot(X, m, color=c, marker=mk, markersize=ms, linestyle=ls,
                linewidth=lw, label=lab, zorder=3,
                markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.45)
    ax.set_xticks(X)
    ax.set_xticklabels([f'{int(f*100)}' for f in FR], fontsize=7.5)
    ax.set_xlabel('Target site available for adaptation (%)', fontsize=7.5)
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.set_xlim(-0.3, len(FR) - 0.7)
    ax.grid(axis='y', color='#e8e8e8', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#999999')
    ax.tick_params(labelsize=7, length=2.5, color='#999999')

axes[0].set_ylabel('Balanced accuracy', fontsize=7.5)
axes[0].set_ylim(0.540, 0.648)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=6, fontsize=6.6, frameon=False,
           loc='lower center', bbox_to_anchor=(0.5, -0.02),
           handlelength=2.3, columnspacing=1.2, handletextpad=0.45)

fig.tight_layout(rect=[0, 0.085, 1, 1])
fig.subplots_adjust(wspace=0.07)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'fig_labeleff.pdf'))
fig.savefig(os.path.join(OUT, 'fig_labeleff.png'), dpi=300)
print('wrote figures/fig_labeleff.pdf and .png')
