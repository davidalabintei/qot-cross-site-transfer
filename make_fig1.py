"""Fig. 1  Segment-level category proportions for the four GPR ballast indices.

Counts come from the CLEANED files, one survey run per site, segmented on
(BMP, EMP) with at least 10 geometry records, which is exactly the segment
definition load() uses in run_everything.py. Segment counts therefore match
the analysis: HTL 803, PTT 2272, RTT 4259, WRM 1017.

Greyscale, light to dark with increasing category number. BFI and MLI increase
with severity, BTI and LRI increase with favorable condition, so the ramp
direction carries no meaning on its own and the caption states the direction.

Output: fig_eda_stacked_grey.pdf, vector, fonts embedded.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,      # embed TrueType, ASCE requirement
    'ps.fonttype': 42,
})

SITES = ['HTL', 'PTT', 'RTT', 'WRM']
CHANS = ['L', 'C', 'R']
CHLAB = {'L': 'Left', 'C': 'Center', 'R': 'Right'}
NSEG = {'HTL': 803, 'PTT': 2272, 'RTT': 4259, 'WRM': 1017}

# counts[index][site][channel][category] -- from the cleaned files
C = {
 'BFI': {
  'HTL': {'L': {1:193,2:288,3:263,4:22,5:37}, 'C': {1:159,2:420,3:204,4:4,5:16},
          'R': {1:221,2:343,3:203,4:19,5:17}},
  'PTT': {'L': {1:675,2:1325,3:249,4:8,5:15}, 'C': {1:221,2:683,3:1168,4:96,5:104},
          'R': {1:1130,2:776,3:342,4:10,5:14}},
  'RTT': {'L': {1:3224,2:721,3:269,4:19,5:26}, 'C': {1:778,2:1933,3:1420,4:56,5:72},
          'R': {1:3711,2:441,3:66,4:12,5:29}},
  'WRM': {'L': {1:722,2:269,3:18,4:2,5:6}, 'C': {1:158,2:658,3:184,4:8,5:9},
          'R': {1:942,2:66,3:8,4:1}},
 },
 'BTI': {
  'HTL': {'L': {1:300,2:341,3:155,4:7}, 'C': {1:348,2:216,3:168,4:66,5:5},
          'R': {1:158,2:426,3:180,4:38,5:1}},
  'PTT': {'L': {1:90,2:1664,3:518}, 'C': {1:38,2:449,3:1629,4:156},
          'R': {1:56,2:380,3:1062,4:755,5:19}},
  'RTT': {'L': {1:53,2:1984,3:1944,4:174,5:104}, 'C': {1:1,2:358,3:1029,4:2424,5:447},
          'R': {2:153,3:921,4:1199,5:1986}},
  'WRM': {'L': {1:16,2:256,3:505,4:213,5:27}, 'C': {2:69,3:441,4:340,5:167},
          'R': {1:5,2:138,3:564,4:303,5:7}},
 },
 'MLI': {
  'HTL': {'L': {1:646,2:129,3:28}, 'C': {1:753,2:39,3:11}, 'R': {1:705,2:73,3:25}},
  'PTT': {'L': {1:1207,2:789,3:276}, 'C': {1:2151,2:121}, 'R': {1:1858,2:387,3:27}},
  'RTT': {'L': {1:2460,2:1684,3:115}, 'C': {1:4135,2:121,3:3}, 'R': {1:3736,2:500,3:23}},
  'WRM': {'L': {1:816,2:194,3:7}, 'C': {1:932,2:85}, 'R': {1:713,2:291,3:13}},
 },
 'LRI': {
  'HTL': {'L': {1:161,2:320,3:322}, 'C': {1:145,2:439,3:219}, 'R': {1:151,2:309,3:343}},
  'PTT': {'L': {1:7,2:130,3:2135}, 'C': {1:286,2:1361,3:625}, 'R': {1:46,2:353,3:1873}},
  'RTT': {'L': {1:77,2:610,3:3572}, 'C': {1:402,2:2640,3:1217}, 'R': {1:319,2:1731,3:2209}},
  'WRM': {'L': {1:64,2:359,3:594}, 'C': {1:99,2:410,3:508}, 'R': {1:50,2:271,3:696}},
 },
}

NCAT = {'BFI': 5, 'BTI': 5, 'MLI': 3, 'LRI': 3}

# The instrument's own category descriptions, taken verbatim from the legend block
# of ZR0637-20-XLS01-A "Trackbed Metrics". Worth showing, because the numbers alone
# hide that BTI is NOT monotonic in quality: category 4 is "Design", the intended
# thickness, and 5 is thicker than design, so both 1 and 5 are departures from it.
DESC = {
    'BFI': ['Clean', 'Mod. clean', 'Mod. fouled', 'Fouled', 'Highly fouled'],
    'BTI': ['Very thin', 'Thin', 'Reduced', 'Design', 'Thick'],
    'MLI': ['Good', 'Moderate', 'Poor'],
    'LRI': ['Poor', 'Moderate', 'Good'],
}
TITLE = {'BFI': 'Ballast Fouling Index (BFI)',
         'BTI': 'Ballast Thickness Index (BTI)',
         'MLI': 'Moisture Likelihood Index (MLI)',
         'LRI': 'Layer Roughness Index (LRI)'}

# Ordered greyscale ramp, light to dark. Steps are spaced so adjacent pairs stay
# separable in monochrome print; a white keyline between segments does the rest.
RAMP5 = ['#f0f0f0', '#cfcfcf', '#9e9e9e', '#6b6b6b', '#2f2f2f']
RAMP3 = ['#e8e8e8', '#a8a8a8', '#454545']


def ramp(n):
    return RAMP5 if n == 5 else RAMP3


BARW = 0.24
OFF = {'L': -BARW, 'C': 0.0, 'R': BARW}

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.4))

for ax, idx in zip(axes.ravel(), ['BFI', 'BTI', 'MLI', 'LRI']):
    n = NCAT[idx]
    cols = ramp(n)
    for si, site in enumerate(SITES):
        for ch in CHANS:
            cnt = C[idx][site][ch]
            tot = sum(cnt.values())
            x = si + OFF[ch]
            bot = 0.0
            for k in range(1, n+1):
                p = cnt.get(k, 0)/tot
                if p <= 0:
                    continue
                ax.bar(x, p, BARW, bottom=bot, color=cols[k-1],
                       edgecolor='white', linewidth=0.8, zorder=2)
                bot += p
            ax.text(x, -0.035, CHLAB[ch][0], ha='center', va='top',
                    fontsize=5.8, color='#444444')

    ax.set_xticks(range(len(SITES)))
    # segment counts are not repeated here; they belong in the site table
    ax.set_xticklabels(SITES, fontsize=8)
    ax.tick_params(axis='x', length=0, pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=7)
    ax.set_ylabel('Segments (%)', fontsize=7.5)
    ax.set_title(TITLE[idx], fontsize=8.5, pad=27)
    ax.set_xlim(-0.5, len(SITES)-0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999999')
    ax.spines['bottom'].set_color('#999999')
    ax.set_axisbelow(True)

    # number AND the instrument's description, so a reader never has to look up
    # what category 4 means. Five long labels do not fit on one row, so the
    # five-category panels wrap to two.
    handles = [Patch(facecolor=cols[k-1], edgecolor='white', linewidth=0.6,
                     label=f'{k}  {DESC[idx][k-1]}') for k in range(1, n+1)]
    ax.legend(handles=handles, ncol=3 if n == 5 else 3, fontsize=6.2,
              loc='lower center', bbox_to_anchor=(0.5, 1.005), frameon=False,
              handlelength=1.0, handleheight=0.85, columnspacing=1.0,
              handletextpad=0.35, borderpad=0.0, labelspacing=0.35)

fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.07,
                    wspace=0.20, hspace=0.58)
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'fig_eda_stacked_grey.pdf'))
fig.savefig(os.path.join(OUT, 'fig_eda_stacked_grey.png'), dpi=300)
print('wrote fig_eda_stacked_grey.pdf and .png')

# ---- sanity: proportions must sum to 1 and counts must match the analysis
for idx in C:
    for site in SITES:
        for ch in CHANS:
            t = sum(C[idx][site][ch].values())
            assert t == NSEG[site], f'{idx} {site} {ch}: {t} != {NSEG[site]}'
print('[ok] every channel totals the analysis segment count for its site')
