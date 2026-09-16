#!/usr/bin/env python3
"""Before/after charts for the 2026-09-15 deck.

PerfectReco / AllParticles are the fraction of truth B candidates in the event whose
chain is recovered, with N = 17 561 for CERN and N = 12 774 for the public sample.

  p1_v31_v38.png     the main line v31 -> v36 -> v37 -> v38, both metrics
  d_mass.png         the mass head: v38 -> v46 -> v47
  d_capacity.png     capacity: Delta AllParticles against v38
  d_attention.png    attention: two bases, opposite outcomes
  d_restructure.png  restructuring: the cheap 50-file comparison
  d_retrain.png      the ablation chain: Delta AllParticles against v500
  d_public.png       the public line
  d_rewards.png      what the two rewards bought (per-class accuracy + PerfectReco)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs_20260915'
os.makedirs(FIG, exist_ok=True)

BLUE = '#1F4E79'
DARK = '#2B2B2B'
GRAY = '#777777'
RED = '#C0392B'
GREEN = '#2E8B57'
ORANGE = '#E8A33D'
GOLD = '#B8860B'
PURPLE = '#7A44AA'
AMBER = '#E8A33D'
PINK = '#C2185B'
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': '#BBBBBB',
                     'axes.labelcolor': DARK, 'text.color': DARK,
                     'xtick.color': DARK, 'ytick.color': DARK, 'figure.dpi': 200})

# thr 0.9:  version: (AllParticles %, PerfectReco %)
S = {31: (22.49, 12.30), 36: (35.17, 18.77), 37: (36.21, 19.50), 38: (37.59, 21.10),
     46: (38.22, 21.28), 47: (39.56, 23.15), 48: (37.36, 21.40), 53: (26.78, 15.84),
     510: (38.01, 21.75), 511: (38.39, 22.11), 512: (24.03, 13.34), 514: (25.44, 15.35),
     515: (39.04, 22.95), 516: (38.95, 22.72), 517: (38.12, 22.00), 518: (38.73, 22.52),
     520: (38.87, 22.69), 500: (33.34, 17.90), 501: (32.70, 17.72), 502: (33.27, 18.01),
     503: (32.78, 17.74), 504: (33.69, 19.13), 505: (33.58, 19.13), 506: (33.20, 19.23),
     507: (34.02, 19.62), 509: (33.27, 17.98), 530: (32.58, 17.71), 549: (33.92, 19.47),
     551: (39.92, 23.19), 553: (37.96, 22.84)}
# public sample, N = 12 774
SP = {27: (51.66, 22.76), 45: (19.70, 8.06), 49: (14.15, 5.95), 60: (54.52, 23.43)}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG + '/' + name, bbox_inches='tight')
    plt.close(fig)
    print('  ' + name)


def stamp(ax):
    fig = ax.get_figure()
    fig.text(0.5, 0.005, 'N = 17 561 truth B candidates  ·  thr 0.9', ha='center', va='bottom',
             fontsize=12, color=GRAY, style='italic')


# ---------------------------------------------------------------- 1. main line
def p1_v31_v38():
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.9))
    vs = [31, 36, 37, 38]
    steps = ['v31\nbaseline', 'v36\n+ soft mask\n+ source head', 'v37\n+ class-2 wt\n+ hinge',
             'v38\n+ chain-CE\ncut 0.85']
    for ax, key, ttl, col in ((axes[0], 1, 'PerfectReco (%)', GREEN),
                              (axes[1], 0, 'AllParticles (%)', BLUE)):
        vals = [S[v][key] for v in vs]
        bars = ax.bar(range(4), vals, color=col, width=0.62, zorder=3)
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, '%.2f' % vals[i],
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color=col)
            if i:
                d = vals[i] - vals[i - 1]
                # a row of its own at the top: next to the bar value it reads as one number
                ax.annotate('%+.2f' % d, (i - 0.5, max(vals) * 1.28),
                            ha='center', va='center', fontsize=13.5, color=DARK, fontweight='bold',
                            bbox=dict(fc='white', ec='#DDDDDD', pad=1.6))
                ax.add_patch(Rectangle((i - 1, 0), 1, vals[i], color='#000000', alpha=0.04, zorder=1))
        ax.set_xticks(range(4)); ax.set_xticklabels(steps, fontsize=11.5)
        ax.set_ylim(0, max(vals) * 1.42)
        ax.set_title(ttl, fontsize=13.5, fontweight='bold', color=col, pad=8)
        ax.grid(axis='y', color='#EEEEEE', zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('The main line v31 → v38', fontsize=16,
                 fontweight='bold', color=BLUE)
    stamp(axes[0])
    _save(fig, 'p1_v31_v38.png')


# ---------------------------------------------------------------- 2. generic delta
def delta(fname, title, base, rows, unit='pp', extra='', figsize=(6.4, 3.6), key=0):
    """rows: (label, version).  Bars are the difference against the base version."""
    fig, ax = plt.subplots(figsize=figsize)
    b = S[base][key]
    labels = [r[0] for r in rows][::-1]
    vals = [S[r[1]][key] - b for r in rows][::-1]
    cols = [GREEN if v > 0 else RED for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.6, zorder=3)
    ax.axvline(0, color='#666666', lw=1.4, zorder=4)
    for i, (v, lbl) in enumerate(zip(vals, labels)):
        ax.text(v + (0.25 if v >= 0 else -0.25), i, '%+.2f %s' % (v, unit),
                va='center', ha='left' if v >= 0 else 'right', fontsize=13, fontweight='bold',
                color=GREEN if v > 0 else RED)
        ax.text(-0.02, i + 0.42, lbl, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=12.5, color=DARK)
    ax.set_yticks([]); ax.set_ylim(-0.7, len(vals) - 0.3)
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = max(1.0, (hi - lo) * 0.30)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(title, fontsize=14.5, fontweight='bold', color=BLUE, pad=24)
    ax.text(0, 1.055, 'base  v%d  =  %.2f' % (base, b) + ('   ' + extra if extra else ''),
            transform=ax.transAxes, ha='left', fontsize=12, color=GRAY)
    ax.grid(axis='x', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    _save(fig, fname)


# ---------------------------------------------------------------- 3. grouped before/after
def grouped(fname, title, groups, key=1, unit='%', figsize=(6.4, 3.6), note=''):
    """groups: list of (series_label, color, [(version, value), ...]) — one cluster per label."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(groups[0][2])
    w = 0.8 / len(groups)
    xs = np.arange(n)
    for g, (lab, col, pts) in enumerate(groups):
        vals = [p[1] for p in pts]
        xs_g = xs + g * w - 0.4 + w / 2
        ax.bar(xs_g, vals, width=w, color=col, label=lab, zorder=3)
        # a shorter bar stands next to a taller one in the same cluster: keep its label
        # inside its own x-range, otherwise the taller bar is drawn over the number
        cluster_top = [max(gr[2][i][1] for gr in groups) for i in range(n)]
        for x, v, ct in zip(xs_g, vals, cluster_top):
            if v < ct:
                ax.text(x + w / 2 - 0.02, v + 0.35, '%.2f' % v, ha='right', va='bottom',
                        fontsize=11.5, color=col, fontweight='bold')
            else:
                ax.text(x, v + 0.35, '%.2f' % v, ha='center', va='bottom', fontsize=11.5,
                        color=col, fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([('v%d' % p[0]) for p in groups[0][2]], fontsize=13)
    top = max(p[1] for _, _, pts in groups for p in pts)
    ax.set_ylim(0, top * 1.30)
    ax.set_ylabel(unit, fontsize=13)
    ax.set_title(title, fontsize=14.5, fontweight='bold', color=BLUE, pad=22)
    if len(groups) > 1:
        # below the axes: inside the frame it lands on the bar labels
        ax.legend(fontsize=11.5, frameon=False, ncol=len(groups), loc='upper center',
                  bbox_to_anchor=(0.5, -0.13), columnspacing=1.6, handlelength=1.4)
    if note:
        ax.text(0, 1.05, note, transform=ax.transAxes, ha='left', fontsize=12, color=GRAY)
    ax.grid(axis='y', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    _save(fig, fname)


# ---------------------------------------------------------------- 4. the charts
def charts():
    p1_v31_v38()

    # the mass head
    grouped('d_mass.png', 'The mass head: v38 → v46 → v47',
            [('PerfectReco', GOLD, [(38, S[38][1]), (46, S[46][1]), (47, S[47][1])]),
             ('AllParticles', BLUE, [(38, S[38][0]), (46, S[46][0]), (47, S[47][0])])],
            unit='%', figsize=(5.4, 2.95),
            note='masshead1 (raw m) → masshead2 (log10 + sentinel mask): +2.05 pp')

    # capacity
    delta('d_capacity.png', 'Capacity attempts — Δ AllParticles against v38',
          base=38, key=0,
          rows=[('v48   three heads at once, 130 ep', 48),
                ('v53   latent 32/24, from scratch', 53),
                ('v512  latent 32/24, from v38', 512),
                ('v514  GN width 128 → 256', 514)],
          extra='(noise ±0.43 pp)')

    # attention: two bases
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7))
    for ax, base, rows, ttl in (
            (axes[0], 38, [(38, 'v38 base'), (510, 'v510  + self-attention'),
                           (511, 'v511  + edge bias')], 'grafted on v38 (plastic)'),
            (axes[1], 47, [(47, 'v47 base'), (515, 'v515  + attention'),
                           (516, 'v516  + 60 ep'), (518, 'v518  + struct gen'),
                           (520, 'v520  179 ep')], 'grafted on v47 (converged)')):
        vs = [r[0] for r in rows]
        vals = [S[v][1] - S[base][1] for v in vs]
        cols = ['#BBBBBB'] + [GREEN if v > 0 else RED for v in vals[1:]]
        ax.barh(np.arange(len(vals))[::-1], vals, color=cols, height=0.6, zorder=3)
        ax.axvline(0, color='#666666', lw=1.4, zorder=4)
        for i, v in enumerate(vals):
            ax.text(v + (0.08 if v >= 0 else -0.08), len(vals) - 1 - i,
                    '%+.2f' % v, va='center', ha='left' if v >= 0 else 'right',
                    fontsize=12.5, fontweight='bold', color=DARK)
        ax.set_yticks(np.arange(len(vals))[::-1])
        ax.set_yticklabels([r[1] for r in rows], fontsize=11.5)
        ax.set_xlim(min(vals) - 0.9, max(vals) + 0.9)
        ax.set_title(ttl, fontsize=13.5, fontweight='bold', color=PINK)
        ax.grid(axis='x', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
    fig.suptitle('Δ PerfectReco vs the base — the same module, two outcomes', fontsize=15,
                 fontweight='bold', color=BLUE)
    stamp(axes[0])
    _save(fig, 'd_attention.png')

    # restructure: the one clean comparison
    grouped('d_restructure.png', 'Restructuring — the 50-file comparison (v42)',
            [('PerfectReco', PURPLE, [(38, 29.26), (42, 26.28)])],
            unit='%', figsize=(5.6, 3.5),
            note='v39/v40/v41 produced no usable numbers (early stop, divergence, 3.2 h/epoch)')

    # retrain
    delta('d_retrain.png', 'Ablation chain — Δ AllParticles against v500',
          base=500, key=0,
          rows=[('v501  B2', 501), ('v502  class-2 weight', 502), ('v503  hinge', 503),
                ('v504  chain-CE', 504), ('v505  source', 505), ('v506  mass', 506),
                ('v507  struct', 507), ('v509  empty control', 509), ('v530  cumulative A1', 530)],
          figsize=(6.4, 4.4), extra='(noise ±0.43 pp)')

    # public
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    vs = [27, 45, 49, 60]
    vals = [SP[v][1] for v in vs]
    cols = [GREEN, RED, RED, GREEN]
    ax.bar(range(4), vals, color=cols, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, '%.2f' % v, ha='center', va='bottom', fontsize=14,
                fontweight='bold', color=cols[i])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['v27\nsimple stack', 'v45\nCERN stack', 'v49\ncontinued',
                        'v60\nlayer-by-layer'], fontsize=11.5)
    ax.set_ylim(0, max(vals) * 1.26)
    ax.set_ylabel('PerfectReco (%)', fontsize=13)
    ax.set_title('The public line, rebuilt from the simple stack', fontsize=14.5,
                 fontweight='bold', color=BLUE, pad=22)
    ax.text(0, 1.05, 'public sample, N = 12 774', transform=ax.transAxes,
            ha='left', fontsize=12, color=GRAY, style='italic')
    ax.grid(axis='y', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    _save(fig, 'd_public.png')

    # rewards: per-class accuracy + PerfectReco
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.75))
    for ax, lab, before, after, unit, col in (
            (axes[0], 'class-1', 67.8, 76.8, '%', BLUE),
            (axes[1], 'class-2', 41.3, 47.9, '%', ORANGE),
            (axes[2], 'PerfectReco', S[31][1], S[38][1], '%', GREEN)):
        ax.bar([0, 1], [before, after], color=['#CCCCCC', col], width=0.6, zorder=3)
        for x, v in ((0, before), (1, after)):
            ax.text(x, v + 0.6, '%.1f' % v, ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color=DARK)
        ax.annotate('', xy=(0.75, after - (after - before) * 0.25),
                    xytext=(0.25, before + (after - before) * 0.25),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.6))
        ax.text(0.5, max(before, after) * 1.15, '%+.1f' % (after - before),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color=col,
                bbox=dict(fc='white', ec='none', pad=0.8))
        ax.set_xticks([0, 1]); ax.set_xticklabels(['v31', 'v38'], fontsize=13)
        ax.set_ylim(0, max(before, after) * 1.46)
        ax.set_title(lab, fontsize=11, fontweight='bold', color=col, pad=6)
        ax.grid(axis='y', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('What the two rewards bought', fontsize=12, fontweight='bold',
                 color=BLUE, y=1.05)
    _save(fig, 'd_rewards.png')


def oracle_ladder():
    """What each pruning axis is worth, measured by rewriting the decisions with the truth."""
    labels = ['baseline', 'fix point\nrecall', 'fix point\nprecision', 'point\n= truth',
              'edges\n= truth', 'perfect', 'reverse']
    allv = [39.56, 54.09, 54.68, 71.99, 71.10, 99.75, 16.91]
    perf = [23.15, 32.06, 30.60, 41.67, 40.08, 57.82, 8.67]
    cols = ['#8A8A8A', GREEN, GREEN, GREEN, GREEN, '#1B5E20', RED]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.8), gridspec_kw={'height_ratios': [1.15, 1.0]})

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, allv, width=0.38, color=cols, zorder=3)
    ax.bar(x + 0.2, perf, width=0.38, color=cols, alpha=0.45, zorder=3)
    for xi, v in zip(x - 0.2, allv):
        ax.text(xi, v + 1.6, '%.2f' % v, ha='center', va='bottom', fontsize=11.5,
                fontweight='bold', color=DARK)
    for xi, v in zip(x + 0.2, perf):
        # left-anchored inside its own (faded) bar: centred, the first digit lands on the
        # solid bar of the same group and grey-on-colour is illegible
        ax.text(xi - 0.17, v + 1.6, '%.2f' % v, ha='left', va='bottom', fontsize=10.5,
                color='#6B6B6B')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel('%', fontsize=13)
    ax.text(0, 1.06, 'solid = AllParticles   ·   faded = PerfectReco', transform=ax.transAxes,
            fontsize=12, color=GRAY)
    ax.set_title('Rewrite the pruning decisions with the truth', fontsize=15, fontweight='bold',
                 color=BLUE, pad=24)
    ax.grid(axis='y', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # decomposition: All% = P(chain survives the point prune) x P(recovered afterwards)
    ax = axes[1]
    names = ['baseline', 'point\n= truth', 'edges\n= truth', 'perfect']
    survive = [70.8, 95.3, 71.1, 100.0]
    recover = [55.9, 75.5, 100.0, 99.8]
    x = np.arange(len(names))
    ax.bar(x - 0.2, survive, width=0.38, color=BLUE, zorder=3, label='P(chain survives the point prune)')
    ax.bar(x + 0.2, recover, width=0.38, color=AMBER, zorder=3, label='P(recovered | it survived)')
    white = dict(fc='white', ec='none', alpha=0.85, pad=1.0)
    for xi, v in zip(x - 0.2, survive):
        ax.text(xi, v + 7, '%.1f' % v, ha='center', va='bottom', fontsize=11, color=BLUE,
                fontweight='bold', bbox=white)
    for xi, v in zip(x + 0.2, recover):
        # both series sit near 100 in the last two groups: dodge them vertically, and box the
        # text so it stays readable where it lands on a bar
        ax.text(xi, v + 1.5, '%.1f' % v, ha='center', va='bottom', fontsize=11, color='#B8770B',
                fontweight='bold', bbox=white)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 146)                     # headroom for the legend, clear of the labels
    ax.set_ylabel('%', fontsize=13)
    ax.legend(fontsize=10.5, frameon=False, ncol=2, loc='upper left',
              columnspacing=1.1, handlelength=1.2, handletextpad=0.5)
    ax.set_title('All% = survive × recover', fontsize=15, fontweight='bold', color=BLUE, pad=24)
    ax.grid(axis='y', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG + '/oracle_ladder.png', bbox_inches='tight')
    plt.close(fig)
    print('  oracle_ladder.png')


if __name__ == '__main__':
    print('figures:')
    charts()
    oracle_ladder()
