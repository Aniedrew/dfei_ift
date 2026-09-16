#!/usr/bin/env python3
"""One explanatory figure per attempt for the 2026-09-15 deck.

Each figure is a two-panel card:
  top    — the DFEI pipeline with the part this attempt changes highlighted
  bottom — what the attempt gave: the base version against the new one(s)

Reconstruction numbers come from logs/fixed_denominator_metrics_inclusive_00342442.csv
(CERN) and from the public-line table in VERSION_LOG.md (public sample); the two
sample sizes are never mixed inside one panel.

  att_<key>.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs_20260915'
os.makedirs(FIG, exist_ok=True)

BLUE = '#1F4E79'
DARK = '#2B2B2B'
GRAY = '#8A8A8A'
RED = '#C0392B'
GREEN = '#2E8B57'
AMBER = '#B8860B'
PURPLE = '#7A44AA'
PINK = '#C2185B'
TEAL = '#0B7285'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#BBBBBB',
                     'text.color': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
                     'figure.dpi': 200})

BLOCKS = [('graph', 'event\ngraph'), ('prune', 'pruning\nheads'), ('gnn', 'GNN\nblocks'),
          ('latent', 'latent\n16-d'), ('heads', 'physics\nheads'), ('chain', 'chain\nassembly')]


def pipeline(ax, hl, note, colour):
    """The six pipeline stages; the ones in `hl` are the ones this attempt touches."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis('off')
    w, gap = 13.4, 3.2
    for i, (key, lab) in enumerate(BLOCKS):
        x = 0.6 + i * (w + gap)
        on = key in hl
        ax.add_patch(FancyBboxPatch((x, 12), w, 13,
                                    boxstyle='round,pad=0.5,rounding_size=1.6',
                                    fc=colour if on else '#F2F2F2',
                                    ec=colour if on else '#C8C8C8', lw=1.8 if on else 1.1))
        ax.text(x + w / 2, 18.5, lab, ha='center', va='center', fontsize=9.6,
                color='white' if on else '#7A7A7A', fontweight='bold' if on else 'normal')
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.3, 18.5), (x - 0.4, 18.5),
                                         arrowstyle='-|>', mutation_scale=8,
                                         color='#B8B8B8', lw=1.1))
    ax.text(50, 4.0, note, ha='center', va='center', fontsize=10.6, color=colour,
            fontweight='bold', wrap=True)


def bars(ax, rows, ylab, unit='%', note=''):
    """rows: (label, allv, perfv) or (label, value); the first row is the base."""
    x = np.arange(len(rows))
    two = len(rows[0]) == 3
    if two:
        w = 0.36
        for k, (idx, lab, col) in enumerate(((1, 'AllParticles', BLUE), (2, 'PerfectReco', AMBER))):
            vals = [r[idx] for r in rows]
            cols = [GRAY if i == 0 else col for i in range(len(rows))]
            ax.bar(x + (k - 0.5) * w, vals, width=w, color=cols, zorder=3)
            for xi, v in zip(x + (k - 0.5) * w, vals):
                ax.text(xi, v + max(vals) * 0.02, '%.2f' % v, ha='center', va='bottom',
                        fontsize=9.4, color=DARK)
        leg = [plt.Rectangle((0, 0), 1, 1, fc=BLUE), plt.Rectangle((0, 0), 1, 1, fc=AMBER)]
        ax.legend(leg, ['AllParticles', 'PerfectReco'], fontsize=9.6, frameon=False,
                  ncol=2, loc='upper left', bbox_to_anchor=(0, 1.20))
        top = max(r[1] for r in rows + [(None, 0, 0)]) if False else max(
            max(r[1] for r in rows), max(r[2] for r in rows))
        ax.set_ylim(0, top * 1.32)
    else:
        vals = [r[1] for r in rows]
        cols = [GRAY] + [GREEN if i else GRAY for i in range(1, len(rows))]
        ax.bar(x, vals, width=0.5, color=cols, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.02, '%.2f' % v, ha='center', va='bottom',
                    fontsize=10.4, color=DARK, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10.2)
    ax.set_ylabel(ylab, fontsize=10.6)
    if note:
        ax.text(0, 1.30, note, transform=ax.transAxes, ha='left', fontsize=10.2, color=GRAY)
    ax.grid(axis='y', color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def curve(ax, xs, ys, ylab, note='', colour=RED, logy=False):
    ax.plot(xs, ys, '-o', color=colour, lw=2.0, ms=6, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y, '%.1f' % y, ha='center', va='bottom', fontsize=9.6, color=DARK)
    ax.set_xlabel('epochs into the run', fontsize=10.4)
    ax.set_ylabel(ylab, fontsize=10.6)
    ax.grid(color='#EEEEEE', zorder=0); ax.set_axisbelow(True)
    if note:
        ax.text(0, 1.30, note, transform=ax.transAxes, ha='left', fontsize=10.2, color=GRAY)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def card(key, hl, hlnote, colour, kind, data, ylab, note=''):
    fig = plt.figure(figsize=(6.3, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.62,
                          left=0.13, right=0.97, top=0.96, bottom=0.10)
    ax1 = fig.add_subplot(gs[0]); pipeline(ax1, hl, hlnote, colour)
    ax2 = fig.add_subplot(gs[1])
    if kind == 'bars':
        bars(ax2, data, ylab, note=note)
    else:
        curve(ax2, *data, ylab, note=note, colour=colour)
    fig.savefig(FIG + '/att_%s.png' % key, bbox_inches='tight')
    plt.close(fig)
    print('  att_%s.png' % key)


# ============================================================ the attempts
def build():
    # ---------------------------------------------------------- capacity
    card('cap_v48', ['heads'], 'three heads into one 16-d space', PURPLE, 'bars',
         [('v38', 37.59, 21.10), ('v48', 37.36, 21.40)], 'value  [%]',
         'auxiliary losses 0.877 vs 0.559 for the main task')
    card('cap_probe', ['heads'], 'one head at a time, on v47', PURPLE, 'bars',
         [('v50\nmass only', 51.41, 29.67), ('v51\n+mom', 53.83, 30.46), ('v52\n+struct', 54.50, 30.46)],
         'value  [%]', 'small probe: 20 training files, 5-file evaluation')
    card('cap_v53', ['latent', 'gnn'], 'widen the latent, train from scratch', PURPLE, 'bars',
         [('v38', 37.59, 21.10), ('v53', 26.78, 15.84)], 'value  [%]',
         'tracks 32-d, tt edges 24-d, 150 epochs')
    card('cap_v512', ['latent'], 'widen the latent, inherit the weights', PURPLE, 'bars',
         [('v38', 37.59, 21.10), ('v512', 24.03, 13.34)], 'value  [%]',
         'same widening, started from v38, 20 epochs')
    card('cap_v514', ['gnn'], 'widen the GN hidden layer', PURPLE, 'bars',
         [('v38', 37.59, 21.10), ('v513/v514', 25.44, 15.35)], 'value  [%]',
         'GN 128 → 256, interfaces untouched')

    # ---------------------------------------------------------- attention
    card('att_v510', ['gnn'], 'self-attention after the last GN block', PINK, 'bars',
         [('v38', 37.59, 21.10), ('v510', 38.01, 21.75)], 'value  [%]',
         'pure content, 20 epochs on v38')
    card('att_v511', ['gnn'], '+ tt edge features as attention bias', PINK, 'bars',
         [('v510', 38.01, 21.75), ('v511', 38.39, 22.11)], 'value  [%]',
         'ParT-style edge bias; class-1 recall 78.27%')
    card('att_v515', ['gnn'], 'the same module on the converged model', PINK, 'bars',
         [('v47', 39.56, 23.15), ('v515', 39.04, 22.95), ('v516', 38.95, 22.72)], 'value  [%]',
         'v516 is v515 continued for 60 more epochs')
    card('att_v519', ['gnn', 'heads'], 'generation: attention + mass from v38', PINK, 'bars',
         [('v38', 37.59, 21.10), ('v517/v519', 38.12, 22.00)], 'value  [%]',
         'both new heads start from v38, 150 epochs')
    card('att_v520', ['heads'], 'generation: the struct head from v47', PINK, 'bars',
         [('v47', 39.56, 23.15), ('v518', 38.73, 22.52), ('v520', 38.87, 22.69)], 'value  [%]',
         'v520 is v518 continued to 179 epochs')

    # ---------------------------------------------------------- restructure
    card('res_v39', ['graph', 'chain'], 'cluster the tracks, reconstruct each cluster', PURPLE,
         'bars', [('whole event', 115.0), ('per PV cluster', 25.0)], 'nodes per graph',
         'the idea was to cut 91–139 nodes down to 20–30 per cluster')
    card('res_v40', ['graph', 'chain'], 'make the split trainable (Gumbel)', PURPLE, 'curve',
         ([102, 103, 104, 105], [35.9, 84.9, 114.0, 155.0]), 'validation loss',
         'diverged: train gets subgraphs, validation stayed on the full graph')
    card('res_v41', ['graph', 'chain'], 'align train and validation, add a curriculum', PURPLE,
         'curve', ([102, 106, 110, 114], [131.0, 121.0, 134.0, 126.0]), 'validation loss',
         'correct code, but 3.2 h/epoch and the curve never fell')
    card('res_v42', ['graph', 'chain'], 'the same question at 50 files', PURPLE, 'bars',
         [('v38', 76.8), ('v42', 56.4)], 'LCAG class-1 accuracy  [%]',
         'same 50-file run; PerfectReco −2.98 pp for v42')

    # ---------------------------------------------------------- retrain: the ablation chain
    card('ret_proto', ['prune', 'heads'], 'one change at a time from v500', PURPLE, 'bars',
         [('v500\nbase', 33.34, 17.90), ('v501\nB2', 32.70, 17.72), ('v502\ncl2w', 33.27, 18.01),
          ('v503\nhinge', 32.78, 17.74), ('v504\nchain-CE', 33.69, 19.13), ('v505\nsource', 33.58, 19.13),
          ('v506\nmass', 33.20, 19.23), ('v507\nstruct', 34.02, 19.62), ('v509\nempty', 33.27, 17.98)],
         'value  [%]', 'each run: 20 epochs, 200 files, judged against the v500 base')
    card('ret_v501', ['prune'], 'B2 differentiable pruning, cut 0.85', PURPLE, 'bars',
         [('v500', 33.34, 17.90), ('v501', 32.70, 17.72)], 'value  [%]',
         'no gain — the base was already trained with the same mask')
    card('ret_v502', ['prune'], 'class-2 loss weight 2.0', PURPLE, 'bars',
         [('v500', 33.34, 17.90), ('v502', 33.27, 18.01)], 'value  [%]',
         'class-2 recall +0.3 pp — a marginal effect once the loss is rebalanced')
    card('ret_v503', ['prune'], 'chain hinge, margin 0.3', PURPLE, 'bars',
         [('v500', 33.34, 17.90), ('v503', 32.78, 17.74)], 'value  [%]',
         'the penalty is max(0, 0.3 − confidence): on this base every chain edge is already above 0.3')
    card('ret_v504', ['prune'], 'chain-CE on truth-chain edges', PURPLE, 'bars',
         [('v500', 33.34, 17.90), ('v504', 33.69, 19.13)], 'value  [%]',
         'the only robust winner: +216 events reconstructed')
    card('ret_v505', ['heads'], 'source head (Rumor Centrality root)', PURPLE, 'bars',
         [('v504', 33.69, 19.13), ('v505', 33.58, 19.13)], 'value  [%]',
         'class-1 +4.35 pp but class-2 −2.76 pp → net flat')
    card('ret_v506', ['heads'], 'mass head (log10 m_ππ)', PURPLE, 'bars',
         [('v504', 33.69, 19.13), ('v506', 33.20, 19.23)], 'value  [%]',
         'class-2 +4.08 and class-3 +1.64, but class-1 −0.74 → AllParticles down')
    card('ret_v507', ['heads'], 'struct head (BFS depth + RC), weight 0.3', PURPLE, 'bars',
         [('v504', 33.69, 19.13), ('v507', 34.02, 19.62)], 'value  [%]',
         'the best single addition in the chain')
    card('ret_v509', ['prune'], 'the empty control: nothing added', PURPLE, 'bars',
         [('v500', 33.34, 17.90), ('v509', 33.27, 17.98)], 'value  [%]',
         'zero drift after 20 fine-tuning epochs — this calibrates every comparison above')
    card('ret_v530', ['prune', 'heads'], 'B2 + source kept as a group', PURPLE, 'bars',
         [('v500', 33.34, 17.90), ('v530', 32.58, 17.71)], 'value  [%]',
         'no gain → “greedy roll-back broke the synergy” is not the explanation')
    card('ret_v549', ['heads'], 'mass + momentum stacked on the best chain model', PURPLE, 'bars',
         [('v507', 34.02, 19.62), ('v549', 33.92, 19.47)], 'value  [%]',
         'no gain — joint use does not beat the sum of the parts')
    card('ret_v540', ['prune'], 'context-aware pruning head', PURPLE, 'bars',
         [('v511  reference', 38.39, 22.11), ('v540  plain 20 ep', 38.23, 22.04),
          ('v545/546', 37.83, 21.60), ('v548', 38.43, 21.98)], 'value  [%]',
         'four variants, none above a plain 20-epoch fine-tune of the same base')

    # ---------------------------------------------------------- public
    card('pub_v27', ['graph'], 'start from the simple stack', GREEN, 'bars',
         [('v27', 51.66, 22.76)], 'value  [%]', 'no B2, no chain losses, no physics heads')
    card('pub_v45', ['prune', 'heads'], 'transfer the whole CERN stack', GREEN, 'bars',
         [('v27', 51.66, 22.76), ('v45', 19.70, 8.06)], 'value  [%]',
         'the v38 recipe on public data — negative transfer')
    card('pub_v49', ['prune', 'heads'], 'continue the transferred model', GREEN, 'bars',
         [('v45', 19.70, 8.06), ('v49', 14.15, 5.95)], 'value  [%]',
         '88 epochs — it keeps getting worse')
    card('pub_v60', ['prune', 'heads'], 'rebuild layer by layer instead', GREEN, 'bars',
         [('v27', 51.66, 22.76), ('v60', 54.52, 23.43)], 'value  [%]',
         '+ B2 and the source head only; v61 adds class-2 weight and the hinge')


# ============================================================ the running arms
def running_arms():
    """Slide 'what is running now': the six arms, what each changes, what it targets."""
    arms = [('v552', 'point-loss weight  1 → 5', 'point', '#C0392B'),
            ('v553', 'edge-loss weight  33 → 3', 'edge', '#7A44AA'),
            ('v554', 'focal loss  γ = 2 on both heads', 'loss shape', '#B8860B'),
            ('v555', 'chain-level min-pooling recall', 'recall', '#0B7285'),
            ('v556', 'all confirmed levers, stacked on v551', 'stack', '#2E8B57'),
            ('v557–561', 'high-lr controls, B2 re-test', 'control', '#777777')]
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(0, 97, 'Six pruning arms in flight', fontsize=14, fontweight='bold', color=BLUE,
            va='top')
    ax.text(0, 89, 'every arm is judged against the v500 / v38 / v551 base it starts from',
            fontsize=10, color=GRAY, va='top')
    y = 78
    for ver, what, tag, col in arms:
        ax.add_patch(FancyBboxPatch((0, y - 4.6), 17, 9.2,
                                    boxstyle='round,pad=0.4,rounding_size=1.6',
                                    fc=col, ec=col, alpha=0.92))
        ax.text(8.5, y, ver, ha='center', va='center', fontsize=11.5, color='white',
                fontweight='bold')
        ax.text(20, y + 1.4, what, ha='left', va='center', fontsize=11, color=DARK)
        ax.text(20, y - 3.6, 'target: ' + tag, ha='left', va='center', fontsize=9, color=col)
        y -= 13.2
    ax.text(0, 2, 'Point loss 1 → 5 and the 33 → 3 edge weight are the two levers the '
                  'diagnosis points at.', fontsize=9.5, color=GRAY, va='bottom', style='italic')
    fig.savefig(FIG + '/running_arms.png', bbox_inches='tight')
    plt.close(fig)
    print('  running_arms.png')


if __name__ == '__main__':
    build()
    running_arms()
