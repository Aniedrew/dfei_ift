#!/usr/bin/env python3
"""Figures for the 2026-10-12 talk, v4 - "Which optimisations actually worked".

v4 = v3 figure set with
  * E1 fixed everywhere: the two chain-survival readings are named apart and
    never conflated -
        benchmark  -> "chain survival @ pruning threshold 0.9"      (fixed 0.9)
        probe      -> "chain survival @ 90% edge-precision point"   (data-driven)
    (they are different thresholds; the two columns are NOT comparable).
  * E3 fixed: F3 caption now reads v633 -> v642 (p2 0.668 -> 0.803), matching
    the body text; 0.663 (v648) is no longer quoted.
  * layout repairs on the five figures that had overlapping text in v3:
    F3 (titles / legend vs annotation), F5 (x tick labels),
    F6 (panel titles + error-bar text), F8 (legend vs curves, value labels),
    F10 (bottom formula row vs the italic "schematic" note).
  * new figures for the PART 0 setup and for the new 3000-event result:
    F12 one-event anatomy, F13 the evaluation pools, F11 the 3000-event
    chain-loss benchmark (v648 -> v654; only the successful arm is shown).
  * F8 now reads report_figs/principles_provenance_features.json (the re-run
    archive); if that file is missing it falls back to the v3 values with a
    warning.

Every number is either given in the talk brief or taken from the archived
re-runs under report_figs/ - nothing is invented.
Only matplotlib (Agg) is used - no seaborn, no training, no dataloader.
All text is English and at least 13 pt.

Run:
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python make_principles_figs_20261012_v4.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20261012_v4'
os.makedirs(FIG, exist_ok=True)

# ---- palette, matching the 2026-09-15 deck -------------------------------
BLUE = '#1F4E79'
DARK = '#2B2B2B'
GRAY = '#777777'
PURPLE = '#8E44AD'
TEAL = '#2C7A7B'
PINK = '#C2185B'
GOLD = '#B8860B'
GREEN = '#2E8B57'
RED = '#C0392B'
LIGHT = '#D9D9D9'

plt.rcParams.update({'font.size': 13, 'figure.dpi': 200, 'text.color': DARK,
                     'axes.edgecolor': '#BBBBBB', 'axes.labelcolor': DARK,
                     'xtick.color': DARK, 'ytick.color': DARK,
                     'axes.titlesize': 15, 'axes.titleweight': 'bold'})

# ---- E1: the two chain-survival readings, named apart --------------------
CS_BENCH = 'chain survival at the 0.9 score threshold'
CS_PROBE = "chain survival at the probe's 90%-edge-precision threshold"
FOOT_BENCH = (CS_BENCH + ' = fraction of truth chains fully recovered at a FIXED\n'
              'pruning score threshold 0.9 (nodes and edges) - 500-event test set')
FOOT_BENCH3K = (CS_BENCH + ' = fraction of truth chains fully recovered at a FIXED\n'
                'pruning score threshold 0.9 (nodes and edges) - 3000-event test set')
FOOT_PROBE = (CS_PROBE + ' = fraction of truth chains fully recovered at the\n'
              'score threshold that gives 90% edge precision - multi-B probe')
# the one-line statement that the two thresholds differ (E1)
E1_NOTE = ('NB: the benchmark uses a FIXED threshold 0.9, the probe a '
           'data-driven\n90%-edge-precision threshold - the two columns are '
           'different rulers, not comparable.')


def save(fig, name):
    fig.savefig(FIG + '/' + name, bbox_inches='tight')
    plt.close(fig)
    print('  ' + name)


def formula(tex, name, fontsize=24, color=DARK):
    """Render one mathtext formula to its own PNG (white background)."""
    fig = plt.figure(figsize=(6.0, 1.2))
    fig.text(0.5, 0.5, tex, ha='center', va='center', fontsize=fontsize,
             color=color)
    fig.savefig(FIG + '/' + name, bbox_inches='tight', pad_inches=0.10,
                dpi=200, facecolor='white')
    plt.close(fig)
    print('  ' + name)


def minrank(d):
    """rank 1 = largest value; ties share the best rank."""
    items = sorted(d.items(), key=lambda kv: -kv[1])
    ranks = {}
    for i, (k, v) in enumerate(items):
        if i > 0 and items[i - 1][1] == v:
            ranks[k] = ranks[items[i - 1][0]]
        else:
            ranks[k] = i + 1
    return ranks


def spread(d):
    """spread tied ranks vertically so the labels do not overlap."""
    grp = {}
    for k, r in d.items():
        grp.setdefault(r, []).append(k)
    dy = {}
    for r, ks in grp.items():
        n = len(ks)
        for i, k in enumerate(ks):
            dy[k] = r + (i - (n - 1) / 2.0) * 0.6
    return dy


# =========================== the data =====================================
# 500-event test benchmark, 18 versions
AP = {'v623': 0.6534, 'v624': 0.6529, 'v648': 0.6524, 'v628': 0.6519,
      'v601': 0.6480, 'v635': 0.6480, 'v631': 0.6394, 'v649': 0.6394,
      'v642': 0.6389, 'v641': 0.6380, 'v643': 0.6372, 'v633': 0.6371,
      'v640': 0.6361, 'v639': 0.6350, 'v636': 0.6344, 'v637': 0.6341,
      'v646': 0.6267, 'v632': 0.6081}
PR90 = {'v623': 0.5368, 'v624': 0.5308, 'v648': 0.5004, 'v628': 0.5346,
        'v601': 0.5356, 'v635': 0.5016, 'v631': 0.5029, 'v649': 0.5017,
        'v642': 0.4960, 'v641': 0.5011, 'v643': 0.4986, 'v633': 0.5078,
        'v640': 0.5003, 'v639': 0.4999, 'v636': 0.5023, 'v637': 0.5054,
        'v646': 0.5051, 'v632': 0.5036}
CS = {'v623': 47.13, 'v624': 42.90, 'v648': 38.67, 'v628': 50.76,
      'v601': 44.11, 'v635': 42.90, 'v631': 47.13, 'v649': 44.11,
      'v642': 61.63, 'v641': 59.21, 'v643': 35.95, 'v633': 42.30,
      'v640': 43.20, 'v639': 45.62, 'v636': 43.20, 'v637': 40.48,
      'v646': 46.53, 'v632': 49.85}
N2 = {'v623': 50.4, 'v624': 44.7, 'v648': 44.0, 'v628': 54.6,
      'v601': 45.4, 'v635': 46.1, 'v631': 51.8, 'v649': 46.8,
      'v642': 64.5, 'v641': 65.2, 'v643': 39.7, 'v633': 44.7,
      'v640': 46.1, 'v639': 49.6, 'v636': 46.8, 'v637': 42.6,
      'v646': 48.9, 'v632': 53.9}
N3 = {'v623': 37.5, 'v624': 35.0, 'v648': 28.8, 'v628': 41.2,
      'v601': 36.2, 'v635': 35.0, 'v631': 40.0, 'v649': 35.0,
      'v642': 60.0, 'v641': 50.0, 'v643': 27.5, 'v633': 32.5,
      'v640': 37.5, 'v639': 35.0, 'v636': 33.8, 'v637': 32.5,
      'v646': 36.2, 'v632': 40.0}
P2 = {'v623': 0.710, 'v624': 0.668, 'v648': 0.663, 'v628': 0.739,
      'v601': 0.674, 'v635': 0.679, 'v631': 0.720, 'v649': 0.684,
      'v642': 0.803, 'v641': 0.808, 'v643': 0.630, 'v633': 0.668,
      'v640': 0.679, 'v639': 0.705, 'v636': 0.684, 'v637': 0.652,
      'v646': 0.700, 'v632': 0.734}

# 3000-event chain-loss benchmark (report_figs/prune_loss_chains_v648_mb.csv /
# v654_mb.csv).  Only the successful arm is shown - the softmin writing (v653)
# is a failed arm and is deliberately left out of the deck.
MB3K = [('v648', 'v648\n(base, no chain loss)', 0.6684,
         [35.77, 36.9, 35.0], 0.608, GRAY),
        ('v654', 'v654\n+ chain_recall\n(min-pooling weakest link)', 0.6591,
         [52.26, 53.0, 52.8], 0.728, GREEN)]


# =========================== F1 ===========================================
def f1():
    ap_r, cs_r = minrank(AP), minrank(CS)
    yl, yr = spread(ap_r), spread(cs_r)
    hl = {'v648': PINK, 'v643': GOLD, 'v641': TEAL, 'v642': GREEN,
          'v623': BLUE, 'v632': PURPLE}
    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    for v in AP:
        c = hl.get(v, LIGHT)
        lw = 2.8 if v in hl else 1.2
        z = 5 if v in hl else 1
        ax.plot([0, 1], [yl[v], yr[v]], color=c, lw=lw, zorder=z,
                solid_capstyle='round')
        ax.scatter([0, 1], [yl[v], yr[v]], color=c, s=52 if v in hl else 16,
                   zorder=z + 1, edgecolor='white', linewidth=0.6)
    for v in AP:
        c = hl.get(v, GRAY)
        fw = 'bold' if v in hl else 'normal'
        ax.text(-0.035, yl[v], v, ha='right', va='center', fontsize=13,
                color=c, fontweight=fw)
        ax.text(1.035, yr[v], v, ha='left', va='center', fontsize=13,
                color=c, fontweight=fw)
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(19.8, -2.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.0, -1.9, 'ranked by\nhard-pool AP', ha='center', va='center',
            fontsize=14, color=BLUE, fontweight='bold')
    ax.text(1.0, -1.9, 'ranked by\nchain survival', ha='center', va='center',
            fontsize=14, color=BLUE, fontweight='bold')
    ax.set_title('Two rulers, two orderings - the same 18 versions\n'
                 '500-event test set, pruning threshold 0.9 (fixed)',
                 fontsize=15, pad=10)

    def lbl(v):
        return '%s:  AP #%d  ->  surv #%d  (%.1f%%)' % (
            v, ap_r[v], cs_r[v], CS[v])
    handles = [plt.Line2D([0], [0], color=hl[v], lw=3, label=lbl(v)) for v in
               ('v642', 'v641', 'v648', 'v643', 'v623', 'v632')]
    fig.supxlabel(FOOT_BENCH, fontsize=13, color=GRAY)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=13, frameon=False, handlelength=1.8)
    save(fig, 'F1_dumbbell_ap_vs_chain.png')


# =========================== F2 ===========================================
def f2():
    ap_r = minrank(AP)
    order = sorted(CS, key=lambda v: CS[v])          # ascending -> top bar = best
    ys = np.arange(len(order))
    hl = {'v642': GREEN, 'v641': TEAL, 'v648': PINK, 'v643': GOLD}
    cols = [hl.get(v, BLUE) for v in order]
    fig, ax = plt.subplots(figsize=(8.0, 6.4), constrained_layout=True)
    ax.barh(ys, [CS[v] for v in order], color=cols, height=0.72, zorder=3)
    for i, v in enumerate(order):
        ax.text(CS[v] + 0.7, i, '%.1f%%   (AP #%d)' % (CS[v], ap_r[v]),
                va='center', fontsize=13, color=DARK)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=13)
    ax.set_xlim(0, 84)
    ax.set_xlabel('chain survival at the 0.9 score threshold  [%]', fontsize=14)
    ax.set_title("Chain survival, best to worst - with each arm's AP rank", pad=12)
    ax.grid(axis='x', color='#EEEEEE', zorder=0)
    fig.supxlabel(FOOT_BENCH, fontsize=13, color=GRAY)
    save(fig, 'F2_chain_surv_bar.png')


# =========================== F3 ===========================================
def f3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.2),
                                 constrained_layout=True)
    # -- panel A: p^n, the AND semantics
    p = np.linspace(0.5, 1.0, 200)
    a1.plot(p, p, color=GRAY, lw=1.8, ls='--', label='p  (1 link)')
    for n, c in ((2, TEAL), (3, PINK), (4, PURPLE)):
        a1.plot(p, p ** n, color=c, lw=2.6, label='p^%d  (%d links)' % (n, n))
    a1.axvline(0.8, color='#BBBBBB', lw=1.2)
    a1.text(0.805, 0.30, 'p = 0.80:\n0.640 / 0.512 / 0.410\n(n = 2 / 3 / 4)',
            fontsize=13, color=DARK)
    a1.set_xlabel('per-link success probability  p', fontsize=14)
    a1.set_ylabel('chain survival  =  p^n', fontsize=14)
    a1.set_title('AND semantics: one dead link kills the chain', fontsize=15)
    a1.set_ylim(0, 1.02)
    a1.legend(fontsize=13, frameon=False, loc='upper left')
    a1.grid(color='#EEEEEE')
    # -- panel B: measured survival vs per-link p
    pp = np.linspace(0.60, 0.82, 100)
    a2.plot(pp, 100 * pp ** 2, color=TEAL, lw=2.2, ls='--', zorder=1,
            label='p^2  (n = 2 truth chains)')
    a2.plot(pp, 100 * pp ** 3, color=PINK, lw=2.2, ls='--', zorder=1,
            label='p^3  (n = 3 truth chains)')
    a2.scatter([P2[v] for v in N2], [N2[v] for v in N2], color=TEAL, s=50,
               zorder=3, edgecolor='white', linewidth=0.7)
    a2.scatter([P2[v] for v in N3], [N3[v] for v in N3], color=PINK, s=50,
               zorder=3, edgecolor='white', linewidth=0.7)
    for v, dx, ha in (('v642', 9, 'left'), ('v633', -9, 'right')):
        a2.annotate('%s\np2 = %.3f' % (v, P2[v]), (P2[v], N2[v]),
                    textcoords='offset points', xytext=(dx, 6), ha=ha,
                    fontsize=13, color=DARK)
    a2.annotate('', xy=(P2['v642'], N2['v642']),
                xytext=(P2['v633'], N2['v633']),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5,
                                connectionstyle='arc3,rad=-0.25'))
    a2.text(0.04, 0.96, 'v633 -> v642:\np2 0.668 -> 0.803\nlifts 2-link survival\n'
            '44.7% -> 64.5%', transform=a2.transAxes, fontsize=13,
            color=GREEN, fontweight='bold', va='top')
    a2.set_ylim(15, 74)
    a2.set_xlabel('per-link success probability  p2', fontsize=14)
    a2.set_ylabel('chain survival  [%]', fontsize=14)
    a2.set_title('18 arms collapse onto the p^n law', fontsize=15)
    a2.legend(fontsize=13, frameon=False, loc='lower right')
    a2.grid(color='#EEEEEE')
    fig.suptitle('Idea 1a: the chain is an AND, so a single link decides it',
                 fontsize=16, fontweight='bold', color=BLUE)
    fig.supxlabel('measured chain survival vs the per-link p2 of the same run '
                  '(500-event benchmark, pruning threshold 0.9);\n'
                  'v633 -> v642: p2 0.668 -> 0.803 lifts 2-link survival '
                  '44.7% -> 64.5%; dashed curves = ideal AND prediction',
                  fontsize=13, color=GRAY)
    save(fig, 'F3_and_semantics.png')


# =========================== F4 ===========================================
def f4():
    g = np.logspace(0, 2, 400)                        # gamma = 1 ... 100
    p = np.array([0.95, 0.50, 0.05])
    surv = -(1.0 / g) * np.log(np.mean(np.exp(-g[:, None] * p[None, :]), axis=1))
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(g, surv, color=BLUE, lw=2.8, label='softmin survival  surv(gamma)')
    ax.axhline(p.min(), color=GREEN, lw=2.0, ls='--',
               label='min(p) = 0.05  (weakest link)')
    ax.axhline(p.mean(), color=GOLD, lw=2.0, ls=':',
               label='mean(p) = 0.50  (per-edge average)')
    ax.annotate('gamma -> large:\nsurv -> min(p)', xy=(55, 0.10), xytext=(2.2, 0.22),
                fontsize=13, color=DARK,
                arrowprops=dict(arrowstyle='->', color=GRAY))
    ax.annotate('gamma -> 1:\nsurv -> mean(p)', xy=(1.15, 0.50), xytext=(1.7, 0.74),
                fontsize=13, color=DARK,
                arrowprops=dict(arrowstyle='->', color=GRAY))
    ax.set_xscale('log')
    ax.set_xlabel('softmin sharpness  gamma  (1 ... 100)', fontsize=14)
    ax.set_ylabel('survival estimate', fontsize=14)
    ax.set_title('Softmin: a differentiable weakest link', pad=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=13, frameon=False, loc='center right')
    ax.grid(color='#EEEEEE')
    fig.supxlabel('worked example (not data):  p = [0.95, 0.50, 0.05]',
                  fontsize=13, color=GRAY)
    save(fig, 'F4_softmin_gamma.png')


# =========================== F5 ===========================================
CFG = [('w = 0', 'baseline', [24.6, 23.6, 24.5], 24.2, 0.8393, 0.8650, GRAY),
       ('prod\nw = 1.0', '~ per-edge BCE', [29.2, 24.6, 30.1], 28.0, 0.8413, 0.8691, GOLD),
       ('softmin\nw = 0.3', 'softer minimum', [32.4, 23.2, 34.8], 30.1, 0.8468, 0.8713, TEAL),
       ('softmin\nw = 1.0', 'sharp minimum', [33.5, 28.5, 35.8], 32.6, 0.8533, 0.8732, GREEN)]


def f5():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.4),
                                 constrained_layout=True)
    x = np.arange(len(CFG))
    a1.bar(x, [c[3] for c in CFG], color=[c[6] for c in CFG], width=0.60,
           zorder=3, alpha=0.85)
    for i, c in enumerate(CFG):
        a1.scatter(i + np.array([-0.13, 0.0, 0.13]), c[2], color=DARK, s=46,
                   zorder=5, edgecolor='white', linewidth=0.7)
        a1.text(i, max(c[2]) + 1.4, '%.1f' % c[3], ha='center', fontsize=13,
                fontweight='bold', color=DARK)
    a1.set_xticks(x)
    a1.set_xticklabels([c[0] for c in CFG], fontsize=13)
    a1.set_ylabel("chain survival at the probe's\n90%-edge-precision threshold  [%]",
                  fontsize=13)
    a1.set_ylim(0, 42)
    a1.set_title('Chain survival (3 seeds + mean)', fontsize=15)
    a1.grid(axis='y', color='#EEEEEE', zorder=0)
    for i, c in enumerate(CFG):
        a2.plot([i, i], [c[4], c[5]], color='#CCCCCC', lw=1.6, zorder=1)
    a2.scatter(x, [c[4] for c in CFG], color=BLUE, s=95, zorder=3, label='AP')
    a2.scatter(x, [c[5] for c in CFG], color=PURPLE, s=95, marker='s', zorder=3,
               label='AUC')
    for i, c in enumerate(CFG):
        a2.text(i - 0.10, c[4], '%.4f' % c[4], ha='right', va='center',
                fontsize=13, color=BLUE)
        a2.text(i + 0.10, c[5], '%.4f' % c[5], ha='left', va='center',
                fontsize=13, color=PURPLE)
    a2.set_xticks(x)
    a2.set_xticklabels([c[0] for c in CFG], fontsize=13)
    a2.set_xlim(-0.65, 3.70)
    a2.set_ylim(0.826, 0.892)
    a2.set_ylabel('edge-level score', fontsize=14)
    a2.set_title('Edge-level AP and AUC (secondary)', fontsize=15)
    a2.legend(fontsize=13, frameon=False, loc='upper left')
    a2.grid(axis='y', color='#EEEEEE')
    fig.suptitle('Idea 1c - the fix has to be non-linear (small-model probe)',
                 fontsize=16, fontweight='bold', color=BLUE)
    fig.supxlabel('multi-B probe: 696 events / 281 truth chains / 5248 test edges\n'
                  + FOOT_PROBE + '\nlabels: baseline = per-edge BCE; prod = product '
                  '/ geometric-mean loss; softmin w = weakest-link weight',
                  fontsize=13, color=GRAY)
    save(fig, 'F5_chainloss_probe.png')


# =========================== F6 ===========================================
def f6():
    names = ['attn L4\n30-dim', 'attn L4\n47-dim', 'MLP\n47-dim']
    ap = [0.8460, 0.8450, 0.8015]
    ap_s = [0.0118, 0.0172, 0.0149]
    auc = [0.8745, 0.8695, 0.8480]
    auc_s = [0.0094, 0.0129, 0.0047]
    cols = [TEAL, BLUE, GOLD]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.2),
                                 constrained_layout=True)
    x = np.arange(3)
    a1.errorbar(x, ap, yerr=ap_s, fmt='none', capsize=8, lw=2.0, ecolor=GRAY, zorder=3)
    for i in range(3):
        a1.scatter(i, ap[i], s=210, color=cols[i], zorder=4,
                   edgecolor='white', linewidth=1.0)
        a1.text(i, ap[i] + ap_s[i] + 0.005, '%.4f\n+/- %.4f' % (ap[i], ap_s[i]),
                ha='center', fontsize=13, color=DARK)
    a1.set_xticks(x)
    a1.set_xticklabels(names, fontsize=13)
    a1.set_xlim(-0.55, 2.55)
    a1.set_ylim(0.770, 0.890)
    a1.set_ylabel('AP', fontsize=14)
    a1.set_title('Attention beats the MLP by ~4.5 AP points', fontsize=15)
    a1.grid(axis='y', color='#EEEEEE')
    a2.errorbar(x, auc, yerr=auc_s, fmt='none', capsize=8, lw=2.0, ecolor=GRAY, zorder=3)
    for i in range(3):
        a2.scatter(i, auc[i], s=210, color=cols[i], zorder=4,
                   edgecolor='white', linewidth=1.0)
        a2.text(i, auc[i] + auc_s[i] + 0.004, '%.4f\n+/- %.4f' % (auc[i], auc_s[i]),
                ha='center', fontsize=13, color=DARK)
    a2.set_xticks(x)
    a2.set_xticklabels(names, fontsize=13)
    a2.set_xlim(-0.55, 2.55)
    a2.set_ylim(0.825, 0.905)
    a2.set_ylabel('AUC', fontsize=14)
    a2.set_title('The MLP has the smallest AUC spread', fontsize=15)
    a2.grid(axis='y', color='#EEEEEE')
    fig.suptitle('Idea 2b - attention helps most where chains compete',
                 fontsize=16, fontweight='bold', color=BLUE)
    fig.supxlabel('small-model probe (pure CPU), multi-B subset, 3-4 seeds\n'
                  'error bars = run-to-run spread of the seeds',
                  fontsize=13, color=GRAY)
    save(fig, 'F6_attn_vs_mlp.png')


# =========================== F8 ===========================================
def _feat_ceiling():
    """Read the archived multi-B feature-ceiling re-run (E5); fall back to v3."""
    path = BASE + '/report_figs/principles_provenance_features.json'
    fall_multi = [0.6916, 0.8043, 0.8291, 0.8360, 0.8391]
    fall_single = [0.7934, 0.8256, 0.8344, 0.8367, 0.8365]
    fall_imp = [('doca', 0.166), ('trdist', 0.073), ('flight', 0.048),
                ('m', 0.022)]
    if not os.path.exists(path):
        print('[warn] missing ' + path + ' -> using v3 fallback numbers')
        return fall_multi, fall_single, fall_imp
    d = json.load(open(path))
    grp = d['multi_b']['groups']
    order = ['node', 'node+geo', 'node+geo+raw+der',
             'node+geo+raw+der+vrt', 'node+geo+raw+der+vrt+ctx']
    multi = [grp[k]['ap'] for k in order]
    single = [d['single_b']['groups'][k]['ap'] for k in order]
    imp = [(x['name'], x['importance']) for x in d['multi_b']['importance'][:4]]
    print('[F8] using archived %s:  multi=%s' % (
        os.path.basename(path), ['%.4f' % v for v in multi]))
    print('[F8] single:', ['%.4f' % v for v in single])
    print('[F8] importance:', [(n, '%.3f' % v) for n, v in imp])
    return multi, single, imp


def f8():
    groups = ['node', '+ geo', '+ raw+der', '+ vrt', '+ ctx']
    multi, single, imp = _feat_ceiling()
    if single is None:
        # single-B column is a v3 companion reading of the same probe
        single = [0.7934, 0.8256, 0.8344, 0.8367, 0.8365]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                                 constrained_layout=True)
    x = np.arange(5)
    a1.plot(x, multi, 'o-', color=BLUE, lw=2.6, ms=11, label='multi-B subset')
    a1.plot(x, single, 's--', color=TEAL, lw=2.6, ms=10, label='single-B subset')
    for i in range(5):                                   # multi labels below
        a1.annotate('%.4f' % multi[i], (i, multi[i]), textcoords='offset points',
                    xytext=(0, -15), ha='center', va='top', fontsize=13, color=BLUE)
    for i in range(5):                                   # single labels above
        a1.annotate('%.4f' % single[i], (i, single[i]), textcoords='offset points',
                    xytext=(0, 11), ha='center', va='bottom', fontsize=13, color=TEAL)
    a1.text(0.52, 0.712, '+0.113', ha='left', va='top', fontsize=13,
            color=BLUE, fontweight='bold')
    a1.text(0.18, 0.852, '+0.032', ha='left', va='bottom', fontsize=13,
            color=TEAL, fontweight='bold')
    a1.set_xticks(x)
    a1.set_xticklabels(groups, fontsize=13)
    a1.set_ylim(0.655, 0.885)
    a1.set_ylabel('GBDT AP', fontsize=14)
    a1.set_title('Cumulative feature groups (GBDT upper bound)', fontsize=15)
    a1.legend(fontsize=13, frameon=False, loc='lower right')
    a1.grid(color='#EEEEEE')
    a2.barh(range(len(imp)), [v for _, v in imp],
            color=[BLUE, TEAL, PURPLE, GOLD][:len(imp)], height=0.62, zorder=3)
    for i, (_, v) in enumerate(imp):
        a2.text(v + 0.005, i, '+%.3f' % v, va='center', fontsize=13, color=DARK)
    a2.set_yticks(range(len(imp)))
    _expand = {
        'doca': 'doca\n(closest approach)',
        'trdist': 'trdist\n(transverse distance\nat the two ends)',
        'flight': 'flight\n(flight-related\nvertex quantity)',
        'm': 'm\n(pair invariant mass)',
    }
    a2.set_yticklabels([_expand.get(k, k) for k, _ in imp], fontsize=12)
    a2.invert_yaxis()
    a2.set_xlim(0, 0.20)
    a2.set_xlabel('permutation importance  (delta AP)', fontsize=14)
    a2.set_title('Multi-B: geometry dominates the trees', fontsize=15)
    a2.grid(axis='x', color='#EEEEEE', zorder=0)
    fig.suptitle('Idea 3b - the evidence: feature upper bound (GBDT probe, '
                 'not a model)', fontsize=16, fontweight='bold', color=BLUE)
    fig.supxlabel('after fixing the feature-indexing bug; AP on the same '
                  'multi-B / single-B pools\n'
                  'bold labels = delta AP bought by the labelled step '
                  '(+0.113 = multi-B node -> +geo; +0.032 = single-B node -> +geo)\n'
                  'feature names on the right axis are spelled out: doca = '
                  'closest approach, trdist = transverse distance at the ends, '
                  'flight = flight-related vertex quantity, m = pair invariant '
                  'mass', fontsize=12, color=GRAY)
    save(fig, 'F8_features.png')


# =========================== F9 (schematics) ==============================
def f9():
    """Attention over the neighbourhood - MLP vs attention on one shared graph."""
    nodes = {'A': (0.24, 0.52), 'B': (0.58, 0.52),
             'C': (0.40, 0.88), 'D': (0.40, 0.16),
             'E': (0.10, 0.86), 'F': (0.82, 0.84),
             'G': (0.86, 0.30), 'H': (0.10, 0.20)}
    centre = ('A', 'B')
    nbrs = [('A', 'C', 0.85, 0.33, 0.755, 'center'),
            ('A', 'E', 0.45, 0.055, 0.735, 'left'),
            ('A', 'H', 0.25, 0.19, 0.315, 'center'),
            ('B', 'D', 0.65, 0.47, 0.285, 'center'),
            ('B', 'F', 0.35, 0.745, 0.735, 'left'),
            ('B', 'G', 0.55, 0.755, 0.44, 'left')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.6),
                                 constrained_layout=True)

    def base(ax):
        for n1, n2, _, _, _, _ in nbrs:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            ax.plot([x1, x2], [y1, y2], color=LIGHT, lw=2.2, zorder=1)
        for n, (x, y) in nodes.items():
            ax.add_patch(Circle((x, y), 0.032, fc='white', ec=GRAY, lw=1.6,
                                zorder=4))
        x1, y1 = nodes[centre[0]]
        x2, y2 = nodes[centre[1]]
        ax.plot([x1, x2], [y1, y2], color=BLUE, lw=5.0, zorder=2,
                solid_capstyle='round')
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 0.10, 'e  (centre edge)',
                ha='center', va='top', fontsize=13, color=BLUE, fontweight='bold')

    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.axis('off')
    base(a1)
    a1.text(0.5, 1.04, 'MLP:  f(x_e)', ha='center', va='bottom', fontsize=16,
            color=GRAY, fontweight='bold')
    a1.text(0.5, -0.04, 'the centre edge\nsees only its own features  x_e',
            ha='center', va='top', fontsize=13, color=GRAY)
    a1.add_patch(Rectangle((0.34, 0.02), 0.32, 0.10, fc='#EEEEEE', ec=GRAY,
                           lw=1.0, zorder=3))
    a1.text(0.5, 0.07, 'x_e  (isolated)', ha='center', va='center',
            fontsize=13, color=GRAY)

    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis('off')
    for n1, n2, _, _, _, _ in nbrs:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        a2.plot([x1, x2], [y1, y2], color=LIGHT, lw=2.2, zorder=1)
    for n, (x, y) in nodes.items():
        a2.add_patch(Circle((x, y), 0.032, fc='white', ec=GRAY, lw=1.6, zorder=4))
    x1, y1 = nodes[centre[0]]
    x2, y2 = nodes[centre[1]]
    a2.plot([x1, x2], [y1, y2], color=BLUE, lw=5.0, zorder=2,
            solid_capstyle='round')
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for n1, n2, w, lx, ly, lha in nbrs:
        mx, my = (nodes[n1][0] + nodes[n2][0]) / 2, (nodes[n1][1] + nodes[n2][1]) / 2
        a2.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]],
                color=TEAL, lw=1.0 + 6.0 * w, alpha=0.85, zorder=2,
                solid_capstyle='round')
        a2.add_patch(FancyArrowPatch((mx, my), (cx, cy), arrowstyle='-|>',
                                     mutation_scale=12, color=GOLD, lw=1.4,
                                     alpha=0.9, zorder=3,
                                     shrinkA=2, shrinkB=6))
        a2.text(lx, ly, r'$\alpha$=%.2f' % w, ha=lha, va='center',
                fontsize=13, color=TEAL)
    a2.text(0.5, 1.04, 'Attention:  h_e  <-  sum_f  alpha_ef  W_v h_f',
            ha='center', va='bottom', fontsize=16, color=TEAL, fontweight='bold')
    a2.text(0.5, -0.04,
            r'line width = attention weight $\alpha_{ef}$;  '
            r'$\alpha_{ef}=\mathrm{softmax}_f((W_q h_e)\cdot(W_k h_f)/\sqrt{d})$',
            ha='center', va='top', fontsize=13, color=DARK)
    fig.suptitle('Idea 2: attention over the neighbourhood', fontsize=16,
                 fontweight='bold', color=BLUE)
    fig.supxlabel('the same node-edge skeleton in both panels: only the '
                  'information each edge may use changes', fontsize=13,
                  color=GRAY)
    save(fig, 'F9_attention_principle.png')


# =========================== F10 ==========================================
def f10():
    """Competition context: true edge vs cross-chain fake, with definitions."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                                 gridspec_kw={'width_ratios': [1.25, 1.0]},
                                 constrained_layout=True)

    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.axis('off')
    a1.text(0.5, 1.04, 'One track, several candidate partners',
            ha='center', va='bottom', fontsize=14, color=BLUE, fontweight='bold')
    tx, ty = 0.13, 0.50
    a1.add_patch(Rectangle((0.05, 0.43), 0.17, 0.14, fc=BLUE, ec='none', zorder=3))
    a1.text(tx, ty, 'track', ha='center', va='center', fontsize=13,
            color='white', fontweight='bold', zorder=4)

    true_p = [(0.60, 0.78), (0.68, 0.65)]
    fake_p = [(0.58, 0.30), (0.68, 0.15)]
    a1.add_patch(Rectangle((0.53, 0.58), 0.28, 0.29, fc='none', ec=GREEN,
                           lw=1.6, ls=(0, (5, 3)), zorder=1))
    a1.text(0.67, 0.88, 'same 2nd vertex', ha='center', va='bottom',
            fontsize=13, color=GREEN)
    for (px, py) in true_p:
        a1.add_patch(FancyArrowPatch((tx + 0.09, ty), (px, py),
                                     arrowstyle='-|>', mutation_scale=12,
                                     color=GREEN, lw=2.6, zorder=2, shrinkB=8))
        a1.add_patch(Circle((px, py), 0.026, fc='white', ec=GREEN, lw=1.8, zorder=3))
    for (px, py) in fake_p:
        a1.add_patch(FancyArrowPatch((tx + 0.09, ty), (px, py),
                                     arrowstyle='-|>', mutation_scale=12,
                                     color=PINK, lw=2.2, ls=(0, (4, 3)), zorder=2,
                                     shrinkB=8))
        a1.add_patch(Circle((px, py), 0.026, fc='white', ec=PINK, lw=1.8, zorder=3))
    a1.text(0.78, 0.70, 'true edge\n(solid)', ha='left', va='center',
            fontsize=13, color=GREEN)
    a1.text(0.78, 0.22, 'cross-chain\nfake (dashed)', ha='left', va='center',
            fontsize=13, color=PINK)
    a1.text(0.5, 0.06, r'$a=e^{-d_3/50}$        $r=a/\max_f a_f$',
            ha='center', va='bottom', fontsize=14, color=DARK)

    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis('off')
    a2.text(0.5, 1.04, 'What the features separate',
            ha='center', va='bottom', fontsize=14, color=BLUE, fontweight='bold')
    a2.text(0.03, 0.83, 'feature', ha='left', va='center', fontsize=14,
            color=DARK, fontweight='bold')
    a2.text(0.68, 0.83, 'true', ha='center', va='center', fontsize=14,
            color=GREEN, fontweight='bold')
    a2.text(0.92, 0.83, 'fake', ha='center', va='center', fontsize=14,
            color=PINK, fontweight='bold')
    rows = [('r  (norm. affinity)', 'high', 'low'),
            ('h_i  (# high-aff nbrs)', 'many', 'few'),
            ('cnt  (# partners)', 'similar', 'similar')]
    y = 0.66
    for r0, r1, r2 in rows:
        a2.plot([0.02, 0.98], [y + 0.12, y + 0.12], color='#DDDDDD', lw=1.0)
        a2.text(0.03, y, r0, ha='left', va='center', fontsize=13, color=DARK)
        a2.text(0.68, y, r1, ha='center', va='center', fontsize=13, color=GREEN)
        a2.text(0.92, y, r2, ha='center', va='center', fontsize=13, color=PINK)
        y -= 0.24
    a2.text(0.5, 0.06, 'h_i = # high-affinity partners;  cnt = # partners',
            ha='center', va='bottom', fontsize=13, color=DARK)
    fig.suptitle('Idea 3: competition context, not just geometry', fontsize=16,
                 fontweight='bold', color=BLUE)
    fig.supxlabel('schematic - qualitative only, no numbers shown',
                  fontsize=13, color=GRAY)
    save(fig, 'F10_competition_context.png')


# =========================== F11 (new: 3000-event) ========================
def f11():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.4),
                                 constrained_layout=True)
    # -- left: survival split by chain length (2 arms x 3 lengths) ----------
    labels = ['all truth\nchains', '2-edge\nchains', '3-edge\nchains']
    x = np.arange(3)
    bw = 0.36
    for k, c in enumerate(MB3K):
        vals = c[3]
        off = (k - 0.5) * bw
        a1.bar(x + off, vals, width=bw, color=c[5], zorder=3, alpha=0.9,
               edgecolor='white', linewidth=1.0,
               label=('%s  (p2 = %.3f)' % (c[0], c[4])))
        for i, v in enumerate(vals):
            a1.text(x[i] + off, v + 1.0, '%.1f' % v, ha='center', va='bottom',
                    fontsize=12, color=DARK, fontweight='bold')
    a1.annotate('', xy=(2 + 0.5 * bw, 52.8), xytext=(2 - 0.5 * bw, 35.0),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.6))
    a1.text(0.98, 0.97, '+17.8 pp on the\n3-edge chains', transform=a1.transAxes,
            ha='right', va='top', fontsize=13, color=GREEN, fontweight='bold')
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=13)
    a1.set_ylabel('chain survival at the 0.9 score threshold  [%]', fontsize=14)
    a1.set_ylim(0, 72)
    a1.set_title('Chain survival: the long chains gain the most', fontsize=15)
    a1.legend(fontsize=12, frameon=False, loc='upper left')
    a1.grid(axis='y', color='#EEEEEE', zorder=0)

    # -- right: edge-level AP cost -----------------------------------------
    xa = np.arange(len(MB3K))
    a2.bar(xa, [c[2] for c in MB3K], color=[c[5] for c in MB3K], width=0.50,
           zorder=3, alpha=0.9)
    for i, c in enumerate(MB3K):
        a2.text(i, c[2] + 0.0012, '%.4f' % c[2], ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=DARK)
    a2.set_xticks(xa)
    a2.set_xticklabels([c[1] for c in MB3K], fontsize=13)
    a2.set_ylabel('hard-pool AP', fontsize=14)
    a2.set_ylim(0.630, 0.692)
    a2.set_title('Edge-level AP: the cost is only -0.009', fontsize=15)
    a2.grid(axis='y', color='#EEEEEE', zorder=0)
    a2.annotate('-0.0093', xy=(1, 0.6591), xytext=(0.30, 0.6765),
                fontsize=13, color=DARK,
                arrowprops=dict(arrowstyle='->', color=GRAY))
    fig.suptitle('Idea 1 on the main model: the 3000-event benchmark',
                 fontsize=16, fontweight='bold', color=BLUE)
    fig.supxlabel('3000-event benchmark, hard pool 33166 edges; chain survival at '
                  'the 0.9 score threshold (FIXED for both arms, so the gain is '
                  'not a re-ranking)\n'
                  '+16.5 pp overall, and the 3-edge chains gain the most '
                  '(35.0 -> 52.8): the weakest link is what decides a long chain\n'
                  'source: report_figs/prune_loss_chains_v648_mb.csv / '
                  'prune_loss_chains_v654_mb.csv', fontsize=12, color=GRAY)
    save(fig, 'F11_chainloss_main.png')


# =========================== F12 (new: one-event anatomy) =================
def f12():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.9),
                                 gridspec_kw={'width_ratios': [1.15, 1.0]},
                                 constrained_layout=True)
    # ---------------- left: one event, four kinds of candidate edge --------
    a1.set_xlim(0, 1); a1.set_ylim(-0.45, 1.0); a1.axis('off')
    a1.set_title('One event, four kinds of candidate edge', fontsize=14)
    pv = (0.06, 0.50)
    sv1 = (0.32, 0.72); d1 = (0.56, 0.94); d2 = (0.60, 0.68)
    sv2 = (0.32, 0.28); e1 = (0.60, 0.36); e2 = (0.86, 0.20)
    bg = [(0.34, 0.04), (0.88, 0.52), (0.90, 0.88)]
    ends = [d1, d2, e1, e2] + bg
    for i in range(len(ends)):                       # faint complete graph
        for j in range(i + 1, len(ends)):
            a1.plot([ends[i][0], ends[j][0]], [ends[i][1], ends[j][1]],
                    color='#EDEDED', lw=0.9, zorder=0)
    for (px, py) in ends + [sv1, sv2]:               # tracks out of the PV
        a1.plot([pv[0], px], [pv[1], py], color=LIGHT, lw=1.6, zorder=1)
    for a, b in [(pv, sv1), (sv1, d1), (sv1, d2), (pv, sv2), (sv2, e1), (sv2, e2)]:
        a1.plot([a[0], b[0]], [a[1], b[1]], color=GREEN, lw=2.6, zorder=3,
                solid_capstyle='round')
    for a, b in [(d1, d2), (e1, e2)]:                # fake_intra
        a1.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=2.2, ls=(0, (4, 3)),
                zorder=4)
    a1.plot([d2[0], e1[0]], [d2[1], e1[1]], color=PINK, lw=2.2, ls=(0, (4, 3)),
            zorder=4)                                # fake_inter
    for (px, py) in [sv1, sv2, d1, d2, e1, e2] + bg:
        a1.scatter([px], [py], s=42, color='white', edgecolor=GRAY, linewidth=1.2,
                   zorder=5)
    a1.scatter([pv[0]], [pv[1]], s=70, color=BLUE, edgecolor='white',
               linewidth=1.2, zorder=6)
    a1.text(pv[0] - 0.01, pv[1] - 0.07, 'PV', ha='left', va='top', fontsize=13,
            color=BLUE, fontweight='bold')
    a1.text(sv1[0] - 0.02, sv1[1] + 0.05, 'secondary vertex (B)', ha='left',
            va='bottom', fontsize=13, color=GREEN)
    # fake_intra suggestion: sv->bg-ish dashed red, fake_inter: bg-bg
    a1.legend(handles=[
        plt.Line2D([0], [0], color=GREEN, lw=3, label='truth structural edge'),
        plt.Line2D([0], [0], color=RED, lw=2.4, ls=(0, (4, 3)),
                   label='fake_intra (same chain, wrong depth)'),
        plt.Line2D([0], [0], color=PINK, lw=2.4, ls=(0, (4, 3)),
                   label='fake_inter (two different chains)'),
        plt.Line2D([0], [0], color=LIGHT, lw=2.4, label='easy background edge')],
        loc='lower left', fontsize=12, frameon=False)

    # ---------------- right: pruning heads + reconstruction stages ---------
    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis('off')
    a2.set_title('Two pruning heads, then two reconstruction stages', fontsize=14)

    def box(cx, cy, w, h, lines, col):
        a2.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc='white',
                               ec=col, lw=1.8, zorder=3))
        a2.text(cx, cy, lines, ha='center', va='center', fontsize=13,
                color=col, fontweight='bold', zorder=4)

    box(0.5, 0.88, 0.94, 0.16,
        'node head:  "is this track a signal track?"\nkeep if score > 0.9', TEAL)
    box(0.5, 0.65, 0.94, 0.16,
        'edge head:  "is this edge a truth structural edge?"\nkeep if score > 0.9', PINK)
    for xa in (0.28, 0.72):
        a2.add_patch(FancyArrowPatch((xa, 0.80), (xa, 0.52), arrowstyle='-|>',
                                     mutation_scale=14, color=GRAY, lw=2.0))
    box(0.5, 0.43, 0.94, 0.16,
        'pruned graph:  both heads pass the fixed\nthreshold 0.9', GRAY)
    a2.add_patch(FancyArrowPatch((0.5, 0.35), (0.5, 0.255), arrowstyle='-|>',
                                 mutation_scale=14, color=GRAY, lw=2.0))
    box(0.5, 0.15, 0.98, 0.19,
        'reconstruction:  assemble kept edges into chains;\na chain counts only '
        'if ALL its links survive', GREEN)
    fig.suptitle('Setup 1: what a single event looks like', fontsize=16,
                 fontweight='bold', color=BLUE)
    fig.supxlabel('candidate edges = ALL track pairs in the event (a complete '
                  'graph); only ~0.13% of them are truth structural edges',
                  fontsize=13, color=GRAY)
    save(fig, 'F12_event_anatomy.png')


# =========================== F13 (new: the pools) =========================
def f13():
    fig, ax = plt.subplots(figsize=(13.6, 5.6), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.5, 1.02, 'Four evaluation pools - the numbers are NOT comparable '
                       'across them', ha='center', va='bottom', fontsize=15,
            color=BLUE, fontweight='bold')
    heads = ['pool', 'what it is', 'size', 'metric / threshold']
    rows = [
        ['500-event benchmark', '18 trained versions, re-scored\non one '
         'held-out test set', 'hard pool 5964 edges\n(~50% positive)',
         'FIXED threshold 0.9\n(hard-pool AP, chain survival)'],
        ['3000-event benchmark', 'v648 / v653 / v654, the\nchain-loss head-to-head',
         'hard pool 33166 edges\n(~52% positive)',
         'FIXED threshold 0.9\n(hard-pool AP, chain survival)'],
        ['696-event multi-B probe', 'small CPU model\n(MLP vs attention)',
         '696 events / 281 chains\n5248 test edges',
         'threshold = score at\n90% edge precision'],
        ['GBDT feature ceiling', 'gradient-boosted trees on\nraw features (no model)',
         'same multi-B subset\n(no training)',
         'hard-pool AP +\npermutation importance'],
    ]
    ncol = len(heads)
    x0, w = 0.015, 0.970
    colw = [0.205, 0.315, 0.235, 0.245]
    xcol = [x0]
    for cw in colw[:-1]:
        xcol.append(xcol[-1] + cw)
    # header
    yh = 0.86
    for j, h in enumerate(heads):
        ax.add_patch(Rectangle((xcol[j], yh), colw[j], 0.10, fc=BLUE, ec='none'))
        ax.text(xcol[j] + colw[j] / 2, yh + 0.05, h, ha='center', va='center',
                fontsize=13, color='white', fontweight='bold')
    y = yh
    for i, r in enumerate(rows):
        y -= 0.19
        bg = '#F2F6FA' if i % 2 else 'white'
        for j in range(ncol):
            ax.add_patch(Rectangle((xcol[j], y), colw[j], 0.19, fc=bg,
                                   ec='#DDDDDD', lw=1.0))
            ax.text(xcol[j] + 0.012, y + 0.095, r[j], ha='left', va='center',
                    fontsize=13, color=DARK)
    ax.text(0.5, y - 0.10, 'the 500-event and 3000-event benchmarks share the '
                           'FIXED 0.9 threshold;\n'
                           'the two probes use a DATA-DRIVEN 90%-edge-precision '
                           'threshold - a different ruler',
            ha='center', va='top', fontsize=13, color=GRAY)
    save(fig, 'F13_eval_pools.png')


# =========================== formulas =====================================
def equations():
    formula(r"$S \;=\; \prod_{e \in \mathrm{chain}} \mathbf{1}\!\left[\hat{y}_e\right]"
            r", \qquad \mathbb{E}[S] \;\approx\; \prod_{e} p_e$", 'EQ_survival.png')
    formula(r"$S_{\mathrm{chain}} = \frac{N_{\mathrm{surv}}}{N_{\mathrm{chains}}},"
            r"\qquad p_n = S_n^{\,1/n}$", 'EQ_metrics.png')
    formula(r"$\mathrm{softmin}_\gamma(p) \;=\; -\frac{1}{\gamma}\,\ln\!\left("
            r"\frac{1}{n}\sum_{e=1}^{n} e^{-\gamma\,p_e}\right)$", 'EQ_softmin.png')
    formula(r"$L_{\mathrm{chain}} \;=\; -\ln\!\left(\mathrm{softmin}_\gamma(p)"
            r"\right)$", 'EQ_chainloss.png')
    formula(r"$\alpha_{ef} \;=\; \mathrm{softmax}_f\!\left(\frac{(W_q h_e)"
            r"\cdot(W_k h_f)}{\sqrt{d}}\right), \qquad h_e \;\leftarrow\; "
            r"\sum_{f \in \mathcal{N}(e)} \alpha_{ef}\,W_v h_f$",
            'EQ_attention.png', fontsize=20)
    formula(r"$a = e^{-d_3/50}, \qquad r = a \,/\, \max_{f} a_f$",
            'EQ_competition.png')


if __name__ == '__main__':
    print('[figs] ->', FIG)
    (f1(), f2(), f3(), f4(), f5(), f6(), f8(), f9(), f10(), f11(), f12(),
     f13(), equations())
    print('[ok] 12 figures + 6 formulas written')
