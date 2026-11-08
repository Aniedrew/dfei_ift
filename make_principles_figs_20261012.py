#!/usr/bin/env python3
"""Figures for the 2026-10-12 talk: "Which optimisations actually worked - and why".

Every number below is the one given in the talk brief; nothing is invented.
Only matplotlib (Agg) is used - no seaborn, no training, no dataloader.
All text is English and at least 13 pt.

Run:
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python make_principles_figs_20261012.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20261012'
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

CS_FOOT = ('chain survival = fraction of truth chains fully recovered at the\n'
           'edge-precision-90% working point (500-event test set)')


def save(fig, name):
    fig.savefig(FIG + '/' + name, bbox_inches='tight')
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


# =========================== F1 ===========================================
def f1():
    ap_r, cs_r = minrank(AP), minrank(CS)
    yl, yr = spread(ap_r), spread(cs_r)
    hl = {'v648': PINK, 'v643': GOLD, 'v641': TEAL, 'v642': GREEN,
          'v623': BLUE, 'v632': PURPLE}
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
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
                 '500-event test set, edge-precision-90% working point',
                 fontsize=15, pad=10)

    def lbl(v):
        return '%s:  AP #%d  ->  surv #%d  (%.1f%%)' % (
            v, ap_r[v], cs_r[v], CS[v])
    handles = [plt.Line2D([0], [0], color=hl[v], lw=3, label=lbl(v)) for v in
               ('v642', 'v641', 'v648', 'v643', 'v623', 'v632')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01),
              ncol=2, fontsize=13, frameon=False, handlelength=1.8)
    save(fig, 'F1_dumbbell_ap_vs_chain.png')


# =========================== F2 ===========================================
def f2():
    ap_r = minrank(AP)
    order = sorted(CS, key=lambda v: CS[v])          # ascending -> top bar = best
    ys = np.arange(len(order))
    hl = {'v642': GREEN, 'v641': TEAL, 'v648': PINK, 'v643': GOLD}
    cols = [hl.get(v, BLUE) for v in order]
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.barh(ys, [CS[v] for v in order], color=cols, height=0.72, zorder=3)
    for i, v in enumerate(order):
        ax.text(CS[v] + 0.7, i, '%.1f%%   (AP #%d)' % (CS[v], ap_r[v]),
                va='center', fontsize=13, color=DARK)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=13)
    ax.set_xlim(0, 84)
    ax.set_xlabel('chain survival  [%]', fontsize=14)
    ax.set_title("Chain survival, best to worst - with each arm's AP rank", pad=12)
    ax.grid(axis='x', color='#EEEEEE', zorder=0)
    fig.text(0.5, -0.025, CS_FOOT, ha='center', fontsize=13, color=GRAY)
    save(fig, 'F2_chain_surv_bar.png')


# =========================== F3 ===========================================
def f3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    # -- panel A: p^n, the AND semantics
    p = np.linspace(0.5, 1.0, 200)
    a1.plot(p, p, color=GRAY, lw=1.8, ls='--', label='p  (1 link)')
    for n, c in ((2, TEAL), (3, PINK), (4, PURPLE)):
        a1.plot(p, p ** n, color=c, lw=2.6, label='p^%d  (%d links)' % (n, n))
    a1.axvline(0.8, color='#BBBBBB', lw=1.2)
    a1.text(0.805, 0.07, 'p = 0.80:\n0.640 / 0.512 / 0.410\n(n = 2 / 3 / 4)',
            fontsize=13, color=DARK)
    a1.set_xlabel('per-link success probability  p', fontsize=14)
    a1.set_ylabel('chain survival  =  p^n', fontsize=14)
    a1.set_title('AND semantics: one dead link kills the chain', fontsize=15)
    a1.set_ylim(0, 1.02)
    a1.legend(fontsize=13, frameon=False, loc='lower right')
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
    for v in ('v642', 'v643'):
        a2.annotate(v, (P2[v], N2[v]), textcoords='offset points', xytext=(8, 6),
                    fontsize=13, color=DARK)
    a2.set_xlabel('per-link success probability  p2', fontsize=14)
    a2.set_ylabel('chain survival  [%]', fontsize=14)
    a2.set_title('18 arms collapse onto p^n  (p2: 0.663 -> 0.803)', fontsize=15)
    a2.legend(fontsize=13, frameon=False, loc='upper left')
    a2.grid(color='#EEEEEE')
    fig.text(0.5, -0.03, 'measured chain survival vs the per-link p2 of the same '
             'run; curves are the ideal AND prediction', ha='center', fontsize=13,
             color=GRAY)
    save(fig, 'F3_and_semantics.png')


# =========================== F4 ===========================================
def f4():
    g = np.logspace(0, 2, 400)                        # gamma = 1 ... 100
    p = np.array([0.95, 0.50, 0.05])
    surv = -(1.0 / g) * np.log(np.mean(np.exp(-g[:, None] * p[None, :]), axis=1))
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
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
    fig.text(0.5, -0.03, 'example p = [0.95, 0.50, 0.05]', ha='center',
             fontsize=13, color=GRAY)
    save(fig, 'F4_softmin_gamma.png')


# =========================== F5 ===========================================
CFG = [('w = 0\n(baseline)', [24.6, 23.6, 24.5], 24.2, 0.8393, 0.8650, GRAY),
       ('prod w = 1.0\n(~per-edge BCE)', [29.2, 24.6, 30.1], 28.0, 0.8413, 0.8691, GOLD),
       ('softmin w = 0.3', [32.4, 23.2, 34.8], 30.1, 0.8468, 0.8713, TEAL),
       ('softmin w = 1.0', [33.5, 28.5, 35.8], 32.6, 0.8533, 0.8732, GREEN)]


def f5():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    x = np.arange(len(CFG))
    a1.bar(x, [c[2] for c in CFG], color=[c[5] for c in CFG], width=0.62,
           zorder=3, alpha=0.85)
    for i, c in enumerate(CFG):
        a1.scatter(i + np.array([-0.13, 0.0, 0.13]), c[1], color=DARK, s=46,
                   zorder=5, edgecolor='white', linewidth=0.7)
        a1.text(i, c[2] + 0.8, '%.1f' % c[2], ha='center', fontsize=13,
                fontweight='bold', color=DARK)
    a1.set_xticks(x)
    a1.set_xticklabels([c[0] for c in CFG], fontsize=13)
    a1.set_ylabel('chain survival  [%]', fontsize=14)
    a1.set_ylim(0, 40)
    a1.set_title('Chain survival (3 seeds + mean)', fontsize=15)
    a1.grid(axis='y', color='#EEEEEE', zorder=0)
    for i, c in enumerate(CFG):
        a2.plot([i, i], [c[3], c[4]], color='#CCCCCC', lw=1.6, zorder=1)
    a2.scatter(x, [c[3] for c in CFG], color=BLUE, s=95, zorder=3, label='AP')
    a2.scatter(x, [c[4] for c in CFG], color=PURPLE, s=95, marker='s', zorder=3,
               label='AUC')
    for i, c in enumerate(CFG):
        a2.text(i - 0.11, c[3], '%.4f' % c[3], ha='right', va='center', fontsize=13,
                color=BLUE)
        a2.text(i + 0.11, c[4], '%.4f' % c[4], ha='left', va='center', fontsize=13,
                color=PURPLE)
    a2.set_xticks(x)
    a2.set_xticklabels([c[0].split('\n')[0] for c in CFG], fontsize=13)
    a2.set_ylim(0.826, 0.886)
    a2.set_ylabel('edge-level score', fontsize=14)
    a2.set_title('Edge-level AP and AUC (secondary)', fontsize=15)
    a2.legend(fontsize=13, frameon=False, loc='lower right')
    a2.grid(axis='y', color='#EEEEEE')
    fig.text(0.5, -0.02, 'multi-B subset: 696 events / 281 truth chains / '
             '5248 test edges', ha='center', fontsize=13, color=GRAY)
    save(fig, 'F5_chainloss_probe.png')


# =========================== F6 ===========================================
def f6():
    names = ['attn L4\n30-dim', 'attn L4\n47-dim', 'MLP\n47-dim']
    ap = [0.8460, 0.8450, 0.8015]
    ap_s = [0.0118, 0.0172, 0.0149]
    auc = [0.8745, 0.8695, 0.8480]
    auc_s = [0.0094, 0.0129, 0.0047]
    cols = [TEAL, BLUE, GOLD]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 5.2))
    x = np.arange(3)
    a1.errorbar(x, ap, yerr=ap_s, fmt='none', capsize=8, lw=2.0, ecolor=GRAY, zorder=3)
    for i in range(3):
        a1.scatter(i, ap[i], s=210, color=cols[i], zorder=4,
                   edgecolor='white', linewidth=1.0)
        a1.text(i, ap[i] + ap_s[i] + 0.004, '%.4f\n+/- %.4f' % (ap[i], ap_s[i]),
                ha='center', fontsize=13, color=DARK)
    a1.set_xticks(x)
    a1.set_xticklabels(names, fontsize=13)
    a1.set_ylim(0.770, 0.880)
    a1.set_ylabel('AP', fontsize=14)
    a1.set_title('Attention beats the MLP by ~4.5 AP points', fontsize=15)
    a1.grid(axis='y', color='#EEEEEE')
    a2.errorbar(x, auc, yerr=auc_s, fmt='none', capsize=8, lw=2.0, ecolor=GRAY, zorder=3)
    for i in range(3):
        a2.scatter(i, auc[i], s=210, color=cols[i], zorder=4,
                   edgecolor='white', linewidth=1.0)
        a2.text(i, auc[i] + auc_s[i] + 0.003, '%.4f\n+/- %.4f' % (auc[i], auc_s[i]),
                ha='center', fontsize=13, color=DARK)
    a2.set_xticks(x)
    a2.set_xticklabels(names, fontsize=13)
    a2.set_ylim(0.825, 0.900)
    a2.set_ylabel('AUC', fontsize=14)
    a2.set_title('The MLP also has the smallest spread', fontsize=15)
    a2.grid(axis='y', color='#EEEEEE')
    fig.text(0.5, -0.02, 'small-model probe (pure CPU), multi-B subset, 3-4 seeds',
             ha='center', fontsize=13, color=GRAY)
    save(fig, 'F6_attn_vs_mlp.png')


# =========================== F7 ===========================================
def f7():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 5.2),
                                     gridspec_kw={'width_ratios': [1.0, 0.95, 1.35]})

    # (a) top-k sweep + sparsemax
    a1.add_patch(Rectangle((-0.5, 0.79), 4.0, 0.01, color=BLUE, alpha=0.20, zorder=1))
    a1.text(1.5, 0.8025, 'k = 4 / 8 / 16 / 32\nall 0.79 - 0.80', ha='center',
            fontsize=13, color=BLUE)
    a1.scatter([4], [0.7847], color=RED, s=150, zorder=4, marker='X')
    a1.annotate('sparsemax\n0.7847', (4, 0.7847), textcoords='offset points',
                xytext=(-64, 12), fontsize=13, color=RED)
    a1.set_xticks([0, 1, 2, 3, 4])
    a1.set_xticklabels(['k=4', 'k=8', 'k=16', 'k=32', 'sparse-\nmax'], fontsize=13)
    a1.set_xlim(-0.6, 4.6)
    a1.set_ylim(0.778, 0.810)
    a1.set_ylabel('AP', fontsize=14)
    a1.set_title('Sparsifying attention: flat', fontsize=15)

    # (b) Sinkhorn
    a2.errorbar([0], [0.735], yerr=[0.015], fmt='o', ms=14, capsize=10, lw=2.6,
                color=RED, ecolor=RED, zorder=3)
    a2.scatter([1], [0.81], color=BLUE, s=170, zorder=3, marker='s')
    a2.text(0, 0.700, 'Sinkhorn match\n0.72 - 0.75', ha='center', va='bottom',
            fontsize=13, color=RED)
    a2.text(1, 0.815, 'pairwise\n~0.81', ha='center', va='bottom', fontsize=13,
            color=BLUE)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(['Sinkhorn\n(edge matching)', 'pairwise\nscoring'],
                       fontsize=13)
    a2.set_xlim(-0.55, 1.55)
    a2.set_ylim(0.690, 0.875)
    a2.set_ylabel('score', fontsize=14)
    a2.set_title('Edge-level Sinkhorn: far weaker\n(comb gain +0.001 - 0.002)',
                 fontsize=15)

    # (c) geometry / line-graph arms on the bench (horizontal -> labels never crowd)
    arms = ['v633  feature stack', 'v637  line-graph edge-edge',
            'v649  + geom pair-bias', 'v643  + vertex geometry']
    vals = [CS['v633'], CS['v637'], CS['v649'], CS['v643']]
    a3.barh(range(4), vals, color=[GRAY, PURPLE, TEAL, GOLD], height=0.62, zorder=3)
    for i, v in enumerate(vals):
        a3.text(v + 0.9, i, '%.1f%%' % v, va='center', fontsize=13, color=DARK)
    a3.set_yticks(range(4))
    a3.set_yticklabels(arms, fontsize=13)
    a3.invert_yaxis()
    a3.set_xlim(0, 54)
    a3.set_ylim(3.6, -0.6)
    a3.set_xlabel('chain survival  [%]', fontsize=14)
    a3.set_title('Geometry-variant arms: no separation', fontsize=15)
    a3.grid(axis='x', color='#EEEEEE', zorder=0)
    fig.text(0.5, -0.05, 'panels (a) and (b): small-model probe; panel (c): '
             '500-event bench', ha='center', fontsize=13, color=GRAY)
    save(fig, 'F7_negatives.png')


# =========================== F8 ===========================================
def f8():
    groups = ['node', '+ geo', '+ raw+der', '+ vrt', '+ ctx']
    multi = [0.6916, 0.8043, 0.8291, 0.8360, 0.8391]
    single = [0.7934, 0.8256, 0.8344, 0.8367, 0.8365]
    imp = [('doca', 0.166), ('trdist', 0.073), ('flight', 0.048), ('m', 0.022)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
    x = np.arange(5)
    a1.plot(x, multi, 'o-', color=BLUE, lw=2.6, ms=11, label='multi-B subset')
    a1.plot(x, single, 's--', color=TEAL, lw=2.6, ms=10, label='single-B subset')
    for i in range(5):
        a1.text(i, multi[i] + 0.007, '%.4f' % multi[i], ha='center', fontsize=13,
                color=BLUE)
        a1.text(i, single[i] - 0.015, '%.4f' % single[i], ha='center', fontsize=13,
                color=TEAL)
    a1.annotate('+0.113', (1.0, 0.8043), textcoords='offset points', xytext=(-4, -34),
                fontsize=13, color=BLUE, fontweight='bold')
    a1.annotate('+0.032', (1.0, 0.8256), textcoords='offset points', xytext=(-4, 18),
                fontsize=13, color=TEAL, fontweight='bold')
    a1.set_xticks(x)
    a1.set_xticklabels(groups, fontsize=13)
    a1.set_ylim(0.660, 0.880)
    a1.set_ylabel('GBDT AP', fontsize=14)
    a1.set_title('Cumulative feature groups (GBDT upper bound)', fontsize=15)
    a1.legend(fontsize=13, frameon=False, loc='lower right')
    a1.grid(color='#EEEEEE')
    a2.barh(range(4), [v for _, v in imp], color=[BLUE, TEAL, PURPLE, GOLD],
            height=0.62, zorder=3)
    for i, (_, v) in enumerate(imp):
        a2.text(v + 0.005, i, '+%.3f' % v, va='center', fontsize=13, color=DARK)
    a2.set_yticks(range(4))
    a2.set_yticklabels([k for k, _ in imp], fontsize=13)
    a2.invert_yaxis()
    a2.set_xlim(0, 0.19)
    a2.set_xlabel('permutation importance  (delta AP)', fontsize=14)
    a2.set_title('Multi-B: geometry dominates the trees', fontsize=15)
    a2.grid(axis='x', color='#EEEEEE', zorder=0)
    fig.text(0.5, -0.02, 'D1 fix applied; AP on the same multi-B / single-B pools',
             ha='center', fontsize=13, color=GRAY)
    save(fig, 'F8_features.png')


if __name__ == '__main__':
    print('[figs] ->', FIG)
    f1(); f2(); f3(); f4(); f5(); f6(); f7(); f8()
    print('[ok] 8 figures written')
