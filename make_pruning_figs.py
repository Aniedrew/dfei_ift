#!/usr/bin/env python3
"""Generate figures for the "Differentiable pruning with annealing" slide:
1) pruning_sigmoid_tau.png — the sigmoid mask shape at different temperatures (τ)
2) pruning_flow.png       — flow chart of where w_eff sits (MLP <-> GNN joint)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs'
os.makedirs(FIG, exist_ok=True)
BLUE = '#1F4E79'
RED = '#C0392B'
ORANGE = '#E67E22'
GREEN = '#2CA02C'
GRAY = '#666666'

# ---------- Figure 1: sigmoid shape vs temperature ----------
import numpy as np
w = np.linspace(0.0, 1.0, 400)
cut = 0.85
taus = [1.0, 0.3, 0.1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {1.0: BLUE, 0.3: ORANGE, 0.1: RED}

# hard cut reference (step at cut, no mask)
mask_hard = (w >= cut).astype(float)
weff_hard = np.where(w >= cut, w, 0.0)
axes[0].plot(w, mask_hard, lw=2.2, color='k', ls='--', label='hard cut (τ → 0)')
axes[1].plot(w, weff_hard, lw=2.2, color='k', ls='--', label='hard cut (τ → 0)')

for tau in taus:
    mask = 1 / (1 + np.exp(-(w - cut) / tau))
    weff = w * mask
    axes[0].plot(w, mask, lw=2.5, color=colors[tau],
                 label=f'τ = {tau}  (start of annealing)' if tau == 1.0 else
                      (f'τ = {tau}' if tau == 0.3 else 'τ = 0.1  (end of annealing)'))
    axes[1].plot(w, weff, lw=2.5, color=colors[tau])

for ax in axes:
    ax.axvline(cut, color=GRAY, ls='--', lw=1.2)
    ax.axvline(0.9, color=GREEN, ls=':', lw=1.4)
    ax.text(cut, ax.get_ylim()[0] + 0.02, ' cut = 0.85\n(fixed per run)', color='#222222', fontsize=10)
    ax.text(0.9, ax.get_ylim()[1] - 0.14, ' 0.9 = inference\nthreshold', color='#222222', fontsize=10)
    ax.set_xlabel('w  (node/edge confidence, 0–1)', fontsize=12)
    ax.grid(alpha=0.3)

axes[0].set_title('mask = σ((w − cut)/τ)\n(how much of w survives)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('mask value', fontsize=12)
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(fontsize=11)

axes[1].set_title('w_eff = w · σ((w − cut)/τ)\n(effective weight used in training)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('w_eff', fontsize=12)
axes[1].set_ylim(-0.05, 1.05)
axes[1].text(0.30, 0.78, 'τ small → behaves like\nhard pruning at cut', color='#222222', fontsize=11)

fig.suptitle('Differentiable pruning: same sigmoid, temperature τ annealed 1.0 → 0.1 during training',
             fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
out1 = f'{FIG}/pruning_sigmoid_tau.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()

# ---------- Figure 2: flow chart of w_eff ----------
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')


def box(x, y, w, h, text, fc='#EAEFF8', ec=BLUE, fs=11, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
                       fc=fc, ec=ec, lw=1.8)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color='#222222', fontweight='bold' if bold else 'normal')
    return (x + w, y + h / 2)


def arrow(x1, y1, x2, y2, color=BLUE, lw=2.0, style='-|>'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=18, color=color, lw=lw)
    ax.add_patch(a)


# layout: reprs -> MLP -> w -> soft mask -> w_eff -> (message passing | loss)
x = 2
x2 = box(x, 18, 15, 10, 'GNN block\n(node/edge reprs)', fs=11)
arrow(x2[0], x2[1], x2[0] + 6, x2[1])

x2 = box(x + 21, 18, 15, 10, 'Weight MLP\n(sigmoid head)', fc='#D9E8F5', fs=11)
arrow(x2[0], x2[1], x2[0] + 6, x2[1])

x2 = box(x + 42, 20, 10, 6, 'w ∈ [0,1]', fc='#FDF2E3', ec=ORANGE, fs=12)
arrow(x2[0], x2[1], x2[0] + 6, x2[1])

# soft mask box (highlighted)
x2 = box(x + 63, 14, 22, 18, 'Soft mask\nw_eff = w·σ((w−cut)/τ)\nτ: 1.0 → 0.1',
         fc='#FDEBD0', ec=RED, fs=12, bold=True)
# annotation arrow: tau annealed
ax.annotate('τ annealed during training\n(1.0 → 0.1)', xy=(x + 74, 32), xytext=(x + 74, 39),
            ha='center', fontsize=10, color='#222222')
ax.annotate('', xy=(x + 74, 32.2), xytext=(x + 74, 36.5),
            arrowprops=dict(arrowstyle='->', color=RED))

# w_eff out
x2 = box(x + 91, 20, 10, 6, 'w_eff', fc='#FDF2E3', ec=ORANGE, fs=12)
arrow(x + 85, 23, x + 91, 23)
# branches
arrow(x + 96, 22, x + 96, 12, color=GREEN)
ax.text(x + 97, 12, 'weighted message passing\n(back into the GN blocks)', fontsize=10.5, color='#222222', va='center')
arrow(x + 96, 24, x + 96, 34, color=BLUE)
ax.text(x + 97, 34, 'pruning loss vs truth (BCE)\n(mask makes it differentiable)', fontsize=10.5, color='#222222', va='center')

ax.text(50, 3, 'inference (no mask): keep node/edge if w ≥ 0.9 — hard threshold',
        ha='center', fontsize=12, color=GRAY, style='italic')
plt.tight_layout()
out2 = f'{FIG}/pruning_flow.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()

print('[ok]', out1)
print('[ok]', out2)
