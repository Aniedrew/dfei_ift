#!/usr/bin/env python3
"""生成 DFEI 优化效果折线图 (v31 -> v47): PerfectReco / AllParticles / class1 / class2.

数据 = CERN 官方 MC (DFEI_IFT_20260702), thr0.9, 20 测试文件 (v47 masshead2 全量评估)。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs'
os.makedirs(FIG, exist_ok=True)

versions = ['v31', 'v36', 'v37', 'v38', 'v47']
labels = ['v31\n(fix bug)', 'v36\n(diff. pruning)', 'v37\n(cl2+chain)', 'v38\n(chain CE)', 'v47\n(mass head)']
perfect = [23.9, 26.3, 27.3, 29.3, 32.7]
allpart = [43.4, 49.2, 50.6, 52.1, 55.9]
class1 = [67.8, 64.5, 69.1, 76.8, 77.4]
class2 = [41.3, 47.7, 44.2, 47.9, 51.1]

fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
fig.suptitle('DFEI optimization on CERN MC (DFEI_IFT_20260702) — threshold 0.9, 20 test files',
             fontsize=14, fontweight='bold')

for ax, (name, series) in zip(axes.ravel(), [
        ('PerfectReco (%)', perfect),
        ('AllParticles (%)', allpart),
        ('LCAG class1 acc (%)', class1),
        ('LCAG class2 acc (%)', class2),
]):
    ax.plot(range(len(versions)), series, marker='o', linewidth=2.2, markersize=6,
            color='#1F4E79')
    for i, v in enumerate(series):
        ax.annotate(f'{v:.1f}', (i, v), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=10, color='#333333')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(min(series) - 5, max(series) + 6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f'{FIG}/progress_line_v31_v47.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('[ok]', out)
