"""English deck: PV association in DFEI — task, traditional minIP, our HGNN,
the 0904 data-production story, and all current results.

Same recipe as the Chinese deck:每页 matplotlib 画成 16:9 PNG, 再用 python-pptx
组装成一页一图, 并把讲稿写进备注栏。
Output: report_figs/pv_ppt_en/slide_XX.png  ,  PV_association_DFEI_EN.pptx
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import os

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
FIG = BASE + "/report_figs"
OUT = FIG + "/pv_ppt_en"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F4E79"; BLUE = "#2E75B6"; RED = "#C0392B"; GREEN = "#1E8449"
GREY = "#7F7F7F"; ORANGE = "#D68910"; DARK = "#2B2B2B"; LIGHT = "#F2F6FA"; CYAN = "#138D75"

SLIDES = []          # (png, notes)


def canvas(title, sub=None, bg="white"):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if title:
        txt(ax, 0.035, 0.945, title, 24, NAVY, "bold", va="top")
        ax.plot([0.035, 0.965], [0.888, 0.888], color=NAVY, lw=2.2)
    if sub:
        txt(ax, 0.035, 0.856, sub, 13, GREY, va="top")
    return fig, ax


def txt(ax, x, y, s, size=13, color=DARK, weight="normal", ha="left", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style=style, linespacing=1.42, zorder=5)


def box(ax, x, y, w, h, s=None, fc="white", ec=GREY, lw=1.4, size=12, color=DARK,
        weight="normal", r=0.02, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if s:
        txt(ax, x + w / 2 if ha == "center" else x + 0.014, y + h / 2, s, size, color, weight, ha=ha, va="center")


def panel(ax, x, y, w, h, title, color=BLUE, fc="white", tsize=12.5):
    box(ax, x, y, w, h, fc=fc, ec=color, lw=1.6)
    txt(ax, x + w / 2, y + h - 0.032, title, tsize, color, "bold", ha="center")
    return x, y, w, h


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.8, style="-|>", ms=11):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=ms, shrinkA=0, shrinkB=0, zorder=4))


def dot(ax, x, y, s=90, c=BLUE, ec="white", lw=1.5, z=6):
    ax.add_patch(Circle((x, y), s / 40000 * 4, fc=c, ec=ec, lw=lw, zorder=z))


def seg(ax, x1, y1, x2, y2, color=GREY, lw=1.2, z=3, alpha=1.0, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha)


def pic(ax, path, x, y, w, h):
    ax.imshow(mpimg.imread(path), extent=[x, x + w, y, y + h], aspect="auto", zorder=1)


def pic_fit(ax, path, xc, y, h, src_aspect):
    """place image preserving aspect: width = h * (13.333/7.5)^-1 * aspect"""
    w = h * (7.5 / 13.333) * src_aspect
    pic(ax, path, xc - w / 2, y, w, h)
    return w


def table(ax, x, y, colw, rows, size=11.5, row_h=0.052, head_fc=NAVY, head_color="white",
          zebra=True, align=None):
    """rows[0] = header; colw = list of widths (fractions of 1.0)"""
    for i, r in enumerate(rows):
        yy = y - i * row_h
        if i == 0:
            ax.add_patch(Rectangle((x, yy - row_h / 2), sum(colw), row_h, fc=head_fc, ec="none", zorder=2))
        elif zebra and i % 2 == 0:
            ax.add_patch(Rectangle((x, yy - row_h / 2), sum(colw), row_h, fc="#F4F7FA", ec="none", zorder=2))
        cx = x
        for j, cell in enumerate(r):
            ha = "left" if (align and align[j] == "l") else "center"
            px = cx + (0.008 if ha == "left" else colw[j] / 2)
            txt(ax, px, yy, str(cell), size, head_color if i == 0 else DARK,
                "bold" if i == 0 else "normal", ha=ha, va="center")
            cx += colw[j]
    return y - len(rows) * row_h


def save(fig, name, notes):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, facecolor=fig.get_facecolor()); plt.close(fig)
    SLIDES.append((p, notes)); print("saved", p)


# ============================================================== 1 title
def s01():
    fig, ax = canvas(None)
    ax.add_patch(Rectangle((0, 0), 1, 0.20, fc=NAVY, ec="none"))
    txt(ax, 0.5, 0.845, "Primary-vertex association in DFEI:", 32, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.765, "task, traditional minIP, our HGNN, and what the 0904 production changed", 22, BLUE, "bold", ha="center")
    txt(ax, 0.5, 0.675, "status report, 2026-09-21", 14, GREY, ha="center", style="italic")
    box(ax, 0.10, 0.30, 0.80, 0.30,
        "A.  The two data productions and the preprocessing bug\n"
        "B.  PV association: the task, minIP, our graph network\n"
        "C.  Results: recovery, PV association, where our gain comes from\n"
        "D.  Core analysis: mis-association vs physics, and its direction\n"
        "E.  Training on the new production (v601 / v602), other findings, next steps",
        fc=LIGHT, ec=BLUE, size=14, ha="left")
    txt(ax, 0.5, 0.235, "All numbers are per-track / per-chain comparisons against the Monte-Carlo truth.", 12, GREY, ha="center")
    txt(ax, 0.5, 0.185, "Data: LHCb simulated DFEI samples (0702 = July, 0904 = September production).", 12, GREY, ha="center")
    save(fig, "slide_01_title",
         "Purpose of this deck: one coherent story about PV association — the task, the traditional baseline, "
         "what our HGNN does differently, and what we learned from the new (0904) data production. "
         "Every number is a comparison against Monte-Carlo truth, so it is reproducible from the eval outputs.")


# ============================================================== 2 data story
def s02():
    fig, ax = canvas("A. Two data productions, one reported preprocessing bug",
                     "Everything downstream depends on which production we evaluate on")
    panel(ax, 0.03, 0.505, 0.455, 0.315, "0702 (July) production", RED)
    txt(ax, 0.05, 0.738, "• Sample our main line was trained/evaluated on:\n    inclusive_00342442, 200 train / 20 val / 20 test files", 11.5, DARK)
    txt(ax, 0.05, 0.638, "• Preprocessing: normalization dict = old_norm_ported.pt\n    graph building = tracks-tracks pairs enumerated from\n    np.sort(ParticleIndex)  →  edge-table ORDER correlates\n    with the truth structure (AUC 0.75 vs 0.52 on the fixed one)", 11.5, DARK)
    txt(ax, 0.05, 0.538, "• No ghost removal / duplicate-MC-track removal", 11.5, DARK)
    panel(ax, 0.515, 0.505, 0.455, 0.315, "0904 (September) — regenerated WITH the fix", GREEN)
    txt(ax, 0.535, 0.738, "• Same task, rebuilt: + ghost cut (Prob_ghost < 0.182),\n    + duplicate MC-track removal, + skip first 64 bunches", 11.5, DARK)
    txt(ax, 0.535, 0.638, "• Normalization dict = normalization_dict.pt\n    (pz scale sign flips vs July: +10077 vs -10015;\n    other momentum columns differ by 3-35%)", 11.5, DARK)
    txt(ax, 0.535, 0.538, "• Also ships real data (no truth) — we ran inference-only stats on it", 11.5, DARK)
    box(ax, 0.03, 0.30, 0.94, 0.17,
        "truth_labeling.py is IDENTICAL in both workspaces  →  the PV-truth definition did not change,\n"
        "so PV-association numbers from the two productions are directly comparable.",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    # arms
    panel(ax, 0.03, 0.06, 0.30, 0.21, "Arm A = 0702 as-is", GREY)
    txt(ax, 0.05, 0.145, "July data, July\nnormalization\n(our historical line)", 11, DARK)
    panel(ax, 0.345, 0.06, 0.30, 0.21, "Arm B = 0904 as-is", GREY)
    txt(ax, 0.365, 0.145, "fixed data just as\nproduced (what the\ngroup will use)", 11, DARK)
    panel(ax, 0.66, 0.06, 0.31, 0.21, "Arm C = 0904 re-normalized", GREY)
    txt(ax, 0.68, 0.145, "0904 events, July\nnormalization → lets us\nrun July-trained models", 11, DARK)
    save(fig, "slide_02_data",
         "The group reported a preprocessing bug in the July (0702) production: in particular the tracks-tracks "
         "edge table was built by enumerating pairs from a sorted particle index, so the EDGE ORDER correlates with "
         "the truth structure — an order leak that makes models look better. Yukai regenerated the data with the fix "
         "as the 0904 production (plus ghost/clone removal). The normalization dictionaries also changed. "
         "To compare fairly we built three arms: A = July data, B = 0904 as produced, C = 0904 re-normalized to the "
         "July dictionary so that July-trained models can be evaluated on the same events.")


# ============================================================== 3 one event
def s03():
    fig, ax = canvas("B. One event: several pp collisions, one vertex each", "PV = primary vertex (pp interaction point); PVs are spread along the beam axis z")
    seg(ax, 0.08, 0.34, 0.94, 0.34, NAVY, 2.4, z=2)
    arrow(ax, 0.94, 0.34, 0.975, 0.34, NAVY, 2.4)
    txt(ax, 0.90, 0.295, "z (beam axis)", 11, NAVY)
    for i, x in enumerate([0.20, 0.42, 0.63, 0.82]):
        dot(ax, x, 0.34, 130, RED)
        txt(ax, x, 0.385, f"PV {i}", 12, RED, "bold", ha="center")
    for base, pts in [(0.20, [(0.20, 0.34), (0.15, 0.52), (0.11, 0.70)]),
                      (0.20, [(0.20, 0.34), (0.26, 0.50), (0.30, 0.71)]),
                      (0.42, [(0.42, 0.34), (0.37, 0.53), (0.34, 0.72)]),
                      (0.42, [(0.42, 0.34), (0.48, 0.51), (0.52, 0.68)]),
                      (0.63, [(0.63, 0.34), (0.59, 0.50), (0.56, 0.73)]),
                      (0.63, [(0.63, 0.34), (0.68, 0.53), (0.72, 0.71)]),
                      (0.82, [(0.82, 0.34), (0.79, 0.51), (0.77, 0.70)]),
                      (0.82, [(0.82, 0.34), (0.87, 0.50), (0.90, 0.69)])]:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=2.0, alpha=.85, zorder=3)
    txt(ax, 0.05, 0.775, "blue = tracks   |   red = primary vertices (PV)", 12, DARK)
    box(ax, 0.05, 0.06, 0.31, 0.16, "Per event in our MC:\n93 tracks, 7.6 PVs\n(PVs are a few cm apart in z)", fc=LIGHT, ec=BLUE, size=12, ha="left")
    box(ax, 0.40, 0.06, 0.55, 0.16,
        "Each track is just a line in space: on its own you cannot tell\nwhich collision it came from — the assignment must be inferred.",
        fc="#FFF8E7", ec=ORANGE, size=12, ha="left")
    save(fig, "slide_03_event",
         "In one bunch crossing several pp collisions happen; each produces a primary vertex. "
         "The detector gives us ~93 tracks per event but not their origin. PV association = answering "
         "'which PV does this track come from' for every track.")


# ============================================================== 4 why it matters
def s04():
    fig, ax = canvas("Why PV association matters: the b-physics measurement chain",
                     "A B hadron is produced at a PV, flies a few hundred micrometres, then decays")
    panel(ax, 0.04, 0.50, 0.44, 0.32, "truth (MC standard answer)", GREY)
    seg(ax, 0.07, 0.585, 0.45, 0.585, NAVY, 2)
    for x in [0.12, 0.24, 0.36]:
        dot(ax, x, 0.585, 110, RED)
    for x0, dx in [(0.12, -0.02), (0.12, 0.03), (0.24, -0.03), (0.24, 0.02), (0.36, -0.02), (0.36, 0.03)]:
        seg(ax, x0, 0.585, x0 + dx, 0.70, BLUE, 1.8)
    txt(ax, 0.26, 0.535, "every track's origin is known", 11, GREY, ha="center")
    panel(ax, 0.52, 0.50, 0.44, 0.32, "what the model must output", BLUE)
    seg(ax, 0.55, 0.585, 0.93, 0.585, NAVY, 2)
    for x in [0.60, 0.72, 0.84]:
        dot(ax, x, 0.585, 110, GREY)
    for x0, dx in [(0.60, -0.02), (0.60, 0.03), (0.72, -0.03), (0.72, 0.02), (0.84, -0.02), (0.84, 0.03)]:
        seg(ax, x0, 0.585, x0 + dx, 0.70, GREY, 1.8)
    txt(ax, 0.74, 0.535, "?  one PV index per track", 11, BLUE, ha="center")
    box(ax, 0.04, 0.29, 0.92, 0.16,
        "Flight distance = B decay vertex − correct PV.  It feeds lifetime, CP-violation and "
        "missing-energy measurements.", fc=LIGHT, ec=BLUE, size=13, ha="left")
    txt(ax, 0.07, 0.33, "A wrong PV association biases the flight distance and degrades the vertex resolution.", 12.5, RED, "bold")
    box(ax, 0.04, 0.06, 0.92, 0.19,
        'Literature framing  —  DFEI paper (arXiv:2504.21844, §4.4): "PV misassociation arises when tracks or decay products\n'
        'from overlapping pp collisions are incorrectly attributed to a PV. This can severely degrade the PV resolution and\n'
        'bias the measurement of observables such as the beauty-hadron decay flight distance and direction."',
        fc="#FFF8E7", ec=ORANGE, size=11.5, ha="left")
    save(fig, "slide_04_why",
         "Why this task exists: to measure a B hadron's flight distance (and hence lifetime, CP violation) you need to "
         "know which PV it came from, and each of its decay products must be attributed to the same PV. "
         "A wrong assignment biases the flight distance — that is the physics motivation quoted from the DFEI paper.")


# ============================================================== 5 constraint
def s05():
    fig, ax = canvas("The hard physics constraint: one B chain ⇒ one PV",
                     "Not a modelling assumption — a physical fact that any method must respect")
    def draw(x0, title, color, correct):
        panel(ax, x0, 0.40, 0.44, 0.42, title, color)
        pvx = [x0 + 0.075, x0 + 0.21, x0 + 0.345]
        seg(ax, x0 + 0.03, 0.545, x0 + 0.41, 0.545, NAVY, 2, z=2)
        for i, xx in enumerate(pvx):
            dot(ax, xx, 0.545, 95, GREY)
            txt(ax, xx, 0.505, f"PV{i}", 10, GREY, ha="center")
        bx, by = x0 + 0.21, 0.72
        dot(ax, bx, by, 130, RED)
        txt(ax, bx, by + 0.036, "B decay vertex", 10, RED, ha="center")
        seg(ax, bx, by, bx, 0.555, GREY, 1.3, ls=(0, (3, 3)), z=2)
        txt(ax, bx + 0.012, 0.655, "flight distance", 9, GREY)
        tgt = [0] * 5 if correct else [0, 0, 1, 2, 1]
        cols = [GREEN] * 5 if correct else [GREEN, GREEN, RED, BLUE, RED]
        for k, (t, c) in enumerate(zip(tgt, cols)):
            dx = -0.055 + k * 0.028
            ax.plot([bx, pvx[t] + dx * 0.15], [by, 0.558], color=c, lw=1.7, alpha=.9, zorder=3)
        for i, xx in enumerate(pvx):
            ok = correct and i == 0
            txt(ax, xx, 0.462, "OK" if ok else ("x" if not correct else "-"), 10,
                GREEN if ok else RED, "bold", ha="center")
        txt(ax, x0 + 0.22, 0.427, "a correct chain must light up exactly one PV", 9, GREY, ha="center")
    draw(0.04, "Correct: all 5 daughters point to PV0", GREEN, True)
    draw(0.52, "Wrong: daughters split over PV0/PV1/PV2", RED, False)
    box(ax, 0.04, 0.235, 0.44, 0.135, "→ flight distance = decay vertex − correct PV\n→ the vertex fit assumes one PV per chain",
        fc="#EAF5EA", ec=GREEN, size=11.5, ha="left")
    box(ax, 0.52, 0.235, 0.44, 0.135, "→ flight distance becomes a mixture of values\n→ vertex fit polluted, resolution degraded",
        fc="#FDECEA", ec=RED, size=11.5, ha="left")
    txt(ax, 0.5, 0.155, "So we evaluate two levels: per-track accuracy AND whether a whole B chain is consistency assigned to one PV.",
        13, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.095, "The baseline below is structurally weak exactly on the second level.", 12, GREY, ha="center")
    save(fig, "slide_05_constraint",
         "The key physics fact: all decay products of one B hadron must be assigned to the same PV. "
         "We therefore judge methods on two levels: per-track correctness, and whether a whole chain is "
         "consistently assigned. Remember this — it explains the baseline's weakness.")


# ============================================================== 6 minIP
def s06():
    fig, ax = canvas("The traditional baseline: minIP (closest PV by impact parameter)",
                     "IP = shortest distance between the track line and a PV (dashed segments)")
    (x1, y1), (x2, y2) = (0.09, 0.31), (0.60, 0.75)
    ax.plot([x1, x2], [y1, y2], color=BLUE, lw=2.6, zorder=3)
    txt(ax, 0.605, 0.735, "track", 11.5, BLUE)
    dx, dy = x2 - x1, y2 - y1; L2 = dx * dx + dy * dy
    def foot(px, py):
        t = ((px - x1) * dx + (py - y1) * dy) / L2
        return x1 + t * dx, y1 + t * dy
    for (px, py, lab, ipl, best) in [(0.20, 0.705, "PV0", 0.12, False), (0.37, 0.560, "PV1", 0.03, True),
                                     (0.53, 0.700, "PV2", 0.31, False)]:
        fx, fy = foot(px, py)
        seg(ax, px, py, fx, fy, GREEN if best else GREY, 2.2 if best else 1.4, z=4)
        dot(ax, px, py, 115, RED, z=6)
        txt(ax, px, py + 0.032, lab, 11.5, RED, "bold", ha="center")
        txt(ax, (px + fx) / 2 + 0.008, (py + fy) / 2 - 0.018, f"IP = {ipl}", 10.5,
            GREEN if best else GREY, "bold" if best else "normal")
    txt(ax, 0.09, 0.265, "compute one IP per (track, PV) and keep the smallest → per-track, independent", 11.5, DARK)
    txt(ax, 0.09, 0.225, "→ here PV1 wins, so this track is assigned to PV1", 12, GREEN, "bold")
    txt(ax, 0.665, 0.745, "Structure of the baseline", 13, NAVY, "bold")
    for i, s in enumerate(["uses only this track's geometry", "each track decides alone — no communication",
                           "simple, fast, the default for decades"]):
        txt(ax, 0.665, 0.705 - i * 0.038, "• " + s, 11, DARK)
    box(ax, 0.665, 0.50, 0.13, 0.075, "track", fc=LIGHT, ec=BLUE, size=11)
    arrow(ax, 0.795, 0.5375, 0.825, 0.5375)
    box(ax, 0.825, 0.50, 0.16, 0.075, "3 IP values", fc=LIGHT, ec=BLUE, size=11)
    arrow(ax, 0.905, 0.50, 0.905, 0.445)
    box(ax, 0.825, 0.37, 0.16, 0.075, "argmin", fc="#FFF8E7", ec=ORANGE, size=11)
    arrow(ax, 0.825, 0.4075, 0.795, 0.4075)
    box(ax, 0.665, 0.37, 0.13, 0.075, "→ PV1", fc="#EAF5EA", ec=GREEN, size=11.5, color=GREEN, weight="bold")
    txt(ax, 0.82, 0.325, "repeated independently for every track", 10.5, GREY, ha="center")
    box(ax, 0.06, 0.06, 0.88, 0.13,
        "minIP sees nothing about other tracks, nothing about the chain, nothing about the global PV structure.\n"
        "It has no mechanism that could make the 5 daughters of one B share a PV.",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    save(fig, "slide_06_minip",
         "The baseline: for each track compute the impact parameter (shortest distance) to every PV and keep the "
         "smallest. It is per-track and independent: it never looks at other tracks or at the chain, so it cannot "
         "enforce the one-chain-one-PV constraint.")


# ============================================================== 7 minIP tendencies
def s07():
    fig, ax = canvas("minIP's two intrinsic tendencies (measured on 0904 inclusive MC)",
                     "37 255 (chain, daughter-track) entries / 7 549 truth chains, 20 test files")
    panel(ax, 0.04, 0.44, 0.44, 0.36, "(1) it tears chains apart", RED)
    pvx = [0.115, 0.26, 0.405]
    seg(ax, 0.07, 0.565, 0.45, 0.565, NAVY, 1.8, z=2)
    for i, xx in enumerate(pvx):
        dot(ax, xx, 0.565, 90, GREY)
        txt(ax, xx, 0.535, f"PV{i}", 9.5, GREY, ha="center")
    bx, by = 0.26, 0.70
    dot(ax, bx, by, 110, RED)
    for k, (t, c) in enumerate([(0, GREEN), (0, GREEN), (1, RED), (2, BLUE), (1, RED)]):
        dx = -0.05 + k * 0.025
        ax.plot([bx, pvx[t] + dx * 0.2], [by, 0.575], color=c, lw=1.7, alpha=.9, zorder=3)
    txt(ax, 0.26, 0.495, "the 5 daughters are spread over 3 different PVs", 11, DARK, ha="center")
    txt(ax, 0.26, 0.462, "chain concordance only 59%   (physics: 100%)", 11.5, RED, "bold", ha="center")
    panel(ax, 0.52, 0.44, 0.44, 0.36, "(2) systematic pull towards later PVs", ORANGE)
    txt(ax, 0.75, 0.735, "mean PV index it assigns (truth = 1.60)", 10.8, DARK, ha="center")
    for i, (lab, v, c) in enumerate([("truth", 1.60, GREY), ("ours", 1.61, GREEN), ("minIP", 1.76, RED)]):
        y = 0.685 - i * 0.048
        txt(ax, 0.56, y, lab, 11, c, "bold", ha="left")
        ax.add_patch(Rectangle((0.63, y - 0.011), (v - 1.5) / 0.3 * 0.17, 0.022, fc=c, ec="none", zorder=3))
        txt(ax, 0.94, y, f"{v:.2f}", 10.5, c, ha="right")
    txt(ax, 0.74, 0.508, ">2/3 of its errors push the track to a LATER PV", 11.5, RED, "bold", ha="center")
    txt(ax, 0.74, 0.474, "(mean shift +0.78 in PV index)", 10.5, GREY, ha="center")
    box(ax, 0.04, 0.275, 0.92, 0.125,
        "Both are systematic, not random: they grow with pile-up (more PVs) and would be absorbed as a bias by the measurement.",
        fc="#FFF8E7", ec=ORANGE, size=12.5, ha="left")
    box(ax, 0.04, 0.06, 0.92, 0.185,
        "Metrics: | chain concordance = fraction of truth chains whose daughters all get the same PV.\n"
        "| PV index = ordering of the PVs inside one event along z (larger index = further downstream).",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_07_minip_bias",
         "Two measured tendencies of the baseline. (1) Because each track decides alone, the daughters of one B "
         "chain are frequently split over different PVs — chain concordance is only ~59% where physics demands 100%. "
         "(2) It systematically pushes tracks towards later PVs (mean PV index 1.76 vs truth 1.60; >2/3 of its errors "
         "go to a later PV). Both are systematic and grow with pile-up.")


# ============================================================== 8 our method
def s08():
    fig, ax = canvas("Our method: PV matching as inference on one whole graph",
                     "(1) a graph network sees global information   (2) a chain-level decision enforces one PV per chain")
    panel(ax, 0.03, 0.36, 0.30, 0.44, "(1) build a track × PV graph", BLUE, tsize=11.5)
    ty = [0.700, 0.645, 0.590, 0.535]; py = [0.700, 0.615, 0.530]
    for y1 in ty:
        for y2 in py:
            seg(ax, 0.085, y1, 0.275, y2, GREY, 0.7, alpha=.5)
    for y in ty:
        dot(ax, 0.085, y, 70, BLUE, z=6)
    for y in py:
        dot(ax, 0.275, y, 70, RED, z=6)
    txt(ax, 0.085, 0.732, "tracks", 10, BLUE, "bold", ha="center")
    txt(ax, 0.275, 0.732, "PVs", 10, RED, "bold", ha="center")
    txt(ax, 0.18, 0.465, "one edge per (track, PV) pair\ncomplete bipartite, carrying log IP", 10.5, GREY, ha="center")
    txt(ax, 0.18, 0.395, "→ every node/edge sees the whole event", 10.5, BLUE, ha="center")
    panel(ax, 0.36, 0.36, 0.29, 0.44, "(2) message passing", NAVY, fc=LIGHT, tsize=11.5)
    txt(ax, 0.505, 0.720, "each node/edge aggregates from its neighbours:", 10.8, DARK, ha="center")
    for i, s in enumerate(["IPs of this track to all PVs", "what other tracks chose", "PV positions and neighbourhood",
                           "geometric consistency of the chain"]):
        txt(ax, 0.505, 0.672 - i * 0.045, "• " + s, 10.8, DARK, ha="center")
    box(ax, 0.375, 0.395, 0.26, 0.055, "→ probability of each PV per track", fc="#EAF5EA", ec=GREEN,
        size=10.5, color=GREEN, r=0.015)
    panel(ax, 0.68, 0.36, 0.29, 0.44, "(3) chain-level decision", GREEN, tsize=11.5)
    txt(ax, 0.825, 0.720, "sum the daughter scores over the PVs of one B chain:", 10.5, DARK, ha="center")
    for i in range(5):
        txt(ax, 0.72, 0.680 - i * 0.036, f"daughter {i+1} score vector", 10, GREY)
    txt(ax, 0.825, 0.505, "+ ... = chain score", 11, DARK, ha="center")
    box(ax, 0.68, 0.415, 0.26, 0.06, "→ argmax: one PV for the whole chain", fc="#EAF5EA", ec=GREEN,
        size=10.5, color=GREEN, r=0.015)
    box(ax, 0.03, 0.06, 0.94, 0.25,
        "Difference in one line:\n"
        "• minIP: a local decision per track, no notion of the event or the chain  →  it fragments chains.\n"
        "• ours: one global decision over the whole graph, then forced to be chain-consistent.\n"
        "Cost: a trained network.  Benefit: uncertainties come for free and the same network serves pruning / reconstruction.",
        fc=LIGHT, ec=NAVY, size=12, ha="left")
    save(fig, "slide_08_method",
         "Three steps. (1) Represent the event as a graph where every track is connected to every PV, with IP features "
         "on the edges. (2) A graph network passes messages, so each decision can use other tracks, the PV structure "
         "and the chain geometry. (3) Sum the per-track scores over the PVs of one B chain and take the argmax — this "
         "structurally guarantees one PV for the whole chain. That is the essential difference from minIP.")


# ============================================================== 9 metrics
def s09():
    fig, ax = canvas("How we measure: two metrics, and one easily-missed subtlety",
                     "The standard answer (truth PV per track) comes from the MC simulation")
    panel(ax, 0.04, 0.55, 0.44, 0.28, "metric (1): per-track accuracy", BLUE)
    txt(ax, 0.26, 0.745, "compare the assignment track by track", 11.5, DARK, ha="center")
    for i, (y, ok) in enumerate([(0.70, 1), (0.665, 1), (0.63, 0), (0.595, 1)]):
        txt(ax, 0.10, y, f"track {i+1}", 11, GREY)
        txt(ax, 0.30, y, "OK" if ok else "X", 11, GREEN if ok else RED, "bold")
    txt(ax, 0.26, 0.575, "→ 3/4 = 75%", 12.5, DARK, "bold", ha="center")
    panel(ax, 0.52, 0.55, 0.44, 0.28, "metric (2): whole-chain accuracy", GREEN)
    txt(ax, 0.74, 0.745, "all daughters of a B chain must be right", 11, DARK, ha="center")
    for i, (y, ok) in enumerate([(0.70, 1), (0.665, 1), (0.63, 0), (0.595, 1)]):
        txt(ax, 0.58, y, f"track {i+1}", 11, GREY)
        txt(ax, 0.78, y, "OK" if ok else "X", 11, GREEN if ok else RED, "bold")
    txt(ax, 0.74, 0.575, "→ one X kills the chain ⇒ 0%", 12.5, RED, "bold", ha="center")
    box(ax, 0.04, 0.30, 0.92, 0.20,
        "Subtlety: minIP can ABSTAIN — tracks that fail its own filter are labelled 'no PV' (-1).\n"
        "Our model only ever outputs an argmax, so it can never abstain. Those tracks are 1.9% of the sample,\n"
        "which is why we quote both 'all' and 'excluding the abstain sentinel' numbers.",
        fc="#FFF8E7", ec=ORANGE, size=12, ha="left")
    txt(ax, 0.5, 0.20, "And we add two bias diagnostics beyond accuracy:", 13, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.145, "(i) the occupancy of PV indices (where does each method push tracks?)", 12.5, DARK, ha="center")
    txt(ax, 0.5, 0.095, "(ii) the direction of the mis-assignments (Δz, later in this deck)", 12.5, DARK, ha="center")
    save(fig, "slide_09_metrics",
         "Two metrics: per-track accuracy and whole-chain accuracy (the latter is what physics needs). "
         "One subtlety: minIP can abstain (tracks failing its filter get -1) while our model cannot - we therefore "
         "report both with and without those 1.9% of tracks. Plus two bias diagnostics: PV-index occupancy and the "
         "direction of mis-assignments.")


# ============================================================== 10 arms results
def s10():
    fig, ax = canvas("C. Does our model still work on the fixed production? (arm A / B / C)",
                     "Same 10 test files of BsToJpsiPhi_00342629; variable-denominator numbers as reported by the eval")
    rows = [["arm", "data / normalization", "model", "N (pruned)", "All %", "Perf %", "PV: HGNN vs minIP"],
            ["A", "0702 July", "v557", "14 196", "53.01", "19.32", "2.15 vs 12.64  (we win)"],
            ["B", "0904 as-is (new norm)", "v557", "1 436", "10.45", "9.89", "23.57 vs --  (broken)"],
            ["C", "0904 re-normalized to July", "v557", "6 635", "47.40", "38.88", "14.56 vs 10.35  (we lose)"],
            ["C", "0904 re-normalized to July", "v31", "8 643", "31.53", "24.55", "14.71 vs 10.35  (we lose)"],
            ["C", "0904 re-normalized to July", "v38", "7 161", "45.43", "37.77", "13.75 vs 10.35"],
            ["C", "0904 re-normalized to July", "v47", "7 138", "44.52", "36.24", "13.28 vs 10.35"]]
    table(ax, 0.04, 0.79, [0.06, 0.27, 0.08, 0.11, 0.09, 0.09, 0.30], rows, size=11.5, row_h=0.052)
    box(ax, 0.04, 0.30, 0.92, 0.16,
        "Reading: on arm B the July-trained models collapse (feature-scale mismatch), which is why arm C exists.\n"
        "On arm C the July-trained models still recover a large part of the chains, but their PV association\n"
        "LOSES to minIP (14.6 vs 10.4) - exactly the degradation the group suspected.",
        fc="#FDECEA", ec=RED, size=12, ha="left")
    box(ax, 0.04, 0.06, 0.92, 0.21,
        "Key observation for the fixed data: the coverage (N) drops a lot while the conditional quality "
        "(All/N) stays roughly the same.\n"
        "v557: coverage 89.4% (July) → 48.5% (0904) with N_total = 13 682 vs 15 883 for the same 10k events.\n"
        "⇒ the fixed production mainly hurts by fragmenting truth chains during pruning, and PV association flips sign.",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    save(fig, "slide_10_arms",
         "Three arms on the same test files. Arm B (fixed data, new normalization) breaks the July-trained models — "
         "that is a feature-scale mismatch, not a physics result. Arm C (fixed data re-normalized) is the fair test: "
         "the models still reconstruct a good fraction of chains, but PV association now LOSES to minIP, and the "
         "surviving-chain count (coverage) roughly halves. So the fixed production hurts mostly by fragmenting chains.")


# ============================================================== 11 head to head
def s11():
    fig, ax = canvas("Head-to-head on the same 20 test files: v31 vs v601",
                     "inclusive_00342451 test split, 0904 production; N_total = 13 255 truth chains (unpruned, fixed denominator)")
    rows = [["model", "trained on", "N (pruned)", "coverage", "All %", "Perf %", "All_fix %", "Perf_fix %", "PV: HGNN vs minIP"],
            ["v31", "July (0702)", "4 676", "35.3%", "25.43", "20.74", "9.0", "7.3", "13.80 vs 10.15  (lose)"],
            ["v601", "0904, v38 recipe", "7 549", "56.9%", "25.38", "16.76", "14.5", "9.5", "8.60 vs 10.15  (win)"]]
    table(ax, 0.03, 0.80, [0.07, 0.15, 0.10, 0.10, 0.08, 0.08, 0.09, 0.09, 0.24], rows, size=11, row_h=0.055)
    txt(ax, 0.5, 0.63, "Fixed-denominator view (All_fix = All# / 13 255) is the apples-to-apples comparison:", 13, NAVY, "bold", ha="center")
    box(ax, 0.06, 0.42, 0.88, 0.17,
        "coverage 35.3% → 56.9%   |   All_fix 9.0% → 14.5%   |   Perf_fix 7.3% → 9.5%\n"
        "PV association on all tracks: 13.80 (worse than minIP 10.15)  →  8.60 (better than minIP 10.15)",
        fc="#EAF5EA", ec=GREEN, size=13.5, ha="left")
    box(ax, 0.06, 0.20, 0.88, 0.18,
        "Also visible in the LCAG head: on the fixed data the July-trained model predicts class-1 correctly only\n"
        "22.7% of the time (v601: 69.1%) and class-2 10.2% (v601: 32.7%)  →  its edge/LCA head is the broken part,\n"
        "consistent with the pruning-side coverage loss.",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    txt(ax, 0.5, 0.10, "v38 / v47 / v557 in the same table are queued on the cluster (no free GPUs at the moment).",
        11.5, GREY, ha="center")
    save(fig, "slide_11_head2head",
         "The decisive comparison: the same 20 test files, evaluated with a fixed denominator of 13 255 truth chains. "
         "Training on the fixed production (v601) raises coverage from 35% to 57%, All_fix from 9.0% to 14.5% and "
         "flips PV association from losing to winning against minIP. The July-trained model's LCA head is the broken "
         "part — class-1 accuracy 22.7% vs 69.1%. v38/v47/v557 evaluations are still queued.")


# ============================================================== 12 occupancy
def s12():
    fig, ax = canvas("D. Bias diagnostic (1): PV-index occupancy",
                     "grey = truth, solid = ours, hatched = minIP; horizontal axis = PV index")
    pic_fit(ax, FIG + "/pv_bias_occupancy_en.png", 0.5, 0.20, 0.58, 16.8 / 9.0)
    box(ax, 0.04, 0.045, 0.92, 0.13,
        "Read it as: equal bar heights = unbiased. Left (new-data model): we match the truth almost exactly while minIP under-uses "
        "the leading PV and over-uses later ones.\nRight (degraded arm C): we instead overshoot PV0 by +3.5pp — a 'PV0 collapse' that "
        "is a very sensitive health indicator.",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_12_occupancy",
         "The occupancy plot shows WHERE each method sends tracks. The truth (grey) is the reference. Our new-data model "
         "matches it almost perfectly; minIP systematically under-fills the leading PV and over-fills later ones. "
         "The degraded model on the right overshoots PV0 — an easy-to-spot failure signature.")


# ============================================================== 13 chain
def s13():
    fig, ax = canvas("Bias diagnostic (2): chain consistency",
                     "whole-chain accuracy, where the difference comes from, and chain concordance")
    pic_fit(ax, FIG + "/pv_bias_chain_en.png", 0.5, 0.235, 0.53, 16.5 / 5.0)
    box(ax, 0.04, 0.045, 0.92, 0.13,
        "Chain concordance (all daughters share one PV, physics needs ~100%): ours 71 / 93 / 72 %   vs   minIP 63 / 59 / 59 %.\n"
        "Whole-chain accuracy: minIP 57 / 56 % → ours 66 / 90 %, i.e. the baseline fragments chains; the chain-level decision adds more.",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_13_chain",
         "Left: whole-chain accuracy. July: minIP 56% → ours 90.6%, and the chain-level decision reaches 94.6%. "
         "The middle panel decomposes the difference: blue = we are right and minIP wrong, red = the reverse. "
         "Right: dependence on the number of PVs. Bottom line: chain concordance is 92%/70% for us versus 58%/59% "
         "for minIP — that is the structural improvement.")


# ============================================================== 14 fair comparison
def s14():
    fig, ax = canvas("Fair comparison: samePV constraint added to the baseline",
                     "LHCb's classical workflow already contains a samePV requirement; our minIP baseline implemented only the per-track half")
    rows = [["dataset", "minIP", "minIP + chain vote", "ours", "ours (correct chains only)"],
            ["July 0702", "77.3", "83.3", "95.7", "97.0"],
            ["0904 new data", "79.7", "80.3", "83.5", "86.8"],
            ["0904 degraded (arm C)", "83.2", "74.3", "78.4", "84.4  (minIP+vote 88.1)"]] 
    table(ax, 0.05, 0.78, [0.24, 0.13, 0.20, 0.13, 0.28], rows, size=12, row_h=0.058)
    box(ax, 0.04, 0.42, 0.92, 0.16,
        "Adding the chain constraint to minIP closes most of the gap:\n"
        "July 77.3 → 83.3 (we still lead by 12.4pp)   |   0904 79.7 → 80.3 (we lead by 3.2pp)\n"
        "Degraded case 83.2 → 74.3 — on broken data 'consistency' becomes 'consistently wrong', so the vote hurts.",
        fc="#FFF8E7", ec=ORANGE, size=12, ha="left")
    txt(ax, 0.5, 0.345, "Honest conclusion: our gain comes mainly from chain-level global consistency,", 14, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.285, "not from a stronger per-track IP discrimination.", 14, NAVY, "bold", ha="center")
    box(ax, 0.04, 0.06, 0.92, 0.19,
        "Why this still matters: the classical workflow has to bolt samePV on by hand, while our network learns the "
        "consistency end-to-end together\nwith pruning and reconstruction, and it gives a calibrated per-track "
        "probability that the downstream vertex fit can use.",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    save(fig, "slide_14_fair",
         "The strictest test of our own claim: LHCb's classical workflow already includes a samePV constraint, so we add "
         "the chain-level vote to minIP. It closes most of the gap (July 77.3→83.3, 0904 79.7→80.3), and on degraded data "
         "it even hurts. Conclusion: our advantage is mainly global consistency, learned end-to-end rather than bolted on.")


# ============================================================== 15 physics dependence
def s15():
    fig, ax = canvas("E. Core analysis (1): mis-association rate vs physics variables",
                     "per-track, 300 events per dataset; solid = ours, dashed = minIP; error bars √(p(1−p)/n)")
    pic_fit(ax, FIG + "/pv_physics_dependence_en.png", 0.5, 0.175, 0.63, 16.5 / 9.0)
    box(ax, 0.04, 0.035, 0.92, 0.115,
        "Star panel (bottom-left): the truth PV is the closest-IP PV for 88–90% of tracks — the battle is entirely in the "
        "remaining 10–12%, where minIP is wrong by construction and we rescue 86% (July) / 33% (new data, v601).",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_15_physics",
         "Top row: mis-association versus pT, pseudo-rapidity and ghost probability — both methods degrade for low-pT, "
         "forward and hard-to-reconstruct tracks, but minIP is systematically worse. Bottom-left is the key panel: for "
         "88-90% of tracks the truth PV is simply the closest-IP PV, so the whole contest happens in the remaining 10-12%. "
         "Bottom-middle shows the same thing by IP rank: minIP jumps from 0.2% to ~100% error at rank≥1, we jump to 64% "
         "(v601) or 11% (July).")


# ============================================================== 16 direction
def s16():
    fig, ax = canvas("Core analysis (2): the DIRECTION of a mis-assignment (Δz)",
                     "Δz = z(assigned PV) − z(true PV), converted back to cm using the normalization constants; wrong tracks only")
    pic_fit(ax, FIG + "/pv_direction_delta_en.png", 0.5, 0.175, 0.63, 16.5 / 9.0)
    box(ax, 0.04, 0.035, 0.92, 0.115,
        "minIP errs systematically towards SMALLER z (70% of its errors; mean Δz = −2.8 / −6.7 / −4.8 cm) and lands further away "
        "(median |Δz| 3.4–9.3 cm);\nour errors are symmetric in direction and land on neighbouring PVs (median |Δz| 2.8–6.4 cm) "
        "⇒ even when wrong, we damage the flight distance less.",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_16_direction",
         "This is the analysis the numbers really hinge on. Top-left: the signed Δz distribution of wrong assignments. "
         "minIP is clearly skewed to negative Δz (assigns a PV with smaller z, i.e. upstream of the truth) in 70% of its "
         "errors, while our distribution is symmetric. Top-middle: how far the error lands — median |Δz| 2.8/5.8/6.4 cm "
         "for us versus 3.4/9.3/6.3 cm for minIP, so our mistakes are local. Top-right/bottom: the rescue rate on the "
         "tracks minIP must get wrong — 86% for the July model, 33-37% once the data degrades — decreasing with the IP gap.")


# ============================================================== 17 battlefield
def s17():
    fig, ax = canvas("The '10% battlefield': the metric we should be quoting",
                     "per-track, 300 events per dataset; 'hard subset' = truth PV is not the closest-IP PV")
    rows = [["dataset", "all tracks: ours / minIP", "hard subset share", "hard subset: ours", "hard subset: minIP", "rescue rate"],
            ["v601 / 0904 new data", "8.7% / 10.7%", "10.8%", "66.8%", "100%", "33.3%"],
            ["v557 / 0702 July", "2.2% / 12.0%", "12.0%", "13.8%", "100%", "86.2%"],
            ["v557 / 0904 degraded", "15.1% / 10.4%", "10.4%", "63.3%", "100%", "36.7%"]]
    table(ax, 0.04, 0.80, [0.22, 0.19, 0.14, 0.14, 0.14, 0.13], rows, size=11.5, row_h=0.056)
    box(ax, 0.04, 0.44, 0.92, 0.18,
        "Why this is the right headline: on the 88–90% 'easy' tracks minIP is right for free (0.2% error), so any\n"
        "overall accuracy is dominated by a subset where there is nothing to win. The informative numbers are:\n"
        "• rescue rate on the hard subset, • error rate even on the easy subset (v601 1.7%, July 0.6%, degraded 9.5%).",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    box(ax, 0.04, 0.20, 0.92, 0.20,
        "Degraded model: it fails BOTH subsets (easy 9.5%, hard 63%) — a broken representation, not just a hard-task problem.\n"
        "July model: 0.6% easy / 13.8% hard  ⇒  the physically meaningful 'PV association works' statement.\n"
        "New-data model (v601): 1.7% easy / 66.8% hard  ⇒  trained on the right production, but PV head still under-trained.",
        fc="#FFF8E7", ec=ORANGE, size=12, ha="left")
    txt(ax, 0.5, 0.12, "Proposal: report rescue-rate + median |Δz| as the primary PV-association metrics from now on.", 13, NAVY, "bold", ha="center")
    save(fig, "slide_17_battlefield",
         "This slide is the proposal for how to report PV association from now on. For 88-90% of tracks the truth PV is "
         "the closest-IP PV and minIP is right for free, so overall accuracy hides the physics. The informative pair is "
         "(a) the rescue rate on the hard 10-12% where minIP must fail, and (b) the error rate even on the easy subset, "
         "which exposes a broken representation (the degraded model fails both: 9.5% and 63%).")


# ============================================================== 18 diagnostic
def s18():
    fig, ax = canvas("A diagnostic we get for free: 'PV0 collapse' as a health check",
                     "Same model and same events — only the input-feature normalization was wrong")
    panel(ax, 0.04, 0.52, 0.44, 0.30, "unhealthy signature: PV0 collapse", RED, fc="#FDECEA")
    for i, s in enumerate(["PV0 occupancy 35.5% vs truth 32.1% (+3.5pp)",
                           "mean PV index 1.45, i.e. 0.15 lower than truth",
                           "65% of errors go towards EARLIER PVs (direction reversed)",
                           "chain concordance inflated (72% vs truth 69%)",
                           "yet whole-chain accuracy is the lowest (61.7% vs minIP 67.3%)"]):
        txt(ax, 0.06, 0.735 - i * 0.043, "· " + s, 11.5, DARK)
    panel(ax, 0.52, 0.52, 0.44, 0.30, "healthy signature", GREEN, fc="#EAF5EA")
    for i, s in enumerate(["occupancy matches truth within ~0.1pp",
                           "mean-index shift < 0.01",
                           "error direction symmetric (~50% each way)",
                           "chain concordance well above minIP (+11 to +34pp)",
                           "chain-level decision adds 5-10pp of whole-chain accuracy"]):
        txt(ax, 0.54, 0.735 - i * 0.043, "· " + s, 11.5, DARK)
    txt(ax, 0.5, 0.465, "⇒ three numbers (PV0 occupancy bias, mean-index shift, chain concordance) form a cheap health check "
                        "for any new model or dataset.", 12.5, NAVY, "bold", ha="center")
    box(ax, 0.04, 0.20, 0.92, 0.20,
        "Two small improvements we can implement immediately:\n"
        "1. give the PV head an abstain class ('no PV') — it matches what the baseline can do and gives the vertex fit an uncertainty;\n"
        "2. report the chain-level decision by default (its whole-chain gain is much larger than the per-track gain).",
        fc=LIGHT, ec=NAVY, size=12.5, ha="left")
    txt(ax, 0.5, 0.115, "(all numbers reproducible: analyze_pv_bias.py, dump_pv_tracks.py, report_figs/pv_*.csv)", 11, GREY, ha="center")
    save(fig, "slide_18_diagnostic",
         "A side product: the same model on the same events, with only the input normalization wrong, flips into a "
         "'PV0 collapse' — PV0 overshoot, mean index shifted down, error direction reversed, chain concordance inflated "
         "but whole-chain accuracy the worst. So these three numbers make a cheap health check. "
         "Two concrete improvements: add an abstain class, and report the chain-level decision by default.")


# ============================================================== 19 threshold
def s19():
    fig, ax = canvas("Other finding: the pruning threshold on the fixed data",
                     "fixed-denominator sweep with the pipeline's own reconstruction; N_total per event set; 300 events")
    rows = [["threshold", "coverage % (0904)", "All_fix % (0904)", "NoneIso % (0904)", "All_fix % (July)"],
            ["0.50", "67.6", "11.8", "55.9", "29.8"],
            ["0.70", "64.3", "17.6", "46.6", "37.5"],
            ["0.80", "60.4", "19.7", "40.7", "44.2"],
            ["0.90 (default)", "54.3", "26.2", "28.1", "50.2"],
            ["0.95", "46.8", "30.3", "16.5", "56.9"],
            ["0.96-0.97", "~40", "25.4", "12.6-14.9", "--"]]
    table(ax, 0.04, 0.79, [0.18, 0.19, 0.18, 0.19, 0.18], rows, size=11.5, row_h=0.05)
    box(ax, 0.04, 0.30, 0.92, 0.16,
        "Lowering the threshold does NOT help: All_fix(0904) rises monotonically up to ~0.95-0.97 and then falls.\n"
        "Reason: All + NoneIso = coverage exactly (PartReco/NotFound are 0), so extra surviving chains are immediately\n"
        "polluted by the background that a lower cut lets through — and on 0904 the edge head cannot tell them apart.",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    box(ax, 0.04, 0.06, 0.92, 0.21,
        "Supporting evidence — the pruning ROC (300 events):  node AUC 0.99 (July) vs 0.91 (0904);\n"
        "edge AUC 0.999 vs 0.70.  The edge head is the part that broke on the fixed production, which is why keeping a high cut\n"
        "(fewer, cleaner edges) beats lowering it. Recommendation: use 0.95-0.97 for the 0904 family, keep 0.90 for July.",
        fc="#FFF8E7", ec=ORANGE, size=12, ha="left")
    save(fig, "slide_19_threshold",
         "We swept the pruning threshold with the pipeline's own reconstruction and a fixed denominator. On the fixed "
         "production the optimum moves up to 0.95-0.97, and lowering the cut makes things worse: the extra chains that "
         "survive are immediately polluted by background that the lower cut admits, and All+NoneIso equals coverage exactly. "
         "The ROC explains it — on 0904 the edge head has AUC 0.70 versus 0.999 in July.")


# ============================================================== 20 ROC
def s20():
    fig, ax = canvas("Pruning ROC: what actually broke on the fixed production",
                     "node pruning = 'is this track in a truth b chain'; edge pruning = 'is this a truth structural edge'")
    pic_fit(ax, FIG + "/roc_pruning_A_vs_C_v557.png", 0.5, 0.22, 0.55, 14.0 / 6.0)
    box(ax, 0.04, 0.055, 0.92, 0.145,
        "node AUC  0.9915 (July) → 0.9123 (0904)      edge AUC  0.9993 → 0.6965\n"
        "⇒ the edge-pruning head is at random level on the fixed data. Working points: at thr 0.9 only ~9% of structural edges survive "
        "(background 0.1%); at thr 0.7, 13% survive but background rises to 2%.",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_20_roc",
         "The ROC makes the failure concrete: the node head is only slightly weaker (AUC 0.99 → 0.91) but the edge head "
         "drops from 0.999 to 0.70 — essentially random. That single fact explains the coverage loss and why a high "
         "threshold is preferable on the fixed data.")


# ============================================================== 21 training
def s21():
    fig, ax = canvas("Training on the new production: v601 and v602",
                     "Both start from the corresponding July checkpoint and are re-trained on inclusive_00342451 (0904), 200 train files")
    rows = [["version", "base / recipe", "init", "epochs", "lr", "status", "test on 0904 tst split"],
            ["v601", "v38 (b2 + chain-CE + class-2)", "v38 ep101", "20", "1e-4", "done (11.5 h)", "All 25.38 / Perf 16.76 (N = 7 549)"],
            ["v602", "v47 (v38 + mass head)", "v47 ep113", "20", "1e-4", "queued", "-"]]
    table(ax, 0.04, 0.78, [0.08, 0.24, 0.10, 0.08, 0.07, 0.13, 0.30], rows, size=11, row_h=0.058)
    box(ax, 0.04, 0.44, 0.92, 0.16,
        "Why start from the July checkpoint instead of from scratch: the feature scale changed with the production, so we\n"
        "need adaptation, and history says lr = 1e-4 is the biggest single lever (3e-5 gives no drift, 1e-4 gave +5.9pp).",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    box(ax, 0.04, 0.20, 0.92, 0.20,
        "Epoch budget: the framework's chunk loader determines the epoch size. With one sample (200 files) one epoch is\n"
        "168k training + 21k validation events ≈ 33 min — the historical scale. (A multi-sample config silently multiplied this\n"
        "by 17 and had to be killed; the loader sizes chunks from the total file count, which is a trap worth remembering.)",
        fc="#FFF8E7", ec=ORANGE, size=12, ha="left")
    txt(ax, 0.5, 0.115, "v602 will show whether the mass head adds anything once its normalization constants finally match the data.", 12.5, NAVY, "bold", ha="center")
    save(fig, "slide_21_training",
         "v601 = the v38 recipe and its July weights, re-trained on the fixed production (20 epochs, lr 1e-4) — it is done "
         "and its test numbers are on the previous slide. v602 = the same protocol with the v47 recipe (adds the mass head) "
         "and is queued. Note the epoch-size trap: the chunk loader scales the epoch with the total file count, so a "
         "multi-sample config silently became 17x bigger and had to be killed.")


# ============================================================== 22 bugs
def s22():
    fig, ax = canvas("Bugs and traps found along the way (worth sharing with the group)",
                     "Each of these silently changes numbers if you are not looking for it")
    items = [
        ("mass-head normalization constants are hard-coded",
         "The mass head un-normalizes momenta with constants that match the 0904 dictionary\n(pz scale +10077), but v46/v47 were trained on July data (pz scale −10015).\nIts regression targets were therefore computed with a mismatched scale — and this was the\nonly auxiliary head that ever paid off. v602 finally trains it with matching constants."),
        ("the 0904 production has fewer truth chains per event",
         "N_total = 15 883 (July) vs 13 682 (0904) for the same 10k events — ghost/clone removal and the\nfirst-64-bunch skip remove tracks. Coverage comparisons must use each production's own N_total."),
        ("sentinel tracks that cannot be compared like-for-like",
         "minIP may abstain (−1); our argmax cannot. 0.4-1.9% of track slots. Always state which\nconvention is used."),
        ("'dead' loss-weight keys are live now",
         "lca_weight / node_prune_weight were ignored by the code when v38 was trained but are read now,\nso training 'the same recipe' today is not bit-identical to history (combined loss ≈138 vs the old 35.9)."),
    ]
    y = 0.80
    for t, d in items:
        box(ax, 0.04, y - 0.155, 0.92, 0.155, None, fc="white", ec=GREY)
        txt(ax, 0.06, y - 0.030, t, 12.5, RED, "bold")
        txt(ax, 0.06, y - 0.100, d, 10.5, DARK)
        y -= 0.175
    save(fig, "slide_22_bugs",
         "Four traps we hit, each of which silently changes numbers. (1) The mass head's un-normalization constants are "
         "hard-coded to the 0904 dictionary while v46/v47 trained on July data — the one auxiliary head that helped was "
         "trained on mismatched targets. (2) The fixed production has ~14% fewer truth chains per event. (3) minIP can "
         "abstain, we cannot. (4) loss-weight keys that were dead when v38 was trained are live now, so 'same recipe' no "
         "longer means bit-identical.")


# ============================================================== 23 status
def s23():
    fig, ax = canvas("Current status (2026-09-21) and next steps", "Cluster: ~12k jobs queued, our GPU jobs are sitting idle")
    panel(ax, 0.04, 0.52, 0.44, 0.30, "running / queued", BLUE, fc=LIGHT)
    for i, s in enumerate(["head-to-head evals: v31 done, v38/v47/v557 queued",
                           "v602 training (v47 recipe on 0904): queued",
                           "local CPU analyses: finished (all figures in this deck)",
                           "real-data inference stats (0904 data): finished"]):
        txt(ax, 0.06, 0.735 - i * 0.05, "· " + s, 11.5, DARK)
    panel(ax, 0.52, 0.52, 0.44, 0.30, "results in hand", GREEN, fc="#EAF5EA")
    for i, s in enumerate(["v601 trained + tested on the fixed production",
                           "v31 vs v601 head-to-head (same 20 files)",
                           "PV bias & direction analysis (3 datasets)",
                           "pruning threshold sweep + ROC (July vs 0904)"]):
        txt(ax, 0.54, 0.735 - i * 0.05, "· " + s, 11.5, DARK)
    txt(ax, 0.5, 0.455, "Next steps", 14, NAVY, "bold", ha="center")
    for i, s in enumerate(["1. finish the head-to-head table (v38 / v47 / v557 on the same 20 files) and re-evaluate v601 at thr 0.95-0.97;",
                           "2. run v602 and compare 'with / without mass head' on the fixed production;",
                           "3. add the abstain class to the PV head, and switch the reported PV metric to rescue-rate + median |Δz|;",
                           "4. re-check the July-trained line on arm C for any remaining normalization mismatch (old_norm vs new_dict)."]):
        txt(ax, 0.06, 0.395 - i * 0.055, s, 11.5, DARK)
    save(fig, "slide_23_status",
         "Status: v31 has completed the head-to-head, the other three evaluations and the v602 training are queued behind a "
         "very busy cluster (about 12k jobs with 2.4k running). Next steps: complete the comparison table, re-evaluate v601 "
         "at the higher threshold, run v602, add the abstain class and change the reported metric to rescue-rate plus "
         "median |Δz|.")


# ============================================================== 24 summary
def s24():
    fig, ax = canvas("Summary: three sentences", None)
    box(ax, 0.05, 0.615, 0.90, 0.185,
        "1.  The task is to give every track its PV; physics demands that all daughters of one B chain share one PV.\n"
        "     The traditional baseline (minIP) decides per track, which fragments chains and biases tracks towards later PVs.",
        fc=LIGHT, ec=BLUE, size=13, ha="left")
    box(ax, 0.05, 0.395, 0.90, 0.185,
        "2.  Our graph network matches the truth distribution almost exactly and keeps chains together (concordance 92%/70% vs 58%/59%),\n"
        "     but a fair comparison shows the gain is mostly chain-level consistency — not a better per-track IP decision.",
        fc="#FFF8E7", ec=ORANGE, size=13, ha="left")
    box(ax, 0.05, 0.175, 0.90, 0.185,
        "3.  The fixed 0904 production breaks our July-trained models (PV association loses to minIP, edge head AUC 0.999→0.70);\n"
        "     re-training on it restores and flips the PV result (8.60 vs minIP 10.15) and raises fixed-denominator recovery 9.0% → 14.5%.",
        fc="#EAF5EA", ec=GREEN, size=13, ha="left")
    txt(ax, 0.5, 0.105, "Next: abstain class, rescue-rate reporting, v602, and the higher pruning threshold for 0904-family data.",
        13, NAVY, "bold", ha="center")
    save(fig, "slide_24_summary",
         "Three sentences. (1) The task and its physics constraint; the baseline fails the constraint by construction. "
         "(2) Our model keeps chains together, but the honest attribution of the gain is global consistency. "
         "(3) The fixed production breaks the July-trained models; re-training on the fixed production restores the PV "
         "performance and improves the fixed-denominator recovery. Next steps as listed.")



# ============================================================== 25 pruning line
def s22b():
    fig, ax = canvas("The pruning line: levers move the operating point, not the ROC",
                     "per-track node/edge scores dumped on the same 250 events per version; cross-checked against the official eval")
    pic_fit(ax, FIG + "/pruning_line_analysis.png", 0.5, 0.185, 0.615, 16.5 / 9.0)
    box(ax, 0.04, 0.035, 0.92, 0.135,
        "* node AUC spans only 0.9888-0.9944 over all 13 levers, while the thr-0.9 recall spans 80.6-87.6% at essentially constant\n"
        "   background (0.69-1.18%): what the levers change is the SCORE SCALE, not the separation.\n"
        "  The equivalent threshold (to reproduce the control's recall) is 0.90 for the control, 0.94-0.96 for weight 5-20, and 0.74 for focal.",
        fc=LIGHT, ec=BLUE, size=11, ha="left")
    save(fig, "slide_22b_pruning_line",
         "This is the pruning-focused line analysed with per-track scores on a fixed set of 250 events. Three findings. "
         "(1) The ROC barely moves: node AUC spans 0.9888-0.9944 across all thirteen versions, whereas the recall at the "
         "historical threshold 0.9 spans 7 percentage points at almost unchanged background — so the levers act as a "
         "score-gain knob. Each version has an equivalent threshold (0.90 for the control, 0.96 for node-weight 5, 0.74 for "
         "focal), which is why re-tuning the threshold is mandatory when a lever changes. (2) The node-weight dose-response "
         "saturates: recall plateaus at weight 5 while weights 10 and 20 keep adding coverage that is paid for in purity — "
         "the official fixed-denominator metric therefore peaks at weight 5. (3) The chain-survival proxy computed here on "
         "CPU tracks the official N (25% -> 12.9k chains, 5% -> 4.7k), so coverage can be predicted without a GPU eval; "
         "and run-to-run calibration jitter is about 2.6pp of recall, which is the size of the claimed v557-over-v559 gain.")


def build():
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for png, notes in SLIDES:
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(png, 0, 0, width=prs.slide_width, height=prs.slide_height)
        s.notes_slide.notes_text_frame.text = notes
    out = f"{BASE}/PV_association_DFEI_EN.pptx"
    prs.save(out)
    print("wrote", out, f"({len(SLIDES)} slides)")


if __name__ == "__main__":
    for f in (s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22b, s22, s23, s24):
        f()
    build()
