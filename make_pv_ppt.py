"""生成《PV 关联：minIP vs 我们的图神经网络》科普 PPT。

做法: 每页用 matplotlib 按 16:9 幻灯片尺寸画好 (图 + 文字都在图里), 存 PNG;
      再用 python-pptx 把每页 PNG 满幅放进一页, 并把"讲稿"写进该页的备注栏。
      这样既保证排版不会溢出, 又给你留了可编辑的文字(备注)。
输出: report_figs/pv_ppt/slide_XX.png 与 PV_association_explained.pptx
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import os

FIG = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/report_figs"
OUT = FIG + "/pv_ppt"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK HK", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F4E79"
BLUE = "#2E75B6"
RED = "#C0392B"
GREEN = "#1E8449"
GREY = "#7F7F7F"
ORANGE = "#D68910"
DARK = "#2B2B2B"
LIGHT = "#F2F6FA"

SLIDES = []          # (文件名, 讲稿)


def canvas(title, sub=None, footer=None, bg="white"):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if title:
        txt(ax, 0.035, 0.945, title, 26, NAVY, "bold", va="top")
        ax.plot([0.035, 0.965], [0.888, 0.888], color=NAVY, lw=2.2)
    if sub:
        txt(ax, 0.035, 0.855, sub, 13.5, GREY, va="top")
    if footer:
        txt(ax, 0.965, 0.022, footer, 9.5, GREY, ha="right")
    return fig, ax


def txt(ax, x, y, s, size=13, color=DARK, weight="normal", ha="left", va="center", style="normal", wrap=False):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style=style, linespacing=1.45, wrap=wrap, zorder=5)


def box(ax, x, y, w, h, s=None, fc="white", ec=GREY, lw=1.4, size=12, color=DARK,
        weight="normal", r=0.02, ha="center", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    if s:
        txt(ax, x + w / 2 if ha == "center" else x + 0.012, y + h / 2, s, size, color, weight, ha=ha, va="center")


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.8, ls="-", style="-|>", ms=11, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw, ls=ls,
                                 mutation_scale=ms, shrinkA=0, shrinkB=0, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def dot(ax, x, y, s=90, c=BLUE, ec="white", lw=1.5, z=6):
    ax.add_patch(Circle((x, y), s / 40000 * 4, fc=c, ec=ec, lw=lw, zorder=z))


def seg(ax, x1, y1, x2, y2, color=GREY, lw=1.2, ls="-", z=3, alpha=1.0):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha)


def pic(ax, path, x, y, w, h):
    ax.imshow(mpimg.imread(path), extent=[x, x + w, y, y + h], aspect="auto", zorder=1)


def bar(ax, x, y, w, h, val, vmax, color, label=None, lsize=10, base=0.0):
    """在 (x,y)-(x+w,y+base+h) 内画一根柱: 高度 ∝ val/vmax"""
    hh = (val / vmax) * h if vmax else 0
    ax.add_patch(Rectangle((x, y + base), w, max(hh, 0.001), fc=color, ec="none", zorder=3))
    if label:
        txt(ax, x + w / 2, y + base + hh + 0.012, label, lsize, DARK, ha="center", va="bottom")


def save(fig, name, notes):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    SLIDES.append((p, notes))
    print("写出", p)


# ==================================================================== S1 标题
def s01():
    fig, ax = canvas(None, bg="white")
    ax.add_patch(Rectangle((0, 0), 1, 0.22, fc=NAVY, ec="none"))
    txt(ax, 0.5, 0.86, "径迹与对撞顶点(PV)的匹配：", 34, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.775, "传统方法 minIP 与我们的图神经网络差在哪", 27, BLUE, "bold", ha="center")
    txt(ax, 0.5, 0.665, "—— 给完全不了解这个问题的人讲清楚 ——", 15, GREY, ha="center", style="italic")
    box(ax, 0.13, 0.33, 0.74, 0.24,
        "内容：① PV 是什么、为什么要匹配\n② 传统方法怎么做、它的两个先天倾向\n"
        "③ 我们的方法怎么做、实测差异在哪\n④ 一个能直接当诊断工具用的发现",
        fc=LIGHT, ec=BLUE, size=14.5, color=DARK, ha="left")
    txt(ax, 0.5, 0.235, "数据：LHCb 模拟（0904 生产版 MC）；模型：HGNN（多任务异构图网络）",
        11.5, GREY, ha="center")
    txt(ax, 0.5, 0.185, "所有数字都是与蒙特卡洛真值逐径迹比对的正确率", 11.5, GREY, ha="center")
    save(fig, "slide_01_title",
         "先说明：这页只讲一件事——‘一条径迹到底来自哪个对撞顶点’这个判断，"
         "传统方法和我们的模型有什么系统性的差别。全程不需要粒子物理背景。"
         "数据来自 LHCb 的模拟样本（0904 版 MC），‘真值’由蒙特卡洛给出，用来当标准答案。")


# ============================================================ S2 一次对撞里有什么
def s02():
    fig, ax = canvas("先看一个事件：一束对撞里有好多个‘对撞点’", "沿束流方向（z 轴）分布的多个 pp 主顶点 = 多个 PV")
    # z 轴
    seg(ax, 0.08, 0.34, 0.94, 0.34, NAVY, 2.4, z=2)
    arrow(ax, 0.94, 0.34, 0.975, 0.34, NAVY, 2.4)
    txt(ax, 0.90, 0.30, "z（束流方向）", 11, NAVY)
    pvs = [(0.20, 0.34, "PV 0"), (0.42, 0.34, "PV 1"), (0.63, 0.34, "PV 2"), (0.82, 0.34, "PV 3")]
    for i, (x, y, lab) in enumerate(pvs):
        dot(ax, x, y, 130, RED)
        txt(ax, x, y + 0.045, lab, 12, RED, "bold", ha="center")
    # 径迹
    tr = [(0.20, 0.34, [(0.20, 0.34), (0.15, 0.52), (0.11, 0.70)]),
          (0.20, 0.34, [(0.20, 0.34), (0.26, 0.50), (0.30, 0.71)]),
          (0.42, 0.34, [(0.42, 0.34), (0.37, 0.53), (0.34, 0.72)]),
          (0.42, 0.34, [(0.42, 0.34), (0.48, 0.51), (0.52, 0.68)]),
          (0.63, 0.34, [(0.63, 0.34), (0.59, 0.50), (0.56, 0.73)]),
          (0.63, 0.34, [(0.63, 0.34), (0.68, 0.53), (0.72, 0.71)]),
          (0.82, 0.34, [(0.82, 0.34), (0.79, 0.51), (0.77, 0.70)]),
          (0.82, 0.34, [(0.82, 0.34), (0.87, 0.50), (0.90, 0.69)])]
    for _, _, pts in tr:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=BLUE, lw=2.0, alpha=.85, zorder=3)
    txt(ax, 0.05, 0.775, "蓝色 = 径迹（带电粒子的轨迹）", 12, BLUE)
    txt(ax, 0.05, 0.745, "红色 = 主顶点 PV（质子-质子的对撞点）", 12, RED)
    box(ax, 0.06, 0.06, 0.30, 0.17,
        "一个真实事件里：\n平均 93 条径迹、7.6 个 PV\n（PV 沿 z 排开，彼此只差几厘米）",
        fc=LIGHT, ec=BLUE, size=12.5, ha="left")
    box(ax, 0.40, 0.06, 0.54, 0.17,
        "关键：每条径迹只是空间中的一条线。\n"
        "只看它自己，你无法判断它来自哪个 PV —— 必须做‘关联’。",
        fc="#FFF8E7", ec=ORANGE, size=12.5, color=DARK, ha="left")
    save(fig, "slide_02_event",
         "讲法：一次 pp 对撞里，实际上同时发生好几次质子-质子碰撞，它们沿束流方向（z）拉开，"
         "彼此只差几厘米。每一次碰撞产生一个主顶点（primary vertex, PV）。"
         "探测器重建出来的是几百条径迹（带电粒子轨迹），但不知道每条径迹属于哪个顶点。"
         "我们的样本里平均一个事件 93 条径迹、7.6 个 PV。")


# ============================================================ S3 什么是 PV 关联
def s03():
    fig, ax = canvas("所谓‘PV 关联’：给每条径迹回答一个问题 —— 你来自哪个对撞点？",
                     "这是 b 物理测量的基础设施：B 介子从某个 PV 飞出、飞一段再衰变")
    # 左：真值
    box(ax, 0.04, 0.52, 0.44, 0.30, fc="white", ec=GREY)
    txt(ax, 0.26, 0.79, "真值（模拟给出的标准答案）", 12.5, GREY, "bold", ha="center")
    seg(ax, 0.07, 0.60, 0.45, 0.60, NAVY, 2)
    for x, c in [(0.12, RED), (0.24, RED), (0.36, RED)]:
        dot(ax, x, 0.60, 110, c)
    for x0, dx in [(0.12, -0.02), (0.12, 0.03), (0.24, -0.03), (0.24, 0.02), (0.36, -0.02), (0.36, 0.03)]:
        seg(ax, x0, 0.60, x0 + dx, 0.72, BLUE, 1.8)
    txt(ax, 0.26, 0.555, "每条径迹的‘出身’已知", 11, GREY, ha="center")
    # 右：任务
    box(ax, 0.52, 0.52, 0.44, 0.30, fc="white", ec=GREY)
    txt(ax, 0.74, 0.79, "我们要做的判断", 12.5, BLUE, "bold", ha="center")
    seg(ax, 0.55, 0.60, 0.93, 0.60, NAVY, 2)
    for x in [0.60, 0.72, 0.84]:
        dot(ax, x, 0.60, 110, GREY)
    for x0, dx in [(0.60, -0.02), (0.60, 0.03), (0.72, -0.03), (0.72, 0.02), (0.84, -0.02), (0.84, 0.03)]:
        seg(ax, x0, 0.60, x0 + dx, 0.72, GREY, 1.8)
    txt(ax, 0.74, 0.555, "？ 每条径迹归哪个 PV：模型要输出的答案", 11, BLUE, ha="center")
    # 物理意义
    box(ax, 0.04, 0.30, 0.92, 0.17,
        "B 介子（含 b 夸克的粒子）从一个 PV 产生，飞出几百微米后再衰变成几条径迹。",
        fc=LIGHT, ec=BLUE, size=13.5, ha="left")
    txt(ax, 0.07, 0.345, "要测量它的飞行距离、寿命、CP 破坏 —— 就必须知道：它的母顶点是哪个 PV，它的子径迹又各自属于谁。",
        12.5, DARK)
    txt(ax, 0.07, 0.505, "→ 关联错了，飞行距离就算错，顶点分辨率变差，物理结果被系统性偏置。", 12.5, RED, "bold")
    box(ax, 0.04, 0.06, 0.92, 0.18,
        "文献里的说法（LHCb Run 3 论文 & 我们参考的 DFEI 论文）：\n"
        "‘PV 误关联会严重劣化 PV 分辨率，并偏置 b 强子衰变飞行距离与方向的测量，最终影响时间依赖 CP 测量的精度。’",
        fc="#FFF8E7", ec=ORANGE, size=12.5, ha="left")
    save(fig, "slide_03_what_is_pv_asso",
         "讲法：所谓 PV 关联，就是给每条径迹标注它来自哪个对撞点。"
         "为什么重要：我们关心的 B 介子是从某个 PV 产生的，它飞出一段距离（几百微米）后才衰变。"
         "要测飞行距离/寿命/CP 破坏，就必须知道 B 的母顶点是哪条 PV、它的子径迹属于谁。"
         "所以关联错 → 飞行距离错 → 物理结果被系统性偏置。这不是精度小问题，是系统误差来源。")


def panel(ax, x, y, w, h, title, color=BLUE, fc="white", tsize=13):
    """带内嵌标题的面板: 标题永远在框内, 不会和页面副标题打架"""
    box(ax, x, y, w, h, fc=fc, ec=color, lw=1.6)
    txt(ax, x + w / 2, y + h - 0.033, title, tsize, color, "bold", ha="center")
    return x, y, w, h


# ============================================================ S4 物理硬约束
def s04():
    fig, ax = canvas("全场最关键的一条物理约束：同一条 B 衰变链，所有子径迹必然共用同一个 PV",
                     "这不是模型的假设，是物理事实；任何方法都必须尊重它")

    def draw_panel(x0, title, color, correct):
        panel(ax, x0, 0.40, 0.44, 0.42, title, color)
        # PV 线
        pvx = [x0 + 0.075, x0 + 0.21, x0 + 0.345]
        seg(ax, x0 + 0.03, 0.545, x0 + 0.41, 0.545, NAVY, 2, z=2)
        for i, xx in enumerate(pvx):
            dot(ax, xx, 0.545, 95, GREY)
            txt(ax, xx, 0.505, f"PV{i}", 10, GREY, ha="center")
        # B 衰变顶点 + 子径迹
        bx, by = x0 + 0.21, 0.72
        dot(ax, bx, by, 130, RED)
        txt(ax, bx, by + 0.038, "B 衰变顶点", 10, RED, ha="center")
        seg(ax, bx, by, bx, 0.555, GREY, 1.3, ls=(0, (3, 3)), z=2)
        txt(ax, bx + 0.012, 0.655, "飞行距离", 9, GREY)
        # 子径迹: 正确 -> 全部到 PV0; 错误 -> 分散到三个 PV
        tgt = [0] * 5 if correct else [0, 0, 1, 2, 1]
        cols = [GREEN] * 5 if correct else [GREEN, GREEN, RED, BLUE, RED]
        for k, (t, c) in enumerate(zip(tgt, cols)):
            dx = -0.055 + k * 0.028
            ax.plot([bx, pvx[t] + dx * 0.15], [by, 0.558], color=c, lw=1.7, alpha=.9, zorder=3)
        for i, xx in enumerate(pvx):
            ok = (not correct) or (i == 0)
            txt(ax, xx, 0.462, "✓" if ok else "×", 13, GREEN if ok else RED, "bold", ha="center", va="center")
        txt(ax, x0 + 0.22, 0.427, "✓ = 有子径迹被指到该 PV；正确的一条链只应有一个 ✓", 9, GREY, ha="center")

    draw_panel(0.04, "正确：5 条子径迹都指向 PV0", GREEN, True)
    draw_panel(0.52, "错误：被拆给 PV0 / PV1 / PV2", RED, False)
    box(ax, 0.04, 0.235, 0.44, 0.135,
        "→ 飞行距离 = B 衰变顶点 − 正确的 PV\n→ 全链共享一个 PV，顶点拟合才成立",
        fc="#EAF5EA", ec=GREEN, size=11.5, ha="left")
    box(ax, 0.52, 0.235, 0.44, 0.135,
        "→ 飞行距离被算成一堆混合值\n→ 顶点拟合被污染，分辨率变差，物理量被偏置",
        fc="#FDECEA", ec=RED, size=11.5, ha="left")
    txt(ax, 0.5, 0.155, "评价一个方法，不只看‘逐条径迹对不对’，更要看‘整条 B 链是不是被一致地指到同一个 PV’。",
        13.5, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.095, "后面会看到：传统方法恰恰在这件事上先天不足。", 12, GREY, ha="center")
    save(fig, "slide_04_constraint",
         "这一页只讲一个物理事实：一条 B 介子衰变出来的所有径迹，一定来自同一个 PV。"
         "所以‘全链一致’是硬约束，不是可选偏好。评价方法时要看两个层面："
         "① 逐条径迹的正确率；② 整条链有没有被一致地指到同一个 PV。"
         "记住这条，下面 minIP 的两个‘倾向’就好懂了。")


# ============================================================ S5 minIP 是什么
def s05():
    fig, ax = canvas("传统方法 minIP：让每条径迹各自挑一个‘离自己最近的 PV’",
                     "IP = 径迹到某个 PV 的最短距离（impact parameter，图中虚线）")
    # 径迹直线
    (x1, y1), (x2, y2) = (0.09, 0.31), (0.60, 0.75)
    ax.plot([x1, x2], [y1, y2], color=BLUE, lw=2.6, zorder=3)
    txt(ax, 0.605, 0.735, "径迹", 11.5, BLUE)
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy

    def foot(px, py):
        t = ((px - x1) * dx + (py - y1) * dy) / L2
        return x1 + t * dx, y1 + t * dy

    for (px, py, lab, ipl, best) in [(0.20, 0.705, "PV0", 0.12, False),
                                     (0.37, 0.560, "PV1", 0.03, True),
                                     (0.53, 0.700, "PV2", 0.31, False)]:
        fx, fy = foot(px, py)
        seg(ax, px, py, fx, fy, GREEN if best else GREY, 2.2 if best else 1.4, z=4)
        dot(ax, px, py, 115, RED, z=6)
        txt(ax, px, py + 0.034, lab, 11.5, RED, "bold", ha="center")
        txt(ax, (px + fx) / 2 + 0.008, (py + fy) / 2 - 0.018, f"IP = {ipl} cm", 10.5,
            GREEN if best else GREY, "bold" if best else "normal")
    txt(ax, 0.09, 0.265, "对每条径迹、每个 PV 各算一个 IP，取最小的那个 PV", 11.5, DARK)
    txt(ax, 0.09, 0.225, "→ 这里 PV1 的 IP 最小 ⇒ 这条径迹判给 PV1", 12, GREEN, "bold")
    # 右列: 结构说明 + 流程图
    txt(ax, 0.665, 0.745, "传统方法的结构", 13, NAVY, "bold")
    for i, s in enumerate(["只看这一条径迹与各 PV 的几何", "每条径迹独立决定、互不通气",
                           "简单、快，几十年来的默认做法"]):
        txt(ax, 0.665, 0.705 - i * 0.038, "• " + s, 11, DARK)
    box(ax, 0.665, 0.50, 0.13, 0.075, "径迹", fc=LIGHT, ec=BLUE, size=11)
    arrow(ax, 0.795, 0.5375, 0.825, 0.5375)
    box(ax, 0.825, 0.50, 0.16, 0.075, "算 3 个 IP", fc=LIGHT, ec=BLUE, size=11)
    arrow(ax, 0.905, 0.50, 0.905, 0.445)
    box(ax, 0.825, 0.37, 0.16, 0.075, "argmin", fc="#FFF8E7", ec=ORANGE, size=11)
    arrow(ax, 0.825, 0.4075, 0.795, 0.4075)
    box(ax, 0.665, 0.37, 0.13, 0.075, "→ PV1", fc="#EAF5EA", ec=GREEN, size=11.5,
        color=GREEN, weight="bold")
    txt(ax, 0.82, 0.325, "每条径迹各自走一遍（彼此独立）", 10.5, GREY, ha="center")
    box(ax, 0.06, 0.06, 0.88, 0.13,
        "minIP 只用了‘这一条径迹’的信息：它看不到其它径迹怎么选、一条 B 链的伙伴选了谁、PV 的全局结构。\n"
        "所以在同一条 B 链内部，它没有任何机制保证 5 条子径迹会挑到同一个 PV。",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    save(fig, "slide_05_minip",
         "讲法：IP（impact parameter）= 径迹这条直线到某个 PV 点的最短距离，图上就是虚线那段。"
         "minIP 的做法非常朴素：对每条径迹，算它到每个 PV 的 IP，选最小那个——这里 PV1 的 IP 最小，"
         "所以这条径迹判给 PV1。两个结构性特点：① 只看这一条径迹自己的几何；② 每条径迹独立决策、互不通气。"
         "它简单、快，是几十年来的默认做法，所以我们拿它当基线。")


# ============================================================ S6 minIP 的两个倾向
def s06():
    fig, ax = canvas("minIP 的两个先天倾向（用真实数据测出来的）",
                     "数据集：LHCb 0904 模拟；把每条径迹的判定与真值逐条比对")
    # 倾向 1：把链撕开
    panel(ax, 0.04, 0.44, 0.44, 0.36, "倾向 ①：把同一条 B 链‘撕开’", RED)
    pvx = [0.115, 0.26, 0.405]
    seg(ax, 0.07, 0.565, 0.45, 0.565, NAVY, 1.8, z=2)
    for i, xx in enumerate(pvx):
        dot(ax, xx, 0.565, 90, GREY)
        txt(ax, xx, 0.535, f"PV{i}", 9.5, GREY, ha="center")
    bx, by = 0.26, 0.70
    dot(ax, bx, by, 110, RED)
    txt(ax, bx, by + 0.033, "B 衰变", 9.5, RED, ha="center")
    for k, (t, c) in enumerate([(0, GREEN), (0, GREEN), (1, RED), (2, BLUE), (1, RED)]):
        dx = -0.05 + k * 0.025
        ax.plot([bx, pvx[t] + dx * 0.2], [by, 0.575], color=c, lw=1.7, alpha=.9, zorder=3)
    txt(ax, 0.26, 0.495, "5 条子径迹被分给 3 个不同 PV", 11, DARK, ha="center")
    txt(ax, 0.26, 0.462, "实测链内一致率 59%（物理上应为 100%）", 11.5, RED, "bold", ha="center")
    # 倾向 2：系统性偏后
    panel(ax, 0.52, 0.44, 0.44, 0.36, "倾向 ②：系统性偏向‘后位’PV", ORANGE)
    txt(ax, 0.75, 0.735, "它把径迹往索引更大的 PV 上推（平均 PV 索引）", 10.8, DARK, ha="center")
    for i, (lab, v, c) in enumerate([("真值", 1.60, GREY), ("我们", 1.61, GREEN), ("minIP", 1.76, RED)]):
        y = 0.685 - i * 0.048
        txt(ax, 0.56, y, lab, 11, c, "bold", ha="left")
        ax.add_patch(Rectangle((0.63, y - 0.011), (v - 1.5) / 0.3 * 0.17, 0.022, fc=c, ec="none", zorder=3))
        txt(ax, 0.94, y, f"{v:.2f}", 10.5, c, ha="right")
    txt(ax, 0.74, 0.508, "错配时 2/3 以上是‘推到更靠后的 PV’", 11.5, RED, "bold", ha="center")
    txt(ax, 0.74, 0.474, "（错配方向平均 Δ = +0.78 个索引）", 10.5, GREY, ha="center")
    box(ax, 0.04, 0.275, 0.92, 0.125,
        "两个倾向都不是‘随机误差’，而是会随事件堆积（PV 越多）越来越明显的系统性行为 —— 这正是需要被替代的原因。",
        fc="#FFF8E7", ec=ORANGE, size=12.5, ha="left")
    box(ax, 0.04, 0.06, 0.92, 0.185,
        "数据来源：0904 生产版 MC 的一个测试样本；逐径迹与真值比对，共 37 255 个‘(链, 子径迹)’样本 / 7 549 条真值 B 链。\n"
        "‘链内一致率’ = 一条 B 链的所有子径迹被指到同一个 PV 的比例；真值（物理事实）在完美重建时为 100%。\n"
        "‘PV 索引’ = 事件里各 PV 沿 z 的排序编号，索引越大越靠后。",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_06_minip_bias",
         "讲法：用真实数据测出来 minIP 有两个先天倾向。"
         "① 因为它逐条独立决策，同一条 B 链的子径迹很容易被分到不同 PV——实测链内一致率只有 59%，"
         "而物理上应该是 100%（同一条链必须共用一个 PV）。② 它系统性地把径迹推向索引更大（更靠后）的 PV："
         "平均 PV 索引 1.76，而真值是 1.60；错配时三分之二以上是‘推到更靠后的 PV’。"
         "这不是随机噪声，是会随堆积（PV 数目）加重并被测量吸收的系统性偏置。")


# ============================================================ S7 我们的方法
def s07():
    fig, ax = canvas("我们的方法：把‘径迹–PV 匹配’当成一整张图上的推理来做",
                     "① 用图神经网络看全局信息   ② 再用‘链级求和’决策保证同链同 PV")
    # 左: 二分图
    panel(ax, 0.03, 0.36, 0.30, 0.44, "① 建图：每对‘径迹 × PV’连一条边", BLUE, tsize=11.5)
    ty = [0.700, 0.645, 0.590, 0.535]
    py = [0.700, 0.615, 0.530]
    for y1 in ty:
        for y2 in py:
            seg(ax, 0.085, y1, 0.275, y2, GREY, 0.7, alpha=.5)
    for y in ty:
        dot(ax, 0.085, y, 70, BLUE, z=6)
    for y in py:
        dot(ax, 0.275, y, 70, RED, z=6)
    txt(ax, 0.085, 0.732, "径迹", 10, BLUE, "bold", ha="center")
    txt(ax, 0.275, 0.732, "PV", 10, RED, "bold", ha="center")
    txt(ax, 0.18, 0.465, "每条边带 log IP 等特征\n（完整二分图：每对都连线）", 10.5, GREY, ha="center")
    txt(ax, 0.18, 0.395, "→ 每节点/边都能看到整体结构", 10.5, BLUE, ha="center")
    # 中: GNN
    panel(ax, 0.36, 0.36, 0.29, 0.44, "② 图神经网络：消息传递", NAVY, fc=LIGHT, tsize=11.5)
    txt(ax, 0.505, 0.720, "每个节点/边不断汇总邻居信息：", 10.8, DARK, ha="center")
    for i, s in enumerate(["这条径迹到各 PV 的 IP", "同事件其它径迹的选择", "PV 的位置与邻域结构",
                           "整条链的几何一致性"]):
        txt(ax, 0.505, 0.672 - i * 0.045, "• " + s, 10.8, DARK, ha="center")
    box(ax, 0.375, 0.395, 0.26, 0.055, "→ 输出：每条径迹属于各 PV 的概率",
        fc="#EAF5EA", ec=GREEN, size=10.5, color=GREEN, r=0.015)
    # 右: 链级决策
    panel(ax, 0.68, 0.36, 0.29, 0.44, "③ 链级求和决策（关键一步）", GREEN, tsize=11.5)
    txt(ax, 0.825, 0.720, "把一条 B 链里 5 条子径迹的分数对各 PV 求和：", 10.5, DARK, ha="center")
    for i in range(5):
        txt(ax, 0.72, 0.680 - i * 0.036, f"径迹{i+1} 的分数向量", 10, GREY)
    txt(ax, 0.825, 0.505, "＋  …  ＝ 链的总分", 11, DARK, ha="center")
    box(ax, 0.375 + 0.305, 0.415, 0.26, 0.06, "→ argmax：整条链共用这一个 PV",
        fc="#EAF5EA", ec=GREEN, size=10.5, color=GREEN, r=0.015)
    box(ax, 0.03, 0.06, 0.94, 0.25,
        "和 minIP 的本质差别：\n"
        "• minIP：只看一条径迹自己的 IP，逐条独立决策 → 没有全局概念，天生会把链撕开。\n"
        "• 我们：整张图一起吃信息（别人的选择、PV 结构、链的几何），再强制链级一致 → 决策是全局一致的。\n"
        "代价是要训练一个神经网络；好处是它天然输出不确定性，并且能和剪枝、径迹重建等任务共享同一个网络。",
        fc=LIGHT, ec=NAVY, size=12.5, ha="left")
    save(fig, "slide_07_our_method",
         "讲法：我们的做法分三步。① 把事件表示成一张图：每条径迹与每个 PV 之间都连一条边（完整二分图），"
         "边上带着 IP 等几何特征。② 用图神经网络在这张图上做消息传递，"
         "于是每条径迹的判定可以用到别人的选择、PV 的结构、整条链的几何一致性——这是 minIP 拿不到的信息。"
         "③ 关键的一步：对一条 B 链内所有子径迹的分数按 PV 求和，再取最大，输出一个 PV。"
         "这从结构上保证了‘同链同 PV’。差别一句话：minIP 是逐条径迹的局部判断，我们是整张图上的全局一致判断。")


# ============================================================ S8 怎么评测
def s08():
    fig, ax = canvas("怎么公平地比：两个指标 + 一个容易被忽略的口径细节",
                     "标准答案来自模拟（MC）逐径迹给出的真 PV")
    box(ax, 0.04, 0.55, 0.44, 0.28, fc="white", ec=BLUE, lw=1.6)
    txt(ax, 0.26, 0.795, "指标 ①：逐径迹正确率", 14, BLUE, "bold", ha="center")
    txt(ax, 0.26, 0.745, "一条一条径迹比对：判对了多少", 11.5, DARK, ha="center")
    for i, (y, ok) in enumerate([(0.70, 1), (0.665, 1), (0.63, 0), (0.595, 1)]):
        txt(ax, 0.10, y, f"径迹{i+1}", 11, GREY)
        txt(ax, 0.30, y, "✓" if ok else "×", 12, GREEN if ok else RED, "bold")
    txt(ax, 0.26, 0.575, "→ 3/4 = 75%", 12.5, DARK, "bold", ha="center")
    box(ax, 0.52, 0.55, 0.44, 0.28, fc="white", ec=GREEN, lw=1.6)
    txt(ax, 0.74, 0.795, "指标 ②：整链全对率", 14, GREEN, "bold", ha="center")
    txt(ax, 0.74, 0.745, "一条 B 链的所有子径迹是否全部判对（物理最相关）", 11, DARK, ha="center")
    for i, (y, ok) in enumerate([(0.70, 1), (0.665, 1), (0.63, 0), (0.595, 1)]):
        txt(ax, 0.58, y, f"径迹{i+1}", 11, GREY)
        txt(ax, 0.78, y, "✓" if ok else "×", 12, GREEN if ok else RED, "bold")
    txt(ax, 0.74, 0.575, "→ 有 × 就整链不算对 ⇒ 0%", 12.5, RED, "bold", ha="center")
    box(ax, 0.04, 0.30, 0.92, 0.20,
        "容易被忽略的细节：minIP 有一个‘弃权’选项 —— 对没通过它筛选的径迹直接给‘不关联’（记 −1）；\n"
        "我们的模型目前只会 argmax，永远不会弃权。这类‘没有真值 PV’的径迹占本题样本的 1.9%，"
        "所以在对比时我们也会把这一小部分单独说明。",
        fc="#FFF8E7", ec=ORANGE, size=12.5, ha="left")
    txt(ax, 0.5, 0.20, "另外我们还会做两个‘倾向性’诊断：", 13, NAVY, "bold", ha="center")
    txt(ax, 0.5, 0.145, "① PV 占用分布（谁把径迹推到哪里）   ② 错配方向（错了往哪边错）", 12.5, DARK, ha="center")
    txt(ax, 0.5, 0.085, "这两个诊断能回答‘准确率之外，方法有没有系统性偏差’——下一节的结果就是它们。", 12, GREY, ha="center")
    save(fig, "slide_08_metrics",
         "讲法：评测需要标准答案，MC 模拟会给每条径迹标出真 PV。两个指标："
         "① 逐径迹正确率；② 整链全对率（一条 B 链所有子径迹都对才算对），后者和物理最相关。"
         "另外两个口径细节：minIP 可以对不过筛选的径迹‘弃权’（给 −1），我们的模型不会弃权，"
         "这类径迹占 1.9%，对比时会单独说明；我们还会额外看两个‘倾向性’诊断："
         "PV 占用分布和错配方向，用来检查有没有系统性偏差。")


# ============================================================ S9 结果:准确率
def s09():
    fig, ax = canvas("结果 ①：准确率 —— 健康时我们领先，数据被搞坏时我们反而输",
                     "三条数据集/模型组合，蓝色=我们，红=minIP，橙=minIP+链级一致性（后文详）")
    data = [("v601 / 0904 新数据训练", 83.5, 79.7, 80.3),
            ("v557 / 0702 七月数据", 95.7, 77.3, 83.3),
            ("v557 / 0904 重归一化（数据退化）", 78.4, 83.2, 74.3)]
    x = 0.10
    for lab, ours, mip, vote in data:
        box(ax, x - 0.015, 0.30, 0.30, 0.46, fc="white", ec=GREY)
        txt(ax, x + 0.135, 0.715, lab, 11.5, DARK, "bold", ha="center")
        for i, (v, c, nm) in enumerate([(ours, BLUE, "我们"), (mip, RED, "minIP"), (vote, ORANGE, "minIP+链一致性")]):
            bx = x + 0.015 + i * 0.095
            bar(ax, bx, 0.40, 0.075, 0.22, v, 100, c)
            txt(ax, bx + 0.037, 0.375, f"{v:.1f}", 10.5, c, "bold", ha="center", va="top")
            txt(ax, bx + 0.037, 0.61, nm, 9, c, ha="center", style="italic")
        txt(ax, x + 0.135, 0.335, "逐径迹正确率 %", 10, GREY, ha="center")
        x += 0.30
    box(ax, 0.04, 0.16, 0.92, 0.11,
        "只统计‘重建正确的链’(AllParticles) 时：七月 97.0 vs 79.1；0904 新数据 86.8 vs 81.2；退化的 arm C 84.4 vs 85.6（minIP+链一致性 88.1 反超）。",
        fc=LIGHT, ec=BLUE, size=12, ha="left")
    txt(ax, 0.5, 0.10, "注意第三组：同一批事件，仅仅因为输入特征被错误地归一化，我们就从领先变成落后 —— 这条线索在最后一节会变成诊断工具。",
        12, RED, ha="center")
    save(fig, "slide_09_accuracy",
         "讲法：先看最直白的指标。① 用新数据重新训练后的模型（v601）83.5% vs minIP 79.7%，我们略胜；"
         "② 七月数据上的成熟模型 95.7% vs 77.3%，我们大幅领先；③ 但这个模型跑在被错误归一化的数据上时，"
         "78.4% vs 83.2%，我们反而输——说明它的判定被数据搞坏了。"
         "只看重建正确的链（物理上最相关）也是同样的排序。")


# ============================================================ S10 结果:占用分布
def s10():
    fig, ax = canvas("结果 ②：占用分布 —— 谁把径迹‘推’到哪里",
                     "横轴=PV 索引；灰=真值，实心=我们，斜纹=minIP")
    pic(ax, FIG + "/pv_bias_occupancy.png", 0.02, 0.20, 0.96, 0.58)
    box(ax, 0.04, 0.045, 0.92, 0.125,
        "怎么读：三条柱一样高 = 与真值一致（灰=真值，实心=我们，斜纹=minIP）。\n"
        "左图：我们几乎与真值重合，minIP 明显偏低索引；右图（退化 arm C）：我们反而在 PV0 处高出真值 3.5pp，即‘往最前面的 PV 塌缩’。",
        fc=LIGHT, ec=BLUE, size=11, ha="left")
    save(fig, "slide_10_occupancy",
         "讲法：这张图回答‘谁把径迹推到哪里’。灰柱是标准答案（真值的占用分布：PV0 占 32%，依次递减）。"
         "我们的柱几乎和灰柱一样高；minIP 的斜纹柱在 PV0 明显偏低、在后面几个 PV 偏高——这就是前面说的"
         "‘把径迹推到更靠后的 PV’。最右边那张（数据被搞坏的模型）反过来：PV0 处高出真值 3.5 个百分点，"
         "说明它塌缩到了 PV0。这类占用分布是非常灵敏的健康度指标。")


# ============================================================ S11 结果:链一致性
def s11():
    fig, ax = canvas("结果 ③：错配方向与‘链一致性’ —— 物理上最要紧的一条",
                     "左：整链全对率与增益分解  右：随 PV 数目（堆积）的变化")
    pic(ax, FIG + "/pv_bias_chain.png", 0.02, 0.20, 0.96, 0.58)
    box(ax, 0.04, 0.045, 0.92, 0.125,
        "链内一致率（一条链的所有子径迹被指到同一个 PV）：真值 77.5% (0904) / 97.4% (七月)，我们 70.1 / 92.4，\n"
        "minIP 只有 59.1 / 58.2 —— 传统方法会把整条链系统性撕开，这是物理上最要紧的差别。",
        fc=LIGHT, ec=BLUE, size=11, ha="left")
    save(fig, "slide_11_chain",
         "讲法：左边这张图是物理上最要紧的：整条 B 链‘全对’的比例。七月：minIP 56% → 我们 90.6%，"
         "链级决策更是 94.6%（红色那根是 minIP 单独对而我们错的份额，很小）。"
         "中间的分块图说明差异来自哪里：蓝=我们单独对，红=minIP 单独对，灰=都错。"
         "右边是随 PV 数目（堆积）的变化。核心数字：链内一致率真值 77.5%/97.4%，我们 70.1%/92.4%，"
         "minIP 只有 59%/58% —— minIP 会把链撕开，这是它最本质的缺陷。")


# ============================================================ S12 公平对照
def s12():
    fig, ax = canvas("一个诚实的对照：把‘链级一致性’补给传统方法，差距还剩多少？",
                     "LHCb 经典流程里本来就有 samePV 约束（与信号候选同一个 PV），而我们的 minIP 基线只实现了逐径迹那一半")
    rows = [("七月 0702", 77.3, 83.3, 95.7), ("0904 新数据", 79.7, 80.3, 83.5),
            ("0904 退化 (arm C)", 83.2, 74.3, 78.4)]
    txt(ax, 0.06, 0.775, "逐径迹正确率 %", 12, NAVY, "bold")
    box(ax, 0.63, 0.745, 0.34, 0.075, "红 = minIP    橙 = minIP+链一致性    蓝 = 我们",
        fc="white", ec=GREY, size=10.5)
    for i, (nm, mip, vote, ours) in enumerate(rows):
        y = 0.675 - i * 0.075
        txt(ax, 0.06, y, nm, 12, DARK)
        for j, (v, c) in enumerate([(mip, RED), (vote, ORANGE), (ours, BLUE)]):
            x0 = 0.24 + j * 0.235
            ax.add_patch(Rectangle((x0, y - 0.017), (v - 60) / 40 * 0.16, 0.034, fc=c, ec="none", zorder=3))
            txt(ax, x0 + (v - 60) / 40 * 0.16 + 0.008, y, f"{v:.1f}", 10.5, c,
                "bold" if j == 2 else "normal", ha="left")
    box(ax, 0.04, 0.30, 0.92, 0.155,
        "把链级一致性补进 minIP 之后：七月 77.3 → 83.3（我们仍领先 12.4pp）；0904 新数据 79.7 → 80.3（我们领先 3.2pp）；\n"
        "退化情况 83.2 → 74.3（多数票在坏数据上反而帮倒忙，我们领先 4.1pp）。\n"
        "若只看‘重建正确的链’：七月 97.0 vs 83.2；0904 新数据 86.8 vs 84.7；退化情况 84.4 vs 88.1（这时 minIP+一致性反超我们）。",
        fc="#FFF8E7", ec=ORANGE, size=11, ha="left")
    txt(ax, 0.5, 0.245, "结论（诚实版）：我们的增益，主要来自‘链级全局一致性’，而不是逐径迹 IP 判别本身更强。",
        13.5, NAVY, "bold", ha="center")
    box(ax, 0.04, 0.06, 0.92, 0.155,
        "这也正是我们相对传统流程的定位：传统流程要人工加上 samePV 这类约束，而我们在同一个网络里端到端学出来。\n"
        "负号的来源也值得记住：把 minIP 的逐径迹判定做链内多数票时，‘一致’会变成‘一起错到同一个 PV’，数据退化时反而更糟\n"
        "—— 一致性必须建立在正确的逐径迹判定之上。",
        fc=LIGHT, ec=BLUE, size=11, ha="left")
    save(fig, "slide_12_fair",
         "讲法：这一页是对我们自己最严格的检验。LHCb 的经典流程里本来就包含 samePV 约束（把信号候选的子径迹"
         "都放到同一个 PV），而我们的 minIP 基线只做了逐径迹那一半。把这一半补回去（链内多数票）之后："
         "七月从 77.3 提到 83.3，我们仍领先 12.4pp；0904 新数据提到 80.3，我们领先 3.2pp；"
         "退化的情况反而从 83.2 掉到 74.3，因为在坏数据上‘一致’会变成‘一起错’。"
         "所以诚实的结论是：我们的优势主要来自链级全局一致性，而不是逐径迹判别更强。")


# ============================================================ S13 诊断工具
def s13():
    fig, ax = canvas("附带发现：这个方法可以当‘模型/数据健康度’的体检工具",
                     "同一个模型、同一批事件，只把输入特征做错归一化，指纹立刻现形")
    box(ax, 0.04, 0.52, 0.44, 0.30, fc="#FDECEA", ec=RED, lw=1.6)
    txt(ax, 0.26, 0.785, "病态指纹：PV0 塌缩", 14, RED, "bold", ha="center")
    for i, s in enumerate(["PV0 占用 35.5%，比真值高 3.5pp",
                           "平均 PV 索引 1.45，比真值偏低 0.15",
                           "错配 65% 是‘往更靠前的 PV’（方向反了）",
                           "链内一致率虚高（72% vs 真值 69%）",
                           "但整链全对率最低（61.7% < minIP 67.3%）"]):
        txt(ax, 0.06, 0.735 - i * 0.043, "· " + s, 11.5, DARK)
    box(ax, 0.52, 0.52, 0.44, 0.30, fc="#EAF5EA", ec=GREEN, lw=1.6)
    txt(ax, 0.74, 0.785, "健康模型的指纹", 14, GREEN, "bold", ha="center")
    for i, s in enumerate(["占用分布与真值几乎重合（差 < 0.1pp）",
                           "平均索引偏差 < 0.01",
                           "错配方向近似对称（向低索引 ~50%）",
                           "链内一致率显著高于 minIP（+11 ~ +34pp）",
                           "链级决策让整链全对率再涨 5 ~ 10pp"]):
        txt(ax, 0.54, 0.735 - i * 0.043, "· " + s, 11.5, DARK)
    txt(ax, 0.5, 0.455, "→ 用法：把‘PV0 占用偏差 / 平均索引偏移 / 链内一致率’三个数做成一张体检表，任何新数据/新模型跑一遍就能看出有没有问题。",
        12.5, NAVY, "bold", ha="center")
    box(ax, 0.04, 0.20, 0.92, 0.20,
        "两个可以马上做的小改进：\n"
        "① 给 PV 判定加一个‘弃权类’（无关联），对齐传统方法的能力，同时给下游提供‘不确定’信号；\n"
        "② 报告指标时默认给出‘链级决策’的整链全对率（它对物理的增益远大于逐径迹指标的提升）。",
        fc=LIGHT, ec=NAVY, size=12.5, ha="left")
    txt(ax, 0.5, 0.10, "（这些数字全部来自真实 eval 输出，可复现：analyze_pv_bias.py + report_figs/pv_bias_*.csv）",
        11, GREY, ha="center")
    save(fig, "slide_13_diagnostic",
         "讲法：一个额外收获。同一批事件、同一个模型，只把输入特征的归一化做错，"
         "我们就从各项指标都健康变成 PV0 塌缩：PV0 占用虚高 3.5 个百分点、平均索引偏低、错配方向反转、"
         "链内一致率虚高但全链全对率最低。也就是说这些‘倾向性’指标可以当体检表用。"
         "建议两个小改进：给 PV 判定加弃权类；报告时默认给出链级决策的全链全对率。")


# ============================================================ S14 小结
def s14():
    fig, ax = canvas("三句话总结", None)
    box(ax, 0.05, 0.615, 0.90, 0.185,
        "① 任务：给每条径迹判断它来自哪个 PV。\n"
        "    物理硬约束 —— 同一条 B 衰变链的所有子径迹必须属于同一个 PV。",
        fc=LIGHT, ec=BLUE, size=13.5, ha="left")
    box(ax, 0.05, 0.395, 0.90, 0.185,
        "② 传统 minIP：每条径迹独立取‘离自己最近’的 PV，两个先天倾向 ——\n"
        "    把整条链撕开（链内一致率仅 58%）、把径迹系统性推向更靠后的 PV（平均索引 +0.16）。",
        fc="#FFF8E7", ec=ORANGE, size=12.5, ha="left")
    box(ax, 0.05, 0.175, 0.90, 0.185,
        "③ 我们：在‘径迹 × PV’整张图上推理 + 链级求和决策 —— 占用无偏、链内一致率 92%、整链全对率 56% → 95%。\n"
        "    但把同一条‘链级一致性’约束补给 minIP 后，它能追上大部分差距 ⇒ 我们的核心优势就是全局一致性。",
        fc="#EAF5EA", ec=GREEN, size=12.5, ha="left")
    txt(ax, 0.5, 0.105, "下一步：给 PV 判定加弃权类；把‘占用偏差 / 索引偏移 / 链一致率’做成常规体检项。",
        13, NAVY, "bold", ha="center")
    save(fig, "slide_14_summary",
         "收尾三句话：① 任务是给每条径迹找它的 PV，物理硬约束是同一条 B 链共用同一个 PV；"
         "② 传统 minIP 逐条独立判断，先天会把链撕开、并把径迹推向更靠后的 PV；"
         "③ 我们在整张图上推理并加链级一致决策，实测占用无偏、链内一致率和整链全对率大幅提升。"
         "诚实地讲，把同一条约束补给 minIP 后它能追上大部分差距——所以我们的卖点是‘全局一致性’。"
         "下一步：加弃权类，并把三个偏置指标做成常规体检项。")



# ============================================================ S13b 物理量依赖
def s13b():
    fig, ax = canvas("核心分析 ①：错配率与物理特征量的关系（逐径迹，300 事件/数据集）",
                     "实心 = 我们，虚线 = minIP；误差棒 = √(p(1−p)/n)　（原图见 report_figs/pv_physics_dependence.png）")
    pic(ax, FIG + "/pv_physics_dependence.png", 0.157, 0.16, 0.686, 0.662)
    box(ax, 0.04, 0.035, 0.92, 0.105,
        "★ 左下角是核心：真值 PV 恰好就是‘最近 IP 的 PV’的径迹占 88–90%，minIP 在这部分几乎白得正确；"
        "整体差距几乎全部集中在剩下的 10–12%（真值 PV ≠ 最近 IP 的 PV）。",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_13b_physics",
         "讲法：这一页把错配率拆到物理量上看。左上 pT、右上 η、左侧 ghost：都是低 pT、大 η（前向）、"
         "高 ghost 概率（难以重建）的径迹更容易被错配，两条方法同趋势，但 minIP 整体更差。"
         "最关键的是左下角这张‘战场分解’：真值 PV 恰好就是最近 IP 的那个 PV 的径迹占 88-90%，"
         "这部分 minIP 几乎不会错；整体差距几乎全部集中在剩下的 10-12%。"
         "所以我们后面所有的比较都应该盯住这个硬子集。")


# ============================================================ S13c 方向性
def s13c():
    fig, ax = canvas("核心分析 ②：错配的‘方向’与‘错得多远’",
                     "Δz = z(判定 PV) − z(真值 PV)，已反归一化到 cm；只统计错配径迹　（原图见 report_figs/pv_direction_delta.png）")
    pic(ax, FIG + "/pv_direction_delta.png", 0.157, 0.16, 0.686, 0.662)
    box(ax, 0.04, 0.035, 0.92, 0.105,
        "minIP 错配时 70% 以上朝 z 更小（更靠前）的方向错，中位 |Δz| = 3.4–9.3 cm；"
        "我们错配时方向近似对称，中位 |Δz| = 2.8–6.4 cm ⇒ 即使错，也错在相邻的 PV，对飞行距离的破坏更小。",
        fc=LIGHT, ec=BLUE, size=11.5, ha="left")
    save(fig, "slide_13c_direction",
         "讲法：这一页回答‘错是怎么错的’。左上：把每个错配径迹的 Δz=判定PV的z − 真值PV的z 画成分布。"
         "minIP 的分布明显偏向负的一侧（70% 以上朝 z 更小、更靠前的 PV 错），是系统性的方向偏置；"
         "我们的分布在 0 附近近似对称，没有系统性方向。左上角那张图还显示 minIP 有明显更长的尾巴。"
         "中上：错配时 |Δz| 的中位数——我们 2.8/5.8/6.4 cm，minIP 3.4/9.3/6.3 cm，"
         "尤其七月模型上错得明显更近。所以即使判错，我们错在‘相邻的 PV’，对飞行距离的破坏更小。"
         "右上：在 minIP 必错的径迹上，我们从 86%（七月）掉到 33-37%（数据变差时）的救援率，"
         "并随 IP 间隙变大而下降。")


def build_pptx():
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for png, notes in SLIDES:
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(png, 0, 0, width=prs.slide_width, height=prs.slide_height)
        s.notes_slide.notes_text_frame.text = notes
    out = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/PV_association_explained.pptx"
    prs.save(out)
    print("写出", out, f"({len(SLIDES)} 页)")


if __name__ == "__main__":
    for f in (s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s13b, s13c, s14):
        f()
    build_pptx()
