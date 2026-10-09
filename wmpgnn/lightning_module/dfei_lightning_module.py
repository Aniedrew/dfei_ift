import pytorch_lightning as L

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.reconstruction.topk_selection import CandidateScorer, build_chain_samples
from wmpgnn.performance.plotter import *
from wmpgnn.performance.reco_accuracy import acc_four_class, obtain_reco_accuracy, acc_pv_asso
from wmpgnn.performance.plot_results import plot_sig_pv_missasso, plot_sig_b_system_pv_missasso


def focal_bce_with_logits(logits, targets, pos_weight=None, gamma=2.0):
    """Focal BCE (Lin et al., 2017): 降低易分样本权重, 聚焦难分样本。

    用于剪枝头 (少数类 recall 是瓶颈): 背景负样本易分、信号正样本难分,
    故 focal 把梯度集中到"低置信"的信号径迹/链边上。gamma=0 退化为普通 BCE。
    """
    if gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    # pos_weight 由图外统计生成(在 CPU 上), 而 nn.BCEWithLogitsLoss 内部会搬设备,
    # 裸 functional 调用不会 -> 必须显式对齐, 否则报 "found at least two devices".
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    return (ce * (1.0 - pt).pow(gamma)).mean()


def truth_chain_labels(y, edge_index, n_nodes):
    """每个节点的 truth 链 id (-1 = 不属于任何 chain)。

    链 = 由 truth 非背景边 (y>0, 即 LCAG class1/2/3) 连成的连通分量。
    图按事件 batch 拼接但事件间无边, 故直接对全图做并查集即可。
    """
    y_bin = (y > 0)
    y_bin = y_bin.squeeze(-1) if y_bin.dim() > 1 else y_bin
    a, b = edge_index[0][y_bin], edge_index[1][y_bin]
    lab = torch.full((n_nodes,), -1, dtype=torch.long, device=edge_index.device)
    if a.numel() == 0:
        return lab
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(a.numel()):
        ra, rb = find(int(a[i])), find(int(b[i]))
        if ra != rb:
            parent[ra] = rb
    touched = torch.zeros(n_nodes, dtype=torch.bool, device=edge_index.device)
    touched[a] = True
    touched[b] = True
    roots = {find(int(i)) for i in touched.nonzero().flatten().tolist()}
    remap = {r: k for k, r in enumerate(sorted(roots))}
    idx = torch.arange(n_nodes, device=edge_index.device)
    lab = torch.tensor([remap[find(int(i))] if bool(touched[i]) else -1 for i in idx.tolist()],
                       dtype=torch.long, device=edge_index.device)
    return lab


def chain_min_scores(scores, chain_lab):
    """按链取"最弱环节"得分 (per-chain min)。scores/chain_lab 均已过滤掉 lab<0。"""
    scores = scores.reshape(-1)
    chain_lab = chain_lab.reshape(-1)
    uniq, inv = torch.unique(chain_lab, return_inverse=True)
    m = torch.full((uniq.numel(),), 1e4, device=scores.device, dtype=scores.dtype)
    return m.scatter_reduce(0, inv.reshape(-1), scores, reduce="amin")


def chain_recall_loss(node_logits, edge_logits, y, edge_index, n_nodes, thr=0.5, tau=0.1):
    """链级 min-pooling recall 损失。

    指标是"整条链"级别的: 链内**最弱**的点/边掉到阈值以下 -> 整条链被剪掉 (记 0)。
    逐元素 BCE 优化的是平均正确率, 与此目标不一致。本项直接惩罚
    "每条真值链中最低分的环节":
        L = -log sigmoid( (min_{v in chain} s_v - thr) / tau )
    即把每条链的最弱环节顶到阈值以上。返回 (节点项, 边项), 均为标量或 None。
    """
    n_lab = truth_chain_labels(y, edge_index, n_nodes)
    l_node = None
    vn = n_lab >= 0
    if vn.any():
        s = torch.sigmoid(node_logits.squeeze(-1))[vn]
        mn = chain_min_scores(s, n_lab[vn])
        l_node = -torch.log(torch.sigmoid((mn - thr) / tau) + 1e-6).mean()
    l_edge = None
    y_bin = (y > 0)
    y_bin = y_bin.squeeze(-1) if y_bin.dim() > 1 else y_bin
    if y_bin.any() and edge_logits is not None:
        el = edge_logits
        if el.dim() > 1:
            el = el.squeeze(-1) if el.size(-1) == 1 else None   # 仅支持逐边二分类头 (非 4 类 LCA 头)
        if el is not None:
            ea = edge_index[0][y_bin]                      # 每条真值边归到其所属链 (取起点节点的链 id)
            e_lab = n_lab[ea]
            ve = (e_lab >= 0)
            if ve.any():
                es = torch.sigmoid(el[y_bin])[ve]
                me = chain_min_scores(es, e_lab[ve])
                l_edge = -torch.log(torch.sigmoid((me - thr) / tau) + 1e-6).mean()
    return l_node, l_edge



def edge_rank_loss(logits, y, edge_index, node_ev, n_neg=64, margin=1.0, mode="hinge"):
    """边头 pairwise ranking 损失 (对准 precision/AUC, 而非被 pos_weight 主导的 BCE)。

    依据 (2026-09-22 诊断, v601/0904):
      - 在**真正的决策 population** (两端都过点剪枝 thr0.9) 上, 边头 AUC 0.868, 但 thr0.9 时
        保留了 ~90% 的边、precision 仅 0.295 (≈ 基频) -> 阈值不动时决策退化为"全留";
        要走到 R90 工作点 (thr≈0.996) 才有 precision 0.73。
      - 该 population 的假阳性 76% 至少一端是背景径迹 (点剪枝漏出来的), 24% 是跨链 ——
        两类缺的都是同一件事: **"信号样"边之间的排序能力**。
      - 现有 BCE 的 pos_weight≈700 (完全图正类率 0.14%) 把梯度几乎全投在"把真边推高",
        对负边之间的相对次序约束很弱 -> 排序上不去。
    本项直接优化排序: 每事件取最难的 n_neg 条负边, 与全部真边两两配对, 罚
        relu(margin + s_neg - s_pos)。
    只学相对次序 => 绝对阈值会漂移, 评测阈值必须按 ROC 重扫 (本项目已有该流程)。

    已排除的同族尝试: 结构先验 support(i,j)=max_k min(s_ik,s_kj) 正则 (罚"高分低支持"的负边)
    在 v601/0904 预检 AUC=0.53 (纯噪声) —— 完全图上对 n 个候选取 max 会把信号抹平。

    mode (配置键 edge_rank_mode):
      - "hinge" (默认): 逐正边-负边配对的 relu(margin + s_neg - s_pos) 均值, 与旧行为逐位一致。
      - "infonce": listwise。每事件把全部正边 {s_pos} 与最难 n_neg 条负边 {s_neg} 并为候选集,
        最大化正类在该候选集中的概率质量 (InfoNCE 形式, 数值稳定用 logsumexp):
            L_evt = -(logsumexp(s_pos) - logsumexp(concat([s_pos, s_neg])))
        对事件取均值。正类率极低 (0.135%) 时 listwise 直接约束"顶部难负例"的相对次序。
    """
    s = logits.squeeze(-1)
    yb = (y.squeeze(-1) > 0.5)
    ev = node_ev[edge_index[0]]                           # 每条边所属事件的 id
    tot, cnt = None, 0
    for e in range(int(ev.max().item()) + 1 if ev.numel() else 0):
        m = ev == e
        pos, neg = s[m & yb], s[m & ~yb]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        if neg.numel() > n_neg:
            neg = neg.topk(n_neg).values                     # 只跟最难的负边比
        if mode == "infonce":
            cand = torch.cat([pos, neg])                     # logsumexp 自带 max 平移, 数值稳定
            v = -(torch.logsumexp(pos, 0) - torch.logsumexp(cand, 0))
        else:
            v = F.relu(margin + (neg.unsqueeze(0) - pos.unsqueeze(1))).mean()
        tot = v if tot is None else tot + v
        cnt += 1
    return (tot / cnt) if cnt else None


def edge_dz_pair_sign(edge_index, prob, salt=0):
    """给每条边一个"按无向对一致"的 ±1 符号 (同一对的两个方向符号相同)。

    用途: 随机化 delta_z0 的方向 (切断 0702 那种"边枚举顺序=真值相关"的泄露)。
    实现: 稳定性哈希 (min,max) 端点 + 每步随机 salt -> 同一对必得同一符号 (与方向无关),
    prob<1 时只按概率翻转一部分对。返回 [E] float32 的 ±1。
    """
    a, b = edge_index[0].long(), edge_index[1].long()
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    h = (lo * 1000003 + hi * 9176 + int(salt) * 2654435761) % 2147483647
    sgn = torch.where(h.remainder(2) == 0, 1.0, -1.0)
    if prob < 1.0:
        keep = torch.rand(lo.shape, device=edge_index.device) < prob
        sgn = torch.where(keep, sgn, torch.ones_like(sgn))
    return sgn


def track_minip(batch):
    """每条 track 的 minIP (用 tr-pv 边特征 log_minIP 逐 track 取最小)。

    归一化是单调仿射 -> 序不变, 故直接在归一化空间取 min 即可。
    节点无 tr-pv 边时留 +inf (不翻转其方向)。
    """
    tt_pv = batch[("tracks", "to", "pvs")]
    v = tt_pv.edges.reshape(-1)
    n = batch["tracks"].x.shape[0]
    out = torch.full((n,), float("inf"), device=v.device, dtype=v.dtype)
    return out.scatter_reduce(0, tt_pv.edge_index[0].long(), v, reduce="amin", include_self=True)


# [2026-10-09 FIX] derived_pair_sym 的列数 = 3 × PSYM_NODE_COLS, **必须与会计式一致**。
#   原实现用 `3 × X.shape[1]`, 而注释与会计式都假设 X 宽 8 (-> 24 维)。训练路径里
#   batch['tracks'].x 实测为 13 列 -> 39 维 -> 边派生总列数 48, 与配置 extra_edge_dim=33 冲突,
#   结果是 v645/v647 在 2500+ 次尝试里**每次都在 epoch 0 崩**
#   (RuntimeError: mat1 and mat2 shapes cannot be multiplied (34058x48 and 33x16)),
#   从未真正训练过。这里把来源固定为**前 8 个节点列**(即注释所指的"原始节点特征"),
#   让训练/评估/配置三处宽度一致。
PSYM_NODE_COLS = 8


def derive_pruning_features(batch, nc, ns, use_triangle=False, use_vertex=False,
                            use_vertex_geom=False, use_pair_sym=False, use_comp=False,
                            sgn_tie_zero=False):
    """从**原始**节点/边特征现算剪枝 MLP 的派生输入 (不重产数据)。

    动机: 剪枝 MLP 的输入只有 8 维节点特征 / 5 维边特征, 缺的正是
      (a) IP 类信息 (只有 log_minIP 经 tr-pv 边间接进来),
      (b) 非局部的隔离度/事件上下文 (MLP 自己算不出来),
      (c) 物理不变量 (m(ππ)/ΔR/电荷和/规范化 pT-IP 不对称/沿合动量的纵向分离),
      (d) 三角传递性 (i~k~j)。

    节点 7 维: [pT, |p|, minIP, pT-rank, isolation, n_tracks/100, npvs/10]
    边   7 维: [ΔR, m(ππ), |q_i+q_j|, ΔpT_canon, ΔIP_canon, iso_pair, Δz_proj_canon]
              (+2 若 use_triangle: [三角传递 support, 局域亲和度行内 rank])
              (+3 若 use_vertex:   [zcpa/100, flight/100, collinearity], 见下方 derived_vertex)
    约定: "上游" = minIP 更小 (更贴 PV), 与 edge_dz_ip_canon 的定向一致。
    nc/ns: 原特征名 -> 归一化 center/scale (用于反归一化, 只求量级合理)。
    """
    tt = ("tracks", "to", "tracks")
    X = batch["tracks"].x
    dev = X.device
    def raw(name, col):
        return X[:, col] * float(ns.get(name, 1.0)) + float(nc.get(name, 0.0))
    px, py, pz = raw("px_reco", 0), raw("py_reco", 1), raw("pz_reco", 2)
    xp, yp, zp = raw("xProd_reco", 3), raw("yProd_reco", 4), raw("zProd_reco", 5)
    q = raw("Charge", 6)
    pT = torch.sqrt(px ** 2 + py ** 2 + 1e-6)
    pmod = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-6)
    u = torch.stack([px, py, pz], 1) / pmod.unsqueeze(1)
    # 逐 track 的 minIP (归一化 log_minIP 的最小值; 单调 -> 序不变)
    tpv = batch[("tracks", "to", "pvs")]
    ip = torch.full((X.shape[0],), float("inf"), device=dev).scatter_reduce(
        0, tpv.edge_index[0].long(), tpv.edges.reshape(-1), reduce="amin", include_self=True)
    fin = torch.isfinite(ip)
    ip = torch.where(fin, ip, ip[fin].max() if bool(fin.any()) else torch.zeros((), device=dev))
    # ==== [2026-09-26] derived_vertex: 每条 track 关联 PV 的 z (minIP 最小的 PV) ====
    # tr-pv 边特征 log_minIP 单调 -> argmin 在归一化/物理空间一致; PV 的 z 用 zPV_reco 反归一化。
    # 无 tr-pv 关联的 track 回退到自身生产顶点 z (flight 仍有限; 逐 track 局部量, 不跨事件)。
    _snd, _rcv = tpv.edge_index[0].long(), tpv.edge_index[1].long()
    _ipv = tpv.edges.reshape(-1)
    _bestip = torch.full((X.shape[0],), float("inf"), device=dev, dtype=_ipv.dtype).scatter_reduce(
        0, _snd, _ipv, reduce="amin", include_self=True)
    _is_best = (_ipv == _bestip[_snd])
    _zpv = batch["pvs"].x[:, 2] * float(ns.get("zPV_reco", 1.0)) + float(nc.get("zPV_reco", 0.0))
    z_pv_assoc = zp.clone()
    if bool(_is_best.any()):
        z_pv_assoc[_snd[_is_best]] = _zpv[_rcv[_is_best]]
    nb = batch["tracks"].batch if "batch" in batch["tracks"] else torch.zeros(
        X.shape[0], dtype=torch.long, device=dev)
    n_ev = int(nb.max()) + 1
    ntr = torch.bincount(nb, minlength=n_ev).float()
    npv = torch.bincount(batch["pvs"].batch, minlength=n_ev).float()
    rank = torch.zeros_like(pT)
    iso = torch.zeros_like(pT)
    ei = batch[tt].edge_index
    a, b = ei[0].long(), ei[1].long()
    n_e = ei.shape[1]
    sup = torch.zeros(n_e, device=dev, dtype=pT.dtype)
    rk_dR = torch.zeros(n_e, device=dev, dtype=pT.dtype)
    rk_aff = torch.zeros(n_e, device=dev, dtype=pT.dtype)
    cos = (u[a] * u[b]).sum(-1).clamp(-1, 1)
    dR = torch.arccos(cos)
    # 注意: LHCb 径迹都在束流附近 (θ≲0.3 rad), 随机两径迹的 ΔR 中位数只有 ~0.1 rad,
    # 所以"锥隔离"必须用很小的锥角; 另外准备了**尺度无关**的 dR 事件内分位 (rank)。
    for e in range(n_ev):                       # 逐事件稠密 (n≲200, 批内事件数 ≲200 -> 可接受)
        mn = nb == e
        n_n = int(mn.sum())
        if n_n < 2:
            continue
        cs = u[mn] @ u[mn].t()
        eye = torch.eye(n_n, dtype=torch.bool, device=dev)
        pe = pT[mn]
        iso[mn] = ((cs > 0.99875) & ~eye).float().mul(pe.unsqueeze(0)).sum(1) / pe.sum().clamp(min=1e-6)
        order = torch.argsort(-pe)
        r = torch.empty(n_n, device=dev, dtype=pT.dtype)
        r[order] = torch.arange(n_n, device=dev, dtype=pT.dtype)
        rank[mn] = r / max(1, n_n - 1)
        me = (nb[a] == e)
        if int(me.sum()) < 2:
            continue
        # 事件内 ΔR 分位: 有多大比例的(同事件)候选对比这一对更不共线 (0=最共线)
        rk_dR[me] = (dR[me].unsqueeze(0) > dR[me].unsqueeze(1)).float().mean(1)
        if not use_triangle:
            continue
        loc = torch.zeros(X.shape[0], dtype=torch.long, device=dev)   # 全局节点 id -> 事件内局部 id
        loc[mn.nonzero(as_tuple=True)[0]] = torch.arange(n_n, device=dev)
        la, lb = loc[a[me]], loc[b[me]]
        # 局域亲和度: 两端生产顶点 3D 距离越小越像同一顶点 -> 用 exp(-d/L)
        d3 = torch.sqrt((xp[a[me]] - xp[b[me]]) ** 2 + (yp[a[me]] - yp[b[me]]) ** 2
                        + (zp[a[me]] - zp[b[me]]) ** 2 + 1e-6)
        aff = torch.exp(-d3 / 50.0)                            # L=50 (与原特征同量纲)
        S = torch.zeros(n_n, n_n, device=dev, dtype=pT.dtype)
        S[la, lb] = aff
        S = 0.5 * (S + S.t())                                  # 对称化
        S2 = S @ S
        row = S.sum(1).clamp(min=1e-6)
        # [2026-09-30 FIX] 对称化: 原式只除 row[la]、且只在 S[la] 这一行内比较, 结果**依赖边方向**
        #   (交换 (i,j) 得到不同的值, 实测 max|Δ|=8.1e-2)。tt 边在物理上是无向的, 这与 0702
        #   delta_z0 那类"顺序信息"同形 —— 必须消掉。S 已对称化 => S2 对称, 故下面两式都对称。
        sup[me] = 0.5 * (S2[la, lb] / row[la] + S2[lb, la] / row[lb])
        # 行内 rank: 该边亲和度在同起点所有边中的分位 (越高越"排他"); 两端行内分位取平均 -> 对称
        rk_aff[me] = 0.5 * ((S[la] > aff.unsqueeze(1)).float().mean(1)
                            + (S[lb] > aff.unsqueeze(1)).float().mean(1))
    node_der = torch.stack([pT / 1000., pmod / 1000., ip / 10., rank, iso,
                            ntr[nb] / 100., npv[nb] / 10.], -1)
    mp = 0.13957
    ea = torch.sqrt(pmod[a] ** 2 + mp ** 2)
    eb = torch.sqrt(pmod[b] ** 2 + mp ** 2)
    m2 = (ea + eb) ** 2 - ((px[a] + px[b]) ** 2 + (py[a] + py[b]) ** 2 + (pz[a] + pz[b]) ** 2)
    mpipi = torch.sqrt(m2.clamp(min=0))
    # [2026-10-08 FIX D5] ip 并列时的定向二义性。
    #   原式 `up = ip[a] <= ip[b]` 在 ip[a]==ip[b] 时把**两个方向都**判为"上游"(sgn=+1),
    #   而 pT[a]-pT[b] 与 P·(A_a-A_b) 交换端点会变号 -> 列3 dpT 与列6 dzp 随边方向翻符号
    #   (与 0702 delta_z0 同类的"顺序信息")。并列并不罕见: 无 tr-pv 关联的径迹被兜底成同一 ip,
    #   实测占 tt 边的 6.55%。
    #   sgn_tie_zero=True 时并列取 sgn=0: **非并列边数值完全不变**, 并列边彻底对称
    #   (那里"谁更上游"本就无定义)。这是**特征定义变更** -> 用配置开关控制, 默认 False
    #   保持历史口径 (与已训练版本可比)。
    up = ip[a] <= ip[b]                                        # a 更靠上游 (含并列)
    if sgn_tie_zero:
        sgn = torch.where(ip[a] < ip[b], torch.ones_like(ip[a]),
                          torch.where(ip[a] > ip[b], -torch.ones_like(ip[a]),
                                      torch.zeros_like(ip[a])))
    else:
        sgn = torch.where(up, 1.0, -1.0)
    dpT = sgn * (pT[a] - pT[b])                                # 规范化: 下游 − 上游
    dIP = torch.where(up, ip[b] - ip[a], ip[a] - ip[b])        # 同样规范化为非负量级
    P = torch.stack([px[a] + px[b], py[a] + py[b], pz[a] + pz[b]], 1)
    dR3 = torch.stack([xp[a] - xp[b], yp[a] - yp[b], zp[a] - zp[b]], 1)
    dzp = (P * dR3).sum(-1) / P.norm(dim=1).clamp(min=1e-6)    # 沿合动量的纵向分离
    dzp = sgn * dzp                                            # 规范化: 下游 − 上游
    cols = [dR, mpipi / 1000., (q[a] + q[b]).abs(), dpT / 1000., dIP / 10., rk_dR, dzp / 10.]
    if use_triangle:
        cols += [sup, rk_aff]
    if use_vertex:
        # ==== [2026-09-26] 次级顶点一致性 (边级 3 维, 配置键 derived_vertex) ====
        # 每条 tt 边两端径迹参数化为直线 P_t(τ) = A_t + τ·u_t (过生产顶点 A, 方向=动量单位向量)。
        # 求两条直线的公共垂足 (closest-approach): 单位方向 => a=c=1,
        #   cosθ=u_a·u_b, denom=1-cos²θ, τ_a=(cosθ·e-d)/denom, τ_b=(e-cosθ·d)/denom
        #   (d=u_a·w0, e=u_b·w0, w0=A_a-A_b); 平行 (denom→0) 时垂足病态 -> 显式掩码回退。
        # 三个量 (均逐边局部, 只用该边两端径迹及其 PV 关联 -> 天然按事件隔离, 不跨事件):
        #   1) zcpa: 公共垂足 (两垂足中点) 的 z (平行时回退到两生产顶点 z 的中点);
        #   2) flight: |zcpa - z_PV_assoc|, PV 取 minIP 更小(上游)那端的关联 PV (次级顶点飞行距离代理);
        #   3) collinearity: 横向接近度 d_perp/(|Δz_两端生产顶点|+1mm)。
        #      注: 用两端**生产顶点**的 z 分离作纵向尺度, 而非两垂足的 z 分离 dz_sep —— 后者对
        #      近共线径迹近乎退化 (本 batch 中位数 ~0.014mm) -> 比值病态、26% 样本饱和 (见报告实测)。
        A3 = torch.stack([xp[a], yp[a], zp[a]], 1)
        B3 = torch.stack([xp[b], yp[b], zp[b]], 1)
        ua, ub = u[a], u[b]
        w0 = A3 - B3
        cab = (ua * ub).sum(-1)
        dd = (ua * w0).sum(-1)
        ee = (ub * w0).sum(-1)
        denom = 1.0 - cab * cab
        par = denom.abs() < 1e-3                               # |cosθ|>~0.9995: 近平行, 垂足病态
        denom_s = torch.where(par, torch.ones_like(denom), denom)
        ta = ((cab * ee - dd) / denom_s).clamp(-1e4, 1e4)      # 数值保险: 防近奇异处巨型 τ
        tb = ((ee - cab * dd) / denom_s).clamp(-1e4, 1e4)
        F1 = A3 + ta.unsqueeze(1) * ua                         # 线 a 上的垂足
        F2 = B3 + tb.unsqueeze(1) * ub                         # 线 b 上的垂足
        zcpa = torch.where(par, 0.5 * (zp[a] + zp[b]), 0.5 * (F1[:, 2] + F2[:, 2]))
        # [2026-10-08 FIX D5/S4] 同上: 原式在 ip 并列时按方向取端点 -> flight 随边方向变。
        #   sgn_tie_zero=True 时并列取两端均值 (对称)。
        if sgn_tie_zero:
            z_assoc = torch.where(ip[a] < ip[b], z_pv_assoc[a],
                                  torch.where(ip[a] > ip[b], z_pv_assoc[b],
                                              0.5 * (z_pv_assoc[a] + z_pv_assoc[b])))
        else:
            z_assoc = torch.where(ip[a] <= ip[b], z_pv_assoc[a], z_pv_assoc[b])
        flight = (zcpa - z_assoc).abs()
        # 两直线最近距离: 非平行用两垂足间距; 平行(垂足不唯一)用 w0 的垂直分量 (良态)
        d_par = torch.linalg.norm(w0 - dd.unsqueeze(1) * ua, dim=1)
        d_perp = torch.where(par, d_par, torch.linalg.norm(F1 - F2, dim=1))
        dz_pair = (zp[a] - zp[b]).abs()
        collin = d_perp / (dz_pair + 1.0)
        # 归一到合理量级 (与既有派生列一致) + 清 NaN/inf 并设硬上界
        for _v in (zcpa / 100.0, flight / 100.0, collin):
            cols.append(torch.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0))
    if use_vertex_geom:
        # ==== [2026-09-30] derived_vertex_geom: 顶点一致性**几何绝对量** (边级 3 维) ====
        # 依据: 特征上界分析(GBDT, docs/feature_ceiling_analysis.md)显示
        #   (a) 现有 5+9+3 维边特征对"真边 vs 跨链假边"的判别力 = AUC 0.50 (随机);
        #   (b) 但"两端生产顶点 3D 距离 |Δr|"单标量 = 0.704, 正确的线线最近距离(DOCA)
        #       在节点特征之上再 +0.06 AUC (=0.738)。
        # 旧 derived_vertex 只输出比值 d_perp/(|Δz|+1), 把 doca 的**绝对量级**除掉了 -> 信息被
        #   稀释(实测贡献≈0)。这里给绝对量: [doca/100, log(doca), |Δr|/100]。
        # 与上游 stored 的 log_DOCA_reco 无关: 那一列因 calculate_doca 的 t1 符号 bug 是噪声
        #   (与正确值 corr=0.019), 这里用标准最小二乘解**现算**, 不依赖重产数据。
        # 直线 P(τ)=A+τu (过生产顶点 A, 方向 = 单位动量); 最小化 |w + ta*ua - tb*ub|^2, w=A_a-A_b:
        #   ta=(c*e-d)/(1-c²), tb=(e-c*d)/(1-c²), c=ua·ub, d=ua·w, e=ub·w。
        A3v = torch.stack([xp[a], yp[a], zp[a]], 1)
        B3v = torch.stack([xp[b], yp[b], zp[b]], 1)
        uav, ubv = u[a], u[b]
        w0v = A3v - B3v
        cb = (uav * ubv).sum(-1)
        dv = (uav * w0v).sum(-1)
        ev = (ubv * w0v).sum(-1)
        dnv = 1.0 - cb * cb
        parv = dnv.abs() < 1e-3                                  # 近平行: 垂足病态
        dnsv = torch.where(parv, torch.ones_like(dnv), dnv)
        tav = ((cb * ev - dv) / dnsv).clamp(-1e4, 1e4)
        tbv = ((ev - cb * dv) / dnsv).clamp(-1e4, 1e4)
        F1v = A3v + tav.unsqueeze(1) * uav
        F2v = B3v + tbv.unsqueeze(1) * ubv
        d_parv = torch.linalg.norm(w0v - dv.unsqueeze(1) * uav, dim=1)   # 平行: w 的垂距(良态)
        doca = torch.where(parv, d_parv, torch.linalg.norm(F1v - F2v, dim=1))
        d_start = torch.linalg.norm(w0v, dim=1)                          # |Δ起点|
        for _v in (doca / 100.0, torch.log(doca + 1e-5), d_start / 100.0):
            cols.append(torch.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0))
    if use_pair_sym:
        # ==== [2026-09-30] derived_pair_sym: 把两端点原始节点特征**对称**放进边 (边级 24 维) ====
        # 依据: 判别信息几乎全在**节点**特征里(两端 6 个生产顶点量就 AUC 0.66, 两端 16 维节点
        #   特征 0.71), 而训练好的 GNN 在同一任务上只有 0.63 -> **模型没把节点信息用足**。
        #   这里给边头一条直接的端点通道, 用三个**交换 (i,j) 不变**的组合(逐 8 维):
        #     [x_i + x_j, |x_i - x_j|, x_i ⊙ x_j]
        # 对称性说明: 故意**不用** [x_i, x_j] 的顺序拼接 —— 那要靠"上游优先"之类的人为定向才能
        #   保持对称, 而 IP 相等时该定向退化为按下标排序, 会像 0702 的 delta_z0 那样把顺序信息
        #   带进来。用和/差/积则天然对称, 且信息量等价(和差可反解两端)。
        # 注意: 本函数的 cols 约定是**每项一个一维列** [n_e], 最后 torch.stack(cols, -1) 拼成 [n_e, n_dim]。
        # 所以 24 维必须先 concat 成 [n_e,24] 再**逐列拆开** append —— 冒烟把三种错法都抓到了:
        # ① `cols += [A,B,C]` 把三个 2D 张量当三列; ② 直接 append [n_e,24] 与其它 [n_e] 无法 stack;
        # ③ reshape(-1) 成 [n_e*24] 同样无法 stack。
        # [2026-10-09 FIX] 只用**前 PSYM_NODE_COLS 列**节点特征, 否则列数会随 X 实际宽度变化
        #   (训练路径 X=13 列 -> 39 维, 与会计式的 24 冲突, 见文件顶部 PSYM_NODE_COLS 说明)。
        if X.shape[1] < PSYM_NODE_COLS:
            raise RuntimeError(f"use_pair_sym 需要至少 {PSYM_NODE_COLS} 列节点特征, 实际 {X.shape[1]} 列")
        _X = X[:, :PSYM_NODE_COLS]
        _ps = torch.cat([_X[a] + _X[b], (_X[a] - _X[b]).abs(), _X[a] * _X[b]], dim=-1)
        for _i in range(_ps.shape[1]):
            cols.append(_ps[:, _i])
    if use_comp:
        # ==== [2026-10-01] derived_comp: 竞争 / 排他性上下文 (边级 6 维) ====
        # 依据: 大样本特征上界分析 (86690 条边, report_figs/feat_ceiling_big_probe.json):
        #   加上下文后 AP 0.7655 -> 0.7824 (+0.0169); permutation importance 里排**第 2** 的
        #   正是 deg_s (伙伴数/竞争度, +0.0235), 仅次于 doca (+0.1066)。
        # 物理含义: 一条径迹若有很多"看起来同样合理"的候选伙伴, 它就更不可信 -> 排他/归属竞争。
        # 注意: 这是"注意力/匹配"方向的**最小手工版本**, 用来当结构方案的下界对照。
        # 按端点分组即自动按事件隔离 (tt 边两端必同事件), 无跨事件泄漏; 只用输入量。
        d3c = torch.sqrt((xp[a] - xp[b]) ** 2 + (yp[a] - yp[b]) ** 2 + (zp[a] - zp[b]) ** 2 + 1e-6)
        aff = torch.exp(-d3c / 50.0)                     # 事件内两两"同起点"亲和度
        _, inv_s, cnt_s = torch.unique(a, return_inverse=True, return_counts=True)
        _, inv_t, cnt_t = torch.unique(b, return_inverse=True, return_counts=True)
        mx_s = torch.zeros(inv_s.max().item() + 1, device=dev, dtype=aff.dtype).scatter_reduce_(
            0, inv_s, aff, reduce="amax")
        mx_t = torch.zeros(inv_t.max().item() + 1, device=dev, dtype=aff.dtype).scatter_reduce_(
            0, inv_t, aff, reduce="amax")
        hi_s = torch.zeros_like(mx_s).scatter_add_(0, inv_s, (aff > 0.5).to(aff.dtype))
        hi_t = torch.zeros_like(mx_t).scatter_add_(0, inv_t, (aff > 0.5).to(aff.dtype))
        fdt = aff.dtype
        # ⚠️ 必须用**对称组合**: 若直接拼 [源端量, 目标端量], 交换 (i,j) 会让两者互换位置 ->
        # 特征向量随**边方向**变化 (冒烟实测 max|Δ|=0.99)。那与 0702 delta_z0 属同一类"顺序信息"
        # 问题, 必须消掉。{min, max} 与无序对一一对应, 信息等价且严格对称。
        r_s = aff / (mx_s[inv_s] + 1e-9)
        r_t = aff / (mx_t[inv_t] + 1e-9)
        d_s = hi_s[inv_s] / 10.0
        d_t = hi_t[inv_t] / 10.0
        c_s = cnt_s[inv_s].to(fdt) / 50.0
        c_t = cnt_t[inv_t].to(fdt) / 50.0
        for _v in (torch.minimum(r_s, r_t), torch.maximum(r_s, r_t),
                   torch.minimum(d_s, d_t), torch.maximum(d_s, d_t),
                   torch.minimum(c_s, c_t), torch.maximum(c_s, c_t)):
            cols.append(torch.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0))
    edge_der = torch.stack(cols, -1)
    return node_der, edge_der


def prune_auc(score, label):
    """[2026-09-29] 剪枝 AUC (秩和公式)。

    与 analyze_prune_loss.py 里的同名函数**逐位一致** —— val 指标必须与
    report_figs/bench_auc.csv 用同一把尺子, 否则"选 ckpt"和"验收"对不上。
    """
    o = np.argsort(score)
    r = np.empty(len(o), float)
    r[o] = np.arange(1, len(o) + 1)
    p = int(label.sum())
    n = len(label) - p
    return float((r[label == 1].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")


def prune_ap(score, label):
    """[2026-09-29] 剪枝 AP (average precision, 正类率极低时比 AUC 有区分度)。同 analyze_prune_loss.py。"""
    o = np.argsort(-score)
    l = label[o]
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(1, l.sum()))


class DFEILightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, configs, pos_weights):
        super().__init__()
        if "model" in configs["settings"]:
            self.version = configs["settings"]["model"]
        else:
            self.version = None
            self.save_hyperparameters({
                **configs,
                "pos_weights": make_loggable(pos_weights)
            })

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "":
            self.signal += "__" + configs["evaluate"]["over_write"]

        self.configs = configs["inference"]
        self.model = model
        self.use_pid = configs["DFEI"]["use_pid"]  # str holding what to do with pid information for DFEI
        # GN blocks 配置 (B2 温度退火参数读取; 兼容无该段的旧配置)
        self.configs_gn = configs.get("DFEI", {}).get("GNblocks", {})

        # ==== 候选衰变链选择 MLP (第5个监督头, 与主干联合训练) ====
        # 启用条件: config inference 段 selection_mlp 非空 ("builtin" 或 ckpt 路径)
        # scorer 只注册为 model 子模块 (model.chain_scorer), 随 state_dict 保存/加载;
        # 本模块通过只读 property self.chain_scorer 代理访问, 避免重复注册。
        self.chain_loss_weight = float(self.configs.get("chain_select_loss_weight", 10.0))
        self.chain_select_on = bool(self.configs.get("selection_mlp", "")) and self.configs.get("selection_mlp", "") != "None"
        if self.chain_select_on:
            node_dim = int(self.configs.get("selection_mlp_node_dim", 1 + 16))   # CERN use_pid: tracks.x=16 (encoder 输出)
            edge_dim = int(self.configs.get("selection_mlp_edge_dim", 1 + 4 + 5))  # 1 + 4 LCA + 5 物理边特征
            model.chain_scorer = CandidateScorer(node_dim, edge_dim)
            self.chain_criterion = nn.BCEWithLogitsLoss()

        # ==== 第6个监督头: 源检测 (Rumor Centrality 训练化) ====
        # 监督 GNN 预测每条 truth 链的"根节点" (B 介子候选): 节点级二分类
        # 标签 = truth 链内 rumor centrality 最大的节点 (truth_chain_roots, 无噪声)
        # 作用: 让主干显式学"衰变链的根-叶结构", 服务 LCA 结构分类;
        #       与推理侧 RC (找根) 对齐: 训练时学找根, 推理时 RC 用根。
        self.source_head_on = bool(self.configs.get("source_head", False))
        if self.source_head_on:
            # 输入 = 节点特征 (tracks.x, 16) + node_weight (1) -> [17]
            src_node_dim = int(self.configs.get("source_head_node_dim", 16))
            model.source_head = nn.Sequential(
                nn.Linear(src_node_dim + 1, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.source_criterion = nn.BCEWithLogitsLoss()
            self.source_loss_weight = float(self.configs.get("source_loss_weight", 5.0))

        # ==== 方案5: 链内 LCA 一致性辅助损失 (chain_lca_filter 训练化) ====
        # 鼓励 truth 链内边 (y>0) 的 LCA 预测"高置信" (被判类别 softmax 概率高),
        # 让模型主动产出物理自洽的链 (推理侧 chain_lca_filter 的判据前移到训练)。
        self.chain_lca_on = bool(self.configs.get("chain_lca_loss", False))
        self.chain_lca_loss_weight = float(self.configs.get("chain_lca_loss_weight", 2.0))
        self.chain_lca_margin = float(self.configs.get("chain_lca_margin", 0.3))
        # ==== 方案6: 链内边"类别正确"监督 (chain_lca 升级) ====
        # 在"高置信"基础上, 额外对链内边 (y>0, 仅 ~0.1% 的边) 施加 LCA 类别 CE,
        # 专门放大链内结构边 (class1/2/3) 的分类监督, 对抗 class0 绝对数量对主
        # LCA loss 的稀释; 链内边分类错误 (尤其 class2<->class1) 直接破坏链结构。
        self.chain_lca_ce = bool(self.configs.get("chain_lca_ce", False))
        self.chain_lca_ce_weight = float(self.configs.get("chain_lca_ce_weight", 1.0))

        # ==== 方案7b (v40): 可训练 PV 分簇 MLP 头 (pv_cluster_head) ====
        # 用户反馈: "cluster 本身就该是一个带训练的 MLP, 不然你要怎么 cluster; 温度退火也得改"。
        # 设计: pv_cluster_head = 独立分簇器, 输入 concat(tracks.x 原始8, pvs.x 3, trpv.edges 1)=12 维,
        #       输出 track->PV 归属 logit, BCE 监督 (与 pv_asso 同标签, 各自独立训练)。
        #       训练时用它的预测 + 温度退火 (Gumbel 噪声随 tau 退火: 高 tau 探索采样 ->
        #       低 tau 收敛 argmax) 给 track 分配 PV -> 切子图; 推理时 reconstruction 用
        #       同一头分簇 -> 训练/推理严格对齐。单 B 子图的处理路径 (GNN/loss/重建) 完全不变。
        self.pv_cluster_on = bool(self.configs.get("pv_cluster", False))
        if self.pv_cluster_on:
            pvclu_in = int(self.configs.get("pv_cluster_head_input_dim", 12))
            pvclu_hid = int(self.configs.get("pv_cluster_head_hidden", 64))
            model.pv_cluster_head = nn.Sequential(
                nn.Linear(pvclu_in, pvclu_hid), nn.ReLU(),
                nn.Linear(pvclu_hid, 1),
            )
            self.pv_cluster_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["pv_asso"])
            self.pv_cluster_loss_weight = float(self.configs.get("pv_cluster_loss_weight", 1.0))
            # 推理侧 track->PV 分配来源: "cluster_head"(推荐, 与本头对齐) | "pred"(pv_asso) | "true"
            self.pv_cluster_assign = self.configs.get("pv_cluster_assign", "cluster_head")
            # 温度退火 (与 B2 tau 同哲学): 1.0(探索,Gumbel采样) -> 0.1(收敛 argmax)
            # ⚠️ 相对本次训练起点计 (续训时 current_epoch 从 v38 ep101 起, 不能用绝对 epoch)
            self.pv_tau_start = float(self.configs.get("pv_cluster_tau_start", 1.0))
            self.pv_tau_end = float(self.configs.get("pv_cluster_tau_end", 0.1))
            self.pv_tau_epochs = max(int(self.configs.get("pv_cluster_tau_epochs", 100)), 1)
            self.pv_tau = self.pv_tau_start
            # ==== v41 修复②: truth->cluster 课程式过渡 ====
            # alpha: 1.0(用 truth PV 分簇, 稳定热启动) -> 0.0(用 cluster 头分簇)。
            # 解决 v40 随机 cluster 头 + tau=0 直接扰动 v38 权重导致发散的问题;
            # 相对本次训练起点计, curriculum 周期内每个 track 以概率 alpha 用 truth 分配。
            self.pv_cluster_curriculum_epochs = max(
                int(self.configs.get("pv_cluster_curriculum_epochs", 30)), 1)
            self.pv_curriculum_alpha = 1.0
            # ==== v41 修复③: 子图数量上限 (训练提速, 0=不限) ====
            # batch 从 8 个全图 -> ~50 子图导致 ~2h/epoch; 训练侧随机保留 cap 个子图,
            # val/test 不受限 (val 每 epoch 一次, 开销可忽略)。
            self.pv_cluster_max_subgraphs = int(self.configs.get("pv_cluster_max_subgraphs", 0))
            # 本次训练起点计数 (续训时 current_epoch 是 v38 的绝对 epoch, 退火/课程需相对本 run)
            self._pv_run_epoch = 0
            self._test_pv_cluster_logits = None  # test 时存全图 cluster logits, 供重建分簇

        # ==== 第7个监督头: 边级不变质量回归 (输出侧物理监督) ====
        # 用 π-π 不变质量 (log10 尺度) 作为物理真值, 监督 tt 边表征携带"动量-夹角"物理信息。
        # 目标 m_ij 由 batch 轨迹动量 (px,py,pz 归一化) 反归一化后计算, 无需额外标签;
        # 输入 = decoder 边表征 (latent_edges, 16维, model 在 op_trafo 前保存)。
        # 纯辅助监督: 不改主任务端到端目标, 梯度经 head 流回主干。
        self.mass_head_on = bool(self.configs.get("mass_head", False))
        if self.mass_head_on:
            mass_edge_dim = int(self.configs.get("mass_head_edge_dim", 16))
            model.edge_mass_head = nn.Sequential(
                nn.Linear(mass_edge_dim, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.mass_criterion = nn.SmoothL1Loss(beta=0.3)
            self.mass_loss_weight = float(self.configs.get("mass_loss_weight", 1.0))
            # 反归一化常数 (center, scale), 与 normalization_dict.pt (LHCb 通道) 逐位核对
            self._mass_norm = {
                "px": (torch.tensor(-4.1619), torch.tensor(470.8137)),
                "py": (torch.tensor(0.7674), torch.tensor(597.9097)),
                "pz": (torch.tensor(7117.4619), torch.tensor(10077.2412)),
            }
            self._m_pi = 139.570  # MeV

        # ==== 第8个监督头: 节点结构监督 (深度 + Rumor Centrality 回归) ====
        # 用户提议 (2026-08-26): source_head 只预测"根"(1 bit) 信息量低;
        # 升级为连续结构监督, 让主干学"节点在衰变树中的位置/层级":
        #   ① depth 主头: 节点到链根的拓扑距离 (归一化 [0,1])
        #   ② rc 辅头:   节点 logR 链内 min-max 归一化 [0,1]
        # 与 LCA 边分类形成全局一致性约束 (class1 边 depth 差1, class2 差0, class3 差>=2)。
        # 标签由 truth_chain_structure 计算 (纯 truth, 无模型噪声)。
        self.struct_head_on = bool(self.configs.get("struct_head", False))
        if self.struct_head_on:
            struct_in = int(self.configs.get("struct_head_node_dim", 16))
            model.node_struct_head = nn.Sequential(
                nn.Linear(struct_in + 1, 64), nn.ReLU(),
                nn.Linear(64, 2),   # [depth_pred, rc_pred]
            )
            self.struct_criterion = nn.SmoothL1Loss(beta=0.1)
            self.struct_head_weight = float(self.configs.get("struct_head_weight", 1.0))
            self.rc_head_weight = float(self.configs.get("rc_head_weight", 0.5))

        # ==== 第9个监督头: 节点级动量回归 (mom_head) ====
        # PhyIP 探针发现: encoder 的 graph_norm+ReLU 把输入动量信息打散,
        # 节点表征几乎不携带线性可读的动量 (线性/MLP 探针 R²≈0)。
        # 方案 A (用户确认): 像 mass head 对边表征那样, 监督节点表征输出
        # 归一化输入动量 [px_n, py_n, pz_n] (encoder 的直接输入), 强制节点
        # 表征线性携带运动学信息 -> 未来节点级物理任务 (如 trigger pT) 可用。
        self.mom_head_on = bool(self.configs.get("mom_head", False))
        if self.mom_head_on:
            mom_in = int(self.configs.get("mom_head_node_dim", 16))
            model.node_mom_head = nn.Sequential(
                nn.Linear(mom_in, 64), nn.ReLU(),
                nn.Linear(64, 3),   # [px_n, py_n, pz_n]
            )
            self.mom_criterion = nn.SmoothL1Loss(beta=0.3)
            self.mom_loss_weight = float(self.configs.get("mom_loss_weight", 1.0))
            # 哨兵检测用反归一化常数 (与 _mass_loss 一致)
            self._mom_norm = {
                "px": (torch.tensor(-4.1619), torch.tensor(470.8137)),
                "py": (torch.tensor(0.7674), torch.tensor(597.9097)),
                "pz": (torch.tensor(7117.4619), torch.tensor(10077.2412)),
            }

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        # 续训时希望使用的初始学习率 (从 settings.lr 读取; None 表示沿用 checkpoint 中的 lr)
        self.resume_lr = configs.get("settings", {}).get("lr", None)

        # ==== [2026-09-29] val 上的剪枝 AUC/AP (服务的把 ckpt 选择/早停切到"验收判据") ====
        # 动机: best ckpt 与 EarlyStopping 一直由 val_combined_loss 决定, 它是 8 个任务的
        # 加权和 (edge 项占 65-70%), 与"剪枝 AUC/AP"的排序不一致 -> 选 ckpt 用的是另一把
        # 尺子 (v614: min-val ckpt=13.81 vs 末轮=16.32, 白丢 2.5pp)。
        # 口径: 固定前 settings.validate_prune_events 个 val 事件; 与 bench_auc.csv 完全一致
        # (边标签 = tt y>0, 点标签 = ft != 1, 分数取最后一个 block 的 weights -- 与推理同位置)。
        _vpm = configs.get("settings", {})
        self.vpm_on = bool(_vpm.get("validate_prune_metric", False))
        # [2026-09-30] 默认 1000 (原 200): 200 个事件上 AP 的逐轮抖动 ~±0.03, 会把
        # "选 ckpt" 变成抽奖 —— v641 的 best ckpt 停在 ep0 的幸运峰 (0.777), 而中位数只有 0.76。
        # 抖动 ~ 1/sqrt(n), 提到 1000 事件可把噪声降约 2.2 倍。
        self.vpm_events = int(_vpm.get("validate_prune_events", 1000))
        self.vpm_node_thr = float(self.configs.get("node_prune_thr", 0.9))   # 工作点池用
        self._vpm = None

        # Loss functions + associated inference class for plotting
        if self.configs["LCA"]:
            lca_w = pos_weights["LCA"].clone().float()
            # ==== 方案4: class2 (同B边) 专项加权 ====
            # class2 决定链内父子结构, 准确率仅 ~36% 是 Perfect 的最大结构瓶颈
            # (class1 已 75%+ 保证连通, class2 是短板)。lca_class2_weight>1 放大其 loss。
            c2 = float(self.configs.get("lca_class2_weight", 1.0))
            if c2 != 1.0:
                lca_w[2] = lca_w[2] * c2
            self.lca_criterion = nn.CrossEntropyLoss(weight=lca_w)
        if self.configs["node_prune"]:
            self.node_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["nodes"])
        if self.configs["edge_prune"]:
            self.edge_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["edges"])
        if self.configs["pv_asso"]:
            self.pv_asso_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["pv_asso"])

        # ==== 剪枝损失再平衡 / focal / 链级 recall (2026-09-13) ====
        # 背景: combined_loss 里 t_nodes 权重仅 1、tt_edges 硬编码 33(v38 配置注释: "tt_edges 占 91%,
        #       node/LCA 欠投入"), 而配置里的 node_prune_weight/lca_weight **从未被代码读取**(死键)。
        #       oracle 实验: 只把被误删的真值径迹补回 -> All 39.56 -> 54.09(+14.5pp), 点 recall 是最大杠杆。
        #       默认值 (1/1/33) 与旧行为逐位一致, 不改变既有 run。
        self.node_loss_w = float(self.configs.get("node_prune_weight", 1.0))
        self.lca_loss_w = float(self.configs.get("lca_weight", 1.0))
        self.edge_loss_w = float(self.configs.get("edge_prune_weight", 33.0))
        # [2026-09-30] PV 关联损失此前**硬编码权重 1**, 无法关闭 -> v646 "只训剪枝" 需要能置 0,
        #   否则 PV 头的梯度仍会经共享 backbone 干扰剪枝表征 (多任务干扰假说无法干净检验)。
        self.pv_asso_w = float(self.configs.get("pv_asso_weight", 1.0))
        self.prune_focal_gamma = float(self.configs.get("prune_focal_gamma", 0.0))
        self._pw_nodes = pos_weights["nodes"]
        self._pw_edges = pos_weights["edges"]
        # ==== [2026-09-22] 极端不平衡(tt 边正类率 ~0.14%, 完全图)下的两个训练侧开关 ====
        # (a) edge_pos_weight_scale: 放大正类权重 -> 决策边界偏向 precision;
        # (b) edge_ohem_frac: 只保留"最难"的一批负边参与损失 (OHEM), 其余负边丢弃。
        #     两者默认关闭(1.0 / 0.0) 时与旧行为逐位一致。
        _pw_scale = float(self.configs.get("edge_pos_weight_scale", 1.0))
        if _pw_scale != 1.0:
            self._pw_edges = self._pw_edges * _pw_scale
            if self.configs["edge_prune"]:
                self.edge_criterion = nn.BCEWithLogitsLoss(pos_weight=self._pw_edges)
        self.edge_ohem_frac = float(self.configs.get("edge_ohem_frac", 0.0))
        # ==== [2026-09-22] 边头 pairwise ranking 损失 (v614) ====
        # 诊断: 边头 BCE 的 pos_weight≈700 把梯度全投在"推高真边", 对负边之间的相对次序
        # 约束很弱 -> thr0.9 时几乎所有"信号样"边都被保留 (precision 退到基频)。
        # 本项直接优化我们评测的排序: 见 edge_rank_loss 的注释。默认 0.0 = 关闭, 与旧行为一致。
        self.edge_rank_w = float(self.configs.get("edge_rank_weight", 0.0))
        self.edge_rank_nneg = int(self.configs.get("edge_rank_nneg", 64))
        self.edge_rank_margin = float(self.configs.get("edge_rank_margin", 1.0))
        # [2026-09-26] ranking 损失形式: "hinge"(默认, 与旧行为逐位一致) | "infonce"(listwise)
        self.edge_rank_mode = str(self.configs.get("edge_rank_mode", "hinge"))
        if self.edge_rank_mode not in ("hinge", "infonce"):
            raise ValueError(
                f"edge_rank_mode 只能取 'hinge' 或 'infonce', 得到 {self.edge_rank_mode!r}")
        # ==== [2026-09-23] delta_z0 方向处理 / 方向头 (leak 修复实验) ====
        # 背景: 0702 用 np.sort(ParticleIndex) 决定边方向, 使唯一反对称特征 delta_z0 的符号
        # 变成"单边且与真值相关"的量; 7 月模型因此学到一条不可迁移的捷径 (反事实: 抹掉符号
        # 边头 AUC 0.9996->0.77~0.86; 0904 训练的模型只掉 0.04~0.09)。三个开关:
        #   edge_dz_flip_prob: 训练时按无向对随机翻转 delta_z0 的符号 -> 方向变纯噪声 (零泄露基线)
        #   edge_dz_ip_canon : 用**可测量**的 minIP 定方向 (IP 小的更靠上游/PV) -> 可迁移的有向特征;
        #                      确定性变换, train/val/test 一致施加 (否则推理口径对不上)
        #   dir_head_weight  : 让剪枝 MLP 多学一个"哪端更靠上游"(标签= truth depth), 见 MLP.lin_dir
        self.dz_flip_prob = float(self.configs.get("edge_dz_flip_prob", 0.0))
        self.dz_ip_canon = bool(self.configs.get("edge_dz_ip_canon", False))
        self.dz_pvz_canon = bool(self.configs.get("edge_dz_pvz_canon", False))
        self.dz_abs = bool(self.configs.get("edge_dz_abs", False))
        self.dir_head_w = float(self.configs.get("dir_head_weight", 0.0))
        # ==== [2026-09-24] 两个新方向 (各自独立开关, 默认 0 -> 与旧行为逐位一致) ====
        # event_count_weight: 事件级链数辅助头 (预测该事件有几条真值链)
        # chain_contrastive_weight: 链级对比损失 (在嵌入空间按真值链拉近/推远)
        self.evt_count_w = float(self.configs.get("event_count_weight", 0.0))
        self.chain_contrast_w = float(self.configs.get("chain_contrastive_weight", 0.0))
        self.chain_contrast_tau = float(self.configs.get("chain_contrastive_tau", 0.1))
        self.dz_c, self.dz_s = 0.0, 1.0
        self._nc, self._ns = {}, {}
        _nd = self.configs.get("dz_norm_dict", "")
        if _nd:
            _d = torch.load(_nd, map_location="cpu", weights_only=False)
            self._nc = {k: float(v) for k, v in _d["center"].items()}
            self._ns = {k: (float(v) or 1.0) for k, v in _d["scale"].items()}
            self.dz_c = self._nc.get("delta_z0_reco", 0.0)
            self.dz_s = self._ns.get("delta_z0_reco", 1.0)
            print(f"[dz] 归一化字典 {_nd}: dz center={self.dz_c:.3f} scale={self.dz_s:.3f}")
        # ==== [2026-09-23] 剪枝 MLP 的派生输入 (物理派生量 / 三角传递性 / 次级顶点一致性) ====
        # 由 derive_pruning_features 现算, 经**零初始化适配器**注入剪枝 MLP (起点与旧模型等价)。
        self.der_prune = bool(self.configs.get("derived_prune", False))
        self.der_tri = bool(self.configs.get("derived_triangle", False))
        # [2026-09-26] derived_vertex: 边级 3 维"次级顶点一致性" (zcpa/flight/collinearity)
        self.der_vertex = bool(self.configs.get("derived_vertex", False))
        # [2026-09-30] 顶点一致性**几何绝对量** (doca / log(doca) / |Δ起点|, 3 维)。
        #   与旧 derived_vertex 的区别: 旧版只给比值 d_perp/(|Δz|+1), 把绝对量级除掉了 ->
        #   实测贡献≈0 (见 docs/feature_ceiling_analysis.md)。
        self.der_vgeom = bool(self.configs.get("derived_vertex_geom", False))
        # [2026-10-08] ip 并列时的定向取 0 (对称化, 见 derive_pruning_features 里 D5 的说明)。
        #   **默认 False = 历史口径**, 保证与已训练版本逐位可比; 新臂可显式打开。
        self.der_sgn_tiez = bool(self.configs.get("der_sgn_tie_zero", False))
        # [2026-09-30] 两端节点特征**对称**直连到边 ([x_i+x_j, |x_i-x_j|, x_i*x_j], 24 维)。
        self.der_psym = bool(self.configs.get("derived_pair_sym", False))
        # [2026-10-01] derived_comp: 竞争/排他性上下文 (6 维) —— 注意力/匹配方向的最小手工版
        self.der_comp = bool(self.configs.get("derived_comp", False))
        self.node_der_dim = 7 if self.der_prune else 0
        self.edge_der_dim = (7 + (2 if self.der_tri else 0) + (3 if self.der_vertex else 0)
                             + (3 if self.der_vgeom else 0) + (3 * PSYM_NODE_COLS if self.der_psym else 0)
                             + (6 if self.der_comp else 0)) \
            if self.der_prune else 0
        if self.der_prune:
            # 维度必须与 GNblocks 里适配器的输入维严格一致 (否则前向 matmul 报错且难定位)
            _kn_cfg = int(self.configs_gn.get("extra_node_dim", 0))
            _ke_cfg = int(self.configs_gn.get("extra_edge_dim", 0))
            assert self.node_der_dim == _kn_cfg, (
                f"派生输入维度不一致: derive_pruning_features 构造节点 {self.node_der_dim} 维, "
                f"但 GNblocks.extra_node_dim={_kn_cfg} (derived_prune={self.der_prune})")
            assert self.edge_der_dim == _ke_cfg, (
                f"派生输入维度不一致: 边 {self.edge_der_dim} 维 = 7(基础)"
                f"+{2 if self.der_tri else 0}(derived_triangle)"
                f"+{3 if self.der_vertex else 0}(derived_vertex)"
                f"+{3 if self.der_vgeom else 0}(derived_vertex_geom)"
                f"+{24 if self.der_psym else 0}(derived_pair_sym)"
                f"+{6 if self.der_comp else 0}(derived_comp), "
                f"但 GNblocks.extra_edge_dim={_ke_cfg}")
            print(f"[der_input] 派生输入启用: 节点 {self.node_der_dim} 维 / 边 {self.edge_der_dim} 维"
                  f"{' (含三角传递)' if self.der_tri else ''}"
                  f"{' (含次级顶点一致性)' if self.der_vertex else ''}"
                  f"{' (含顶点几何绝对量)' if self.der_vgeom else ''}"
                  f"{' (含端点对称直连)' if self.der_psym else ''}"
                  f"{' (含竞争/排他)' if self.der_comp else ''}")
        # 节点侧 pairwise ranking (边侧 ranking 已验证 +12.7%, 点的正类率高得多, 值得搬到点侧)
        self.node_rank_w = float(self.configs.get("node_rank_weight", 0.0))
        self.node_rank_nneg = int(self.configs.get("node_rank_nneg", 64))
        self.node_rank_margin = float(self.configs.get("node_rank_margin", 1.0))
        # 链级 min-pooling recall 损失 (对准"整条链被剪掉"的失败模式)
        self.chain_recall_w = float(self.configs.get("chain_recall_weight", 0.0))
        self.chain_recall_edge_w = float(self.configs.get("chain_recall_edge_weight", 0.0))
        self.chain_recall_thr = float(self.configs.get("chain_recall_thr", 0.5))
        self.chain_recall_tau = float(self.configs.get("chain_recall_tau", 0.1))

        # ==== [2026-10-09] "最弱环"链级存活损失 (chain_weakest) ====
        # 物理判据: 整条真值链必须**每个环节都对** (AND 语义, 一失毁全链), 而训练一直只做逐边
        #   BCE (优化的是平均正确率, 与"整链存活"不一致)。纯 CPU 小探针实证 (analyze_vertex_assoc.py
        #   --chain_loss_w/--chain_pool softmin): 多 B 子集"链存活@边精度90%" 24.2% -> 32.6%
        #   (3/3 种子为正, 均值 +8.4pp); 关键对照——换成乘积/几何平均形式 (≈逐边 BCE) 只有 28.0%
        #   -> 有效的不是"换成乘积", 而是罚链内**最弱环**的这种非线性 (softmin_γ) 形式; 同时边级
        #   AP/AUC 也变好 (0.8393->0.8533 / 0.8650->0.8732), 并非此消彼长。
        # 本项把该形式搬到主模型: 对每条真值链用 softmin_γ 聚合链内 tt 边存活概率, 罚最弱环节。
        # 缺省 w=0 -> 整段不执行, 与旧行为逐位一致 (见 shared_step 中同名实现)。
        self.chain_weakest_w = float(self.configs.get("chain_weakest_w", 0.0))
        self.chain_weakest_gamma = float(self.configs.get("chain_weakest_gamma", 10.0))
        if self.chain_weakest_w > 0:
            print(f"[chain_weakest] 最弱环链级存活损失启用: w={self.chain_weakest_w} "
                  f"gamma={self.chain_weakest_gamma}", flush=True)

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")
        # init event reconstruction class
        self.evt_reco = EventReconstruction(configs)

        # Pruning threshold for reco
        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

    @property
    def chain_scorer(self):
        """候选衰变链 scorer (只读代理到 model.chain_scorer; 未启用时为 None)。"""
        return getattr(self.model, "chain_scorer", None)

    def on_load_checkpoint(self, checkpoint):
        """兼容旧 checkpoint 续训: 无 chain_scorer/source_head 头时, 重置 optimizer/lr_scheduler 状态。

        新头使 optimizer 参数数增加 (如 CERN 297 -> 309), 旧 checkpoint 的
        optimizer_states 会因 param group 大小不匹配而 load 失败; 用新结构重建
        空状态, optimizer 从当前 lr 重新起步 (新头本无历史动量)。
        """
        new_heads = []
        if self.chain_scorer is not None:
            has_cs = any(k.startswith("model.chain_scorer.") for k in checkpoint.get("state_dict", {}))
            if not has_cs:
                new_heads.append("chain_scorer")
        if self.source_head_on:
            has_sh = any(k.startswith("model.source_head.") for k in checkpoint.get("state_dict", {}))
            if not has_sh:
                new_heads.append("source_head")
        if self.pv_cluster_on:
            has_pch = any(k.startswith("model.pv_cluster_head.") for k in checkpoint.get("state_dict", {}))
            if not has_pch:
                new_heads.append("pv_cluster_head")
        if self.mass_head_on:
            has_mh = any(k.startswith("model.edge_mass_head.") for k in checkpoint.get("state_dict", {}))
            if not has_mh:
                new_heads.append("edge_mass_head")
        if self.struct_head_on:
            has_sh2 = any(k.startswith("model.node_struct_head.") for k in checkpoint.get("state_dict", {}))
            if not has_sh2:
                new_heads.append("node_struct_head")
        if self.mom_head_on:
            has_mh2 = any(k.startswith("model.node_mom_head.") for k in checkpoint.get("state_dict", {}))
            if not has_mh2:
                new_heads.append("node_mom_head")
        # ==== 试验: track 级自注意力 (新参数, 旧 ckpt 无) ====
        track_attn = getattr(self.model, "_track_attn", None)
        if track_attn is not None:
            has_ta = any(k.startswith("model._track_attn.") for k in checkpoint.get("state_dict", {}))
            if not has_ta:
                new_heads.append("track_attn")
        # ==== [2026-09-26] tt 边图注意力 (新参数, 旧 ckpt 无) ====
        if any("_line_attn." in k for k in self.state_dict()):
            has_lg = any("_line_attn." in k for k in checkpoint.get("state_dict", {}))
            if not has_lg:
                new_heads.append("line_graph_attn")
        if new_heads:
            print(f"[heads] 旧 checkpoint 无 {new_heads} 头: "
                  "重置 optimizer/lr_scheduler 状态 (新头无历史动量, 从当前 lr 重新起步)")
            opt = self.optimizer_class(self.model.parameters(), **self.optimizer_params)
            checkpoint["optimizer_states"] = [opt.state_dict()]
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
            checkpoint["lr_schedulers"] = [sch.state_dict()]

    def load_state_dict(self, state_dict, strict=True):
        """兼容旧 checkpoint 续训 / 升维继承:
        1) 新头缺参 (chain_scorer/source_head/pv_cluster_head/edge_mass_head/
           node_struct_head/node_mom_head/_track_attn)
        2) 维度升级: 同名层 shape 不匹配 (tracks 16→32 / tt 边 16→24 的 encoder/GN/
           decoder 末层及下游头输入层) — 该层保持随机初始化, 其余按名严格加载。

        trainer.fit(ckpt_path=...) 与 load_from_checkpoint 最终都经 load_state_dict;
        放宽仅在存在上述两类时触发, 跳过的层会打印, 供冒烟人工核对。
        """
        if strict:
            heads = ("model.chain_scorer", "model.source_head", "model.pv_cluster_head",
                     "model.edge_mass_head", "model.node_struct_head", "model.node_mom_head",
                     "model._track_attn")
            cur = self.state_dict()
            # 方案A 上下文剪枝头挂在 GN block 内 (key 形如 model._blocks.N._context_head.*),
            # 前缀不固定 -> 用子串匹配纳入"新头缺参"白名单。
            miss = [k for k in cur if k not in state_dict
                    and (k.startswith(heads) or "_context_head." in k or "lin_dir." in k
                         or "der_adapter." in k
                         # [2026-09-24] 三个新模块 (同样挂在 GN block 内, 零初始化/新头 -> 允许缺失)
                         or "_pv_ov_adapter." in k or "_evt_bias." in k or "_evt_count." in k
                         # [2026-09-26] tt 边图注意力 (挂最后一个 GN block 内, 零初始化 -> 允许缺失)
                         or "_line_attn." in k)]
            shape_mm = []
            for k in cur:
                if k not in state_dict:
                    continue
                try:
                    same = tuple(state_dict[k].shape) == tuple(cur[k].shape)
                except RuntimeError:          # uninit (Lazy) 参数, shape 不可读
                    continue
                if not same:
                    shape_mm.append(k)
            if miss or shape_mm:
                print(f"[load-relax] 兼容加载: 缺新头 {len(miss)} 个, "
                      f"shape 不匹配 {len(shape_mm)} 个 (保持随机初始化)")
                for k in shape_mm:
                    print(f"    skip(shape) {k}: ckpt {tuple(state_dict[k].shape)} "
                          f"-> new {tuple(cur[k].shape)}")
                compatible = {}
                for k in cur:
                    if k not in state_dict:
                        continue
                    try:
                        same = tuple(state_dict[k].shape) == tuple(cur[k].shape)
                    except RuntimeError:
                        same = True    # cur 侧 uninit (Lazy 待 materialize): 由 ckpt 模板加载
                    if same:
                        compatible[k] = state_dict[k]
                return super().load_state_dict(compatible, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, batch):
        if self.use_pid == "realistic":  # only for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":  # mc response for lhcb or onehot for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,  # Reduce LR after 5 epochs of no improvement
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_combined_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def on_train_start(self):
        # 从 checkpoint 续训时, 若配置了新学习率, 覆盖 checkpoint 中保存的 lr
        # (checkpoint 恢复会还原 optimizer 状态, 包括旧 lr)
        if self.resume_lr is not None:
            for g in self.optimizers().param_groups:
                g["lr"] = float(self.resume_lr)
            print(f"[resume] overwrite lr -> {self.resume_lr}")

        # ==== v41 修复: 重置 EarlyStopping (子图训练 val 口径变化) ====
        # v38 切到子图训练后, val_combined_loss 起点会远高于 v38 全图口径的 35.562;
        # 若继承 checkpoint 里的 best_score/wait_count, 15 epoch 内没跌破旧 best 就会误停
        # (v40 就是被继承的 wait_count 在 4 epoch 后停掉的)。子图训练应只跟本次 run 的 best 比。
        if self.pv_cluster_on:
            try:
                for cb in self.trainer.callbacks:
                    if cb.__class__.__name__ == "EarlyStopping":
                        dev = cb.best_score.device if cb.best_score is not None else self.device
                        cb.best_score = torch.tensor(float("inf"), device=dev)
                        cb.wait_count = 0
                        print("[pv_cluster] 重置 EarlyStopping (子图训练 val 口径变化, 只跟本次 run 比)")
            except Exception as e:
                print(f"[pv_cluster] 重置 EarlyStopping 失败(跳过): {e}")

    def _pv_cluster_scores(self, batch):
        """pv_cluster_head 前向: 全图 tr-pv 边 -> 归属 logits [E, 1]。

        特征 = concat(tracks.x[src] (原始, pid concat 前), pvs.x[dst], trpv.edges),
        训练/测试都在同一特征空间 (与 GNN 编码无关), 保证 train/infer 对齐。
        """
        trpv = batch[('tracks', 'to', 'pvs')]
        ei = trpv.edge_index
        feat = torch.cat([
            batch['tracks'].x[ei[0]],
            batch['pvs'].x[ei[1]],
            trpv.edges,
        ], dim=-1)
        return self.model.pv_cluster_head(feat)

    def _pv_cluster_assign(self, batch, logits, sample=True):
        """用 pv_cluster_head 预测 (+ 温度退火 Gumbel 路由) 给每个 track 分配 PV (全局索引, -1=无)。

        分配 = argmax((logit + Gumbel噪声) / tau) (Gumbel-Softmax 标准形式):
          - sample=True (训练): tau 高(早期) 近似按 softmax 概率采样(探索); tau 低(后期) 收敛 argmax
          - sample=False (val): 确定性 argmax(logit), 与推理侧分簇一致
        """
        trpv = batch[('tracks', 'to', 'pvs')]
        ei = trpv.edge_index
        s = logits.squeeze(-1)
        if sample:
            tau = max(self.pv_tau, 1e-3)
            u = torch.rand_like(s).clamp(1e-8, 1.0 - 1e-8)
            g = -torch.log(-torch.log(u))  # Gumbel(0,1)
            score = (s + g) / tau
        else:
            score = s
        track_pv = torch.full((batch['tracks'].x.shape[0],), -1, dtype=torch.long, device=s.device)
        for t in torch.unique(ei[0]):
            m = (ei[0] == t)
            if not m.any():
                continue
            track_pv[int(t.item())] = ei[1][m][score[m].argmax()]
        return track_pv

    def _truth_pv_assign(self, batch):
        """truth PV 分配 (v39 方式): 每条 track 取其 y==1 关联边的第一个 PV; 无关联 -> -1。"""
        t_batch = batch['tracks'].batch
        trpv = batch[('tracks', 'to', 'pvs')]
        tr_pv = trpv.edge_index[:, trpv.y == 1]
        track_pv = torch.full((t_batch.numel(),), -1, dtype=torch.long, device=t_batch.device)
        for t, p in zip(tr_pv[0].tolist(), tr_pv[1].tolist()):
            if track_pv[t].item() == -1:
                track_pv[t] = p
        return track_pv

    def _split_by_pv(self, batch, logits=None, sample=True, cap=0):
        """方案7 (v39/v40/v41): 训练/验证时按 PV 分簇子图 (pv_cluster 纳入训练)。

        把 batch 中每个事件按 PV 拆成子图, 再用 Batch.from_data_list 重拼。
        子图内仅含该 PV 的 tracks + 该 PV 节点 + 簇内边 (tt 两端同簇 + tr-pv)
        —— 模型只见到"簇内低连通小图", 与推理时分簇重建完全对齐 (消除 train-inference gap)。

        track->PV 分配来源 (v41 课程式过渡):
          - alpha = 1.0 (run 起点): 全部用 truth 分配 (稳定热启动, v38 权重不被随机头扰动)
          - alpha -> 0.0: 逐步切换到 cluster 头分配 (训练 MLP + 温度退火)
          - logits=None: 回退纯 truth 分配
        碎片簇 (< pv_cluster_min_tracks) 并入无PV簇, 与推理侧同规则。
        cap>0 时训练侧随机保留 cap 个子图 (提速; val/test 传 cap=0 全量)。
        """
        try:
            from torch_geometric.data import Batch
            t_batch = batch['tracks'].batch              # [n_tr] 全局事件索引
            p_batch = batch['pvs'].batch                 # [n_pv]
            n_evt = int(t_batch.max()) + 1
            if logits is not None:
                truth_pv = self._truth_pv_assign(batch)
                cluster_pv = self._pv_cluster_assign(batch, logits, sample=sample)
                # 课程式过渡: 以概率 alpha 用 truth, 否则用 cluster 头 (逐 track)
                alpha = self.pv_curriculum_alpha
                if alpha <= 0.0:
                    track_pv = cluster_pv
                elif alpha >= 1.0:
                    track_pv = truth_pv
                else:
                    use_truth = torch.rand(truth_pv.shape, device=truth_pv.device) < alpha
                    track_pv = torch.where(use_truth, truth_pv, cluster_pv)
            else:
                track_pv = self._truth_pv_assign(batch)

            min_tracks = int(self.configs.get("pv_cluster_min_tracks", 3))
            subgraphs = []
            for ev in range(n_evt):
                ev_tracks = (t_batch == ev).nonzero().flatten().tolist()
                ev_pvs = (p_batch == ev).nonzero().flatten().tolist()
                groups = {}
                for t in ev_tracks:
                    groups.setdefault(int(track_pv[t].item()), []).append(t)
                # 碎片 PV 簇并入无PV簇 (防碎片链, 与推理侧同规则)
                for p in list(groups.keys()):
                    if p != -1 and len(groups[p]) < min_tracks:
                        groups.setdefault(-1, []).extend(groups.pop(p))
                for p, tl in groups.items():
                    if not tl:
                        continue
                    # 无PV (-1) 簇保留该事件全部 pvs (维持 pv_asso 监督); 有 PV 簇只留该 PV
                    pvs_keep = ev_pvs if p == -1 else [p]
                    if not pvs_keep:
                        continue
                    subgraphs.append(self._build_pv_subgraph(batch, tl, pvs_keep, ev))
            if not subgraphs:
                return batch
            # v41 修复③: 训练侧子图数量上限 (随机保留, 提速; 每 batch 随机 -> 全覆盖)
            if cap > 0 and len(subgraphs) > cap:
                keep = torch.randperm(len(subgraphs))[:cap].tolist()
                subgraphs = [subgraphs[i] for i in keep]
            return Batch.from_data_list(subgraphs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[pv_cluster_train] WARN: {type(e).__name__}: {e} -> 用原 batch")
            return batch

    def _build_pv_subgraph(self, batch, tl, pvs_keep, ev):
        """手动构造 PV 子图 (普通 HeteroData, 无 batch 残留):
        选中 tracks/pvs 节点 + 两端均在簇内的 tt/tr-pv 边 + 节点/边/全局属性。
        """
        from torch_geometric.data import HeteroData
        dev = batch['tracks'].x.device  # 所有构造张量与 batch 同设备 (GPU 上 isin/indexing 需同设备)
        tl_t = torch.tensor(tl, dtype=torch.long, device=dev)
        pv_t = torch.tensor(pvs_keep, dtype=torch.long, device=dev)
        old2new = {old: i for i, old in enumerate(tl)}
        pv_old2new = {old: i for i, old in enumerate(pvs_keep)}
        sub = HeteroData()
        n_tr = batch['tracks'].x.shape[0]
        # 节点属性 (排除 batch/ptr; 只复制节点级属性, 信号级属性如 sig_keys 长度 != 节点数, 训练不需要)
        for k, v in batch['tracks'].items():
            if k in ('batch', 'ptr'):
                continue
            if v.shape[0] == n_tr:
                sub['tracks'][k] = v[tl_t]
        for k, v in batch['pvs'].items():
            if k in ('batch', 'ptr'):
                continue
            sub['pvs'][k] = v[pv_t]
        # tt 边: 两端同簇
        tt = batch[('tracks', 'to', 'tracks')]
        n_e = tt.edge_index.shape[1]
        m = torch.isin(tt.edge_index[0], tl_t) & torch.isin(tt.edge_index[1], tl_t)
        sub_tt_ei = tt.edge_index[:, m]
        sub_tt_ei = torch.tensor(
            [[old2new[int(a)] for a in sub_tt_ei[0].tolist()],
             [old2new[int(b)] for b in sub_tt_ei[1].tolist()]],
            dtype=torch.long, device=dev)
        sub[('tracks', 'to', 'tracks')].edge_index = sub_tt_ei
        for k, v in tt.items():
            # 只复制边级属性 (长度==边数; senders/receivers/sig_y 是稀疏衰变边属性, 训练不需要)
            if k != 'edge_index' and v.shape[0] == n_e:
                sub[('tracks', 'to', 'tracks')][k] = v[m]
        # tr-pv 边: 簇内 track + 保留的 PV
        trpv = batch[('tracks', 'to', 'pvs')]
        n_ep = trpv.edge_index.shape[1]
        m2 = torch.isin(trpv.edge_index[0], tl_t) & torch.isin(trpv.edge_index[1], pv_t)
        sub_trpv_ei = trpv.edge_index[:, m2]
        sub_trpv_ei = torch.tensor(
            [[old2new[int(a)] for a in sub_trpv_ei[0].tolist()],
             [pv_old2new[int(b)] for b in sub_trpv_ei[1].tolist()]],
            dtype=torch.long, device=dev)
        sub[('tracks', 'to', 'pvs')].edge_index = sub_trpv_ei
        for k, v in trpv.items():
            if k != 'edge_index' and v.shape[0] == n_ep:
                sub[('tracks', 'to', 'pvs')][k] = v[m2]
        # 全局属性 (Batch 无 .store property, 用 _global_store)
        gstore = getattr(batch, '_global_store', None)
        if gstore is not None:
            for k, v in gstore.items():
                sub[k] = v
        # globals: Batch 中堆叠为 [n_evt, dim], 单子图取本事件 (模型 encoder 需要 globals.x)
        try:
            gx = batch['globals'].x
            sub['globals'].x = gx[ev].unsqueeze(0)
        except Exception:
            pass
        return sub

    def shared_step(self, batch, batch_idx, log, mode="train"):
        loss = init_loss(self.device)

        # ==== 方案7b (v40/v41): 可训练 PV 分簇头 ====
        # cluster 头吃原始特征 (pid concat / GNN 前), 训练与推理同一头同一特征 -> 严格对齐。
        # 训练时: 头 loss (BCE, 全图 tr-pv 边) + 温度退火 Gumbel 分配 -> 切子图;
        # 验证时: 同样切子图 (v41 修复①: 消除 train/val 图结构 gap), 确定性分配;
        # 测试时: logits 存起来供 reconstruction 用同一头分簇 (pv_cluster_assign=cluster_head)。
        pv_cluster_logits = None
        if self.pv_cluster_on and mode in ("train", "val", "test"):
            try:
                trpv = batch[('tracks', 'to', 'pvs')]
                pv_cluster_logits = self._pv_cluster_scores(batch)      # [E, 1]
                if mode == "train":
                    y_pvclu = trpv.y.to(torch.float32).view(-1, 1)
                    pvclu_filter = (trpv.filter == 1) if hasattr(trpv, 'filter') else None
                    if pvclu_filter is not None and pvclu_filter.any():
                        loss["pv_cluster"] = self.pv_cluster_criterion(
                            pv_cluster_logits[pvclu_filter], y_pvclu[pvclu_filter])
                        log["pv_cluster_loss"].append(loss["pv_cluster"].item())
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[pv_cluster] WARN: {type(e).__name__}: {e} -> 回退 truth 分簇 / 跳过 cluster 头")
                pv_cluster_logits = None

        # ==== 方案7 (v39/v41): 训练/验证时 PV 分簇子图 (训练/推理图结构对齐) ====
        # v41 修复①: val 也切子图 (与 train 同结构, 消除 train/val gap; 确定性分配, 与推理一致)
        # v41 修复③: 训练侧 cap 子图数提速 (val/test 全量, val 每 epoch 一次开销可忽略)
        if mode in ("train", "val") and self.configs.get("pv_cluster", False):
            cap = self.pv_cluster_max_subgraphs if mode == "train" else 0
            batch = self._split_by_pv(batch, logits=pv_cluster_logits,
                                      sample=(mode == "train"), cap=cap)

        # modify batch to include pid information depending on use_pid or not
        if self.use_pid == "realistic":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        if mode == "test" and self.configs["pv_asso"]:
            minip = batch[("tracks", "to", "pvs")].edges.flatten()

        # 保存原始 tt 物理边特征 (model forward 会原地修改 batch edges -> 必须提前 clone)
        orig_tt_edges = batch[('tracks', 'to', 'tracks')].edges.clone()
        # ==== [2026-09-23] delta_z0 方向处理 (leak 实验; 必须在 forward 前改, 与造数据时的口径一致) ====
        # 镜像 raw 值: raw -> sign*raw, 归一化空间里 v -> sign*v + (sign-1)*c/s
        _do_flip = (mode == "train" and self.dz_flip_prob > 0)
        if _do_flip or self.dz_ip_canon or self.dz_pvz_canon or self.dz_abs:
            _tt = batch[('tracks', 'to', 'tracks')]
            _tt.edges = _tt.edges.clone()
            _v = _tt.edges[:, 3]
            if self.dz_abs:
                # 诚实基线 (对应 yukai 的 delta_z_mode="abs"): 只保留 |Δz| 的**大小**信息, 完全不给方向。
                # 用途: 证明"不用方向也能达到同一水平", 挡回"你们是不是又靠顺序"的质疑。
                _tt.edges[:, 3] = (_v * self.dz_s + self.dz_c).abs().sub(self.dz_c).div(self.dz_s)
            else:
                if self.dz_pvz_canon:
                    # yukai 的方案: 按"minIP 关联 PV 的 z 更小者在前"定方向 (z 相同/无关联的对不翻转)。
                    # 注意: 只对**两端关联到不同 PV** 的对生效 -> 同 PV 对仍是原顺序 (部分规范化)。
                    _tpv = batch[("tracks", "to", "pvs")]
                    _n_tr = batch["tracks"].x.shape[0]
                    _snd, _rcv = _tpv.edge_index[0].long(), _tpv.edge_index[1].long()
                    _ip = _tpv.edges.reshape(-1)
                    _best = torch.full((_n_tr,), float("inf"), device=_v.device, dtype=_ip.dtype)
                    _best = _best.scatter_reduce(0, _snd, _ip, reduce="amin", include_self=True)
                    _is_best = _ip == _best[_snd]
                    _assoc = torch.full((_n_tr,), -1, dtype=torch.long, device=_v.device)
                    _assoc[_snd[_is_best]] = _rcv[_is_best]
                    _ok = _assoc >= 0                    # 无 tr-pv 边的径迹不参与 (避免 yukai 版取 zpv[-1] 的错值)
                    _zpv = (batch["pvs"].x[:, 2] * float(self._ns.get("zPV_reco", 1.0))
                            + float(self._nc.get("zPV_reco", 0.0)))
                    _tz = torch.zeros(_n_tr, device=_v.device, dtype=_zpv.dtype)
                    _tz[_ok] = _zpv[_assoc[_ok]]
                    _a, _b = _tt.edge_index[0].long(), _tt.edge_index[1].long()
                    _sgn = torch.where((_ok[_a] & _ok[_b]) & (_tz[_a] > _tz[_b]),
                                       -torch.ones_like(_v), torch.ones_like(_v))
                elif self.dz_ip_canon:
                    _ip = track_minip(batch)                    # 可测量: minIP 小的更靠上游
                    _a, _b = _tt.edge_index[0].long(), _tt.edge_index[1].long()
                    _sgn = torch.where(_ip[_a] <= _ip[_b],
                                       torch.ones_like(_v), -torch.ones_like(_v))
                else:
                    _sgn = edge_dz_pair_sign(_tt.edge_index, self.dz_flip_prob,
                                             salt=int(torch.randint(0, 2 ** 30, (1,)).item()))
                _tt.edges[:, 3] = torch.where(_sgn < 0, -_v - 2.0 * self.dz_c / self.dz_s, _v)
        # ==== [2026-09-23] 剪枝 MLP 的派生输入 (物理派生量 / 三角传递性) ====
        # 从原始 px/py/pz/生产顶点/PV 关联现算, 经零初始化适配器注入剪枝 MLP (不重产数据)。
        if self.der_prune:
            try:
                _nd_d, _ed_d = derive_pruning_features(batch, self._nc, self._ns, self.der_tri,
                                                       use_vertex=self.der_vertex,
                                                       use_vertex_geom=self.der_vgeom,
                                                       use_pair_sym=self.der_psym,
                                                       use_comp=self.der_comp,
                                                       sgn_tie_zero=self.der_sgn_tiez)
                batch['tracks'].x_der = _nd_d
                batch[('tracks', 'to', 'tracks')].der_edges = _ed_d
                if mode == "train" and self.trn_log is not None and "der_stat" not in self.trn_log:
                    self.trn_log["der_stat"] = [0.0]
                    print(f"[der_input] 尺寸检查: 节点派生 {tuple(_nd_d.shape)} / 边派生 {tuple(_ed_d.shape)}"
                          f" | 样例 边 [ΔR,m,|Σq|,ΔpT,ΔIP,rk,Δz"
                          f"{',sup,rk_aff' if self.der_tri else ''}"
                          f"{',zcpa,flight,collin' if self.der_vertex else ''}"
                          f"{',doca,logdoca,|dr|' if self.der_vgeom else ''}"
                          f"{',xi+xj,|xi-xj|,xi*xj' if self.der_psym else ''}"
                          f"{',comp_ratio,deg,cnt ×2' if self.der_comp else ''}]= "
                          + " ".join(f"{x:+.2f}" for x in _ed_d[0].tolist()), flush=True)
            except Exception as _e:
                # [2026-09-26 FIX] 训练时不允许静默降级: 派生特征算失败会退化成"没开 derived_prune"
                # (适配器仍在但不生效) 却照常出结论, 属最危险的静默失效。推理侧可容忍。
                if mode == "train":
                    raise
                print(f"[der_input] WARN: {type(_e).__name__}: {_e}", flush=True)
        # 保存原始轨迹动量 (px,py,pz, 归一化) —— model forward 会原地覆盖 tracks.x
        # 为 encoder 表征, mass head 的物理真值 (ππ 不变质量) 需在覆盖前取出。
        orig_tracks_p = batch['tracks'].x[:, :3].clone()

        outputs = self.model(batch)
        # 将原始物理边特征挂到模型输出上 (model 会覆盖 edges 为 LCA 输出, 物理特征需保留)
        outputs[('tracks', 'to', 'tracks')].phys_edges = orig_tt_edges
        if self.configs["LCA"]:
            y_LCA = batch[('tracks', 'to', 'tracks')].y.to(torch.int64)
            outputs[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'to', 'tracks')].edges
            loss["LCA"] = self.lca_criterion(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            log["LCA_loss"].append(loss["LCA"].item())
            acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            for key, values in acc_LCA.items():
                log[key].append(values)
        if self.configs["node_prune"]:
            y_nodes = (batch["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
        if self.configs["edge_prune"]:
            y_edges = batch[('tracks', 'to', 'tracks')].y > 0
            y_edges = y_edges.to(torch.float32).unsqueeze(-1)
        if self.configs["pv_asso"]:
            y_pv_asso = batch[("tracks", "to", "pvs")].y.to(torch.float32).view(-1, 1)
            pv_filter = batch[('tracks', 'pvs')].filter == 1

        for i, block in enumerate(self.model._blocks):
            use_focal = (mode == "train" and self.prune_focal_gamma > 0)
            # [2026-09-29] val 剪枝 AUC/AP: 只在最后一个 block 取分 (与推理剪枝取值位置一致)
            # 注意: validation_step 传的 mode 是 "val" (不是 "validation")。
            if mode == "val" and self.vpm_on and i == len(self.model._blocks) - 1:
                self._vpm_accumulate(batch, block)
            if self.configs["node_prune"]:
                if use_focal:
                    loss["t_nodes"] += focal_bce_with_logits(block.node_logits['tracks'], y_nodes,
                                                             pos_weight=self._pw_nodes,
                                                             gamma=self.prune_focal_gamma)
                else:
                    loss["t_nodes"] += self.node_criterion(block.node_logits['tracks'], y_nodes)
                if mode == "test" and self.configs["plt_nodes"]:
                    get_block_score(log, block.node_weights['tracks'].squeeze(), y_nodes, i, var="nodes")

            if self.configs["edge_prune"]:
                _e_logits = block.edge_logits[('tracks', 'to', 'tracks')]
                if self.edge_ohem_frac > 0 and mode == "train":
                    # [2026-09-22] OHEM: 保留全部正边 + 最难的一批负边 (按 pos_weight 加权后的逐边损失排序)
                    _le = F.binary_cross_entropy_with_logits(_e_logits, y_edges,
                                                             pos_weight=self._pw_edges.to(_e_logits.device),
                                                             reduction="none")
                    _neg = y_edges < 0.5
                    _n_neg = int(_neg.sum().item())
                    _k = max(1, int(self.edge_ohem_frac * _n_neg))
                    if _n_neg > 0 and _k < _n_neg:
                        _thr = torch.topk(_le[_neg], _k).values.min()
                        _keep = (~_neg) | (_le >= _thr)
                        loss["tt_edges"] += _le[_keep].mean()
                    else:
                        loss["tt_edges"] += _le.mean()
                elif use_focal:
                    loss["tt_edges"] += focal_bce_with_logits(_e_logits, y_edges,
                                                              pos_weight=self._pw_edges,
                                                              gamma=self.prune_focal_gamma)
                else:
                    loss["tt_edges"] += self.edge_criterion(_e_logits, y_edges)
                if mode == "test" and self.configs["plt_edges"]:
                    get_block_score(log, block.edge_weights[('tracks', 'to', 'tracks')].squeeze(), y_edges, i,
                                    var="edges")
                # ==== [2026-09-22] v614: 边头 pairwise/listwise ranking 损失 (只在最后一个 block) ====
                if (self.edge_rank_w > 0 and mode == "train"
                        and i == len(self.model._blocks) - 1):
                    _rl = edge_rank_loss(_e_logits, y_edges,
                                         outputs[('tracks', 'to', 'tracks')].edge_index,
                                         batch['tracks'].batch,
                                         n_neg=self.edge_rank_nneg, margin=self.edge_rank_margin,
                                         mode=self.edge_rank_mode)
                    if _rl is not None:
                        loss["tt_rank"] = _rl
                # ==== [2026-09-23] 节点侧 pairwise ranking (把边侧已验证的 +12.7% 搬到点侧) ====
                # 动机: 链存活瓶颈在**点**(oracle: 只卡点 39.7% vs 只卡边 71.5%); 且点正类率远高于
                # 边(0.14%) -> pos_weight 主导的问题在点侧更轻, ranking 更可能有效。
                if (self.node_rank_w > 0 and mode == "train"
                        and i == len(self.model._blocks) - 1 and "tracks" in block.node_logits):
                    _nlg = block.node_logits["tracks"]
                    _nidx = torch.arange(_nlg.shape[0], device=_nlg.device)
                    _nl = edge_rank_loss(_nlg, y_nodes, torch.stack([_nidx, _nidx]),
                                         batch['tracks'].batch,
                                         n_neg=self.node_rank_nneg, margin=self.node_rank_margin)
                    if _nl is not None:
                        loss["node_rank"] = _nl
            if self.configs["pv_asso"]:
                loss["pv_asso"] += self.pv_asso_criterion(block.edge_logits[("tracks", "to", "pvs")][pv_filter],
                                                          y_pv_asso[pv_filter])
                if mode == "test" and self.configs["plt_pvs"]:
                    get_block_score(log, block.edge_weights[("tracks", "to", "pvs")].squeeze(), y_pv_asso, i,
                                    var="pv_asso")

        # Cap each loss component to prevent FP16 overflow before combination.
        # The 33x multiplier on tt_edges can push values beyond FP16 max (65504).
        # Per-component protection means only the overflowed component is zeroed,
        # while other loss signals still produce gradients for this batch.
        # clamp(max=1e3) 额外防止有限但极大的 loss (如 v31 ep68-70 出现 ~1e17 的数值爆炸)
        for k in loss:
            loss[k] = torch.nan_to_num(loss[k], nan=0.0, posinf=1e6, neginf=0.0).clamp(max=1e3)

        # ==== 第5个监督头: 候选衰变链选择 (train 时与主干联合训练) ====
        if mode == "train" and self.chain_scorer is not None:
            chain_loss = self._chain_select_loss(batch, outputs, block)
            if chain_loss is not None:
                loss["chain_select"] = chain_loss
                log["chain_select_loss"].append(chain_loss.item())

        # ==== 第6个监督头: 源检测 (Rumor Centrality 训练化, train 时) ====
        if mode == "train" and self.source_head_on:
            src_loss = self._source_loss(batch, outputs, block)
            if src_loss is not None:
                loss["source"] = src_loss
                log["source_loss"].append(src_loss.item())

        # ==== 方案5: 链内 LCA 一致性辅助损失 (chain_lca_filter 训练化, train 时) ====
        if mode == "train" and self.chain_lca_on:
            cl_loss = self._chain_lca_loss(batch, outputs, block)
            if cl_loss is not None:
                loss["chain_lca"] = cl_loss
                log["chain_lca_loss"].append(cl_loss.item())

        # ==== 第7个监督头: 边级不变质量回归 (输出侧物理监督, train 时) ====
        if mode == "train" and self.mass_head_on:
            m_loss = self._mass_loss(batch, outputs, orig_tracks_p)
            if m_loss is not None:
                loss["mass"] = m_loss
                log["mass_loss"].append(m_loss.item())

        # ==== 第8个监督头: 节点结构监督 (depth + RC 回归, train 时) ====
        if mode == "train" and self.struct_head_on:
            st_loss = self._struct_loss(batch, outputs, block)
            if st_loss is not None:
                loss["struct"] = st_loss
                log["struct_loss"].append(st_loss.item())

        # ==== 第9个监督头: 节点级动量回归 (mom_head, train 时) ====
        if mode == "train" and self.mom_head_on:
            mo_loss = self._mom_loss(batch, outputs, block, orig_tracks_p)
            if mo_loss is not None:
                loss["mom"] = mo_loss
                log["mom_loss"].append(mo_loss.item())

        # ==== [2026-09-23] 方向头监督: 让**剪枝 MLP 自己**学会"哪一端更靠上游" ====
        # 标签 = truth 链内 depth 比较 (depth 由 truth_chain_structure 给出, 链根=0; 只在同链对上定义)。
        # 动机: 0702 的"方向"是真值(ParticleIndex)免费送的; 这里改成让剪枝 MLP 从特征里学出来,
        # 这样方向在 data/MC 上口径一致、可迁移 (推理时也可用同头定方向)。
        if mode == "train" and self.dir_head_w > 0:
            _blk = self.model._blocks[-1]
            _dl = getattr(_blk, "edge_dir_logits", {}).get(('tracks', 'to', 'tracks'))
            if _dl is not None:
                try:
                    from wmpgnn.reconstruction.topk_selection import truth_chain_structure
                    _tt = batch[('tracks', 'to', 'tracks')]
                    _a, _b = _tt.edge_index[0], _tt.edge_index[1]
                    _lab = truth_chain_labels(_tt.y, _tt.edge_index, batch['tracks'].x.shape[0])
                    _d = truth_chain_structure(_tt.y, _tt.edge_index, batch['tracks'].batch,
                                               self.device)['depth']
                    _m = (_lab[_a] >= 0) & (_lab[_a] == _lab[_b]) & (_d[_a] != _d[_b])
                    if bool(_m.any()):
                        _ydir = (_d[_a] < _d[_b]).to(torch.float32).unsqueeze(-1)
                        loss["dir"] = F.binary_cross_entropy_with_logits(_dl[_m], _ydir[_m])
                        if "dir_acc_loss" in log:      # 方向头准确率 (借 _loss 后缀走平均/记录)
                            log["dir_acc_loss"].append(
                                float(((_dl[_m] > 0) == (_ydir[_m] > 0.5)).float().mean()))
                except Exception as _e:
                    print(f"[dir_head] WARN: {type(_e).__name__}: {_e}")

        # ==== [2026-09-24] 事件级链数辅助头 + 链级对比损失 ====
        # 真值链 id 直接来自 ft (0=b, 2=bbar, 1=background -> 背景置 -1)。
        # 动机: (a) 边剪枝标签本质是"簇关系", 对比损失直接在嵌入空间按链拉近/推远,
        #          属于"换目标函数"的新维度 (与调阈值/加特征这类同旋钮手段互补);
        #       (b) 计数头迫使事件级表征编码"这事件有几条链", 为 K 路解码/事件自适应阈值提供依据。
        if self.evt_count_w > 0 or self.chain_contrast_w > 0:
            _blk = self.model._blocks[-1]
            # 链级对比损失需要块的节点表征: 打开暂存开关 (幂等, 见 hetero_graph_network 的 _stash_emb)
            if self.chain_contrast_w > 0 and not getattr(self, "_stash_set", False):
                if hasattr(self, "model") and hasattr(self.model, "_blocks"):
                    for _b in self.model._blocks:
                        _b._stash_emb = True
                    self._stash_set = True
            if hasattr(batch['tracks'], 'ft'):
                _ft = batch['tracks'].ft.long().to(self.device)
                _bt = batch['tracks'].batch if 'batch' in batch['tracks'] else \
                    torch.zeros(_ft.shape[0], dtype=torch.long, device=self.device)
                _n_ev = int(_bt.max().item()) + 1
                _cid = torch.where(_ft == 1, torch.full_like(_ft, -1), _ft)
                _vv = _cid >= 0
                if self.evt_count_w > 0 and getattr(_blk, "_evt_count_logits", None) is not None:
                    _lab = torch.zeros(_n_ev, dtype=torch.long, device=self.device)
                    if bool(_vv.any()):
                        # 每事件的不同有效链 id 数 = 该事件的 B 数 (0/1/2, >=3 归到 3)
                        _uniq = torch.unique(torch.stack([_bt[_vv], _cid[_vv]], 1), dim=0)
                        _lab = torch.bincount(_uniq[:, 0], minlength=_n_ev).clamp(max=3).long()
                    loss["evt_count"] = F.cross_entropy(_blk._evt_count_logits, _lab)
                if self.chain_contrast_w > 0 and getattr(_blk, "_last_node_emb", None) is not None:
                    from wmpgnn.lightning_module.chain_contrastive import chain_contrastive_loss
                    loss["chain_contrast"] = chain_contrastive_loss(
                        _blk._last_node_emb, _cid, _bt, tau=self.chain_contrast_tau)
                    # [2026-09-26 FIX] 落日志: 此前该项从不记录, combined_loss 无法分项复核
                    # [2026-09-29 FIX] 去掉 `in` 守卫: on_train_epoch_end 会把 trn_log 重建成
                    # defaultdict(list), 守卫在 epoch>=1 恒为 False -> 只在第 0 轮记过一次。
                    if mode == "train" and self.trn_log is not None:
                        self.trn_log["chain_contrast_loss"].append(float(loss["chain_contrast"]))

        # 权重可配 (默认 1/1/33 与旧行为完全一致); 见 __init__ 中"剪枝损失再平衡"注释
        combined_loss = (self.lca_loss_w * loss["LCA"] + self.node_loss_w * loss["t_nodes"]
                         + self.edge_loss_w * loss["tt_edges"] + self.pv_asso_w * loss["pv_asso"])
        # ==== 边头 pairwise ranking 损失 (v614) ====
        if self.edge_rank_w > 0 and "tt_rank" in loss:
            combined_loss = combined_loss + self.edge_rank_w * loss["tt_rank"]
            # [2026-09-29 FIX] 去掉 `in` 守卫: trn_log 在 epoch_end 被重建成 defaultdict(list),
            # 守卫自 epoch>=1 起恒为 False -> 实测 train_edge_rank_loss 只记了 1/54 行。
            log["edge_rank_loss"].append(loss["tt_rank"].item())
        # ==== 方向头 (v618/v619) ====
        if self.dir_head_w > 0 and "dir" in loss:
            combined_loss = combined_loss + self.dir_head_w * loss["dir"]
            if "dir_loss" in log:
                log["dir_loss"].append(loss["dir"].item())
        # ==== 节点侧 ranking (v622) ====
        if self.node_rank_w > 0 and "node_rank" in loss:
            combined_loss = combined_loss + self.node_rank_w * loss["node_rank"]
            if "node_rank_loss" in log:
                log["node_rank_loss"].append(loss["node_rank"].item())

        # ==== 链级 min-pooling recall 损失 (点/边) ====
        # 对准"整条真值链被剪掉"的失败模式: 罚每条链里**最弱**的那个点/边。
        if mode == "train" and (self.chain_recall_w > 0 or self.chain_recall_edge_w > 0):
            _tt = batch[('tracks', 'to', 'tracks')]
            _cr_node, _cr_edge = chain_recall_loss(
                block.node_logits['tracks'], block.edge_logits[('tracks', 'to', 'tracks')],
                _tt.y, _tt.edge_index, batch['tracks'].x.shape[0],
                thr=self.chain_recall_thr, tau=self.chain_recall_tau)
            if _cr_node is not None:
                combined_loss = combined_loss + self.chain_recall_w * _cr_node
                # [2026-09-29 FIX] 去掉 `in` 守卫 (trn_log 每轮被重建成 defaultdict, 守卫自
                # epoch>=1 起恒 False -> 只在第 0 轮记过一次)。下同。
                log["chain_recall_loss"].append(_cr_node.item())
            if _cr_edge is not None:
                combined_loss = combined_loss + self.chain_recall_edge_w * _cr_edge
                log["chain_recall_edge_loss"].append(_cr_edge.item())
        # ==== [2026-10-09] "最弱环"链级存活损失 (chain_weakest) ====
        # 见 __init__ 同名说明。物理判据 = "整条真值链每个环节都对" (AND 语义, 一失毁全链),
        # 与逐边 BCE 的"平均正确率"目标不一致。这里对每条真值链用 softmin_γ 聚合链内 tt 边
        # **存活概率** (γ 越大越接近 min), 罚链内最弱的一环:
        #     surv_c = -log( mean_{e in c} exp(-γ p_e) ) / γ
        #     L      = -log( surv_c.clamp_min(1e-6) ).mean()   (对链取平均)
        # 事件隔离: 用 (事件 id, 节点链 id) 拼成唯一键, 绝不让不同事件的径迹进同一条链。
        # 缺省 chain_weakest_w=0 -> 整段不执行, 与旧行为逐位一致。
        if mode == "train" and self.chain_weakest_w > 0:
            try:
                # 注: truth_chain_labels 是本文件**模块级**的现成函数 (见文件顶部), 直接调用即可;
                #     topk_selection 里只有 truth_chain_roots / truth_chain_structure, 没有它。
                _tt = batch[('tracks', 'to', 'tracks')]
                # tt 边剪枝 logits: 与 edge_prune 的 BCE 用的是**同一个**输出 (逐边一个标量 logit);
                # 来源见 hetero_graph_network: edge_logits[et] = _edge_mlps[et](node_input[et].edges, ...)
                # -> 行序严格对应 batch[('tracks','to','tracks')].edge_index (与 y_edges 同序)。
                _cw_logits = self.model._blocks[-1].edge_logits[
                    ('tracks', 'to', 'tracks')].reshape(-1)
                _n_nodes = batch['tracks'].x.shape[0]
                # 每个节点/径迹的真值链 id (-1 = 背景)
                _lab = truth_chain_labels(_tt.y, _tt.edge_index, _n_nodes)
                _ea, _eb = _tt.edge_index[0].long(), _tt.edge_index[1].long()
                _ev = (batch['tracks'].batch if 'batch' in batch['tracks']
                       else torch.zeros(_n_nodes, dtype=torch.long, device=_lab.device))
                # 只计"两端属于同一非负链 id"的边
                _same = (_lab[_ea] >= 0) & (_lab[_ea] == _lab[_eb])
                if bool(_same.any()):
                    # 事件隔离: 唯一键 = 事件 id * n_nodes + 链 id (链 id < n_nodes, 跨事件必不碰撞)
                    _key = _ev[_ea[_same]] * _n_nodes + _lab[_ea[_same]]
                    _uniq, _inv = torch.unique(_key, return_inverse=True)
                    _p = torch.sigmoid(_cw_logits[_same]).clamp(1e-6, 1.0 - 1e-6)   # p_e
                    _wgt = torch.exp(-self.chain_weakest_gamma * _p)                 # exp(-γ p_e)
                    _sum = torch.zeros(_uniq.numel(), device=_p.device, dtype=_p.dtype
                                       ).scatter_add_(0, _inv.reshape(-1), _wgt)
                    _cnt = torch.zeros(_uniq.numel(), device=_p.device, dtype=_p.dtype
                                       ).scatter_add_(0, _inv.reshape(-1), torch.ones_like(_wgt))
                    _mean = _sum / _cnt.clamp_min(1.0)
                    _surv = -torch.log(_mean.clamp_min(1e-12)) / self.chain_weakest_gamma
                    # 只登记**未乘权重**的原始值 (权重在下方组合处统一乘, 避免重复计权)
                    loss["chain_weakest"] = -torch.log(_surv.clamp_min(1e-6)).mean()
                    # 落日志 (chain_weakest_loss 未在 init_logs 预置 -> 用 setdefault;
                    #  epoch_end_loggable 按 *_loss 后缀汇总 -> 会出 train_chain_weakest_loss)
                    log.setdefault("chain_weakest_loss", []).append(float(loss["chain_weakest"]))
            except Exception as e:
                print(f"[chain_weakest] WARN: {type(e).__name__}: {e}")
        if "chain_select" in loss and self.chain_scorer is not None:
            combined_loss = combined_loss + self.chain_loss_weight * loss["chain_select"]
        if "source" in loss and self.source_head_on:
            combined_loss = combined_loss + self.source_loss_weight * loss["source"]
        if "chain_lca" in loss and self.chain_lca_on:
            combined_loss = combined_loss + self.chain_lca_loss_weight * loss["chain_lca"]
        if "mass" in loss and self.mass_head_on:
            combined_loss = combined_loss + self.mass_loss_weight * loss["mass"]
        if "struct" in loss and self.struct_head_on:
            combined_loss = combined_loss + self.struct_head_weight * loss["struct"]
        if "mom" in loss and self.mom_head_on:
            combined_loss = combined_loss + self.mom_loss_weight * loss["mom"]
        if "pv_cluster" in loss and self.pv_cluster_on:
            combined_loss = combined_loss + self.pv_cluster_loss_weight * loss["pv_cluster"]
        # [2026-09-24] 事件级链数辅助头 / 链级对比损失
        if self.evt_count_w > 0 and "evt_count" in loss:
            combined_loss = combined_loss + self.evt_count_w * loss["evt_count"]
        if self.chain_contrast_w > 0 and "chain_contrast" in loss:
            combined_loss = combined_loss + self.chain_contrast_w * loss["chain_contrast"]
        # [2026-10-09] 最弱环链级存活损失 (chain_weakest): 原始值在 loss dict 中, 此处统一乘权重
        if self.chain_weakest_w > 0 and "chain_weakest" in loss:
            combined_loss = combined_loss + self.chain_weakest_w * loss["chain_weakest"]

        # 极端防御: 组合 loss 仍非有限或异常巨大时, 置为 0 损失, 避免梯度爆炸污染训练
        if not torch.isfinite(combined_loss) or combined_loss > 1e5:
            combined_loss = torch.zeros((), device=combined_loss.device, requires_grad=True)

        # Apply reco
        if mode == "test":
            # Attaching pruning information to graph
            if self.configs["node_prune"]:
                outputs["node_weights"] = block.node_weights["tracks"].squeeze()
            if self.configs["edge_prune"]:
                outputs["edge_weights"] = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()
            # Getting the PV decisions
            if self.configs["pv_asso"]:
                pv_asso_des = {"pred": block.edge_weights[('tracks', 'to', 'pvs')].squeeze(), "minIP": minip,
                               "true": y_pv_asso.squeeze(), "pv_filter": pv_filter}
                # 方案7b: 用同一 pv_cluster_head 的得分供重建分簇 (train/infer 对齐)
                if self.pv_cluster_on and pv_cluster_logits is not None:
                    pv_asso_des["cluster_pred"] = torch.sigmoid(pv_cluster_logits).squeeze(-1)
            else:
                pv_asso_des = None
            # 注入内置 scorer (联合训练挂载在 model 上), 供链级选择使用
            if self.chain_scorer is not None:
                self.evt_reco.chain_scorer = self.chain_scorer
            self.evt_reco.reconstruct_heavyhadrons(outputs, pv_des=pv_asso_des)

        """Logging"""
        log = loss_logging(log, loss, self.configs, mode="DFEI")

        log["combined_loss"].append(combined_loss.item())
        return combined_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, self.tst_log, mode="test")
        return {}

    def _chain_select_loss(self, batch, outputs, block):
        """第5个监督头: 候选衰变链选择 loss (train 时调用)。

        truth 正边连通分量 -> 真链 (正样本); 随机节点组合 -> 假链 (负样本);
        scorer 对每条链打分, BCE 到真/假标签。梯度经链内 node/edge 权重流回主干。
        """
        try:
            node_w = block.node_weights['tracks']                                  # [N_all, 1]
            edge_w = block.edge_weights[('tracks', 'to', 'tracks')]                # [E_all, 1]
            lca = outputs[('tracks', 'to', 'tracks')].edges                        # [E_all, 4] logits
            # 节点物理特征用模型输出 (encoder 后, 16 维), 与推理时链特征空间一致
            x = outputs['tracks'].x                                                # [N_all, 16]
            # 边物理特征必须用原始特征 (model 覆盖了 edges 为 LCA 输出)
            edges_phys = outputs[('tracks', 'to', 'tracks')].phys_edges            # [E_all, d_edge_phys]
            ei = batch[('tracks', 'to', 'tracks')].edge_index                      # [2, E_all]
            y = batch[('tracks', 'to', 'tracks')].y                                # [E_all] 或 [E_all,4]
            ft = batch['tracks'].ft                                                # [N_all]

            track_batch = batch['tracks'].batch                                    # [N_all]
            edge_batch = track_batch[ei[0]]                                        # [E_all] 边归属事件

            # 边 truth 二元 (兼容 one-hot 与标量)
            if y.dim() == 2 and y.shape[-1] > 1:
                y_bin = y.argmax(dim=-1) > 0
            else:
                y_bin = y > 0
            node_sig = ft != 1                                                      # 信号节点掩码

            n_evts = int(track_batch.max().item()) + 1
            scores, labels = [], []
            for evt_id in range(n_evts):
                tm = track_batch == evt_id
                em = edge_batch == evt_id
                if not tm.any():
                    continue
                nf = torch.cat([node_w[tm], x[tm]], dim=-1)                          # [N_i, 1+d_phys] (node_w 已含权重列)
                ef = torch.cat([edge_w[em], lca[em], edges_phys[em]], dim=-1)        # [E_i, 1+4+d_e]
                samples = build_chain_samples(ei[:, em], y_bin[em], node_sig[tm],
                                              int(tm.sum()), nf, ef)
                if samples is None:
                    continue
                pos_n, pos_e, pos_y, neg_n, neg_e, neg_y = samples
                dev = nf.device
                # 正样本链打分
                for nfe, efe in zip(pos_n, pos_e):
                    scores.append(self.chain_scorer(nfe.to(dev), efe.to(dev)))
                    labels.append(1.0)
                # 负样本链打分
                for nfe, efe in zip(neg_n, neg_e):
                    scores.append(self.chain_scorer(nfe.to(dev), efe.to(dev)))
                    labels.append(0.0)

            if not scores:
                return None
            s = torch.stack(scores)
            t = torch.tensor(labels, device=s.device)
            return self.chain_criterion(s, t)
        except Exception as e:
            print(f"[chain_select] WARN: {type(e).__name__}: {e}")
            return None

    def _source_loss(self, batch, outputs, block):
        """第6个监督头: 源检测 loss (Rumor Centrality 训练化)。

        truth 链内 rumor centrality 最大的节点 = 链根 (B 介子候选) -> 节点级标签;
        source_head 从节点特征 (tracks.x + node_weight) 预测"是否为根",
        BCE 到 truth 标签。梯度经 head 流回主干, 让主干学"根-叶结构"。
        """
        try:
            from wmpgnn.reconstruction.topk_selection import truth_chain_roots
            ei = batch[('tracks', 'to', 'tracks')].edge_index                      # [2, E_all]
            y = batch[('tracks', 'to', 'tracks')].y                                # [E_all] 或 [E_all,4]
            track_batch = batch['tracks'].batch                                    # [N_all]
            # 边 truth 二元 (兼容 one-hot 与标量)
            if y.dim() == 2 and y.shape[-1] > 1:
                y_bin = y.argmax(dim=-1)
            else:
                y_bin = y

            # 根节点标签: 每条 truth 链 (非背景边连通分量) 的 rumor centrality argmax
            roots = truth_chain_roots(y_bin, ei, track_batch, self.device)          # [N_all] 0/1

            # 节点特征: 最终 block 的 node_weight + 节点表征 -> [N_all, 1+16]
            node_w = block.node_weights['tracks']                                  # [N_all, 1]
            x = outputs['tracks'].x                                                # [N_all, 16]
            feat = torch.cat([node_w, x], dim=-1)
            logits = self.model.source_head(feat).squeeze(-1)                      # [N_all]
            return self.source_criterion(logits, roots)
        except Exception as e:
            print(f"[source_head] WARN: {type(e).__name__}: {e}")
            return None

    def _mass_loss(self, batch, outputs, orig_tracks_p):
        """第7个监督头: 边级 π-π 不变质量回归 (输出侧物理监督, train 时)。

        物理真值完全由数据自带 (无需额外标签): 从每条 tt 边两端的轨迹动量
        (orig_tracks_p, 归一化) 反归一化, 按双 π 假设 (E = sqrt(p² + m_pi²)) 计算
        不变质量 m_ij, 目标 = log10(m_ij/MeV) (动态范围 2.4~4.3, 匹配 SmoothL1 β=0.3);
        edge_mass_head 从 decoder 边表征 (latent_edges, 16维) 回归该目标。

        作用: 让 tt 边表征显式携带"动量-夹角"物理信息 (同母 ππ 对 -> 共振峰),
        辅助主干学习物理结构; 纯辅助监督, 不改主任务端到端目标。

        掩码: 任一端为未重建径迹 (px≈py≈pz≈-1 哨兵, VALUE_OR(-1)) 的边跳过。
        单维 px<0 不能做掩码 (真实轨迹 px<0 占比 ~48%)。
        """
        try:
            px = orig_tracks_p[:, 0] * self._mass_norm["px"][1] + self._mass_norm["px"][0]
            py = orig_tracks_p[:, 1] * self._mass_norm["py"][1] + self._mass_norm["py"][0]
            pz = orig_tracks_p[:, 2] * self._mass_norm["pz"][1] + self._mass_norm["pz"][0]

            # 未重建径迹哨兵: 三动量同时≈-1
            sentinel = ((px > -1.5) & (px < -0.5) & (py > -1.5) & (py < -0.5)
                        & (pz > -1.5) & (pz < -0.5))
            valid = ~sentinel

            ei = batch[('tracks', 'to', 'tracks')].edge_index
            a, b = ei[0], ei[1]
            edge_valid = valid[a] & valid[b]
            if not edge_valid.any():
                return None

            p1 = torch.stack([px[a], py[a], pz[a]], dim=-1)          # [E, 3]
            p2 = torch.stack([px[b], py[b], pz[b]], dim=-1)
            E1 = torch.sqrt((p1 ** 2).sum(-1) + self._m_pi ** 2)
            E2 = torch.sqrt((p2 ** 2).sum(-1) + self._m_pi ** 2)
            m2 = (E1 + E2) ** 2 - ((p1 + p2) ** 2).sum(-1)
            m = torch.sqrt(torch.clamp(m2, min=1.0))                 # MeV, 下限防 sqrt(0)
            target = torch.log10(m)                                  # ~2.4-4.3

            feat = outputs[('tracks', 'to', 'tracks')].latent_edges  # [E_all, 16]
            pred = self.model.edge_mass_head(feat).squeeze(-1)       # [E_all]
            return self.mass_criterion(pred[edge_valid], target[edge_valid])
        except Exception as e:
            print(f"[mass_head] WARN: {type(e).__name__}: {e}")
            return None

    def _struct_loss(self, batch, outputs, block):
        """第8个监督头: 节点结构监督 (深度 + Rumor Centrality 回归, train 时)。

        用户提议: source_head 只预测"链根"(1 bit), 信息量低; 改为连续结构监督,
        让主干学"节点在衰变树中的位置/层级":
          - depth 主头: 节点到链根的拓扑距离 (链内归一化 [0,1], 根=0)
          - rc 辅头:    节点 logR 链内 min-max 归一化 [0,1] (质心=1, 叶子=0)
        与 LCA 边分类 (class1 母子/class2 同母/class3 祖孙) 形成一致性约束:
        class1 边 depth 差 1, class2 差 0, class3 差 >=2 -> 模型被迫产出全局自洽的树。

        标签由 truth_chain_structure 计算 (纯 truth, 无模型噪声), 只监督链内节点。
        """
        try:
            from wmpgnn.reconstruction.topk_selection import truth_chain_structure
            ei = batch[('tracks', 'to', 'tracks')].edge_index            # [2, E_all]
            y = batch[('tracks', 'to', 'tracks')].y                      # [E_all] 或 [E_all,4]
            track_batch = batch['tracks'].batch                          # [N_all]
            if y.dim() == 2 and y.shape[-1] > 1:
                y_bin = y.argmax(dim=-1)
            else:
                y_bin = y
            st = truth_chain_structure(y_bin, ei, track_batch, self.device)
            m = st['in_chain']
            if not m.any():
                return None

            node_w = block.node_weights['tracks']                        # [N_all, 1]
            x = outputs['tracks'].x                                      # [N_all, 16]
            feat = torch.cat([node_w, x], dim=-1)
            out = self.model.node_struct_head(feat)                      # [N_all, 2]
            d_loss = self.struct_criterion(out[:, 0][m], st['depth'][m])
            r_loss = self.struct_criterion(out[:, 1][m], st['rc'][m])
            return d_loss + self.rc_head_weight * r_loss
        except Exception as e:
            print(f"[struct_head] WARN: {type(e).__name__}: {e}")
            return None

    def _mom_loss(self, batch, outputs, block, orig_tracks_p):
        """第9个监督头: 节点级动量回归 (mom_head, train 时)。

        PhyIP 探针发现: encoder 的 graph_norm+ReLU 把输入动量打散, 节点表征
        几乎不携带线性可读的动量 (线性/MLP 探针 R²≈0)。方案 A: 监督节点表征
        输出归一化输入动量 [px_n, py_n, pz_n] (encoder 的直接输入), 强制节点
        表征线性携带运动学信息。目标纯由数据自带 (tracks.x 前3维), 无需标签。

        掩码: 未重建径迹 (px≈py≈pz≈-1 哨兵) 跳过。
        """
        try:
            px = orig_tracks_p[:, 0] * self._mom_norm["px"][1] + self._mom_norm["px"][0]
            py = orig_tracks_p[:, 1] * self._mom_norm["py"][1] + self._mom_norm["py"][0]
            pz = orig_tracks_p[:, 2] * self._mom_norm["pz"][1] + self._mom_norm["pz"][0]
            sentinel = ((px > -1.5) & (px < -0.5) & (py > -1.5) & (py < -0.5)
                        & (pz > -1.5) & (pz < -0.5))
            valid = ~sentinel
            if not valid.any():
                return None

            x = outputs['tracks'].x                                # [N_all, 16]
            pred = self.model.node_mom_head(x)                     # [N_all, 3]
            target = orig_tracks_p                                 # [N_all, 3] 归一化输入动量
            return self.mom_criterion(pred[valid], target[valid])
        except Exception as e:
            print(f"[mom_head] WARN: {type(e).__name__}: {e}")
            return None

    def _chain_lca_loss(self, batch, outputs, block):
        """方案5: 链内 LCA 一致性辅助损失 (chain_lca_filter 训练化)。

        "最物理"判据训练化: 真链的链内边应是模型**高置信**的非背景边。
        truth 链内边 = LCA y>0 的边; 对该边被判类别 (argmax) 的 softmax 概率
        施加 hinge loss (低于 margin 则惩罚) -> 模型主动产出物理自洽的链,
        推理时 chain_lca_filter 的 conf 阈值自然更高、过滤更准。

        方案6 (chain_lca_ce=true): 额外对链内边施加 LCA 类别 CE, 直接监督
        链内边类别正确性 (class1/2/3), 对抗 class0 对主 LCA loss 的稀释。
        """
        try:
            import torch.nn.functional as F
            lca_logits = outputs[('tracks', 'to', 'tracks')].lca              # [E_all, 4]
            y = batch[('tracks', 'to', 'tracks')].y                            # [E_all] 或 [E_all,4]
            if y.dim() == 2 and y.shape[-1] > 1:
                y_cat = y.argmax(dim=-1)
            else:
                y_cat = y.long()
            y_bin = y_cat > 0                                                  # 链内边 (真类别 1/2/3)
            if not y_bin.any():
                return None
            probs = F.softmax(lca_logits[y_bin], dim=-1)                       # [n_chain_e, 4]
            conf = probs.max(dim=-1).values                                    # 被判类别概率
            margin = self.chain_lca_margin
            # hinge: conf < margin 的边受罚 (0.5*|conf-margin|² 让<margin的边远离)
            gap = torch.clamp(margin - conf, min=0.0)
            loss = (gap ** 2).mean()
            if self.chain_lca_ce:
                # 链内边类别 CE (target 1/2/3 合法), 放大结构边分类监督
                ce = F.cross_entropy(lca_logits[y_bin], y_cat[y_bin], reduction='mean')
                loss = loss + self.chain_lca_ce_weight * ce
            return loss
        except Exception as e:
            print(f"[chain_lca] WARN: {type(e).__name__}: {e}")
            return None

    def on_train_epoch_start(self):
        # ==== B2: 温度退火 (epoch 0: tau_start -> epoch b2_tau_epochs: tau_end) ====
        if getattr(self.model, "_b2_enable", False):
            gn = self.configs_gn
            tau_start = float(gn.get("b2_tau_start", 1.0))
            tau_end = float(gn.get("b2_tau_end", 0.1))
            tau_epochs = max(int(gn.get("b2_tau_epochs", 100)), 1)
            frac = min(float(self.current_epoch) / tau_epochs, 1.0)
            tau = tau_start + (tau_end - tau_start) * frac
            self.model.set_b2_tau(tau)
            if self.current_epoch % 10 == 0:
                print(f"[B2] epoch {self.current_epoch}: tau = {tau:.4f}", flush=True)

        # ==== 方案7b (v41): PV 分簇温度退火 + truth->cluster 课程式过渡 ====
        # 均相对本次训练起点计 (续训时 current_epoch 是 v38 的绝对 epoch, 不能直接用)
        if self.pv_cluster_on:
            self._pv_run_epoch += 1
            re = float(self._pv_run_epoch)
            # 温度退火: 1.0(探索,Gumbel采样) -> 0.1(收敛 argmax), 与 B2 同谱系
            frac = min(re / self.pv_tau_epochs, 1.0)
            self.pv_tau = self.pv_tau_start + (self.pv_tau_end - self.pv_tau_start) * frac
            # 课程式过渡: alpha 1.0(truth 分簇, 稳定热启动) -> 0.0(cluster 头)
            self.pv_curriculum_alpha = max(1.0 - re / self.pv_cluster_curriculum_epochs, 0.0)
            if self.current_epoch % 10 == 0 or self._pv_run_epoch == 1:
                print(f"[pv_cluster] run_epoch {self._pv_run_epoch}: tau={self.pv_tau:.3f} "
                      f"alpha={self.pv_curriculum_alpha:.3f}", flush=True)

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False, on_epoch=True, on_step=False)

    # ==== [2026-09-29] val 剪枝 AUC/AP: 累积 / 汇总 ====
    def _vpm_reset(self):
        self._vpm = {"n_evt": 0, "nsc": [], "nlab": [], "esc": [], "elab": [], "ehard": [], "edp": []}

    def _vpm_accumulate(self, batch, block):
        """累积最后一个 block 的节点/边剪枝分数与标签 (固定前 vpm_events 个事件)。

        除全局池外, 还记录两个"主判据池"的成员掩码 (2026-09-29 v3):
          - 难池   : 真边 ∪ {两端都是真值径迹的假边}。全局池 99.7% 是背景-背景这类
                     闭眼可分的负例, 全局 AUC 基本是常数 -> 必须在难池里看。
          - 工作点池: 两端点都通过点剪枝阈值的边 = 推理时真正进入边决策的 population。
        两个掩码都只需 y 与 ft, 不需要真值链重建, 因此 val 每轮都算得起。

        注: 按 batch 累积, 因此实际事件数会略微超过 vpm_events (最多一个 batch), 日志里打印真实值。
        """
        if self._vpm is None or self._vpm["n_evt"] >= self.vpm_events:
            return
        tb = batch["tracks"].batch
        n_ev = int(tb.max().item()) + 1 if tb.numel() else 0
        with torch.no_grad():
            ei = batch[("tracks", "to", "tracks")].edge_index.detach().cpu().numpy()
            nsc = block.node_weights["tracks"].detach().float().cpu().numpy().reshape(-1)
            esc = block.edge_weights[("tracks", "to", "tracks")].detach().float().cpu().numpy().reshape(-1)
            nlab = (batch["tracks"].ft != 1).detach().cpu().numpy().astype(np.int8)
            elab = (batch[("tracks", "to", "tracks")].y.reshape(-1) > 0).detach().cpu().numpy().astype(np.int8)
            ehard = (((nlab[ei[0]] == 1) & (nlab[ei[1]] == 1)) | (elab == 1))
            edp = (nsc[ei[0]] > self.vpm_node_thr) & (nsc[ei[1]] > self.vpm_node_thr)
        self._vpm["nsc"].append(nsc)
        self._vpm["nlab"].append(nlab)
        self._vpm["esc"].append(esc)
        self._vpm["elab"].append(elab)
        self._vpm["ehard"].append(ehard)
        self._vpm["edp"].append(edp)
        self._vpm["n_evt"] += n_ev

    def on_validation_epoch_start(self):
        if self.vpm_on:
            self._vpm_reset()

    def _log_val_prune_metrics(self):
        d = self._vpm
        if not self.vpm_on or not d or not d["esc"]:
            return
        esc, elab = np.concatenate(d["esc"]), np.concatenate(d["elab"])
        nsc, nlab = np.concatenate(d["nsc"]), np.concatenate(d["nlab"])
        ehard, edp = np.concatenate(d["ehard"]), np.concatenate(d["edp"])
        ap_hard = prune_ap(esc[ehard], elab[ehard]) if ehard.any() else float("nan")
        dl = elab[edp]
        ap_dp = prune_ap(esc[edp], dl) if (dl.sum() and dl.sum() < len(dl)) else float("nan")
        vals = {
            "prune_edge_auc": prune_auc(esc, elab),
            "prune_edge_ap": prune_ap(esc, elab),
            "prune_node_auc": prune_auc(nsc, nlab),
            "prune_node_ap": prune_ap(nsc, nlab),
            "prune_ap_hard": ap_hard,          # 主判据 1: 难池 AP
            "prune_ap_dp": ap_dp,              # 主判据 2: 工作点池 AP
        }
        for k, v in vals.items():
            if np.isfinite(v):
                self.log(f"val_{k}", float(v), prog_bar=(k == "prune_ap_hard"),
                         on_epoch=True, on_step=False)
        # monitor 用的别名 val_prune_ap = 难池 AP (主判据 1); 旧配置无需改动
        if np.isfinite(ap_hard):
            self.log("val_prune_ap", float(ap_hard), on_epoch=True, on_step=False)
        print(f"[vpm] val 剪枝指标 (events={d['n_evt']}, 边 {len(elab)} 条 / 点 {len(nlab)} 个): "
              f"全局 edge AUC={vals['prune_edge_auc']:.4f} AP={vals['prune_edge_ap']:.4f} | "
              f"难池 AP={ap_hard:.4f} (n={int(ehard.sum())}) | 工作点池 AP={ap_dp:.4f} (n={int(edp.sum())}) | "
              f"node AUC={vals['prune_node_auc']:.4f} AP={vals['prune_node_ap']:.4f}", flush=True)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)
        self._log_val_prune_metrics()

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        # grab from the class and save to disk
        sig_df, evt_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        evt_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/event_reco_df_{self.signal}.csv', index=False)
        if self.configs["LCA"]:
            obtain_reco_accuracy(sig_df, self.version, self.signal, self.log_dir, model="DFEI")
            # LCAG 分类准确率 (论文 Table1), 追加到 info txt
            lca_num_keys = [k for k in self.tst_log if k.startswith("LCA_class") and k.endswith("_num")]
            if lca_num_keys:
                info_path = f"{self.log_dir}/DFEI/version_{self.version}/info_{self.signal}_reco.txt"
                with open(info_path, "a") as f:
                    f.write("=" * 50 + "\n")
                    f.write("LCAG classification accuracy (Table1):\n")
                    for i in range(4):
                        nums = list(self.tst_log.get(f"LCA_class{i}_num", []))
                        total_num = sum(nums)
                        if total_num == 0:
                            f.write(f"  LCA_class{i}: n=0\n")
                            continue
                        preds = []
                        for j in range(4):
                            fracs = list(self.tst_log.get(f"LCA_class{i}_pred_class{j}", []))
                            if not fracs:
                                continue
                            # 按每批数量加权
                            weighted = sum(n * p for n, p in zip(nums, fracs)) / total_num
                            preds.append(f"pred{j}={weighted*100:.2f}%")
                        f.write(f"  LCA_class{i} (n={total_num}): " + " ".join(preds) + "\n")

        if self.configs["plt_nodes"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                             [f"NN_nodes_{i}_decision", "sig", "bkg"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                               [f"NN_nodes_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
        if self.configs["plt_edges"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                             [f"NN_edges_{i}_decision", "sig", "bkg"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                               [f"NN_edges_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
        if self.configs["plt_pvs"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_pv_asso_score_{i}"], self.tst_log[f"bkg_pv_asso_score_{i}"],
                             [f"NN_pv_asso_{i}_decision", "correct", "false"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_pv_asso_score_{i}"], self.tst_log[f"bkg_pv_asso_score_{i}"],
                               [f"NN_pv_asso_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
            # Get the PV association performance
            log = self.evt_reco.log
            pv_perf = {}
            pv_perf["all_tracks"] = plot_pv_missasso(log["pv_corr_ml"], log["pv_corr_ip"], log["pv_total"], log["npvs"],
                                                     self.version, self.signal, log_dir=self.log_dir)
            pv_sig_tracks = plot_sig_pv_missasso(sig_df, self.version, self.signal, log_dir=self.log_dir)
            pv_perf.update(pv_sig_tracks)
            pv_perf["sig_b_system"] = plot_sig_b_system_pv_missasso(sig_df, self.version, self.signal,
                                                                    log_dir=self.log_dir)
            acc_pv_asso(pv_perf, self.version, self.signal, self.log_dir, model="DFEI")
