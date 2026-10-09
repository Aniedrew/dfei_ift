"""集合预测 (MaskFormer / DETR 式) 次级顶点重建原型  [2026-10-09]

为什么做这个 (动机)
    当前仓库里所有 ML 顶点重建都是 "边分类 (pairwise) + 后处理取连通分量" 的范式。
    文献 (以及本仓库 analyze_vertex_assoc.py 的结论) 指出: 这种做法**无法完整重建一个
    事件里的多条真值链** —— 一个事件里有多条 B (多 B 事件) 时, 边分类只看"这一对像不像
    同一条链", 没有对"事件里到底有几条链、哪条链归哪些径迹"做**全局的、排他的**决策。
    这正是多 B 的失效模式。

    本原型换个范式: **直接预测 "事件里有几条链" 以及 "每条链的径迹归属 mask"**,
    用一组可学习 slot (query) 去和 track 做注意力, 得到 (K, ntr) 的 mask logits 与
    每个 slot 的 objectness; 训练时用匈牙利/Sinkhorn 把 slot 与真值链**全局匹配**
    (DETR 的标准做法)。评估用**顶点级**指标 (顶点发现效率/假顶点率/链级完美率),
    而不是边级 AUC —— 因为真正要回答的是 "事件里的链能不能被完整找出来"。

    并给出 "边分类 MLP + 连通分量" 的同指标基线, 直接对比两种范式。

数据: report_figs/feat_ceiling_nb.npz (只读, 不改)。
运行:
    python3 analyze_vertex_setpred.py --npz report_figs/feat_ceiling_nb.npz \
        --epochs 60 --seed 0 --slots 6 --assign hungarian [--min_nb 2] \
        [--out report_figs/setpred.json]
"""
# [OMP] 必须在 import torch 之前设置, 否则 OpenMP 线程数不生效。登录节点核多, 别占满。
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import math

import numpy as np

# 复用现成的函数 (不要重复实现): 见 analyze_vertex_assoc.py
#   load_npz / split_events / ap_score / _auc / _multib_mask
from analyze_vertex_assoc import load_npz, split_events, ap_score, auc as _auc, _multib_mask

import torch
import torch.nn as nn

torch.set_num_threads(4)

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

NB_BOXES = ["1", "2", ">=3"]


# ====================================================================== 真值链
def gt_chains(n, ii, jj, yy):
    """对单个事件, 由 y==1 的边构成子图, 其**连通分量**就是真值链 (顶点=局部径迹 id)。

    只保留顶点数 >=2 的分量 (单径迹分量不构成一条链, 与 nb 的定义一致)。
    返回 list[np.ndarray[bool]](每个是长度 n 的径迹指示向量)。
    """
    par = list(range(n))

    def find(x):
        r = x
        while par[r] != r:
            r = par[r]
        while par[x] != r:            # 路径压缩
            par[x], x = r, par[x]
        return r

    for k in np.nonzero(yy == 1)[0]:
        a, b = find(int(ii[k])), find(int(jj[k]))
        if a != b:
            par[a] = b
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    chains = []
    for v in comp.values():
        if len(v) >= 2:
            m = np.zeros(n, dtype=bool)
            m[v] = True
            chains.append(m)
    return chains


def build_events(grp, Xs, t0, t1, ntr, y, nb):
    """把边级数组按事件打包成 per-event 结构 (含节点特征 / 真值链)。

    节点特征: npz 只有**边**特征, 所以节点的表示用 "该径迹所有关联边的 X 的均值 + 最大池化"
    拼接而成 (2*47=94 维)。均值捕获平均行为, 最大池化捕获最突出的单条边。
    """
    events = {}
    for e in np.unique(grp):
        m = np.nonzero(grp == e)[0]
        n = int(ntr[m[0]])
        ii = t0[m].astype(np.int64)
        jj = t1[m].astype(np.int64)
        # ---- 节点级池化 (mean + max) ----
        s = np.zeros((n, Xs.shape[1]), np.float64)
        mx = np.full((n, Xs.shape[1]), -np.inf, np.float64)
        cnt = np.zeros(n, np.float64)
        np.add.at(s, ii, Xs[m]); np.add.at(s, jj, Xs[m])
        np.add.at(cnt, ii, 1); np.add.at(cnt, jj, 1)
        np.maximum.at(mx, ii, Xs[m]); np.maximum.at(mx, jj, Xs[m])
        mean = s / np.maximum(cnt, 1)[:, None]
        mx[~np.isfinite(mx)] = 0.0
        nodef = np.concatenate([mean, mx], axis=1).astype(np.float32)
        events[int(e)] = dict(
            nodef=nodef, n=n, ii=ii, jj=jj, yy=y[m].astype(np.int64),
            chains=gt_chains(n, ii, jj, y[m]), nb=int(nb[m[0]]))
    return events


def ev_column_standardize(events, ev_list):
    """用 train 事件统计量对节点特征做逐列标准化 (稳定训练)。"""
    allf = np.concatenate([events[e]["nodef"] for e in ev_list], axis=0)
    mu = allf.mean(0, keepdims=True)
    sd = allf.std(0, keepdims=True) + 1e-6
    for e in events:
        events[e]["nodef"] = ((events[e]["nodef"] - mu) / sd).astype(np.float32)


# ====================================================================== 模型
class SetPredictor(nn.Module):
    """MaskFormer/DETR 式集合预测: K 个可学习 slot 查询 + track 注意力。

    - 节点特征 -> 小 MLP -> track embedding h (ntr, d)
    - slot 查询对 tracks 做多头注意力 (nn.MultiheadAttention) -> slot 上下文
    - mask logits (K, ntr) = slot_q @ track_k^T / sqrt(d): 每个 slot 对每条径迹的归属打分
    - objectness (K,) : 该 slot 是否对应一条真实链
    - 可选 slot-track bias: 由 "该 track 关联边的平均 X" 投影到 K 维, 加到 mask logits 上,
      即把边特征直接作为注意力的 pair/track 偏置 (简化版, 见下方注释)。
    """

    def __init__(self, d_node, n_edge, d=64, heads=4, slots=6, edge_bias=True):
        super().__init__()
        self.K = slots
        self.d = d
        self.node_proj = nn.Sequential(
            nn.Linear(d_node, 128), nn.GELU(), nn.Linear(128, d))
        # track 自注意力: 让每条径迹看到同事件其它径迹 (上下文), 再被 slot 查询。
        self.track_norm = nn.LayerNorm(d)
        self.track_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.slots = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        # mask logits 用 per-(slot, track) 的 MLP (MaskFormer 做法): 输入 [slot, track, slot*track]。
        #   比单纯点积更强, 能表达 "slot k 关心 track t 的某几个特征分量" 这种交互。
        self.mask_mlp = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.obj = nn.Linear(d, 1)
        # [可选 bias] 由 track 的平均边特征 (n_edge 维) 投影到 K 个 slot 的 logit 偏置。
        #   说明: 完整的 pair-bias (用**两条径迹之间**的边特征做偏置, 如 Graphormer) 需要
        #   slot 在 forward 时已经选定了 track 对, 而这里 slot 只是 query、尚无可微的硬选择,
        #   所以退化为 "track 级" 偏置 (每个 slot 对每条 track 加一个常量)。若嫌复杂可 --no_edge_bias。
        self.edge_bias = edge_bias
        if edge_bias:
            self.bias_proj = nn.Linear(n_edge, slots)

    def forward(self, nodef, node_mean):
        # nodef: (ntr, d_node);  node_mean: (ntr, n_edge) 平均边特征 (nodef 的前半段)
        h = self.node_proj(nodef)                       # (ntr, d)
        hs = self.track_norm(h + self.track_attn(h.unsqueeze(0), h.unsqueeze(0),
                                                 h.unsqueeze(0))[0].squeeze(0))
        q = self.slots.unsqueeze(0)                     # (1, K, d)
        ctx, _ = self.attn(q, hs.unsqueeze(0), hs.unsqueeze(0))  # (1, K, d)
        ctx = ctx.squeeze(0)                            # (K, d)
        K = ctx.shape[0]; ntr = hs.shape[0]
        ck = ctx.unsqueeze(1).expand(K, ntr, self.d)
        ht = hs.unsqueeze(0).expand(K, ntr, self.d)
        logits = self.mask_mlp(torch.cat([ck, ht, ck * ht], -1)).squeeze(-1)  # (K, ntr)
        if self.edge_bias:
            logits = logits + self.bias_proj(node_mean).t()   # (K, ntr)
        obj = self.obj(ctx).squeeze(-1)                 # (K,)
        return logits, obj


# ====================================================================== 匹配代价
def iou_matrix(pred_bool, gt_masks):
    """pred_bool: (K, ntr) bool; gt_masks: (G, ntr) bool -> IoU (K, G)。用 torch 算便于 batched。"""
    K, ntr = pred_bool.shape
    G = gt_masks.shape[0]
    if G == 0:
        return torch.zeros((K, 0))
    inter = (pred_bool.float() @ gt_masks.float().t())          # (K, G)
    pu = pred_bool.float().sum(1, keepdim=True)                 # (K,1)
    gu = gt_masks.float().sum(1, keepdim=True).t()              # (1,G)
    union = pu + gu - inter
    return inter / union.clamp(min=1e-6)


def greedy_match(cost):
    """scipy 不可用时的 fallback: 按代价从小到大贪心, 每个 slot / 每条链只用一次。"""
    K, G = cost.shape
    pairs = []
    used_k, used_g = set(), set()
    for idx in np.argsort(cost, axis=None):
        k, g = divmod(int(idx), G)
        if k in used_k or g in used_g:
            continue
        used_k.add(k); used_g.add(g)
        pairs.append((k, g))
    return pairs


def hungarian_match(cost):
    """cost: (K, G) numpy, 越小越好。返回匹配对 list[(k, g)] (每 slot/链至多用一次)。

    DETR 语义: 这不是 "逐对分类", 而是给每个预测 slot 找**唯一**的一条真值链, 反之亦然 ——
    这正是 "集合预测" 与 "边分类" 的本质区别 (后者允许一条链被多条边重复解释)。
    """
    K, G = cost.shape
    if K == 0 or G == 0:
        return []
    if HAS_SCIPY:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    return greedy_match(cost)


def sinkhorn_weights(cost, tau, iters, dust_logit):
    """log-domain Sinkhorn: 把 (K, G) 匹配代价变成**软匹配权重** (含 dustbin)。

    语义: 把 "slot x 链" 的运输问题解成一个双随机 (双边际) 矩阵。
      - 代价 -> logit: S = -cost/tau
      - 扩张成 (K+1, G+1): 末行 / 末列是 **dustbin** —— 允许 "某个 slot 不对应任何链"
        或 "某条链没人认领"。没有 dustbin 就会强迫每个 slot 都匹配一条链 (硬性排他),
        在链数 < slot 数时会把噪声也强行摊派成目标, 所以 dustbin 是必须的。
      - 行列边缘取均匀 (各 1/(K+1), 1/(G+1)), 迭代归一化得到软分配 P。
    返回 P (K+1, G+1) 的 tensor。P[:K,:G] 是真实 (slot, 链) 的软权重。
    """
    K, G = cost.shape
    dev = cost.device
    S = torch.full((K + 1, G + 1), float(dust_logit),
                   device=dev, dtype=cost.dtype)
    S[:K, :G] = -cost / tau
    logr = torch.full((K + 1,), -math.log(K + 1), device=dev, dtype=cost.dtype)
    logc = torch.full((G + 1,), -math.log(G + 1), device=dev, dtype=cost.dtype)
    u = torch.zeros(K + 1, device=dev, dtype=cost.dtype)
    v = torch.zeros(G + 1, device=dev, dtype=cost.dtype)
    for _ in range(iters):
        u = logr - torch.logsumexp(S + v.unsqueeze(0), dim=1)
        v = logc - torch.logsumexp(S + u.unsqueeze(1), dim=0)
    P = torch.exp(S + u.unsqueeze(1) + v.unsqueeze(0))
    return P


# ====================================================================== 损失
def setpred_loss(net, ev, args, dev, bce):
    """单个事件的 DETR 式损失。返回 (loss tensor, 诊断标量)。"""
    nodef = torch.as_tensor(ev["nodef"], device=dev)
    node_mean = nodef[:, :ev["nodef"].shape[1] // 2]        # 均值段 = 平均边特征
    logits, obj = net(nodef, node_mean)                     # (K, ntr), (K,)
    K, ntr = logits.shape
    G = len(ev["chains"])

    # 目标: objectness 全零 (未匹配 slot = 无目标)
    obj_t = torch.zeros(K, device=dev)

    if G == 0:
        # 事件里没有链: 只监督 objectness=0
        return bce(obj, obj_t), dict(mask=0.0, obj=float(bce(obj, obj_t).item()), G=0)

    gt = torch.as_tensor(np.stack(ev["chains"], 0), device=dev)   # (G, ntr) bool

    with torch.no_grad():
        # 匹配用的预测 mask 用硬阈值 (DETR: 匹配本身不可导, 当目标分配处理)
        pred_bool = torch.sigmoid(logits) > 0.5
        cost = 1.0 - iou_matrix(pred_bool, gt)              # (K, G) 越小越好

    if args.assign == "hungarian":
        pairs = hungarian_match(cost.detach().cpu().numpy())
        if pairs:
            ks = torch.as_tensor([k for k, _ in pairs], device=dev)
            gs = torch.as_tensor([g for _, g in pairs], device=dev)
            loss_mask = bce(logits[ks], gt[gs].float())
            obj_t[ks] = 1.0
        else:
            loss_mask = torch.zeros((), device=dev)
        loss = loss_mask + bce(obj, obj_t)
        return loss, dict(mask=float(loss_mask.item()), obj=float(bce(obj, obj_t).item()), G=G)

    # ---- Sinkhorn 软匹配 ----
    P = sinkhorn_weights(cost, args.match_tau, args.match_iters, args.dust_logit)
    W = P[:K, :G]                       # slot -> 真实链 的软权重
    Pdust = P[:K, G]                    # slot -> dustbin (不对应任何链)
    # objectness 软目标 = 该 slot 分给真实链的质量 / (真实链 + dustbin)
    obj_t = W.sum(1) / (W.sum(1) + Pdust + 1e-8)
    obj_t = obj_t.clamp(0.0, 1.0)
    tot = W.sum() + 1e-8
    # mask 损失按软权重加权: 越确信的 (slot, 链) 对, mask BCE 权重越大
    loss_mask = torch.zeros((), device=dev, dtype=logits.dtype)
    for g in range(G):
        bce_g = nn.functional.binary_cross_entropy_with_logits(
            logits, gt[g].float().unsqueeze(0).expand(K, ntr), reduction="none")
        loss_mask = loss_mask + (bce_g.mean(1) * W[:, g]).sum()
    loss_mask = loss_mask / tot
    loss = loss_mask + bce(obj, obj_t)
    return loss, dict(mask=float(loss_mask.item()), obj=float(bce(obj, obj_t).item()), G=G)


# ====================================================================== 指标
def box_of(nb):
    return "1" if nb == 1 else ("2" if nb == 2 else ">=3")


def _new_box():
    return dict(n_gt=0, n_found=0, n_pred=0, n_fake=0, n_perfect=0)


def evaluate_setpred(net, ev_list, events, dev):
    """顶点级评估集合预测模型。

    顶点级 (而不是边级) 是因为目标就是 "事件里的真值链能不能被完整找出":
      1) 顶点发现效率: 对每条真值链, 若有 slot (objectness>0.5) 且 IoU>=0.5 -> 找到
      2) 假顶点率: 预测为 object 的 slot 中, 与所有真值链 IoU<0.5 的比例
      3) 链级完美率: 预测 slot 的 mask 与某个真值链**完全一致** (IoU==1.0) 的比例
      并按 nb (1 / 2 / >=3) 分层, 因为多 B 才是目标场景。
    """
    net.eval()
    boxes = {b: _new_box() for b in NB_BOXES}
    allb = _new_box()
    es, el = [], []
    with torch.no_grad():
        for e in ev_list:
            ev = events[e]
            nodef = torch.as_tensor(ev["nodef"], device=dev)
            node_mean = nodef[:, :nodef.shape[1] // 2]        # 均值段 = 平均边特征
            logits, obj = net(nodef, node_mean)
            mp = torch.sigmoid(logits).cpu().numpy()            # (K, ntr)
            op = torch.sigmoid(obj).cpu().numpy()               # (K,)
            chains = ev["chains"]
            box = box_of(ev["nb"])

            # ---- 预测链: objectness>0.5 的 slot ----
            pred_idx = np.nonzero(op > 0.5)[0]
            pred_masks = [mp[k] > 0.5 for k in pred_idx]
            # IoU (n_pred x n_gt)
            n_gt = len(chains)
            ious = np.zeros((len(pred_masks), n_gt))
            for pi, pm in enumerate(pred_masks):
                for gi, gm in enumerate(chains):
                    inter = np.logical_and(pm, gm).sum()
                    union = np.logical_or(pm, gm).sum()
                    ious[pi, gi] = inter / union if union > 0 else 0.0

            for b in (box, "ALL"):
                tgt = boxes[b] if b != "ALL" else allb
                # 1) 发现效率
                tgt["n_gt"] += n_gt
                for gi in range(n_gt):
                    if ious.shape[0] > 0 and ious[:, gi].max() >= 0.5:
                        tgt["n_found"] += 1
                # 2) 假顶点率 / 3) 完美率 (分母都是预测出的 slot)
                tgt["n_pred"] += len(pred_masks)
                for pi in range(len(pred_masks)):
                    best = ious[pi].max() if n_gt > 0 else 0.0
                    if best < 0.5:
                        tgt["n_fake"] += 1
                    if best >= 1.0 - 1e-9:
                        tgt["n_perfect"] += 1

            # ---- 边分数: max_k obj_k * mask_k[i] * mask_k[j] (由顶点预测导出) ----
            for r in range(len(ev["ii"])):
                i, j = int(ev["ii"][r]), int(ev["jj"][r])
                sc = 0.0
                for k in pred_idx:
                    v = op[k] * mp[k, i] * mp[k, j]
                    if v > sc:
                        sc = v
                es.append(sc); el.append(ev["yy"][r])
    return _summarize(boxes, allb), np.array(es), np.array(el)


def _summarize(boxes, allb):
    out = {}
    for name, t in ([(b, boxes[b]) for b in NB_BOXES] + [("ALL", allb)]):
        out[name] = dict(
            n_gt=t["n_gt"],
            finding_eff=(t["n_found"] / t["n_gt"]) if t["n_gt"] > 0 else float("nan"),
            fake_rate=(t["n_fake"] / t["n_pred"]) if t["n_pred"] > 0 else float("nan"),
            perfect_rate=(t["n_perfect"] / t["n_pred"]) if t["n_pred"] > 0 else float("nan"),
            n_pred=t["n_pred"], n_found=t["n_found"],
            n_fake=t["n_fake"], n_perfect=t["n_perfect"],
        )
    return out


def val_finding_eff(net, ev_list, events, dev):
    """轻量 val 指标: 全体真值链的顶点发现效率 (用于挑选最优 checkpoint)。

    为什么不用 val loss 选模型: 训练目标 (DETR 匹配损失) 与最终关心的**顶点级**指标
    (链能不能被完整找到) 并不单调相关 —— 实测 val loss 最低的 epoch 其发现效率并不最高。
    所以直接用发现效率做早停/选模型, 与评估目标一致。
    """
    net.eval()
    n_gt = n_found = 0
    with torch.no_grad():
        for e in ev_list:
            ev = events[e]
            nodef = torch.as_tensor(ev["nodef"], device=dev)
            logits, obj = net(nodef, nodef[:, :nodef.shape[1] // 2])
            mp = torch.sigmoid(logits).cpu().numpy()
            op = torch.sigmoid(obj).cpu().numpy()
            pred_masks = [mp[k] > 0.5 for k in np.nonzero(op > 0.5)[0]]
            for gm in ev["chains"]:
                n_gt += 1
                for pm in pred_masks:
                    inter = np.logical_and(pm, gm).sum()
                    union = np.logical_or(pm, gm).sum()
                    if union > 0 and inter / union >= 0.5:
                        n_found += 1
                        break
    return n_found / n_gt if n_gt > 0 else float("nan")


# ====================================================================== 基线: 边分类 + 连通分量
def evaluate_baseline_cc(probs, ev_list, events):
    """同指标基线: 边概率 p>0.5 -> 事件内连通分量 -> 预测链。"""
    boxes = {b: _new_box() for b in NB_BOXES}
    allb = _new_box()
    es, el = [], []
    for e in ev_list:
        ev = events[e]
        p = probs[e]                                        # 每条边概率
        n = ev["n"]
        par = list(range(n))

        def find(x):
            r = x
            while par[r] != r:
                r = par[r]
            while par[x] != r:
                par[x], x = r, par[x]
            return r

        for r in range(len(ev["ii"])):
            if p[r] > 0.5:
                a, b = find(int(ev["ii"][r])), find(int(ev["jj"][r]))
                if a != b:
                    par[a] = b
        comp = {}
        for i in range(n):
            comp.setdefault(find(i), []).append(i)
        pred_masks = []
        for v in comp.values():
            if len(v) >= 2:
                m = np.zeros(n, bool); m[v] = True; pred_masks.append(m)
        chains = ev["chains"]
        box = box_of(ev["nb"])
        n_gt = len(chains)
        ious = np.zeros((len(pred_masks), n_gt))
        for pi, pm in enumerate(pred_masks):
            for gi, gm in enumerate(chains):
                inter = np.logical_and(pm, gm).sum()
                union = np.logical_or(pm, gm).sum()
                ious[pi, gi] = inter / union if union > 0 else 0.0
        for b in (box, "ALL"):
            tgt = boxes[b] if b != "ALL" else allb
            tgt["n_gt"] += n_gt
            for gi in range(n_gt):
                if ious.shape[0] > 0 and ious[:, gi].max() >= 0.5:
                    tgt["n_found"] += 1
            tgt["n_pred"] += len(pred_masks)
            for pi in range(len(pred_masks)):
                best = ious[pi].max() if n_gt > 0 else 0.0
                if best < 0.5:
                    tgt["n_fake"] += 1
                if best >= 1.0 - 1e-9:
                    tgt["n_perfect"] += 1
        es.extend(p.tolist()); el.extend(ev["yy"].tolist())
    return _summarize(boxes, allb), np.array(es), np.array(el)


# ====================================================================== 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="report_figs/feat_ceiling_nb.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--assign", choices=["hungarian", "sinkhorn"], default="hungarian")
    ap.add_argument("--match_tau", type=float, default=0.5)
    ap.add_argument("--match_iters", type=int, default=10)
    ap.add_argument("--dust_logit", type=float, default=-1.0,
                    help="dustbin 的 logit (越大越容易判为'无目标')")
    ap.add_argument("--no_edge_bias", action="store_true",
                    help="关闭 '平均边特征->K 维 slot-track 偏置'")
    ap.add_argument("--min_nb", type=int, default=0)
    ap.add_argument("--max_nb", type=int, default=10 ** 9)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # ---------------- 固定随机性 (np / torch / 数据划分 / 模型初始化) ----------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- 数据 ----------------
    X, y, grp, names = load_npz(args.npz)
    d = np.load(args.npz, allow_pickle=True)
    t0, t1, ntr, nb = d["t0"], d["t1"], d["ntr"], d["nb"]
    _m, _info = _multib_mask(args.npz, grp, args.min_nb, args.max_nb)
    if _info is not None:
        print(f"[data] 子集 nb in [{args.min_nb},{args.max_nb}]: {int(_m.sum())}/{len(_m)} 条边, "
              f"{_info[0]}/{_info[1]} 个事件 ({100*_info[0]/max(_info[1],1):.1f}%)")
        X, y, grp, t0, t1, ntr, nb = X[_m], y[_m], grp[_m], t0[_m], t1[_m], ntr[_m], nb[_m]
    y = y.astype(np.int64)

    tr, va, te = split_events(grp, seed=args.seed)
    m_tr, m_te = np.isin(grp, tr), np.isin(grp, te)
    # 边级标准化 (用 train 事件统计量)
    mu = X[m_tr].mean(0, keepdims=True)
    sd = X[m_tr].std(0, keepdims=True) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    print(f"[data] 事件 train/val/test = {len(tr)}/{len(va)}/{len(te)} | "
          f"边 {X.shape} 真边率 {100*y.mean():.1f}% | 特征 {len(names)} 维")

    # ---------------- 组装 per-event ----------------
    events = build_events(grp, Xs, t0, t1, ntr, y, nb)
    ev_column_standardize(events, tr)
    d_node = events[tr[0]]["nodef"].shape[1]
    n_edge = d_node // 2
    gtsz = sum(len(events[e]["chains"]) for e in tr)
    print(f"[data] 节点特征 {d_node} 维 (mean+max) | train 真值链总数 {gtsz}")

    # ---------------- 模型 ----------------
    net = SetPredictor(d_node, n_edge, d=args.d, heads=args.heads,
                       slots=args.slots, edge_bias=not args.no_edge_bias).to(dev)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"[cfg] assign={args.assign} slots={args.slots} d={args.d} heads={args.heads} "
          f"seed={args.seed} scipy={HAS_SCIPY}")

    # ---------------- 训练 (per-event) ----------------
    # 选模型用 val 发现效率 (越大越好), 与最终评估目标一致; val_loss 仅打印参考。
    best, best_ep, bad, best_state = -1.0, -1, 0, None
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        net.train()
        order = tr.copy()
        rng.shuffle(order)
        tot = 0.0
        for e in order:
            loss, _ = setpred_loss(net, events[e], args, dev, bce)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        vfe = val_finding_eff(net, va, events, dev)
        if not math.isnan(vfe) and vfe > best + 1e-6:
            best, best_ep, bad = vfe, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"  ep{ep:>3} train_loss={tot/max(1,len(order)):.4f} "
                  f"val_finding_eff={vfe:.4f} (best {best:.4f}@{best_ep})", flush=True)
        if bad >= args.patience:
            print(f"  early stop @ep{ep} (best val_finding_eff {best:.4f}@{best_ep})")
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    # ---------------- 评估: 集合预测 ----------------
    sp_metrics, sp_es, sp_el = evaluate_setpred(net, list(te), events, dev)

    # ---------------- 基线: 边分类 MLP + 连通分量 ----------------
    Xs_tr = Xs[m_tr]; y_tr = y[m_tr].astype(np.float32)
    torch.manual_seed(args.seed)
    bl_net = nn.Sequential(nn.Linear(Xs.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                           nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    pw = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())],
                      dtype=torch.float32, device=dev)
    bl_bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    bl_opt = torch.optim.AdamW(bl_net.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr = torch.as_tensor(Xs_tr, device=dev); ytr = torch.as_tensor(y_tr, device=dev)
    ntr_ = len(xtr)
    for ep in range(args.epochs):
        bl_net.train()
        perm = rng.permutation(ntr_)
        for i in range(0, ntr_, 512):
            j = perm[i:i + 512]
            bl_opt.zero_grad()
            bl_bce(bl_net(xtr[j]).squeeze(-1), ytr[j]).backward()
            bl_opt.step()
    bl_net.eval()
    # 逐事件取出边概率 (边顺序: 事件内按 grp 出现顺序)
    probs = {}
    with torch.no_grad():
        p_all = torch.sigmoid(bl_net(torch.as_tensor(Xs, device=dev)).squeeze(-1)).cpu().numpy()
    m_te_edge = np.isin(grp, te)
    for e in te:
        sel = np.nonzero((grp == e))[0]
        probs[e] = p_all[sel]
    bl_metrics, bl_es, bl_el = evaluate_baseline_cc(probs, list(te), events)

    # ---------------- 打印表格 ----------------
    def print_table(title, met):
        print(f"\n== {title} ==")
        print(f"{'nb箱':<6}{'链数':>6}{'发现效率':>10}{'假顶点率':>10}{'完美率':>10}"
              f"{'#预测':>8}{'#找到':>8}{'#假':>6}")
        for b in NB_BOXES + ["ALL"]:
            r = met[b]
            def f(x):
                return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"
            print(f"{b:<6}{r['n_gt']:>6}{f(r['finding_eff']):>10}{f(r['fake_rate']):>10}"
                  f"{f(r['perfect_rate']):>10}{r['n_pred']:>8}{r['n_found']:>8}{r['n_fake']:>6}")

    print(f"\n[评测] 测试集事件 {len(te)} 个")
    print_table("集合预测 (set prediction)", sp_metrics)
    print_table("基线: 边分类 MLP + 连通分量", bl_metrics)

    print(f"\n[边级] 集合预测(由mask导出) AUC={_auc(sp_es, sp_el):.4f} "
          f"AP={ap_score(sp_es, sp_el):.4f}")
    print(f"[边级] 基线边分类        AUC={_auc(bl_es, bl_el):.4f} "
          f"AP={ap_score(bl_es, bl_el):.4f}")

    # ---------------- 输出 JSON ----------------
    result = dict(
        config=dict(npz=args.npz, epochs=args.epochs, seed=args.seed, slots=args.slots,
                    assign=args.assign, min_nb=args.min_nb, max_nb=args.max_nb,
                    d=args.d, heads=args.heads, edge_bias=(not args.no_edge_bias),
                    match_tau=args.match_tau, match_iters=args.match_iters,
                    dust_logit=args.dust_logit, scipy=HAS_SCIPY),
        n_events=dict(train=len(tr), val=len(va), test=len(te)),
        setpred=sp_metrics,
        baseline_cc=bl_metrics,
        edge_level=dict(
            setpred=dict(auc=_auc(sp_es, sp_el), ap=ap_score(sp_es, sp_el)),
            baseline=dict(auc=_auc(bl_es, bl_el), ap=ap_score(bl_es, bl_el)),
        ),
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"\n[out] 已写 {args.out}")


if __name__ == "__main__":
    main()
