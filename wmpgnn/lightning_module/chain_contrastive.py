"""链级对比损失 (chain-level contrastive loss)。

动机
----
多任务异构图 GNN 做事件重建: 每条径迹 (节点) 有一个学到的嵌入 emb, 真值给出它属于
哪条衰变链 (chain_id)。现有边剪枝标签本质是"簇关系"(同一条链内任意两条径迹是一对
正样本), 但损失一直是**逐边 BCE** —— 它优化的是"单条边独立判对错", 并不直接塑造
嵌入空间, 也不区分"簇内紧凑 / 簇间远离"。

本模块引入一个**链级 InfoNCE 对比损失**: 直接作用在节点嵌入上,
  - 同一链 (同事件内) 的任意两条径迹 = 正样本对 -> 拉近;
  - 不同链 (同事件内) 的两条径迹 = 负样本对 -> 推远;
从而把"事件属于哪条链"这一簇结构直接编码进嵌入的几何。

与"调阈值/调 k"这类旋钮不同, 这是**换目标函数**的新维度: 阈值只改变已学表示的决策
边界, 对比损失改变的是表示本身。

数学形式 (对每个有效 anchor i, 只在其所属事件内取样本):
    P(i) = {j : batch_j == batch_i, chain_id_j == chain_id_i, j != i}   正样本集合
    Q(i) = {k : batch_k == batch_i, chain_id_k 有效且 != chain_id_i}   负样本集合
    L_i  = -log( Σ_{j∈P(i)} exp(sim(z_i,z_j)/τ) / Σ_{k∈Q(i)} exp(sim(z_i,z_k)/τ) )
其中 sim 为余弦相似度 (先对 emb 做 L2 归一化), τ 为温度。
总损失 = 所有有效 anchor 的 L_i 平均。

注: 按需求给定的形式, 分母只累积负样本集合 Q(i) (不含正样本)。这使损失可为负值
(当正样本相似度整体高于负样本时), 与"分母含正样本"的经典 InfoNCE 变体略有差异,
但同样把"正样本靠近、负样本推远"两个目标同时刻进梯度。

实现约束
--------
- 严格**按事件分组**: 不同事件的径迹永不互相比较 (事件彼此独立)。
- chain_id 为负值 (-1) 表示背景/无链: 既不做 anchor, 也不做正/负样本。
- 数值稳定: 用 logsumexp 实现, 不手写 exp 相除。
- 可微且无 in-place 破坏; 全程只用 torch / torch 标准库, 无新依赖。
"""

import torch
import torch.nn.functional as F


def chain_contrastive_loss(emb, chain_id, batch, tau=0.1,
                           max_anchors_per_event=64, max_negatives=512):
    """链级 InfoNCE 对比损失 (逐事件, 逐链)。

    Args:
        emb:      (N, D) float, N 条径迹的嵌入 (可含多个事件, 需 requires_grad)
        chain_id: (N,) long, 每条径迹的真值链 id; **负值表示背景/无链, 必须排除**
        batch:    (N,) long, 每条径迹的事件 id (不同事件之间绝不比较)
        tau:      温度 (余弦相似度 / tau), 越小分布越尖锐; 会 clamp 到 >=1e-8
        max_anchors_per_event: 每个事件最多取多少个 anchor (None/<=0 表示不限制);
                               超限时用 randperm **随机**子采样
        max_negatives:         每个 anchor 的负样本上限 (None/<=0 表示不限制);
                               超限时按随机 key 做 top-k **随机**子采样

    Returns:
        标量 tensor = 有效 anchor 的 L_i 平均; 若没有任何有效 anchor,
        返回 emb.sum()*0.0 (值 0.0, 但仍与计算图连通, 可安全参与反向)。
    """
    tau = max(float(tau), 1e-8)
    device = emb.device

    # 立即捕获一条与计算图连通的零 (兜底返回值, 保证 requires_grad 连通)
    zero = emb.sum() * 0.0

    if emb.shape[0] == 0:
        return zero

    chain_id = chain_id.reshape(-1)
    batch = batch.reshape(-1)

    losses = []
    for g in torch.unique(batch).tolist():
        # ---- 取该事件的径迹, 并剔除背景 (chain_id < 0) ----
        in_ev = (batch == g)
        idx_ev = torch.nonzero(in_ev).flatten()
        lab_ev = chain_id[idx_ev]
        valid = lab_ev >= 0
        if not bool(valid.any()):
            continue
        idx_v = idx_ev[valid]              # 事件内有效径迹的全局索引
        lab_v = lab_ev[valid]              # 事件内有效径迹的链 id

        # ---- 链分组: 该事件至少要 2 条链才可能构成负样本 ----
        _, inv, counts = torch.unique(lab_v, return_inverse=True, return_counts=True)
        if counts.numel() < 2:
            continue                       # 只有 1 条链 -> 没有负样本
        # 每条有效径迹所属链的大小 (用来跳过"单点链", 这类链没有正样本)
        node_chain_size = counts[inv]

        # ---- 选 anchor: 只取"链大小 >= 2"的径迹 (保证至少 1 个正样本) ----
        cand = torch.nonzero(node_chain_size >= 2).flatten()   # idx_v 内的局部下标
        if cand.numel() == 0:
            continue
        if max_anchors_per_event and cand.numel() > max_anchors_per_event:
            perm = torch.randperm(cand.numel(), device=device)[:max_anchors_per_event]
            cand = cand[perm]
        A = cand.numel()

        # ---- L2 归一化后的相似度矩阵 (仅事件内有效径迹) ----
        z = F.normalize(emb[idx_v], p=2, dim=-1)      # [M, D]
        sim = z[cand] @ z.t()                         # [A, M]
        logits = sim / tau                            # [A, M]

        lab_a = lab_v[cand]                           # [A]
        same = (lab_v.unsqueeze(0) == lab_a.unsqueeze(1))    # [A, M] 同链

        # 正样本: 同链且非自身
        pos_mask = same.clone()
        pos_mask[torch.arange(A, device=device), cand] = False

        # 负样本: 不同链 (idx_v 内全部有效, 故直接取反)
        neg_mask = ~same
        if neg_mask.any():
            max_neg_cnt = int(neg_mask.sum(dim=1).max().item())
        else:
            max_neg_cnt = 0

        # 负样本过多 -> 每个 anchor 随机子采样到 max_negatives
        if max_negatives and max_neg_cnt > max_negatives:
            k = min(int(max_negatives), neg_mask.shape[1])
            rnd = torch.rand_like(logits)
            rnd = torch.where(neg_mask, rnd, torch.full_like(rnd, -1.0))
            _, topi = rnd.topk(k, dim=1)
            sel = torch.zeros_like(neg_mask)
            sel.scatter_(1, topi, True)
            neg_mask = sel & neg_mask

        # ---- 该 anchor 必须同时有正样本和负样本, 否则不计入 ----
        row_ok = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not bool(row_ok.any()):
            continue
        pos_mask = pos_mask[row_ok]
        neg_mask = neg_mask[row_ok]
        logits = logits[row_ok]

        # ---- logsumexp 数值稳定实现 ----
        neg_inf = float("-inf")
        log_num = torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=1)
        log_den = torch.logsumexp(logits.masked_fill(~neg_mask, neg_inf), dim=1)
        losses.append(-(log_num - log_den))           # [n_ok]

    if not losses:
        return zero

    return torch.cat(losses).mean()


# ============================ 自测 ============================

if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}  torch = {torch.__version__}")

    # ---- 合成数据: 2 个事件 × 各 2 条链 × 每条链 3 条径迹, 外加每条链 1 条背景 ----
    D = 16
    chains, evts = [], []
    for ev in range(2):
        for c in range(2):
            chains += [c] * 3          # 3 条径迹属于链 c
            evts += [ev] * 3
            chains.append(-1)          # 每条链再加 1 条背景径迹 (无链)
            evts.append(ev)
    chain_id = torch.tensor(chains, dtype=torch.long, device=device)
    batch = torch.tensor(evts, dtype=torch.long, device=device)
    N = chain_id.numel()
    print(f"合成: N={N} 条径迹, 事件数={int(batch.max()) + 1}, "
          f"每事件有效链数=2, 每链 3 径迹 + 1 背景(chain_id=-1), D={D}")

    # ---- (1) 完全可分嵌入: 同链完全相同 (正交 one-hot), 链间正交 ----
    emb_sep = torch.zeros(N, D, device=device)
    for i in range(N):
        if int(chain_id[i]) >= 0:
            key = int(batch[i]) * 2 + int(chain_id[i])   # 每个 (事件, 链) 一个正交基
        else:
            key = 4                                       # 背景 (会被排除, 不影响)
        emb_sep[i, key] = 1.0

    # ---- (2) 随机噪声嵌入 ----
    emb_rand = torch.randn(N, D, device=device)

    loss_sep = chain_contrastive_loss(emb_sep, chain_id, batch)
    loss_rand = chain_contrastive_loss(emb_rand, chain_id, batch)
    print(f"[1] loss(可分) = {loss_sep.item():.4f}  loss(随机) = {loss_rand.item():.4f}")
    assert loss_sep.item() < loss_rand.item(), "可分嵌入的损失应低于随机嵌入"
    print("    -> 断言通过: loss(可分) < loss(随机)")

    # ---- (3) 梯度下降循环: 随机初始化, 50 步 Adam, 断言损失下降 ----
    emb_gd = torch.randn(N, D, device=device, requires_grad=True)
    opt = torch.optim.Adam([emb_gd], lr=0.05)
    loss_first, loss_last = None, None
    for step in range(50):
        opt.zero_grad()
        loss = chain_contrastive_loss(emb_gd, chain_id, batch)
        loss.backward()
        g = emb_gd.grad
        assert g is not None, "梯度缺失"
        assert torch.isfinite(g).all(), f"第 {step} 步梯度出现 NaN/Inf"
        if step == 0:
            loss_first = loss.item()
        opt.step()
        loss_last = loss.item()
    assert torch.isfinite(emb_gd).all(), "优化后嵌入出现 NaN/Inf"
    assert loss_last < loss_first, f"损失未下降: {loss_first} -> {loss_last}"
    print(f"[2] 梯度下降: loss {loss_first:.4f} -> {loss_last:.4f} (50 步 Adam)")
    print("    -> 断言通过: 损失下降, 梯度有限且无 NaN")

    # ---- (4) 全背景: 返回 0 且 backward 不报错 ----
    emb_bg = torch.randn(N, D, device=device, requires_grad=True)
    cid_bg = torch.full((N,), -1, dtype=torch.long, device=device)
    loss_bg = chain_contrastive_loss(emb_bg, cid_bg, batch)
    assert loss_bg.item() == 0.0, f"全背景应返回 0, 实得 {loss_bg.item()}"
    loss_bg.backward()
    assert emb_bg.grad is not None and torch.isfinite(emb_bg.grad).all()
    print(f"[3] 全背景 (chain_id 全为 -1): loss = {loss_bg.item():.4f}, "
          f"backward 正常, 梯度有限")

    # ---- (5) 混合情形: 某事件只有 1 条链时该事件应被整体跳过 ----
    cid_one_chain = chain_id.clone()
    cid_one_chain[batch == 1] = 0          # 事件1 只剩 1 条链 -> 无负样本
    loss_mix = chain_contrastive_loss(emb_rand, cid_one_chain, batch)
    assert torch.isfinite(loss_mix).all() and loss_mix.item() != 0.0
    print(f"[4] 事件1 退化为单链 (仍由事件0贡献): loss = {loss_mix.item():.4f} "
          f"(事件1 的径迹被跳过)")

    # ---- (6) 计时 ----
    t0 = time.perf_counter()
    for _ in range(20):
        _ = chain_contrastive_loss(emb_gd.detach(), chain_id, batch)
    dt = (time.perf_counter() - t0) / 20 * 1000.0

    print(f"[5] 最终损失值 = {loss_last:.4f}; 单次前向耗时 = {dt:.3f} ms")
    print("全部自测断言通过 ✔")
