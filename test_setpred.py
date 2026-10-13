"""方向2 单元测试: 集合预测头 (slot + mask + 匈牙利匹配), 合成小图, 纯 CPU。

覆盖 (对应任务要求的 a-f):
  (a) 逐径迹 softmax (softmax_bg) 模式下每条径迹只属于一个槽位 (对随机 logits 也成立);
  (b) 置换不变: 打乱槽位顺序 / 真值链顺序 / 径迹顺序, 匹配后的损失不变;
  (c) 事件隔离: 两个事件各一条链, 不能被匹配到同一个槽位 (对抗用例, K=1);
  (d) 空事件 / 单链 / 链数 > K 的边界安全;
  (e) 梯度有限;
  (f) 缺省关闭 (未建头) 时损失函数不参与 -> 由 smoke_setpred_cpu.py 做逐位核对。

用法: python test_setpred.py
"""
import numpy as np
import torch

from wmpgnn.model.set_pred_head import (SetPredHead, decode_slots, dice_cost,
                                        hungarian_match, setpred_event_loss)

torch.manual_seed(0)
np.random.seed(0)


def _make_head(K=4, d_node=8, d_edge=6, assign="softmax_bg"):
    return SetPredHead(d_node, d_edge, n_slots=K, hidden=16, heads=2, assign_mode=assign)


def _rand_event(N, M, d_node, d_edge, seed):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, d_node, generator=g)
    he = torch.randn(M, d_edge, generator=g)
    if M > 0:
        ei = torch.randint(0, N, (2, M), generator=g)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
    return h, ei, he


# ---------------------------------------------------------------- (a) 排他性
def test_softmax_bg_exclusivity():
    K = 5
    head = _make_head(K=K, assign="softmax_bg")
    for seed in range(5):
        h, ei, he = _rand_event(9, 14, 8, 6, seed)
        out = head.forward_event(h, ei, he)
        prob = out["assign_prob"]                          # [N, K]
        full = torch.cat([out["assign_logit"], out["bg_logit"].unsqueeze(-1)], -1)
        # 逐径迹: (K 槽位 + 背景) 是一个概率分布
        assert torch.allclose(full.softmax(-1).sum(-1), torch.ones(9), atol=1e-5), \
            "softmax_bg 未构成逐径迹概率分布"
        assert float(prob.max()) <= 1.0 + 1e-6
        # 解码: 每个径迹恰好一个标签 (单一整数 -> 天然只属于一个槽位)
        labels, alive = decode_slots(out, exist_thr=0.0, assign_mode="softmax_bg")
        assert labels.shape == (9,)
        assert bool(((labels >= -1) & (labels < K)).all())
        # 组链: 无径迹出现在两条链里
        seen = {}
        for t, l in enumerate(labels.tolist()):
            if l >= 0:
                seen.setdefault(l, []).append(t)
        flat = [t for v in seen.values() for t in v]
        assert len(flat) == len(set(flat)), "径迹被多个槽位同时占用!"
    print("[a] softmax_bg exclusivity OK (随机 logits 下每条径迹唯一标签)")


# ---------------------------------------------------------------- (b) 置换不变
def test_permutation_invariance():
    N, K, G = 10, 4, 2
    torch.manual_seed(1)
    exist = torch.randn(K)
    assign = torch.randn(N, K)
    bg = torch.randn(N)
    prob = torch.cat([assign, bg.unsqueeze(-1)], -1).softmax(-1)[:, :K]
    out = dict(exist_logit=exist, assign_logit=assign, bg_logit=bg, assign_prob=prob)
    gm = torch.zeros(G, N, dtype=torch.bool)
    gm[0, [0, 1, 2]] = True
    gm[1, [5, 6]] = True
    ref, _ = setpred_event_loss(out, gm, assign_mode="softmax_bg")
    ref = float(ref)

    # (b1) 打乱槽位顺序 (对 exist/assign/prob 施加同一排列) -> 匈牙利重匹配 -> 不变
    p = torch.tensor([2, 0, 3, 1])
    out_p = dict(exist_logit=exist[p], assign_logit=assign[:, p], bg_logit=bg,
                 assign_prob=prob[:, p])
    lp, _ = setpred_event_loss(out_p, gm, assign_mode="softmax_bg")
    assert abs(float(lp) - ref) < 1e-5, f"槽位置换改变了损失: {float(lp)} vs {ref}"

    # (b2) 打乱真值链顺序 -> 不变
    gm2 = gm[[1, 0]]
    lt, _ = setpred_event_loss(out, gm2, assign_mode="softmax_bg")
    assert abs(float(lt) - ref) < 1e-5, "真值链置换改变了损失"

    # (b3) 打乱径迹顺序 (行 + gt 列同步) -> 不变
    q = torch.randperm(N)
    out_q = dict(exist_logit=exist, assign_logit=assign[q], bg_logit=bg[q],
                 assign_prob=prob[q])
    lt2, _ = setpred_event_loss(out_q, gm[:, q], assign_mode="softmax_bg")
    assert abs(float(lt2) - ref) < 1e-5, "径迹置换改变了损失"
    print("[b] permutation invariance OK (slot/chain/track)")


# ---------------------------------------------------------------- (c) 事件隔离
def test_event_isolation():
    # 对抗用例: K=1 槽位, 两个事件各一条链。若匹配是全局的, 单个槽位只能匹配一条链
    # (另一个事件"未匹配"); 逐事件匹配则应各自匹配到该事件的那条链。
    N1, N2 = 4, 3
    h1, ei1, he1 = _rand_event(N1, 5, 8, 6, 10)
    h2, ei2, he2 = _rand_event(N2, 4, 8, 6, 11)
    head = _make_head(K=1, assign="softmax_bg")
    gm1 = torch.tensor([[True, True, False, False]])
    gm2 = torch.tensor([[True, False, True]])           # 第二条链只含 0,2
    out1 = head.forward_event(h1, ei1, he1)
    out2 = head.forward_event(h2, ei2, he2)
    _, d1 = setpred_event_loss(out1, gm1, assign_mode="softmax_bg")
    _, d2 = setpred_event_loss(out2, gm2, assign_mode="softmax_bg")
    assert d1["n_matched"] == 1 and d2["n_matched"] == 1, \
        f"逐事件匹配应各自匹配 1 条链, 实得 {d1['n_matched']}/{d2['n_matched']}"
    # 反证: 若把两个事件拼成一个"事件"匹配, K=1 最多只能匹配 1 条链
    N = N1 + N2
    exist = out1["exist_logit"]
    assign = torch.cat([out1["assign_logit"], out2["assign_logit"]], 0)
    bg = torch.cat([out1["bg_logit"], out2["bg_logit"]], 0)
    prob = torch.cat([out1["assign_prob"], out2["assign_prob"]], 0)
    gm_all = torch.zeros(2, N, dtype=torch.bool)
    gm_all[0, :N1] = gm1[0]
    gm_all[1, N1:] = gm2[0]
    out_all = dict(exist_logit=exist, assign_logit=assign, bg_logit=bg, assign_prob=prob)
    _, d_all = setpred_event_loss(out_all, gm_all, assign_mode="softmax_bg")
    assert d_all["n_matched"] == 1, "全局匹配对照应只匹配 1 条 (K=1)"
    print("[c] event isolation OK (对抗: K=1 两事件, 逐事件各匹配 1, 全局只 1)")


# ---------------------------------------------------------------- (d) 边界
def test_boundaries():
    head = _make_head(K=3, assign="softmax_bg")
    # 空事件
    h, ei, he = _rand_event(0, 0, 8, 6, 20)
    out = head.forward_event(h, ei, he)
    assert out["assign_logit"].shape == (0, 3)
    assert out["exist_logit"].shape == (3,)
    l, d = setpred_event_loss(out, torch.zeros((0, 0), dtype=torch.bool),
                              assign_mode="softmax_bg")
    assert torch.isfinite(l) and d["n_gt"] == 0
    lbl, alive = decode_slots(out, exist_thr=0.5, assign_mode="softmax_bg")
    assert lbl.shape == (0,)
    # 单链
    h, ei, he = _rand_event(5, 6, 8, 6, 21)
    out = head.forward_event(h, ei, he)
    l, d = setpred_event_loss(out, torch.ones((1, 5), dtype=torch.bool),
                              assign_mode="softmax_bg")
    assert torch.isfinite(l) and d["n_matched"] == 1
    # 链数 > K (G=5 > K=3)
    gm = torch.zeros(5, 5, dtype=torch.bool)
    for g in range(5):
        gm[g, g] = True
    l, d = setpred_event_loss(out, gm, assign_mode="softmax_bg")
    assert torch.isfinite(l) and d["n_matched"] == 3, f"G>K 应匹配 min(G,K)=3, 实得 {d['n_matched']}"
    # sigmoid 模式同样安全
    heads = _make_head(K=3, assign="sigmoid")
    outs = heads.forward_event(h, ei, he)
    l2, d2 = setpred_event_loss(outs, gm, assign_mode="sigmoid")
    assert torch.isfinite(l2) and d2["n_matched"] == 3
    print("[d] boundaries OK (空/单链/G>K)")


# ---------------------------------------------------------------- (e) 梯度
def test_grads():
    head = _make_head(K=4, assign="softmax_bg")
    h, ei, he = _rand_event(8, 12, 8, 6, 30)
    gm = torch.zeros(2, 8, dtype=torch.bool)
    gm[0, [0, 1, 2]] = True
    gm[1, [5, 6]] = True
    out = head.forward_event(h, ei, he)
    loss, _ = setpred_event_loss(out, gm, assign_mode="softmax_bg")
    head.zero_grad(set_to_none=True)
    loss.backward()
    for n, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"非有限梯度: {n}"
    # sigmoid 模式
    heads = _make_head(K=4, assign="sigmoid")
    outs = heads.forward_event(h, ei, he)
    loss2, _ = setpred_event_loss(outs, gm, assign_mode="sigmoid")
    heads.zero_grad(set_to_none=True)
    loss2.backward()
    for n, p in heads.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"sigmoid 非有限梯度: {n}"
    print("[e] gradients finite OK (softmax_bg + sigmoid)")


def test_hungarian_basic():
    """匹配本身: 对角小代价 -> 对角匹配。"""
    cost = torch.tensor([[0.1, 2.0, 3.0], [2.0, 0.2, 3.0], [3.0, 3.0, 0.3]])
    pairs = sorted(hungarian_match(cost))
    assert pairs == [(0, 0), (1, 1), (2, 2)], f"匈牙利匹配错误: {pairs}"
    # 矩形 (K>G)
    cost2 = torch.tensor([[0.1, 5.0], [5.0, 0.2], [9.0, 9.0]])
    pairs2 = sorted(hungarian_match(cost2))
    assert pairs2 == [(0, 0), (1, 1)]
    # 空
    assert hungarian_match(torch.zeros(0, 3)) == []
    assert hungarian_match(torch.zeros(3, 0)) == []
    print("[+] hungarian basic OK")


def test_dustbin_weight():
    """dustbin_w: 只缩放"未匹配槽位 (target=0)"的 objectness 项。
    (i) dustbin_w=1.0 与不传逐位一致; (ii) 未匹配槽位预测"有对象"时, 增大 dustbin_w -> 损失增大。
    """
    K, N = 4, 8
    torch.manual_seed(7)
    exist = torch.randn(K) + 0.5                     # 偏向"有对象"
    assign = torch.randn(N, K)
    bg = torch.randn(N)
    prob = torch.cat([assign, bg.unsqueeze(-1)], -1).softmax(-1)[:, :K]
    out = dict(exist_logit=exist, assign_logit=assign, bg_logit=bg, assign_prob=prob)
    gm = torch.zeros(1, N, dtype=torch.bool)
    gm[0, [0, 1, 2]] = True                          # 仅 1 条链 -> 3 个槽位未匹配
    l_ref, _ = setpred_event_loss(out, gm, assign_mode="softmax_bg", dustbin_weight=1.0)
    l_def, _ = setpred_event_loss(out, gm, assign_mode="softmax_bg")
    assert abs(float(l_ref) - float(l_def)) < 1e-7, "dustbin_weight=1.0 应逐位等于默认"
    l_big, _ = setpred_event_loss(out, gm, assign_mode="softmax_bg", dustbin_weight=5.0)
    assert float(l_big) > float(l_ref), "dustbin_weight 增大应抬高损失"
    # 空事件 (G=0): 全部槽位为 dustbin -> 同样应随 dustbin_w 增大
    l0, _ = setpred_event_loss(out, torch.zeros((0, N), dtype=torch.bool),
                               assign_mode="softmax_bg", dustbin_weight=1.0)
    l0b, _ = setpred_event_loss(out, torch.zeros((0, N), dtype=torch.bool),
                                assign_mode="softmax_bg", dustbin_weight=5.0)
    assert float(l0b) > float(l0)
    print(f"[f] dustbin_weight OK (1.0={float(l_ref):.4f} == 默认; 5.0={float(l_big):.4f} 更大)")


def test_endpoint_symmetry():
    """端点顺序置换不变 (任务 (e)): 交换部分 tt 边的端点顺序 -> forward_event 输出不变。

    边->节点聚合对两端点对称 (val 同时加到 a 与 b), 槽位上下文 C 不依赖端点顺序。
    """
    h, ei, he = _rand_event(9, 14, 8, 6, 50)
    head = _make_head(K=4, assign="softmax_bg")
    out1 = head.forward_event(h, ei, he)
    ei2 = ei.clone()
    ei2[:, :7] = ei2[[1, 0]][:, :7]                    # 前 7 条边交换端点
    out2 = head.forward_event(h, ei2, he)
    assert torch.allclose(out1["assign_logit"], out2["assign_logit"], atol=1e-6), \
        "端点交换改变了 assign_logit"
    assert torch.allclose(out1["exist_logit"], out2["exist_logit"], atol=1e-6), \
        "端点交换改变了 exist_logit"
    print("[e2] 端点顺序置换不变 OK")


def test_node_mask_control():
    """对照臂: pairwise=False (mask 建在节点池化上) 前向/梯度安全, 且与 pairwise=True 输出不同。"""
    h, ei, he = _rand_event(9, 14, 8, 6, 40)
    hp = _make_head(K=4, assign="softmax_bg")            # pairwise=True (默认)
    hn = SetPredHead(8, 6, n_slots=4, hidden=16, heads=2, assign_mode="softmax_bg",
                     pairwise=False)
    op = hp.forward_event(h, ei, he)
    on = hn.forward_event(h, ei, he)
    assert on["assign_logit"].shape == op["assign_logit"].shape == (9, 4)
    assert not torch.allclose(on["assign_logit"], op["assign_logit"]), \
        "pairwise=False 应与 pairwise=True 输出不同 (成对信息被丢弃)"
    gm = torch.zeros(2, 9, dtype=torch.bool)
    gm[0, [0, 1, 2]] = True
    gm[1, [5, 6]] = True
    ln, _ = setpred_event_loss(on, gm, assign_mode="softmax_bg")
    hn.zero_grad(set_to_none=True)
    ln.backward()
    for n, p in hn.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), f"node_mask 非有限梯度: {n}"
    print("[g] node_mask 对照臂 OK (输出与 pairwise 不同, 梯度有限)")


if __name__ == "__main__":
    test_softmax_bg_exclusivity()
    test_permutation_invariance()
    test_event_isolation()
    test_boundaries()
    test_grads()
    test_hungarian_basic()
    test_dustbin_weight()
    test_endpoint_symmetry()
    test_node_mask_control()
    print("ALL SETPRED UNIT TESTS PASSED")
