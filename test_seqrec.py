"""方向3 单元测试: 可学习顺序重组合解码器 (合成小图, 纯 CPU, 不依赖真实数据)。

覆盖:
  (a) 完美分数 -> 完整还原真值链; 且 cut=+inf + nearest 逐位还原"连通分量" (回归护栏);
  (b) 排他性: 任何径迹只属于一个簇 (对随机分数也成立);
  (c) 置换不变: 打乱边顺序 / 交换边端点 -> 输出划分不变;
  (d) 并列距离的确定性 (重复调用逐位一致);
  (e) 空事件 / 单径迹 / 无候选边 边界安全;
  (f) 可学习参数 (lam / cut) 的梯度有限。

用法: python test_seqrec.py
"""
import numpy as np
import torch

from wmpgnn.reconstruction.sequential_recombine import (
    SeqRecParams, metric_distance, seqrec_chains, seqrec_cut_loss,
    seqrec_metric_loss, sequential_recombine)


def _parts(labels, n):
    """labels -> {cluster_id: sorted member list}, 忽略 -1。"""
    out = {}
    for i, l in enumerate(labels.tolist()):
        if l >= 0:
            out.setdefault(int(l), []).append(i)
    return {k: sorted(v) for k, v in out.items()}


def _set_parts(labels, n):
    return sorted(tuple(v) for v in _parts(labels, n).values())


def test_perfect_recovery_and_cc_regression():
    """(a) 完美分数 -> 还原真值链; cut=inf -> 逐位还原连通分量。"""
    n = 9
    truth = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    edges, p = [], []
    for ci, ch in enumerate(truth):
        for a in ch:
            for b in ch:
                if a < b:
                    edges.append((a, b)); p.append(0.99)
    inter = [(0, 3), (2, 5), (4, 8), (1, 6)]
    for a, b in inter:
        edges.append((a, b)); p.append(0.01)
    edges = np.array(edges, dtype=int); p = np.array(p, dtype=float)
    d = metric_distance(p, metric="learned").numpy()
    # cut=1.0: intra d=-ln(0.99)=0.01 < 1; inter d=-ln(0.01)=4.6 > 1 -> 只并链内
    lab = sequential_recombine(n, edges, d, cut=1.0, linkage="nearest")
    got = _set_parts(lab, n)
    exp = sorted(tuple(c) for c in truth)
    assert got == exp, f"完美分数未还原真值链: {got} != {exp}"
    # cut=inf + nearest == 连通分量 (在"判成候选"的边上)
    lab_cc = sequential_recombine(n, edges, np.zeros(len(edges)), cut=None, linkage="nearest")
    assert _set_parts(lab_cc, n) == [tuple(sorted(set(sum(truth, []))))], \
        "cut=inf 应把所有边并成一个连通分量"
    # 去掉 inter 边 -> cc == 真值链
    mask = p > 0.5
    lab_cc2 = sequential_recombine(n, edges[mask], np.zeros(int(mask.sum())), cut=None)
    assert _set_parts(lab_cc2, n) == exp, "无跨链边时 cc 应等于真值链"
    print("[a] perfect recovery + cc regression OK")


def test_exclusivity_random():
    """(b) 排他性: 随机分数/随机边, 每条径迹只属于一个簇。"""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(2, 40))
        m = int(rng.integers(1, 3 * n))
        e = rng.integers(0, n, size=(m, 2))
        p = rng.random(m)
        d = metric_distance(torch.tensor(p), metric="learned").numpy()
        lab = sequential_recombine(n, e, d, cut=0.5)
        # 每节点标签唯一 -> 天然只在 1 个簇; 显式核对成员集合两两不交
        parts = _parts(lab, n)
        seen = []
        for v in parts.values():
            seen += v
        assert len(seen) == len(set(seen)), "径迹被多条链同时占用!"
        assert len(lab) == n
    print("[b] exclusivity (random) OK")


def test_permutation_invariance():
    """(c) 边顺序打乱 / 端点交换 -> 划分不变。"""
    n = 8
    edges = np.array([(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (2, 3), (6, 7)], dtype=int)
    p = np.array([0.9, 0.8, 0.9, 0.7, 0.85, 0.2, 0.3])   # (2,3),(6,7) 弱 -> 断开
    d = metric_distance(torch.tensor(p), metric="learned").numpy()
    ref = _set_parts(sequential_recombine(n, edges, d, cut=0.5), n)
    # 打乱边顺序
    rng = np.random.default_rng(1)
    for _ in range(5):
        idx = rng.permutation(len(edges))
        got = _set_parts(sequential_recombine(n, edges[idx], d[idx], cut=0.5), n)
        assert got == ref, "打乱边顺序改变了输出"
    # 交换端点
    sw = edges[:, ::-1]
    assert _set_parts(sequential_recombine(n, sw, d, cut=0.5), n) == ref, "端点交换改变了输出"
    print("[c] permutation invariance OK")


def test_tie_determinism():
    """(d) 并列距离 -> 确定性 (重复调用逐位一致)。"""
    n = 6
    edges = np.array([(0, 1), (2, 3), (4, 5), (1, 4)], dtype=int)
    d = np.array([1.0, 1.0, 1.0, 1.0])       # 全并列
    a = sequential_recombine(n, edges, d, cut=1.0)
    b = sequential_recombine(n, edges, d, cut=1.0)
    assert np.array_equal(a, b), "并列距离输出不确定"
    # 反序输入也确定 (排序键含 (i,j))
    a2 = sequential_recombine(n, edges[::-1], d[::-1], cut=1.0)
    assert _set_parts(a, n) == _set_parts(a2, n)
    print("[d] tie determinism OK")


def test_boundaries():
    """(e) 空事件 / 单径迹 / 无候选边。"""
    assert sequential_recombine(0, np.zeros((0, 2), int), np.zeros(0)).tolist() == []
    assert sequential_recombine(1, np.zeros((0, 2), int), np.zeros(0)).tolist() == [-1]
    lab = sequential_recombine(5, np.zeros((0, 2), int), np.zeros(0))
    assert (lab == -1).all(), "无候选边应全为孤立"
    # 自环 / 越界被安全剔除
    e = np.array([(0, 0), (0, 9), (1, 2)], dtype=int)
    lab = sequential_recombine(5, e, np.array([0.1, 0.1, 0.1]), cut=1.0)
    assert _set_parts(lab, 5) == [(1, 2)]
    # 单链 (全连) -> 一个簇
    e = np.array([(0, 1), (1, 2), (2, 3)], dtype=int)
    lab = sequential_recombine(4, e, np.zeros(3), cut=1.0)
    assert _set_parts(lab, 4) == [(0, 1, 2, 3)]
    print("[e] boundaries OK")


def test_learnable_grads():
    """(f) lam / cut 可学习参数梯度有限。"""
    p = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.85, 0.3], requires_grad=True)
    dgeom = torch.tensor([0.1, 0.2, 1.0, 1.2, 0.15, 0.9])
    y = torch.tensor([1, 1, 0, 0, 1, 0])
    prm = SeqRecParams(lam=1.0, cut=0.5, lam_learn=True, cut_learn=True)
    l1 = seqrec_metric_loss(p, dgeom, y, metric="mix", lam=prm.lam)
    l1.backward(retain_graph=True)
    assert prm.log_lam.grad is not None and torch.isfinite(prm.log_lam.grad)
    l2 = seqrec_cut_loss(p, dgeom, y, metric="mix", lam=prm.lam, cut=prm.cut)
    l2.backward()
    assert prm.cut_raw.grad is not None and torch.isfinite(prm.cut_raw.grad)
    assert torch.isfinite(p.grad).all()
    # 关闭学习 -> 不是 Parameter
    prm2 = SeqRecParams(lam=2.0, cut=None, lam_learn=False, cut_learn=False)
    assert not isinstance(prm2.log_lam, torch.nn.Parameter)
    assert prm2.cut is None
    print("[f] learnable gradients finite OK")


def test_seqrec_chains_interface():
    """与重建管线对接: seqrec_chains 返回 reconstruct_decay 同构的 rc_dict。"""
    import pandas as pd
    n = 6
    keys = [10, 20, 30, 40, 50, 60]
    rows = [dict(senders=0, receivers=1, LCA_dec=1, p=0.99, dgeom=0.0),
            dict(senders=1, receivers=2, LCA_dec=1, p=0.98, dgeom=0.0),
            dict(senders=3, receivers=4, LCA_dec=3, p=0.95, dgeom=0.0),
            dict(senders=2, receivers=4, LCA_dec=2, p=0.01, dgeom=1.0)]  # 跨链弱边
    mat = pd.DataFrame(rows)
    rc, nc = seqrec_chains(mat, keys, n, metric="learned", cut=1.0)
    parts = sorted(tuple(sorted(v["node_keys"])) for v in rc.values())
    assert parts == [(10, 20, 30), (40, 50)], f"seqrec_chains 划分错误: {parts}"
    assert all("LCA_values" in v for v in rc.values())
    print("[+] seqrec_chains interface OK")


if __name__ == "__main__":
    test_perfect_recovery_and_cc_regression()
    test_exclusivity_random()
    test_permutation_invariance()
    test_tie_determinism()
    test_boundaries()
    test_learnable_grads()
    test_seqrec_chains_interface()
    print("ALL SEQREC UNIT TESTS PASSED")
