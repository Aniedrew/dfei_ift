"""方向4 单元测试: 几何候选窗口 (合成小图, 纯 CPU, 不依赖真实数据)。

覆盖:
  (a) 候选集判定与窗口参数一致 (边界值含等号; NaN/退化安全; 输出为二值 float);
  (c) 端点交换不变 (dR / DOCA / |Δ起点| 都是对称量 -> 掩码与边方向无关);
  (d) 事件隔离 (窗口量只依赖该边两端径迹 -> 不跨事件比较几何量)。

(b) 非候选边被正确排除在损失之外 与 (e) 梯度有限 —— 需要在真实模型 + 真实数据上验证,
    见 smoke_geomcand_cpu.py (在真实 shared_step 里直接核对 edge_criterion 收到的目标只含
    候选边, 并对 combined_loss.backward() 检查梯度有限)。

用法: python test_geomcand.py
"""
import torch
from torch_geometric.data import HeteroData

from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule


class _Stub:
    """只需 _geomcand_keep_mask 用到的属性 (绕过完整 DFEILightningModule 构造)。"""

    def __init__(self, r, doca, dz, nc=None, ns=None):
        self.geomcand_max_r = r
        self.geomcand_max_doca = doca
        self.geomcand_max_dz = dz
        self._nc = nc or {}
        self._ns = ns or {}


def keep(stub, batch):
    """调用真实实现的窗口掩码 (unbound method -> 用 stub 当 self)。"""
    return DFEILightningModule._geomcand_keep_mask(stub, batch)


def make_batch(tracks, edges, n_ev):
    """合成 batch: tracks=[(px,py,pz, vx,vy,vz, ev)], edges=[(a,b)]。

    归一化字典为空 (ns=1, nc=0) -> X 里的值就是物理量 (mm / GeV 尺度), 便于解析核对。
    """
    n = len(tracks)
    d = HeteroData()
    x = torch.zeros(n, 8)
    ev = torch.zeros(n, dtype=torch.long)
    for i, (p, v, e) in enumerate(tracks):
        x[i, :3] = torch.tensor(p, dtype=torch.float32)
        x[i, 3:6] = torch.tensor(v, dtype=torch.float32)
        ev[i] = e
    d["tracks"].x = x
    d["tracks"].batch = ev
    # 每个事件一个 PV
    d["pvs"].x = torch.zeros(n_ev, 3)
    d["pvs"].batch = torch.arange(n_ev)
    # 每条径迹关联到它所在事件的 PV (log_minIP 常数)
    ti, pi = [], []
    for i in range(n):
        ti.append(i); pi.append(int(ev[i]))
    d[("tracks", "to", "pvs")].edge_index = torch.tensor([ti, pi], dtype=torch.long)
    d[("tracks", "to", "pvs")].edges = torch.full((n, 1), 0.1)
    d[("tracks", "to", "tracks")].edge_index = torch.tensor(
        [[a for a, _ in edges], [b for _, b in edges]], dtype=torch.long)
    d[("tracks", "to", "tracks")].y = torch.ones(len(edges), 1)
    return d


def _parallel_pair(dx):
    """两条共线 (pz 同向) 径迹: dR=0, DOCA=dx, |Δ起点|=dx (解析可核对)。"""
    tracks = [((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0),
              ((0.0, 0.0, 1.0), (dx, 0.0, 0.0), 0)]
    return make_batch(tracks, [(0, 1)], n_ev=1)


def test_window_boundary():
    """(a) 窗口判定与参数一致, 含等号边界; 输出二值 float。"""
    dx = 2.0
    b = _parallel_pair(dx)
    # 边界: max_doca==doca, max_dz==dstart, max_r>dR -> 保留 (含等号)
    s = _Stub(r=0.5, doca=dx, dz=dx)
    k = keep(s, b)
    assert k.dtype == torch.float32 and k.shape == (1,)
    assert set(k.unique().tolist()) <= {0.0, 1.0}, k
    assert k.item() == 1.0, "边界 (相等) 应保留"
    # 略小于 doca -> 排除
    assert keep(_Stub(0.5, dx - 1e-3, dx), b).item() == 0.0
    # 略小于 dstart -> 排除
    assert keep(_Stub(0.5, dx, dx - 1e-3), b).item() == 0.0
    # dR 上限略小于实算 dR (=0 时用负阈值不可行) -> 用另一对制造非零 dR
    tracks = [((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0),
              ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)]     # 垂直 -> dR=pi/2
    b2 = make_batch(tracks, [(0, 1)], n_ev=1)
    assert keep(_Stub(0.5, 1e9, 1e9), b2).item() == 0.0    # dR=1.57 > 0.5
    assert keep(_Stub(2.0, 1e9, 1e9), b2).item() == 1.0    # dR=1.57 <= 2.0
    print("[a] window boundary OK")


def test_nan_safe():
    """(a) 退化/极端输入 (零动量, 重合顶点) 不产生 NaN, 输出仍二值有限。"""
    tracks = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0),
              ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)]
    b = make_batch(tracks, [(0, 1)], n_ev=1)
    for s in (_Stub(0.3, 5.0, 200.0), _Stub(1e9, 1e9, 1e9), _Stub(0.0, 0.0, 0.0)):
        k = keep(s, b)
        assert torch.isfinite(k).all(), k
        assert set(k.unique().tolist()) <= {0.0, 1.0}, k
    print("[a] nan/degenerate safe OK")


def test_endpoint_swap():
    """(c) 端点交换不变: 同一对径迹, (a,b) 与 (b,a) 的掩码逐位一致。"""
    tracks = [((0.3, 0.1, 1.0), (0.5, 0.2, 0.4), 0),
              ((0.1, -0.2, 0.9), (0.6, 0.25, 0.35), 0)]
    b_ab = make_batch(tracks, [(0, 1)], n_ev=1)
    b_ba = make_batch(tracks, [(1, 0)], n_ev=1)
    s = _Stub(0.3, 1.0, 1.0)
    assert torch.equal(keep(s, b_ab), keep(s, b_ba)), "端点交换后掩码变了"
    print("[c] endpoint-swap invariant OK")


def test_event_isolation():
    """(d) 事件隔离: 把两对径迹分配到同一或不同事件, 各自的掩码只由自己的几何决定。"""
    # 事件 A: 近对 (dR 小, 起点近); 事件 B: 远对 (起点远)
    A = [((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0),
         ((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), 0)]
    B = [((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1),
         ((0.0, 0.0, 1.0), (500.0, 0.0, 0.0), 1)]
    s = _Stub(0.3, 1.0, 1.0)
    b_two = make_batch(A + B, [(0, 1), (2, 3)], n_ev=2)
    k = keep(s, b_two)
    assert k.tolist() == [1.0, 0.0], f"事件隔离失败: {k.tolist()}"
    # 同样两对放到同一事件 (n_ev=1) -> 掩码不变 (窗口量逐边局部, 与事件划分无关)
    A1 = [(a[0], a[1], 0) for a in A]
    B1 = [(a[0], a[1], 0) for a in B]
    b_one = make_batch(A1 + B1, [(0, 1), (2, 3)], n_ev=1)
    assert torch.equal(keep(s, b_one), k), "事件划分改变了掩码 (存在跨事件比较)"
    print("[d] event isolation OK")


if __name__ == "__main__":
    test_window_boundary()
    test_nan_safe()
    test_endpoint_swap()
    test_event_isolation()
    print("ALL SYNTHETIC GEOMCAND TESTS PASSED")
