"""方向4 第一步: 用真实 MC 数据离线标定"几何候选窗口" (纯 CPU, 不训练)。

动机
    当前架构靠一个**学出来的**边二分类头决定哪些径迹对进入下游 -> 召回被一个概率性决策绑定
    (剪枝漏掉一条真值边, 整条链报废), 且下游要在 O(N^2) 的候选里学极稀疏的真值边。
    方向4 想把"生成候选"从学习里剥离: 用**物理量** (几何/运动学) 先给出候选集, 让召回由物理
    保证 (传统径迹跟踪的做法), 学习只负责在干净子集上打分。

本脚本回答: "几何窗口能不能既保住链、又压缩候选集?"
    三个几何量 (均由 derive_pruning_features 现算, 绕开上游 log_DOCA_reco 的 bug):
        dR   = 两端动量单位向量夹角 (rad)
        doca = 两条径迹 (过生产顶点) 直线的最近距离 (mm)
        dstart = 两端生产顶点 3D 距离 |Δr| (mm)
    扫描量出每个窗口:
        recall    = 真值结构边 (tt y>0) 的保留率  (最关键: 窗口能不能保住链)
        reduction = 候选集总边数 / 保留边数       (压缩比; 相对完全图)
        chain_ub  = "整链完整恢复比例"的**上界** (假设候选集内的边全判对: 一条真值链的所有
                    结构边都落在窗口内 => 该链可完整恢复)
    并据此选 2~3 组窗口 (宽松/中等/激进) 作为训练臂的参数 (阈值的选取必须有数据依据)。

用法
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python analyze_geomcand_windows.py --config config_files/eval_v601_inc51_0904.yaml \
        --version 654 --events 200 --out report_figs/geomcand_study.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation        # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                # noqa: E402
from wmpgnn.lightning_module.dfei_lightning_module import derive_pruning_features  # noqa: E402

NORM = ("/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/"
        "normalization_dict.pt")


def geom_columns(batch, nc, ns):
    """返回 (dR[rad], doca[mm], dstart[mm]) —— 复用现算的派生几何量。

    只用 use_vertex_geom=True 时, 边派生列 = 7 base + 3 vgeom:
      [0]=dR(rad), [7]=doca/100, [8]=log(doca), [9]=|Δ起点|/100
    """
    _, ed = derive_pruning_features(batch, nc, ns, use_vertex_geom=True)
    dr = ed[:, 0]
    doca = ed[:, 7] * 100.0
    dstart = ed[:, 9] * 100.0
    return dr, doca, dstart


def collect(a, nc, ns):
    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    _, _, ckl = load_tst_loader(cfg)

    dr_all, doca_all, dst_all, y_all, ev_all = [], [], [], [], []
    # 真值链的"组件"信息 (逐事件并查集): 用于链恢复上界
    tr_dr, tr_doca, tr_dst, tr_comp = [], [], [], []
    n_evt = 0
    comp_off = 0
    for batch in ckl.test_dataloader():
        ei = batch[("tracks", "to", "tracks")].edge_index
        y = batch[("tracks", "to", "tracks")].y.reshape(-1) > 0
        tb = batch["tracks"].batch
        ev = tb[ei[0]].cpu().numpy().astype(np.int64)
        dr, doca, dst = geom_columns(batch, nc, ns)
        dr, doca, dst = dr.cpu().numpy(), doca.cpu().numpy(), dst.cpu().numpy()
        ynp = y.cpu().numpy()
        a_np = ei[0].cpu().numpy()
        b_np = ei[1].cpu().numpy()

        n_here = int(ev.max()) + 1 if ev.size else 0
        take = min(n_here, a.events - n_evt)
        if take <= 0:
            break
        # 只保留前 take 个事件 (逐事件切, 保证事件隔离)
        keep = ev < take
        dr_all.append(dr[keep]); doca_all.append(doca[keep]); dst_all.append(dst[keep])
        y_all.append(ynp[keep]); ev_all.append(ev[keep])

        # ---- 真值链组件 (逐事件并查集, 只用 y>0 的边) ----
        for e in range(take):
            m = keep & (ev == e) & ynp
            if not m.any():
                continue
            ea = a_np[m]; eb = b_np[m]
            nodes = np.unique(np.concatenate([ea, eb]))
            idx = {int(n): i for i, n in enumerate(nodes)}
            parent = list(range(len(nodes)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for u, v in zip(ea, eb):
                ru, rv = find(idx[int(u)]), find(idx[int(v)])
                if ru != rv:
                    parent[ru] = rv
            roots = np.array([find(idx[int(n)]) for n in nodes])
            # 每条真值边的组件根 (全局偏移 -> 跨事件不碰撞)
            edge_comp = np.array([find(idx[int(u)]) for u in ea], dtype=np.int64)
            # 重映射到 0..k-1 并加全局偏移
            uniq = np.unique(edge_comp)
            remap = {int(r): i + comp_off for i, r in enumerate(uniq)}
            comp_off += len(uniq)
            tr_dr.append(dr[m]); tr_doca.append(doca[m]); tr_dst.append(dst[m])
            tr_comp.append(np.array([remap[int(c)] for c in edge_comp], dtype=np.int64))
        n_evt += take
        if n_evt >= a.events:
            break

    return dict(
        dr=np.concatenate(dr_all) if dr_all else np.zeros(0, np.float32),
        doca=np.concatenate(doca_all) if doca_all else np.zeros(0, np.float32),
        dst=np.concatenate(dst_all) if dst_all else np.zeros(0, np.float32),
        y=np.concatenate(y_all) if y_all else np.zeros(0, bool),
        ev=np.concatenate(ev_all) if ev_all else np.zeros(0, np.int64),
        tr_dr=np.concatenate(tr_dr) if tr_dr else np.zeros(0, np.float32),
        tr_doca=np.concatenate(tr_doca) if tr_doca else np.zeros(0, np.float32),
        tr_dst=np.concatenate(tr_dst) if tr_dst else np.zeros(0, np.float32),
        tr_comp=np.concatenate(tr_comp) if tr_comp else np.zeros(0, np.int64),
    )


def windows_stats(D, dr_max, doca_max, dst_max):
    """给定窗口, 返回 recall / reduction / keep_frac / n_keep / chain_ub。

    chain_ub = 完整链比例 (该真值链的所有结构边都在窗口内)。按组件聚合: 组内所有边
    "满足窗口"的布尔全为真 -> 该链可完整恢复。
    """
    keep_all = (D["dr"] <= dr_max) & (D["doca"] <= doca_max) & (D["dst"] <= dst_max)
    n_all = int(keep_all.size)
    n_keep = int(keep_all.sum())
    n_true = int(D["y"].sum())
    n_true_keep = int((keep_all & D["y"]).sum())
    recall = n_true_keep / n_true if n_true else float("nan")
    reduction = n_all / n_keep if n_keep else float("inf")
    keep_frac = n_keep / n_all if n_all else float("nan")

    # ---- 链恢复上界 (按真值链组件) ----
    tc = D["tr_comp"]
    if tc.size:
        sat = ((D["tr_dr"] <= dr_max) & (D["tr_doca"] <= doca_max)
               & (D["tr_dst"] <= dst_max)).astype(np.int64)
        order = np.argsort(tc, kind="stable")
        tc_s, sat_s = tc[order], sat[order]
        uniq, starts, counts = np.unique(tc_s, return_index=True, return_counts=True)
        # 每个组内满足窗口的边数 vs 组内真值边数
        sums = np.add.reduceat(sat_s, starts)
        # 只统计"真链长度>=2" 的组件 (长度 1 的孤立真边不构成链)? 保留全部以对齐边召回口径。
        full = sums == counts
        chain_ub = float(full.mean()) if counts.size else float("nan")
        n_chain = int(counts.size)
        n_chain_full = int(full.sum())
    else:
        chain_ub, n_chain, n_chain_full = float("nan"), 0, 0

    return dict(n_keep=n_keep, keep_frac=keep_frac, recall=recall, reduction=reduction,
                chain_ub=chain_ub, n_chain=n_chain, n_chain_full=n_chain_full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_files/eval_v601_inc51_0904.yaml")
    ap.add_argument("--version", type=int, default=654)
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--out", default="report_figs/geomcand_study.json")
    a = ap.parse_args()

    nd = torch.load(NORM, map_location="cpu", weights_only=False)
    nc = {k: float(v) for k, v in nd["center"].items()}
    ns = {k: (float(v) or 1.0) for k, v in nd["scale"].items()}

    print(f"[geomcand] 加载真实数据 (version={a.version}, events<={a.events}) ...", flush=True)
    D = collect(a, nc, ns)
    n_all, n_true = int(D["dr"].size), int(D["y"].sum())
    print(f"[geomcand] 边总数 {n_all}, 真值结构边 {n_true} "
          f"(占比 {n_true / max(1, n_all):.4%}), 真值链组件 {int(np.unique(D['tr_comp']).size)}")

    dr_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    doca_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    dst_grid = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    out = {"meta": {
        "model": a.version, "events": a.events, "n_edges": n_all,
        "n_true": n_true, "true_frac": n_true / max(1, n_all),
        "n_chain": int(np.unique(D["tr_comp"]).size),
        "note": "recall=真值结构边保留率; reduction=候选集总数/保留数(压缩比); "
                "chain_ub=整链完整恢复比例上界(该链所有结构边都在窗口内); 距离 mm, dR rad",
    }}

    # ---- 单量边缘扫描 ----
    out["scan_dr"] = {str(t): windows_stats(D, t, np.inf, np.inf) for t in dr_grid}
    out["scan_doca"] = {str(t): windows_stats(D, np.inf, t, np.inf) for t in doca_grid}
    out["scan_dstart"] = {str(t): windows_stats(D, np.inf, np.inf, t) for t in dst_grid}

    # ---- 组合网格 -> Pareto 前沿 (recall vs reduction) ----
    combos = []
    for dr in dr_grid:
        for doca in doca_grid:
            for dst in dst_grid:
                s = windows_stats(D, dr, doca, dst)
                if np.isfinite(s["recall"]) and np.isfinite(s["reduction"]):
                    combos.append(dict(dr_max=dr, doca_max=doca, dstart_max=dst, **s))
    # Pareto: 不被"另一个 recall 更高且 reduction 也更高"的支配
    pareto = []
    for c in combos:
        dominated = any((o["recall"] >= c["recall"] and o["reduction"] >= c["reduction"]
                         and (o["recall"] > c["recall"] or o["reduction"] > c["reduction"]))
                        for o in combos)
        if not dominated:
            pareto.append(c)
    pareto.sort(key=lambda c: c["recall"])
    out["pareto"] = pareto
    out["n_combos"] = len(combos)

    # ---- 选定 3 档窗口 (宽松/中等/激进) ----
    def pick(pred):
        cand = [c for c in pareto if pred(c)]
        if not cand:
            return None
        # 取推理上"最省候选"(reduction 最大) 的那个, 即满足约束的边界点
        return max(cand, key=lambda c: c["reduction"])

    picks = {
        "loose": pick(lambda c: c["recall"] >= 0.80),   # 召回优先: 至少保住 80% 真值边
        "mid": pick(lambda c: c["recall"] >= 0.55),     # 中等: recall/压缩折中
        "tight": pick(lambda c: c["recall"] >= 0.30),   # 压缩优先: 候选集缩 ~12x
    }
    out["selected"] = picks

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 打印表 ----
    print("\n==== 单量边缘扫描 (recall / reduction) ====")
    for name, key in (("dR", "scan_dr"), ("doca", "scan_doca"), ("dstart", "scan_dstart")):
        print(f"-- {name} --")
        for t, s in out[key].items():
            print(f"   {name}<={t:>6}: recall={s['recall']:.4f} reduction={s['reduction']:.2f} "
                  f"chain_ub={s['chain_ub']:.4f}")
    print("\n==== Pareto 前沿 (recall / reduction / chain_ub) ====")
    for c in pareto:
        print(f"   dr<={c['dr_max']:<5} doca<={c['doca_max']:<5} dst<={c['dstart_max']:<6} "
              f"| recall={c['recall']:.4f} reduction={c['reduction']:6.2f} chain_ub={c['chain_ub']:.4f}")
    print("\n==== 选定 3 档窗口 ====")
    for k, c in picks.items():
        if c is None:
            print(f"   {k}: 无满足约束的点")
        else:
            print(f"   {k}: dr<={c['dr_max']} doca<={c['doca_max']} dst<={c['dstart_max']} "
                  f"-> recall={c['recall']:.4f} reduction={c['reduction']:.2f} chain_ub={c['chain_ub']:.4f}")
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
