"""方向2 冻结主干探针: 只训集合预测头 (trunk=v654 冻结), 报告与方向3 同口径的整链恢复。

动机
----
在提交完整训练臂之前, 先用**冻结的 v654 主干**做一次廉价探针 (CPU, 几百事件, 几分钟):
把 v654 的节点/边表征 (pair-wise) 缓存下来, 只训练 `SetPredHead`, 在与方向3 离线评测
(eval_seqrec_offline.py) **同一批事件、同一判据 (整链完整恢复 chain_full)** 上比较:
    连通分量基线 7.79%  vs  seqrec 17.34%  vs  setpred ?
-> 若探针明显优于基线再提交训练臂; 若明显更差, 如实报告并把训练臂降到最少。

方法
----
- trunk: 加载 v654 ckpt (`LHCb_logs/DFEI/version_654/...` 的 best), 冻结; 前向时打开
  `_stash_emb` 暂存最后一个 GN block 的节点表征 (`global_input['tracks'].x`) 与 tt 边表征;
- 头部: 只用缓存表征训练 SetPredHead (逐事件 forward_event + 匈牙利匹配损失);
- 事件划分: 按事件 train/test 切分 (默认 70/30), **测试事件不参与头部训练**;
- 口径: 预测链只允许落在"点剪枝存活"的径迹上 (与 CC/seqrec 同一工作点); 真值链用
  truth_chain_labels (真值结构边连通分量), 与 eval_seqrec_offline 同义;
  chain_full = 真值链中能被某条预测链**完全精确**复现的比例。

用法 (CPU):
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python3 probe_setpred.py --events 300 --slots 8 --out report_figs
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation        # noqa: E402
from wmpgnn.analysis.load_module import load_module                          # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader               # noqa: E402
from wmpgnn.lightning_module.dfei_lightning_module import truth_chain_labels  # noqa: E402
from wmpgnn.model.set_pred_head import SetPredHead, decode_slots, setpred_event_loss  # noqa: E402

PW = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}


# ---------------------------------------------------------------- 指标
def _iou(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def greedy_match(truth, pred, thr=0.5):
    pairs = []
    for ti, ts in enumerate(truth):
        for pi, ps in enumerate(pred):
            v = _iou(ts, ps)
            if v >= thr:
                pairs.append((v, ti, pi))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    ut, up, m = set(), set(), []
    for v, ti, pi in pairs:
        if ti in ut or pi in up:
            continue
        ut.add(ti); up.add(pi); m.append((ti, pi))
    return m


def exact_recovered(truth, pred):
    ps = {frozenset(s) for s in pred}
    return sum(1 for ts in truth if frozenset(ts) in ps)


def overlap_rate(pred, n_tracks):
    cnt = np.zeros(n_tracks, dtype=int)
    for s in pred:
        for x in s:
            cnt[x] += 1
    occ = cnt > 0
    d = int(occ.sum())
    return (float((cnt > 1).sum()) / d if d else 0.0), d


# ---------------------------------------------------------------- 连通分量基线
def cc_chains(n_nodes, edges, min_size=2):
    """edges: list[(i,j)] 候选边 -> 连通分量链 (set of local node idx)。"""
    par = list(range(n_nodes))

    def find(x):
        r = x
        while par[r] != r:
            r = par[r]
        while par[x] != r:
            par[x], r = r, par[x]
        return r

    for i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            par[a] = b
    comp = {}
    for i in range(n_nodes):
        comp.setdefault(find(i), []).append(i)
    out = []
    for v in comp.values():
        if len(v) >= min_size:
            out.append(set(int(x) for x in v))
    return out


# ---------------------------------------------------------------- 数据缓存
def build_cache(module, args):
    cfg = module.configs
    node_thr = float(cfg.get("node_prune_thr", 0.9))
    edge_thr = float(cfg.get("edge_prune_thr", 0.9))
    blk = module.model._blocks[-1]
    for b in module.model._blocks:
        b._stash_emb = True
    _, _, ckl = load_tst_loader(module._cfg_full)
    cache = []
    n_ev = 0
    t0 = time.time()
    for batch in ckl.test_dataloader():
        if n_ev >= args.events:
            break
        batch = batch.to(module.device)
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        tt = ('tracks', 'to', 'tracks')
        ei = batch[tt].edge_index
        tb = batch["tracks"].batch
        n_nodes = batch["tracks"].x.shape[0]
        with torch.no_grad():
            out = module.model(batch)
        h_node = blk._last_node_emb.detach().cpu()
        h_edge = blk._last_edge_emb.detach().cpu() if blk._last_edge_emb is not None else None
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1)
        ew = blk.edge_weights[tt].detach().cpu().squeeze(-1)
        lca = out[tt].edges.detach().cpu()
        lab = truth_chain_labels(batch[tt].y, ei, n_nodes).detach().cpu()
        eic = ei.detach().cpu()
        tbc = tb.detach().cpu()
        n_b = int(tbc.max().item()) + 1 if tbc.numel() else 0
        for e in range(n_b):
            if n_ev >= args.events:
                break
            nm = (tbc == e)
            ev_nodes = nm.nonzero().flatten()
            g2l = torch.full((n_nodes,), -1, dtype=torch.long)
            g2l[ev_nodes] = torch.arange(ev_nodes.numel())
            em = nm[eic[0]]
            ei_e = g2l[eic[:, em]]                          # 事件内局部索引
            he_e = h_edge[em] if h_edge is not None else torch.zeros(0, h_node.shape[1])
            lab_e = lab[nm]
            uniq = torch.unique(lab_e[lab_e >= 0])
            gm = (torch.stack([(lab_e == u) for u in uniq.tolist()], 0)
                  if uniq.numel() > 0 else torch.zeros((0, int(nm.sum())), dtype=torch.bool))
            surv = (nw[nm] > node_thr)
            # CC 候选边 (局部): 两端存活 & 边分>thr & 模型判 LCA>0 (与 eval_seqrec_offline 同规则)
            arg = lca[em].argmax(-1)
            ew_e = ew[em]
            sl = surv[ei_e[0]] & surv[ei_e[1]]
            cc_keep = sl & (ew_e > edge_thr) & (arg > 0)
            cc_edges = [(int(ei_e[0, k]), int(ei_e[1, k])) for k in cc_keep.nonzero().flatten().tolist()]
            cache.append(dict(h_node=h_node[nm], ei=ei_e, h_edge=he_e, gm=gm,
                              surv=surv, cc_edges=cc_edges, n=int(nm.sum())))
            n_ev += 1
        if n_ev % 50 == 0:
            print(f"  ... cached {n_ev} events, {time.time()-t0:.1f}s", flush=True)
    print(f"[probe] 缓存 {len(cache)} 事件 (trunk=v{args.version}), 耗时 {time.time()-t0:.1f}s",
          flush=True)
    return cache


# ---------------------------------------------------------------- 训练/评测头部
def train_head(cache, tr_idx, slots, assign, args, seed):
    d_node = int(cache[0]["h_node"].shape[1])
    d_edge = int(cache[0]["h_edge"].shape[1]) if cache[0]["h_edge"].numel() else d_node
    torch.manual_seed(seed)
    head = SetPredHead(d_node, d_edge, n_slots=slots, hidden=args.hidden,
                       heads=args.heads, assign_mode=assign)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.default_rng(seed)
    for ep in range(args.epochs):
        head.train()
        order = list(rng.permutation(tr_idx))
        tot = 0.0
        for i in order:
            c = cache[i]
            out = head.forward_event(c["h_node"], c["ei"], c["h_edge"])
            loss, _ = setpred_event_loss(out, c["gm"], assign_mode=assign)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"    [K={slots}/{assign}] ep{ep:>3} train_loss={tot/max(1,len(order)):.4f}",
                  flush=True)
    head.eval()
    return head


def evaluate(head, cache, te_idx, assign, exist_thr, assign_thr):
    n_truth = n_pred = full = mt = mp = 0
    occ = occ_d = 0
    slot_pred = slot_hit = 0
    n_evt = 0
    for i in te_idx:
        c = cache[i]
        n_evt += 1
        with torch.no_grad():
            out = head.forward_event(c["h_node"], c["ei"], c["h_edge"])
            lbl, alive = decode_slots(out, exist_thr=exist_thr,
                                      assign_mode=assign, assign_thr=assign_thr)
        lab = lbl.numpy().astype(int).copy()
        lab[~c["surv"].numpy()] = -1                     # 非存活径迹不能进预测链 (同 CC 工作点)
        pred = []
        for l in np.unique(lab[lab >= 0]):
            m = np.nonzero(lab == l)[0]
            if m.size >= 2:
                pred.append(set(int(x) for x in m.tolist()))
        truth = []
        for g in range(c["gm"].shape[0]):
            m = np.nonzero(c["gm"][g].numpy())[0]
            if m.size >= 2:
                truth.append(set(int(x) for x in m.tolist()))
        n_truth += len(truth)
        n_pred += len(pred)
        full += exact_recovered(truth, pred)
        mt2 = greedy_match(truth, pred, 0.5)
        mt += len(mt2)
        mp += len(mt2)
        # 槽位级: 存活槽位 (去重后) 作为一条链预测
        n_slots = int(alive.sum().item())
        slot_pred += n_slots
        # 槽位命中: 存在一条真值链 IoU>=0.5
        for k in alive.nonzero().flatten().tolist():
            sk = set(int(x) for x in np.nonzero(lab == k)[0].tolist())
            if any(_iou(sk, ts) >= 0.5 for ts in truth):
                slot_hit += 1
        ov, d = overlap_rate(pred, c["n"])
        occ += ov * d
        occ_d += d
    fail = max(n_truth, 1)
    fpred = max(n_pred, 1)
    return dict(n_evt=n_evt, n_truth=n_truth, n_pred=n_pred,
                chain_full=round(100 * full / fail, 2),
                chain_P=round(100 * mp / fpred, 2), chain_R=round(100 * mt / fail, 2),
                mean_chains_truth=round(n_truth / max(n_evt, 1), 3),
                mean_chains_pred=round(n_pred / max(n_evt, 1), 3),
                slot_precision=round(100 * slot_hit / max(slot_pred, 1), 2),
                slot_recall=round(100 * mt / fail, 2),
                overlap=round(100 * occ / max(occ_d, 1), 3))


def evaluate_cc(cache, te_idx):
    n_truth = n_pred = full = mt = mp = 0
    occ = occ_d = 0
    n_evt = 0
    for i in te_idx:
        c = cache[i]
        n_evt += 1
        pred = cc_chains(c["n"], c["cc_edges"])
        truth = []
        for g in range(c["gm"].shape[0]):
            m = np.nonzero(c["gm"][g].numpy())[0]
            if m.size >= 2:
                truth.append(set(int(x) for x in m.tolist()))
        n_truth += len(truth); n_pred += len(pred)
        full += exact_recovered(truth, pred)
        m2 = greedy_match(truth, pred, 0.5)
        mt += len(m2); mp += len(m2)
        ov, d = overlap_rate(pred, c["n"])
        occ += ov * d; occ_d += d
    fail = max(n_truth, 1); fpred = max(n_pred, 1)
    return dict(n_evt=n_evt, n_truth=n_truth, n_pred=n_pred,
                chain_full=round(100 * full / fail, 2),
                chain_P=round(100 * mp / fpred, 2), chain_R=round(100 * mt / fail, 2),
                mean_chains_truth=round(n_truth / max(n_evt, 1), 3),
                mean_chains_pred=round(n_pred / max(n_evt, 1), 3),
                overlap=round(100 * occ / max(occ_d, 1), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_files/eval_bench_v3_n90.yaml")
    ap.add_argument("--version", type=int, default=654)
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exist_thr", type=float, default=0.5)
    ap.add_argument("--assign_thr", type=float, default=0.5)
    ap.add_argument("--out", default="report_figs")
    ap.add_argument("--tag", default="setpred_probe")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, PW)
    module.eval()
    module._cfg_full = cfg
    print(f"[probe] trunk=v{a.version} events={a.events} slots={a.slots} epochs={a.epochs} "
          f"assign modes tested; device={module.device}")

    cache = build_cache(module, a)
    n = len(cache)
    if n == 0:
        raise RuntimeError("无缓存事件")
    # 事件划分 (按事件, 保序可复现)
    idx = np.arange(n)
    np.random.default_rng(a.seed).shuffle(idx)
    n_te = max(1, int(round(a.test_frac * n)))
    te_idx = idx[:n_te].tolist()
    tr_idx = idx[n_te:].tolist()
    print(f"[probe] 事件划分: train={len(tr_idx)} test={len(te_idx)}")

    results = {}
    results["cc_baseline"] = evaluate_cc(cache, te_idx)
    print(f"[CC baseline] {results['cc_baseline']}")

    for slots in [a.slots] if a.slots else []:
        for assign in ["softmax_bg", "sigmoid"]:
            head = train_head(cache, tr_idx, slots, assign, a, a.seed)
            res = evaluate(head, cache, te_idx, assign, a.exist_thr, a.assign_thr)
            results[f"setpred_K{slots}_{assign}"] = res
            print(f"[probe] K={slots} assign={assign}: chain_full={res['chain_full']}% "
                  f"P={res['chain_P']}% R={res['chain_R']}% "
                  f"slot_P={res['slot_precision']}% slot_R={res['slot_recall']}% "
                  f"overlap={res['overlap']}%", flush=True)

    print("\n===== 方向2 冻结主干探针 (同口径 chain_full) =====")
    print("  参照: 连通分量基线 7.79% | seqrec-learned 17.34% (方向3 离线评测)")
    for k, v in results.items():
        print(f"  {k:<26} chain_full={v['chain_full']}%  chain_P={v['chain_P']}%  "
              f"chain_R={v['chain_R']}%  overlap={v.get('overlap')}%")

    os.makedirs(a.out, exist_ok=True)
    jp = f"{a.out}/{a.tag}_v{a.version}.json"
    json.dump(dict(config=dict(version=a.version, events=a.events, slots=a.slots,
                               epochs=a.epochs, hidden=a.hidden, heads=a.heads,
                               seed=a.seed, n_train=len(tr_idx), n_test=len(te_idx)),
                   results=results), open(jp, "w"), indent=1)
    print(f"\n[out] 已写 {jp}")


if __name__ == "__main__":
    main()
