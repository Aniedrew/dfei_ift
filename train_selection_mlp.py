"""候选衰变链选择 MLP 的训练脚本

> ⚠️ **已废弃**：选择 MLP 已改为与主干**联合训练**（DFEI 第5监督头，
> 见 `SELECTION_MLP_DESIGN.md` 第 3 节与 `dfei_lightning_module._chain_select_loss`）。
> 本独立训练脚本仅保留作参考，新流程直接用 `selection_mlp: "builtin"` 配置即可，无需独立 ckpt。

两步流程:
  python train_selection_mlp.py collect --version 31 --n_events 400 --k_list 10,15,20,30 \
        --out selection_mlp_data.npz
  python train_selection_mlp.py train --data selection_mlp_data.npz --epochs 60 --out scorer.ckpt

collect 阶段 (链级):
  - 加载 HGNN 模型 (v25/v31 等), 在测试数据上推理
  - 节点剪枝 (thr 0.2) 后, 对每个 k 生成候选边集 (top-k)
  - 用候选边集重建出多条候选衰变链 (reconstruct_decay 的 cluster_dict)
  - 对每条候选链: 提取链内节点/边特征 (chain_features) + 判断是否完美匹配某条 truth 链
      label = 1 若该链 PerfectReco (node_keys+LCA_values 与某 truth 链一致), 否则 0
  - 保存 {node_feats, edge_feats, label}, 供 scorer 训练

train 阶段:
  - 变长集合特征 -> CandidateScorer (set pooling) -> 标量 score
  - 二分类回归到 label (BCE); 训练后保存 checkpoint (含 scorer_meta 维度元数据)
"""
import argparse
import copy
import glob
import io
import os
import sys

import numpy as np
import torch
import zstandard as zstd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, BASE)

from wmpgnn.reconstruction.reco_helper import lca_truth_matrix, lca_reco_matrix, reconstruct_decay
from wmpgnn.reconstruction.topk_selection import (
    CandidateScorer, chain_features, topk_edge_selbool)
import analyze_reco_visual as A


# ============ collect: 生成链级候选特征 + 标签 ============

def collect_data(args):
    A.VERSION = args.version
    A.DATA_DIR = args.data_dir
    A.SAMPLE = args.sample
    module, hparams = A.load_module(None)
    module.eval()

    import yaml
    with open(f"{A.LOG_DIR}/DFEI/version_{args.version}/hparams.yaml") as fh:
        use_pid = yaml.safe_load(fh).get("DFEI", {}).get("use_pid", "None")

    k_list = [int(x) for x in args.k_list.split(",")]
    files = sorted(glob.glob(f"{A.DATA_DIR}/{A.SAMPLE}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()

    X_node, X_edge, y_label = [], [], []
    n_evt = 0
    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as r:
                data = torch.load(io.BytesIO(r.read()), weights_only=False)
        for evt in data:
            n_evt += 1
            if n_evt > args.n_events:
                break
            et = ("tracks", "to", "tracks")
            store = evt[et]
            store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            store.edges = store.edges.repeat(2, 1)
            store.y = store.y.repeat(2)

            outputs, batch = A.predict(module, evt, use_pid=use_pid)
            node_w = outputs["node_weights"].cpu()
            edge_w = outputs["edge_weights"].cpu()
            lca = outputs[("tracks", "to", "tracks")].lca.cpu()
            edge_index = store.edge_index

            # 节点剪枝 (thr 0.2, 与正式评估一致)
            node_sel = node_w > args.node_thr
            lca_dec = lca.argmax(dim=-1)

            # 特征空间与正式评估一致: 直接用模型输出 (tracks.x 为模型内部维度, 含 PID 处理)
            feat_evt = outputs.cpu()

            # truth 链 (原始 evt 上计算, 用于标签: 候选链是否完美匹配某 truth 链)
            tc_dict = {}
            try:
                tl = lca_truth_matrix(evt)
                if len(tl) > 0:
                    sig_keys = evt["tracks"].sig_keys.tolist()
                    tc_dict, _, _ = reconstruct_decay(tl, sig_keys)
            except Exception:
                pass

            # 当前事件所有候选链 (每个 k 的重建结果) 与 truth 链
            for k in k_list:
                mask = topk_edge_selbool(edge_w, edge_index, node_sel, k)
                _collect_chains(feat_evt, node_w, edge_w, lca, node_sel, mask,
                                lca_dec, tc_dict, X_node, X_edge, y_label)
            if n_evt % 25 == 0:
                print(f"  {n_evt} events, {len(y_label)} chains", flush=True)
        if n_evt > args.n_events:
            break

    np.savez(args.out,
             node=np.array(X_node, dtype=object),
             edge=np.array(X_edge, dtype=object),
             label=np.array(y_label))
    pos = np.mean(y_label)
    print(f"saved {len(y_label)} chains -> {args.out} | label=1 占比 {pos:.3f}")


def _collect_chains(feat_evt, node_w, edge_w, lca, node_sel, edge_mask, lca_dec,
                    tc_dict, X_node, X_edge, y_label):
    """对一个候选边集: 重建出所有链, 每条链提取特征 + 是否完美匹配 truth。"""
    from wmpgnn.reconstruction.reco_helper import get_final_keys
    try:
        part_keys = get_final_keys(feat_evt).cpu().tolist()
        ei = feat_evt[("tracks", "to", "tracks")].edge_index
        undir = (ei[0] < ei[1]) & node_sel[ei[0]] & node_sel[ei[1]] & edge_mask
        d = lca_dec[undir]
        pos = d > 0
        a = ei[0][undir][pos].tolist()
        b = ei[1][undir][pos].tolist()
        dd = d[pos].tolist()
        if not dd:
            return
        reco_lca = np.column_stack([a, b, dd]).astype(np.int64)
        rc_dict, _, _ = reconstruct_decay(reco_lca, part_keys)

        for ck, cluster in rc_dict.items():
            nf, ef = chain_features(feat_evt, node_w, edge_w, lca, edge_mask,
                                    cluster["node_keys"], part_keys)
            if nf is None or ef is None or ef.shape[0] == 0:
                continue
            X_node.append(nf)
            X_edge.append(ef)
            y_label.append(float(_chain_is_perfect(cluster, tc_dict)))
    except Exception:
        return


def _chain_is_perfect(cluster, tc_dict):
    """候选链是否 PerfectReco: node_keys + LCA_values 与某条 truth 链完全一致。"""
    for tc in tc_dict.values():
        if cluster["node_keys"] == tc["node_keys"] and cluster["LCA_values"] == tc["LCA_values"]:
            return 1.0
    return 0.0


# ============ train: 训练 scorer ============

def train(args):
    data = np.load(args.data, allow_pickle=True)
    nodes = [torch.tensor(x, dtype=torch.float32) for x in data["node"]]
    edges = [torch.tensor(x, dtype=torch.float32) for x in data["edge"]]
    labels = torch.tensor(data["label"], dtype=torch.float32)
    assert len(nodes) == len(edges) == len(labels), "data length mismatch"
    node_dim = nodes[0].shape[1]
    edge_dim = edges[0].shape[1]
    print(f"chains={len(labels)} node_dim={node_dim} edge_dim={edge_dim}")
    print(f"label mean={labels.mean():.3f} (perfect 链占比)")

    model = CandidateScorer(node_dim, edge_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()

    n = len(labels)
    idx = np.random.permutation(n)
    n_tr = int(n * 0.8)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    for ep in range(args.epochs):
        model.train()
        tot, cnt = 0.0, 0
        for i in tr_idx:
            opt.zero_grad()
            logit = model(nodes[i], edges[i])
            loss = crit(logit, labels[i])
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        tr_loss = tot / cnt
        model.eval()
        with torch.no_grad():
            va = torch.mean(torch.stack([
                crit(model(nodes[i], edges[i]), labels[i]) for i in va_idx]))
        sched.step(va)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep}: tr_bce={tr_loss:.4f} va_bce={va.item():.4f} lr={opt.param_groups[0]['lr']:.2e}")

    torch.save({"scorer_state": model.state_dict(),
                "scorer_meta": {"node_dim": node_dim, "edge_dim": edge_dim}},
               args.out)
    print(f"saved scorer -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pc = sub.add_parser("collect")
    pc.add_argument("--version", type=int, default=31)
    pc.add_argument("--data_dir", default="/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed")
    pc.add_argument("--sample", default="inclusive_00342442")
    pc.add_argument("--n_events", type=int, default=400)
    pc.add_argument("--k_list", default="10,15,20")
    pc.add_argument("--node_thr", type=float, default=0.2)
    pc.add_argument("--out", default="selection_mlp_data.npz")

    pt = sub.add_parser("train")
    pt.add_argument("--data", default="selection_mlp_data.npz")
    pt.add_argument("--epochs", type=int, default=60)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--out", default="scorer.ckpt")

    args = p.parse_args()
    if args.cmd == "collect":
        collect_data(args)
    elif args.cmd == "train":
        train(args)
    else:
        p.print_help()
