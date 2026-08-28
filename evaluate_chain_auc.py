#!/usr/bin/env python3
"""评估"链判据"区分 truth 链 vs 组合背景链的 AUC (trigger 候选筛选视角)。

用当前最好模型 (masshead2, version_47 ep113) 的 LCA/权重输出构造链特征:
  - conf        : 链内边被判类别的平均 softmax 概率 (高置信 = 真链)
  - class2_frac : 链内 class2/3 边占比 (真 B 链更高)
  - edge_w      : 链内边平均 edge_weight (剪枝分数)
  - node_w      : 链内节点平均 node_weight
正样本 = truth 链 (y>0 连通分量); 负样本 = 同大小随机节点组合。
单特征 AUC + 逻辑回归 (多特征) AUC。
"""
import sys, io, argparse
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')

import yaml
import torch
import torch.nn.functional as F
import zstandard as zstd
import numpy as np
from torch_geometric.data import Batch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

torch.manual_seed(0)
np.random.seed(0)

DATA = '/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst'
CKPT = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_47/checkpoints/best-epoch=113-val_combined_loss=35.677.ckpt'


def load_events(path, n=6):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            data = torch.load(io.BytesIO(reader.read()), weights_only=False)
    evts = data[:n]
    for evt in evts:  # 双向化 (与 load_dataset 一致)
        store = evt[('tracks', 'to', 'tracks')]
        store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
        store.edges = store.edges.repeat(2, 1)
        if getattr(store, 'y', None) is not None:
            store.y = store.y.repeat(2)
        if getattr(store, 'lca', None) is not None:
            store.lca = store.lca.repeat(2, 1)
    return evts


def truth_chains(y, edge_index, n_nodes):
    """y>0 无向连通分量 -> 每链节点列表 (>=2)。y: [E] 0/1。"""
    a, b = edge_index[0].tolist(), edge_index[1].tolist()
    yb = (y > 0).squeeze(-1).tolist() if y.dim() == 2 else y.tolist()
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, yv in enumerate(yb):
        if yv > 0:
            ra, rb = find(a[i]), find(b[i])
            if ra != rb:
                parent[ra] = rb
    comp = {}
    for i in range(n_nodes):
        comp.setdefault(find(i), []).append(i)
    return [nodes for nodes in comp.values() if len(nodes) >= 2]


def chain_features(node_idx, lca_probs, edge_weights, node_weights, tt_ei):
    """单条链 (节点索引列表) 的判据特征。"""
    nset = set(node_idx)
    # 链内边 = 两端都在链内 (去重 a<b)
    a, b = tt_ei[0].tolist(), tt_ei[1].tolist()
    e_in = [i for i in range(len(a)) if a[i] in nset and b[i] in nset and a[i] < b[i]]
    if not e_in:
        return None
    probs = lca_probs[e_in]
    cls = probs.argmax(-1)
    conf = float(probs.max(-1).values.mean())            # 被判类别平均概率
    struct_conf = float(probs[:, 1:].sum(-1).mean())     # 非 class0 (结构边) 总概率
    struct_frac = float((cls > 0).float().mean())        # 结构边占比
    class2_frac = float(((cls == 2) | (cls == 3)).float().mean())
    edge_w = float(edge_weights[e_in].mean())
    node_w = float(node_weights[node_idx].mean())
    return [conf, struct_conf, struct_frac, class2_frac, edge_w, node_w, len(node_idx), len(e_in)]


def pruned_components(node_w, edge_w, tt_ei, n_nodes, node_thr=0.5, edge_thr=0.5):
    """剪枝图连通分量: node_w/edge_w 超过阈值的结构 (模型认为像信号的部分)。"""
    keep_n = node_w > node_thr
    keep_e = (edge_w > edge_thr).tolist()
    a, b = tt_ei[0].tolist(), tt_ei[1].tolist()
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(a)):
        if keep_e[i] and keep_n[a[i]] and keep_n[b[i]]:
            ra, rb = find(a[i]), find(b[i])
            if ra != rb:
                parent[ra] = rb
    comp = {}
    for i in range(n_nodes):
        if keep_n[i]:
            comp.setdefault(find(i), []).append(i)
    return [nodes for nodes in comp.values() if len(nodes) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nevents', type=int, default=20)
    ap.add_argument('--neg_per_pos', type=int, default=3, help='每个正样本对应的负样本数(随机组合对照)')
    args = ap.parse_args()

    with open('config_files/train_CERN_v38_masshead.yaml') as f:
        configs = yaml.safe_load(f)
    configs = adjust_config_training(configs)
    configs['DFEI']['cpt'] = CKPT
    pos_weights = transform_pos_weight(None, None, mode="eval")
    module = load_module(configs, pos_weights)
    module.eval()
    block = module.model._blocks[-1]

    events = load_events(DATA, n=args.nevents)
    print(f"[auc] 事件数: {len(events)}")

    X_pos, X_neg, X_neg_pruned, X_neg_rand = [], [], [], []
    n_chains = 0
    for evt in events:
        batch = Batch.from_data_list([evt])
        with torch.no_grad():
            outputs = module.forward(batch)
        lca_probs = F.softmax(outputs[('tracks', 'to', 'tracks')].edges, dim=-1).cpu()
        edge_w = block.edge_weights[('tracks', 'to', 'tracks')].squeeze(-1).cpu()
        node_w = block.node_weights['tracks'].squeeze(-1).cpu()
        tt_ei = batch[('tracks', 'to', 'tracks')].edge_index.cpu()
        y = batch[('tracks', 'to', 'tracks')].y.cpu()
        n_nodes = batch['tracks'].x.shape[0]

        chains = truth_chains(y, tt_ei, n_nodes)
        n_chains += len(chains)
        truth_sets = [set(c) for c in chains]
        for c in chains:
            f = chain_features(c, lca_probs, edge_w, node_w, tt_ei)
            if f is not None:
                X_pos.append(f)
        # 负样本 1 (真实场景): 剪枝图连通分量 = 模型误认为信号的结构, 剔除与 truth 链重合
        for comp in pruned_components(node_w, edge_w, tt_ei, n_nodes):
            if any(set(comp) == ts for ts in truth_sets):
                continue
            f = chain_features(comp, lca_probs, edge_w, node_w, tt_ei)
            if f is not None:
                X_neg.append(f); X_neg_pruned.append(f)
        # 负样本 2 (对照): 随机组合, 大小从正样本大小分布采样
        sizes = [len(c) for c in chains]
        all_nodes = list(range(n_nodes))
        for _ in range(len(chains) * args.neg_per_pos):
            k = np.random.choice(sizes) if sizes else 3
            k = min(k, n_nodes)
            comb = tuple(sorted(np.random.choice(all_nodes, size=k, replace=False)))
            if any(set(comb) == ts for ts in truth_sets):
                continue
            f = chain_features(list(comb), lca_probs, edge_w, node_w, tt_ei)
            if f is not None:
                X_neg.append(f); X_neg_rand.append(f)

    X_pos = np.array(X_pos); X_neg = np.array(X_neg)
    X_neg_p = np.array(X_neg_pruned); X_neg_r = np.array(X_neg_rand)
    print(f"[auc] 正样本链: {len(X_pos)}, 负样本链: {len(X_neg)} (剪枝分量 {len(X_neg_p)} + 随机组合 {len(X_neg_r)})")
    y_pos = np.ones(len(X_pos)); y_neg = np.zeros(len(X_neg))
    X = np.concatenate([X_pos, X_neg]); y = np.concatenate([y_pos, y_neg])

    names = ['conf', 'struct_conf', 'struct_frac', 'class2/3占比', '平均edge_w', '平均node_w', '链大小', '链内边数']
    print("\n========== 单特征 AUC (正=truth链 vs 负=剪枝分量+随机组合) ==========")
    for j, nm in enumerate(names):
        try:
            auc = roc_auc_score(y, X[:, j])
            print(f"  {nm:<14}: AUC = {auc:.4f}")
        except ValueError as e:
            print(f"  {nm:<14}: 无法计算 ({e})")

    # 逻辑回归多特征 AUC (5折交叉验证, 避免同集乐观)
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    reg = LogisticRegression(max_iter=2000)
    pred = cross_val_predict(reg, X, y, cv=cv, method='predict_proba')[:, 1]
    auc_multi = roc_auc_score(y, pred)
    print(f"\n========== 多特征 (LogReg, 5折CV) AUC ==========")
    print(f"  全部负样本:     AUC = {auc_multi:.4f}")
    # 只对"剪枝分量"负样本 (真实场景) 单独评估
    if len(X_neg_p) > 0:
        Xp = np.concatenate([X_pos, X_neg_p]); yp = np.concatenate([y_pos, np.zeros(len(X_neg_p))])
        reg2 = LogisticRegression(max_iter=2000)
        pred2 = cross_val_predict(reg2, Xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        auc_pruned = roc_auc_score(yp, pred2)
        print(f"  仅剪枝分量负样本: AUC = {auc_pruned:.4f}  ← 真实场景 (模型误判信号)")


if __name__ == '__main__':
    main()
