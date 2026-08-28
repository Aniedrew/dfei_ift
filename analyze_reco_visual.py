"""
CERN旧数据 (version_8) 重建结果分析与可视化 (v2)

分析内容:
1. 统计: 哪些事件能正确/部分/不能重建, 与事件特征的关联 (Wilson误差棒 + 样本数标注)
2. 事件级可视化: 挑选基准事件, 绘制
   - 事件显示 (z-x投影, 径迹按真值衰变链着色, 标出母粒子/粒子名/被剪枝丢失的径迹)
   - 真值 vs 重建的衰变树 (graphviz 树状图)

用法:
    python3 analyze_reco_visual.py [--events N] [--outdir DIR] [--nselect N]
"""

import argparse
import glob
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zstandard as zstd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from wmpgnn.model.model import DFEI_HGNN
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.reconstruction.reco_helper import reconstruct_decay, lca_truth_matrix, lca_reco_matrix
from wmpgnn.reconstruction.signal_dict import particle_name
from wmpgnn.util.pruners import true_node_pruning, edge_pruning

# ============ 配置 ============
VERSION = 8
DATA_DIR = "/lzufs/user/guoqingxiang/DFEI_data/CERN_data_LHCb"
SAMPLE = "00342442_inclusive"
LOG_DIR = "LHCb_logs"
NODE_THR = 0.9
EDGE_THR = 0.9
CHAIN_PALETTE = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#17becf"]


def load_config():
    import yaml
    with open(f"{LOG_DIR}/DFEI/version_{VERSION}/input_config.yaml") as f:
        configs = yaml.safe_load(f)
    return configs


def load_module(configs):
    """加载 version_8 的最佳 checkpoint"""
    import yaml
    with open(f"{LOG_DIR}/DFEI/version_{VERSION}/hparams.yaml") as f:
        hparams = yaml.safe_load(f)
    hparams["settings"]["data_dir"] = DATA_DIR
    ckpts = sorted(glob.glob(f"{LOG_DIR}/DFEI/version_{VERSION}/checkpoints/best-epoch*.ckpt"))
    assert ckpts, "No checkpoint found"
    ckpt = ckpts[0]
    print(f"Loading checkpoint: {ckpt}")
    model = DFEI_HGNN(hparams["DFEI"])
    pos_weights = transform_pos_weight(None, None, mode="eval")  # 全1权重
    module = DFEILightningModule.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model,
        pos_weights=pos_weights,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        configs=hparams,
    )
    module.eval()
    if torch.cuda.is_available():
        module = module.cuda()
    return module, hparams


def load_events(n_events=5000):
    """加载测试数据 (与 load_dataset 一致: 双向化 track-track 边)"""
    files = sorted(glob.glob(f"{DATA_DIR}/{SAMPLE}/tst_data_*"))
    events = []
    dctx = zstd.ZstdDecompressor()
    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                data = torch.load(io.BytesIO(reader.read()), weights_only=False)
        for evt in data:
            et = ("tracks", "to", "tracks")
            store = evt[et]
            store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            store.edges = store.edges.repeat(2, 1)
            store.y = store.y.repeat(2)
            events.append(evt)
        if len(events) >= n_events:
            break
    return events[:n_events]


def predict(module, evt, use_pid="true"):
    """单事件前向, 返回模型输出 (附加上推理头权重, 与 shared_step 的 test 模式一致)"""
    import copy
    from torch_geometric.loader import DataLoader
    batch = copy.deepcopy(evt)
    if use_pid == "true":
        batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
    # 用 DataLoader 打包以生成 batch 属性 (模型需要 graph[...].batch)
    loader = DataLoader([batch], batch_size=1)
    device = next(module.parameters()).device
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            outputs = module.model(b)
            block = module.model._blocks[-1]
            outputs["node_weights"] = block.node_weights["tracks"].squeeze()
            outputs["edge_weights"] = block.edge_weights[("tracks", "to", "tracks")].squeeze()
            outputs[("tracks", "to", "tracks")].lca = outputs[("tracks", "to", "tracks")].edges
            return outputs, b


def classify_event(evt, outputs, node_thr=NODE_THR, edge_thr=EDGE_THR):
    """复刻 reconstruction.py 的剪枝逻辑, 返回剪枝后的单事件图 (CPU)"""
    device = outputs["tracks"].x.device
    graph = evt.clone().to(device)

    node_weights = outputs["node_weights"]
    edge_weights = outputs["edge_weights"]
    lca = outputs[("tracks", "to", "tracks")].lca

    # 先附加 lca 到图上 (与 reconstruction.py 一致, 这样剪枝函数会同步裁剪 lca)
    graph[("tracks", "to", "tracks")].lca = lca

    # 剪枝
    node_selbool = node_weights > node_thr
    edge_mask = true_node_pruning(node_selbool, graph, "tracks", [("tracks", "to", "tracks")])
    edge_selbool = edge_weights[edge_mask] > edge_thr
    edge_pruning(edge_selbool, graph, ("tracks", "to", "tracks"))

    return graph.cpu()


def classify_reco(graph):
    """事件级严格分类: 事件内所有真值衰变链都被完美重建才算 perfect

    与官方 PerfectEventReconstruction 一致 (各链 PerfectReco 的乘积)。
    之前"任意一条链完美即perfect"的定义会高估 (per-chain 18.7% vs event 14.0%)。
    """
    try:
        true_LCA = lca_truth_matrix(graph)
        particle_keys_t = graph["truth_part_keys"].tolist()
        particle_ids = list(map(particle_name, graph["truth_part_ids"].numpy()))
        tc_dict, _, _ = reconstruct_decay(true_LCA, particle_keys_t,
                                          particle_ids=particle_ids, truth_level_simulation=1)
        reco_LCA = lca_reco_matrix(graph, mode="reco")
        particle_keys_r = graph["final_keys"].tolist()
        rc_dict, _, _ = reconstruct_decay(reco_LCA, particle_keys_r)

        if tc_dict == {}:
            return "nosignal"

        # 每条真值链的状态
        chain_status = []
        for tc in tc_dict.values():
            best = "notfound"
            for rc in rc_dict.values():
                true_in_reco = np.sum(np.isin(tc["node_keys"], rc["node_keys"])) / len(tc["node_keys"])
                if rc["node_keys"] == tc["node_keys"] and rc["LCA_values"] == tc["LCA_values"]:
                    best = "perfect"
                    break
                elif rc["node_keys"] == tc["node_keys"]:
                    best = "allfound"
                    break
                elif true_in_reco == 1 and len(rc["node_keys"]) > len(tc["node_keys"]):
                    best = "noniso"
                    break
                elif 0.2 <= true_in_reco < 1:
                    best = "partial"
                    break
            chain_status.append(best)

        # 事件级聚合 (严格): 所有链都完美 -> perfect; 所有链至少找到全部粒子 -> allfound;
        # 至少一条链有部分重建 -> partial; 否则 notfound
        if all(s == "perfect" for s in chain_status):
            return "perfect"
        if all(s in ("perfect", "allfound") for s in chain_status):
            return "allfound"
        if any(s in ("perfect", "allfound", "noniso", "partial") for s in chain_status):
            return "partial"
        return "notfound"
    except Exception as e:
        return f"error:{e}"


# ============ 真值衰变链信息 ============

def get_chain_info(evt, graph):
    """从真值LCA聚类出衰变链, 并判断每条链的重建情况

    返回 (chains, pruned_signal_tracks)
    chains: [{'tracks': [idx], 'names': [...], 'mother': 'B-', 'nreco': n, 'ntot': n}, ...]
    pruned_signal_tracks: 被模型剪枝掉的信号径迹索引集合
    """
    true_LCA = lca_truth_matrix(evt)
    particle_keys = evt["truth_part_keys"].tolist()
    particle_ids = evt["truth_part_ids"].numpy().tolist()
    tc_dict, _, _ = reconstruct_decay(true_LCA, particle_keys,
                                      particle_ids=list(map(particle_name, particle_ids)),
                                      truth_level_simulation=1)

    # 重建树的径迹keys (剪枝后)
    reco_keys = set()
    try:
        reco_LCA = lca_reco_matrix(graph, mode="reco")
        rc_dict, _, _ = reconstruct_decay(reco_LCA, graph["final_keys"].tolist())
        for c in rc_dict.values():
            reco_keys.update(c["node_keys"])
    except Exception:
        pass

    # 被剪枝的信号径迹: 原始 final_keys 中不在重建图 final_keys 里的
    # (final_keys 是逐径迹的键, truth_part_keys 是这些键的子集, 用它反查径迹索引)
    all_keys = evt["final_keys"].numpy().tolist() if hasattr(evt, "final_keys") else []
    track_of_key = {k: i for i, k in enumerate(all_keys)}
    surv_keys = set(graph["final_keys"].tolist())
    pruned_set = {i for i, k in enumerate(all_keys) if k not in surv_keys}

    key_pos = {k: i for i, k in enumerate(particle_keys)}  # key -> 真值数组中的位置
    chains = []
    for ck, c in tc_dict.items():
        keys = list(c["node_keys"])
        idxs = [track_of_key[k] for k in keys if k in track_of_key]
        if not idxs:
            continue
        # 母粒子名: 该链内 LCA_dec 最大的一对 (真值矩阵的 sender/receiver 是 truth 数组位置)
        pos_set = {key_pos[k] for k in keys if k in key_pos}
        rows = true_LCA[(true_LCA['senders'].isin(pos_set)) & (true_LCA['receivers'].isin(pos_set))]
        mother = "?"
        if len(rows) > 0:
            mother = rows.loc[rows['LCA_dec'].idxmax(), 'LCA_id_label']
        nreco = sum(1 for k in keys if k in reco_keys)
        chains.append({
            "tracks": idxs,
            "names": [particle_name(particle_ids[key_pos[k]]) for k in keys if k in key_pos],
            "pids": [particle_ids[key_pos[k]] for k in keys if k in key_pos],
            "mother": mother,
            "nreco": nreco,
            "ntot": len(keys),
        })
    return chains, pruned_set


# ============ 事件显示 (径迹视图) ============

def plot_event_display(ax, evt, chains, pruned_set, title):
    """事件显示: z-x投影 (z为水平束流方向), 径迹按真值衰变链着色

    - 实线: 保留下来(重建使用)的信号径迹; 虚线: 被模型剪枝掉的信号径迹
    - 灰色: 背景径迹 (不属于任何信号衰变链)
    - 黑色星: 初级顶点 (PV); 彩色星: 各衰变链的母粒子位置(径迹起点质心)
    - 每个信号径迹旁标注粒子名
    """
    x = evt["tracks"].x
    oz, ox = x[:, 2].numpy(), x[:, 0].numpy()   # z 水平, x 垂直
    pz, px = x[:, 5].numpy(), x[:, 3].numpy()
    n = len(ox)
    scale = 60.0

    chain_of = np.full(n, -1)
    for ci, ch in enumerate(chains):
        for t in ch["tracks"]:
            if t < n:
                chain_of[t] = ci

    # 背景径迹
    for i in range(n):
        if chain_of[i] == -1:
            ax.plot([oz[i], oz[i] + pz[i] * scale], [ox[i], ox[i] + px[i] * scale],
                    color="#c0c0c0", alpha=0.45, lw=0.8)
            ax.scatter([oz[i]], [ox[i]], color="#c0c0c0", s=12, zorder=2)

    # 各衰变链
    for ci, ch in enumerate(chains):
        color = CHAIN_PALETTE[ci % len(CHAIN_PALETTE)]
        # 母粒子位置: 径迹起点质心
        cx = np.mean([ox[t] for t in ch["tracks"]])
        cz = np.mean([oz[t] for t in ch["tracks"]])
        # 先画母粒子到各子径迹的连线 (体现衰变结构)
        for t in ch["tracks"]:
            ax.plot([cz, oz[t]], [cx, ox[t]], color=color, alpha=0.25, lw=0.8, ls=":")
        ax.scatter([cz], [cx], marker="*", s=320, color=color, edgecolor="k", zorder=5)
        lost = ch["ntot"] - ch["nreco"]
        status_txt = "reco" if lost == 0 else f"{ch['nreco']}/{ch['ntot']} reco"
        ax.annotate(f"{ch['mother']} ({status_txt})", (cz, cx),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold", color=color)
        # 链内径迹
        for j, t in enumerate(ch["tracks"]):
            if t >= n:
                continue
            pruned = t in pruned_set
            ls = "--" if pruned else "-"
            alpha = 0.35 if pruned else 0.95
            ax.plot([oz[t], oz[t] + pz[t] * scale], [ox[t], ox[t] + px[t] * scale],
                    color=color, alpha=alpha, lw=1.6, ls=ls)
            ax.scatter([oz[t]], [ox[t]], color=color, s=28, zorder=3)
            ax.annotate(ch["names"][j], (oz[t] + pz[t] * scale, ox[t] + px[t] * scale),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=8, color=color, alpha=min(1.0, alpha + 0.4))

    # PV
    if "pvs" in evt.node_types:
        pvs = evt["pvs"].x
        for pi in range(pvs.shape[0]):
            ax.scatter([pvs[pi, 2]], [pvs[pi, 0]], marker="*", s=260, color="k", zorder=5)
            ax.annotate(f"PV{pi}", (pvs[pi, 2], pvs[pi, 0]),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=9, color="k")

    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.2)

    # 图例
    handles = [Line2D([0], [0], color="#c0c0c0", lw=1.5, label="background tracks")]
    for ci, ch in enumerate(chains):
        color = CHAIN_PALETTE[ci % len(CHAIN_PALETTE)]
        lost = ch["ntot"] - ch["nreco"]
        label = f"{ch['mother']} chain ({ch['ntot']} tracks, {lost} pruned)" if lost else \
                f"{ch['mother']} chain ({ch['ntot']} tracks, reco'd)"
        handles.append(Line2D([0], [0], color=color, lw=2, label=label))
    handles.append(Line2D([0], [0], color="gray", lw=1.5, ls="--", label="pruned (by model)"))
    handles.append(Line2D([0], [0], marker="*", color="k", linestyle="", label="PV"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)


def plot_decay_tree(ax, lca_matrix, keys, title, particle_ids=None, truth=False):
    """包装 reconstruct_decay 画树, 失败时在面板写提示"""
    try:
        if truth:
            ids = list(map(particle_name, particle_ids))
            reconstruct_decay(lca_matrix, keys, particle_ids=ids,
                              truth_level_simulation=1, ax=ax)
        else:
            reconstruct_decay(lca_matrix, keys, ax=ax)
        ax.set_title(title)
        ax.axis("off")
        return True
    except Exception as e:
        ax.text(0.5, 0.5, f"tree failed: {e}", ha="center", va="center")
        ax.set_title(title + " (failed)")
        return False


# ============ 统计图 ============

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion"""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def analyze_stats(sig_df, outdir):
    """统计分析 (事件级): 重建质量分布 + 特征关联 (带误差棒)

    sig_df 是每链一行的官方CSV, 这里聚合到事件级 (严格定义):
    - Perfect : 事件内所有链 PerfectReco==1
    - AllFound: 所有链都找到全部粒子 (至少一条非perfect)
    - Partial : 至少一条链部分重建
    - NotFound: 所有链都没重建出来
    """
    sig_df = sig_df.copy()

    def evt_status(rows):
        if all(rows["PerfectReco"] == 1):
            return "Perfect"
        if all((rows["PerfectReco"] == 1) | (rows["AllParticles"] == 1)):
            return "AllFound"
        if any((rows["PerfectReco"] == 1) | (rows["AllParticles"] == 1) |
               (rows["PartReco"] == 1) | (rows["NoneIso"] == 1)):
            return "Partial"
        return "NotFound"

    evt = sig_df.groupby("EVENTNUMBER", as_index=False).agg(
        NumParticlesInEvent=("NumParticlesInEvent", "first"),
        NumSignalParticles=("NumSignalParticles", "sum"),
        num_pvs=("num_pvs", "first"),
    )
    evt["status"] = sig_df.groupby("EVENTNUMBER").apply(
        evt_status, include_groups=False).reset_index(name="status")["status"]
    # 校验: 事件级 Perfect 应与官方 PerfectEventReconstruction 一致
    evt_csv = f"{LOG_DIR}/DFEI/version_{VERSION}/event_reco_df_{SAMPLE}.csv"
    if os.path.exists(evt_csv):
        off = pd.read_csv(evt_csv)
        off_perf = (off["PerfectEventReconstruction"] == 1).mean() * 100
        my_perf = (evt["status"] == "Perfect").mean() * 100
        print(f"[check] event-level Perfect: mine={my_perf:.1f}%  official={off_perf:.1f}%")

    print(f"事件级分类 (n={len(evt)}):")
    for k, v in evt["status"].value_counts().items():
        print(f"  {k}: {v} ({v/len(evt)*100:.1f}%)")

    # 1. 类别占比饼图
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = evt["status"].value_counts()
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    ax.pie(counts.values, labels=[f"{k}\n{v} ({v/len(evt)*100:.1f}%)" for k, v in counts.items()],
           colors=colors, startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Event-level Reconstruction Quality (n={len(evt)})")
    fig.tight_layout()
    fig.savefig(f"{outdir}/01_reco_quality_pie.png", dpi=150)
    plt.close(fig)

    # 2. 效率 vs 事件径迹数 (Wilson误差棒 + 样本数)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for status, color, marker in [("Perfect", "#2ca02c", "o"),
                                  ("AllFound", "#1f77b4", "^"),
                                  ("Partial", "#ff7f0e", "s"),
                                  ("NotFound", "#d62728", "v")]:
        xs, ys, lo, hi, ns = [], [], [], [], []
        for ntr in range(1, 13):
            sub = evt[evt["NumParticlesInEvent"] == ntr]
            if len(sub) >= 30:
                k = (sub["status"] == status).sum()
                l, h = wilson_ci(k, len(sub))
                pct = k / len(sub) * 100
                xs.append(ntr); ys.append(pct); lo.append(pct - l * 100)
                hi.append(h * 100 - pct); ns.append(len(sub))
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=marker, color=color,
                    label=status, capsize=3, lw=1.5, ms=6)
        for x, y, nn in zip(xs, ys, ns):
            ax.annotate(str(nn), (x, y), textcoords="offset points", xytext=(0, 7),
                        fontsize=7, color=color, ha="center", alpha=0.85)
    ax.set_xlabel("NumParticlesInEvent (after pruning)")
    ax.set_ylabel("Fraction of events (%)")
    ax.set_title("Event-level Reconstruction Quality vs Track Count (labels = N events)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-3, 103)
    fig.tight_layout()
    fig.savefig(f"{outdir}/02_eff_vs_ntracks.png", dpi=150)
    plt.close(fig)

    # 3. 完美率 vs PV 数 (Wilson误差棒)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs, ys, lo, hi, ns = [], [], [], [], []
    for n in sorted(evt["num_pvs"].dropna().unique()):
        sub = evt[evt["num_pvs"] == n]
        if len(sub) >= 30:
            k = (sub["status"] == "Perfect").sum()
            l, h = wilson_ci(k, len(sub))
            pct = k / len(sub) * 100
            xs.append(int(n)); ys.append(pct); lo.append(pct - l * 100)
            hi.append(h * 100 - pct); ns.append(len(sub))
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color="#1f77b4",
                capsize=4, lw=1.8, ms=7)
    for x, y, nn in zip(xs, ys, ns):
        ax.annotate(f"n={nn}", (x, y), textcoords="offset points", xytext=(6, -14),
                    fontsize=8, color="#1f77b4")
    ax.set_xlabel("NumPVs")
    ax.set_ylabel("Perfect event reconstruction fraction (%)")
    ax.set_title("Perfect (event-level) vs Number of PVs")
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 103)
    fig.tight_layout()
    fig.savefig(f"{outdir}/03_eff_vs_npvs.png", dpi=150)
    plt.close(fig)

    # 4. 各状态下的径迹数与信号径迹数分布 (箱线图 + 计数)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat in zip(axes, ["NumParticlesInEvent", "NumSignalParticles"]):
        cats = ["Perfect", "AllFound", "Partial", "NotFound"]
        data = [evt[evt["status"] == s][feat].values for s in cats]
        bp = ax.boxplot(data, tick_labels=cats, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for xi, s in enumerate(cats, start=1):
            n = len(evt[evt["status"] == s])
            ax.annotate(f"n={n}", (xi, ax.get_ylim()[1]), ha="center", fontsize=8,
                        xytext=(0, 4), textcoords="offset points")
        ax.set_title(feat)
        ax.grid(alpha=0.3)
    fig.suptitle("Event complexity vs reconstruction status (event-level)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/04_complexity_vs_status.png", dpi=150)
    plt.close(fig)

    print(f"统计图已保存到 {outdir}")


# ============ 事件级可视化 ============

def visualize_events(module, events, n_select, outdir):
    """对选定事件做推断 + 可视化"""
    print("Running inference on test events...")
    results = []  # (status, evt_idx, pruned_graph, evt)
    for idx, evt in enumerate(events):
        outputs, batch = predict(module, evt)
        graph = classify_event(evt, outputs)
        status = classify_reco(graph)
        results.append((status, idx, graph, evt))
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(events)}")

    # 各类别统计
    from collections import Counter
    cnt = Counter(r[0] for r in results)
    print("\n事件分类统计 (基于真值vs重建衰变树):")
    for k, v in cnt.most_common():
        print(f"  {k}: {v} ({v/len(results)*100:.1f}%)")

    # 为每类挑选代表性事件 (径迹数适中)
    print("\n选择代表性事件...")
    selected = {}
    for status in ["perfect", "partial", "notfound"]:
        candidates = [r for r in results if r[0] == status]
        valid = [r for r in candidates if 6 <= r[3]["tracks"].x.shape[0] <= 40]
        if not valid:
            valid = candidates[:n_select]
        selected[status] = valid[:n_select]

    # 绘图
    for status, items in selected.items():
        for j, (st, idx, graph, evt) in enumerate(items):
            fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
            evt_num = evt["EVENTNUMBER"].item() if "EVENTNUMBER" in evt else idx

            chains, pruned_set = get_chain_info(evt, graph)
            chain_summary_txt = "; ".join(
                f"{ch['mother']} ({ch['nreco']}/{ch['ntot']})" for ch in chains) or "no chains"

            # 1. 事件显示
            plot_event_display(axes[0], evt, chains, pruned_set,
                               f"Event {evt_num}: track display (by truth chain)")

            # 2. 真值衰变树
            true_LCA = lca_truth_matrix(evt)
            plot_decay_tree(axes[1], true_LCA, evt["truth_part_keys"].tolist(),
                            "True decay tree",
                            particle_ids=evt["truth_part_ids"].numpy(), truth=True)

            # 3. 重建衰变树 (剪枝后)
            reco_LCA = lca_reco_matrix(graph, mode="reco")
            plot_decay_tree(axes[2], reco_LCA, graph["final_keys"].tolist(),
                            "Reconstructed decay tree")

            fig.suptitle(f"{status.upper()} - Event {evt_num} "
                         f"(tracks={evt['tracks'].x.shape[0]}, pruned={graph['tracks'].x.shape[0]}) | chains: {chain_summary_txt}",
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{outdir}/evt_{status}_{j}_event{evt_num}.png", dpi=130)
            plt.close(fig)
            print(f"  已保存: evt_{status}_{j}_event{evt_num}.png | {chain_summary_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=3000, help="测试事件数")
    parser.add_argument("--outdir", type=str, default="reco_analysis")
    parser.add_argument("--nselect", type=int, default=2, help="每类可视化事件数")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(42)

    # 统计部分 (直接读已有CSV)
    sig_path = f"{LOG_DIR}/DFEI/version_{VERSION}/signal_reco_df_{SAMPLE}.csv"
    if os.path.exists(sig_path):
        sig_df = pd.read_csv(sig_path)
        analyze_stats(sig_df, args.outdir)
    else:
        print(f"未找到 {sig_path}, 跳过统计部分")

    # 可视化部分
    configs = load_config()
    module, hparams = load_module(configs)
    events = load_events(args.events)
    print(f"Loaded {len(events)} test events")
    visualize_events(module, events, args.nselect, args.outdir)

    print(f"\n全部完成! 结果保存在 {args.outdir}/")


if __name__ == "__main__":
    main()
