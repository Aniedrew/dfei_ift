"""
Mock 数据集创建脚本

从原始 npy 文件中抽取少量事件，转换为代码预期的 zst chunk 格式，
包含 trn_data_*, val_data_*, tst_data_* 文件。
输出目录结构模仿原始 CERN EOS 数据布局。

用法:
    cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
    python3 prepare_mock_data.py
"""

import os
import io
import zstandard as zstd
import numpy as np
import torch
from torch_geometric.data import HeteroData

# ========== 配置 ==========
SRC_TRAIN = "/lzufs/user/guoqingxiang/DFEI_data/cached_data/training_dataset"
SRC_VAL   = "/lzufs/user/guoqingxiang/DFEI_data/cached_data/validation_dataset"
SRC_TEST  = "/lzufs/user/guoqingxiang/DFEI_data/truth_inclusive_10k_49193/test_dataset"

OUT_DIR   = "/lzufs/user/guoqingxiang/DFEI_data/mock_dataset"
SAMPLE    = "00342442_inclusive"  # 模拟的 sample 名称

# 每个 chunk 包含几个事件
EVENTS_PER_CHUNK = 4

N_TRAIN_CHUNKS = 2   # 8 events
N_VAL_CHUNKS   = 1   # 4 events
N_TEST_CHUNKS  = 1   # 4 events


def compute_ft_labels(n_nodes, senders, receivers, lcag_classes):
    """
    从 LCAG edge labels 推导 node-level FT labels。
    简化方法：如果某 track 有 incident edge 的 LCAG class != 0，则视为 b 强子径迹。
    FT: 0=bbar, 1=background, 2=b
    """
    ft = torch.ones(n_nodes, dtype=torch.long)  # 默认 background (1)
    # 如果某个 track 参与的任何边的 LCAG 类别非 0，标记为 b (2)
    non_zero_mask = lcag_classes != 0
    b_nodes = torch.unique(torch.cat([senders[non_zero_mask], receivers[non_zero_mask]]))
    ft[b_nodes] = 2
    return ft


def compute_pv_association(nodes_array):
    """
    从节点特征的 PV 坐标 (columns 10-12) 创建 PV 节点和 track-PV 关联。
    """
    pv_positions = nodes_array[:, 10:13]  # (n_tracks, 3)
    # 近似唯一 PV：四舍五入到 1mm 精度
    rounded = np.round(pv_positions, decimals=1)
    unique_pvs, inverse_idx = np.unique(rounded, axis=0, return_inverse=True)
    return torch.from_numpy(unique_pvs).float(), torch.from_numpy(inverse_idx).long()


def npy_event_to_heterodata(input_path, target_path):
    """将一对 npy 文件转换为 PyG HeteroData 对象。"""
    inp = np.load(input_path, allow_pickle=True).item()
    tgt = np.load(target_path, allow_pickle=True).item()

    n_nodes = inp['nodes'].shape[0]
    n_edges = inp['edges'].shape[0]

    # --- 节点特征 ---
    # 前 10 列是 track 特征（PV_IP 是给同质 GNN 用的，HGNN 只用前6列）
    # 但完整保留13列也无妨，模型会只取需要的部分
    track_x = torch.from_numpy(inp['nodes'][:, :10]).float()
    # 补充第10-12列作为PV坐标，但这些不放入 track_x

    # --- 边 ---
    senders = torch.from_numpy(inp['senders']).long()
    receivers = torch.from_numpy(inp['receivers']).long()
    edge_index = torch.stack([senders, receivers], dim=0)
    edge_feats = torch.from_numpy(inp['edges']).float()

    # --- LCAG 标签 ---
    lcag_onehot = torch.from_numpy(tgt['edges']).float()
    lcag_classes = lcag_onehot.argmax(dim=1)

    # --- PV 节点与关联 ---
    unique_pvs, pv_assignments = compute_pv_association(inp['nodes'])
    n_pvs = unique_pvs.shape[0]

    # 构建 track->PV 边
    tr_senders = torch.arange(n_nodes)
    pv_receivers = pv_assignments
    tr_pv_edge_index = torch.stack([tr_senders, pv_receivers], dim=0)

    # PV 关联标签：该 track 是否关联到正确的 PV（使用"true"关联，全部为 1）
    pv_asso_y = torch.ones(n_nodes, 1)

    # PV 节点特征：使用 PV 坐标 (x, y, z)
    pv_x = unique_pvs

    # --- FT 标签 ---
    ft = compute_ft_labels(n_nodes, senders, receivers, lcag_classes)

    # --- 构建 HeteroData ---
    data = HeteroData()

    # Track 节点
    data['tracks'].x = track_x
    data['tracks'].ft = ft
    data['tracks'].frag = torch.zeros(n_nodes, dtype=torch.long)  # 无碎片信息

    # PV 节点
    data['pvs'].x = pv_x
    data['pvs'].pos = pv_x  # 位置信息

    # Track-Track 边
    data[('tracks', 'to', 'tracks')].edge_index = edge_index
    data[('tracks', 'to', 'tracks')].edges = edge_feats
    data[('tracks', 'to', 'tracks')].y = lcag_classes

    # Track-PV 边
    data[('tracks', 'to', 'pvs')].edge_index = tr_pv_edge_index
    data[('tracks', 'to', 'pvs')].edges = torch.zeros(n_nodes, 1)  # 占位 edge features
    data[('tracks', 'to', 'pvs')].y = pv_asso_y

    return data


def save_chunk(data_list, out_path, compression_level=3):
    """将 HeteroData 列表保存为 zstd 压缩的 torch 文件。"""
    cctx = zstd.ZstdCompressor(level=compression_level)
    buffer = io.BytesIO()
    torch.save(data_list, buffer)
    compressed = cctx.compress(buffer.getvalue())
    with open(out_path, 'wb') as f:
        f.write(compressed)
    print(f"  Saved {len(data_list)} events -> {out_path} ({len(compressed)/1024:.0f} KB)")


def main():
    print("=" * 60)
    print("准备 Mock 数据集")
    print("=" * 60)

    # 为 sample 创建目录
    sample_dir = os.path.join(OUT_DIR, SAMPLE)
    os.makedirs(sample_dir, exist_ok=True)

    # 1. 训练数据
    print("\n[1/3] 创建训练数据 (trn_data_*) ...")
    train_indices = list(range(0, N_TRAIN_CHUNKS * EVENTS_PER_CHUNK))
    for chunk_id in range(N_TRAIN_CHUNKS):
        chunk_data = []
        for i in range(EVENTS_PER_CHUNK):
            idx = chunk_id * EVENTS_PER_CHUNK + i
            inp = os.path.join(SRC_TRAIN, f"input_{idx}.npy")
            tgt = os.path.join(SRC_TRAIN, f"target_{idx}.npy")
            if not os.path.exists(inp) or not os.path.exists(tgt):
                print(f"  ⚠️  Event {idx} not found, skipping")
                continue
            data = npy_event_to_heterodata(inp, tgt)
            chunk_data.append(data)
        if chunk_data:
            save_chunk(chunk_data, os.path.join(sample_dir, f"trn_data_{chunk_id}.zst"))

    # 2. 验证数据 (从 13400 开始)
    print("\n[2/3] 创建验证数据 (val_data_*) ...")
    val_start = 13400
    for chunk_id in range(N_VAL_CHUNKS):
        chunk_data = []
        for i in range(EVENTS_PER_CHUNK):
            idx = val_start + chunk_id * EVENTS_PER_CHUNK + i
            inp = os.path.join(SRC_VAL, f"input_{idx}.npy")
            tgt = os.path.join(SRC_VAL, f"target_{idx}.npy")
            if not os.path.exists(inp) or not os.path.exists(tgt):
                print(f"  ⚠️  Event {idx} not found, skipping")
                continue
            data = npy_event_to_heterodata(inp, tgt)
            chunk_data.append(data)
        if chunk_data:
            save_chunk(chunk_data, os.path.join(sample_dir, f"val_data_{chunk_id}.zst"))

    # 3. 测试数据 (含真值)
    print("\n[3/3] 创建测试数据 (tst_data_*) ...")
    for chunk_id in range(N_TEST_CHUNKS):
        chunk_data = []
        for i in range(EVENTS_PER_CHUNK):
            idx = chunk_id * EVENTS_PER_CHUNK + i
            inp = os.path.join(SRC_TEST, f"input_{idx}.npy")
            tgt = os.path.join(SRC_TEST, f"target_{idx}.npy")
            if not os.path.exists(inp) or not os.path.exists(tgt):
                print(f"  ⚠️  Event {idx} not found, skipping")
                continue
            data = npy_event_to_heterodata(inp, tgt)
            chunk_data.append(data)
        if chunk_data:
            save_chunk(chunk_data, os.path.join(sample_dir, f"tst_data_{chunk_id}.zst"))

    # 4. 验证
    print("\n" + "=" * 60)
    print("验证 Mock 数据集")
    print("=" * 60)
    print(f"\n输出目录: {OUT_DIR}/{SAMPLE}/")
    for f in sorted(os.listdir(sample_dir)):
        size = os.path.getsize(os.path.join(sample_dir, f))
        print(f"  {f}: {size/1024:.0f} KB")

    # 测试加载
    print("\n测试加载...")
    dctx = zstd.ZstdDecompressor()
    for fname in sorted(os.listdir(sample_dir)):
        fpath = os.path.join(sample_dir, fname)
        with open(fpath, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                decompressed = reader.read()
                data_list = torch.load(io.BytesIO(decompressed), weights_only=False)
        n_events = len(data_list)
        # 检查每个 event 的结构
        for i, d in enumerate(data_list):
            n_tracks = d['tracks'].x.shape[0]
            n_pvs = d['pvs'].x.shape[0]
            n_trtr = d[('tracks', 'to', 'tracks')].edge_index.shape[1]
            n_trpv = d[('tracks', 'to', 'pvs')].edge_index.shape[1]
            ft_vals = torch.unique(d['tracks'].ft).tolist()
            lcag_vals = torch.unique(d[('tracks', 'to', 'tracks')].y).tolist()
            print(f"  [{fname}] event {i}: tracks={n_tracks}, pvs={n_pvs}, "
                  f"tr-tr edges={n_trtr}, tr-pv edges={n_trpv}, "
                  f"FT classes={ft_vals}, LCAG classes={lcag_vals}")

    print("\n✅ Mock 数据集创建完成!")


if __name__ == "__main__":
    main()
