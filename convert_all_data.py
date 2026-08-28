"""
完整数据集批量转换脚本

将 npy 逐事件数据转换为 zst chunk 格式，
与项目现有的 chunk_loader / default_data_loader 兼容。

用法:
    cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
    python3 convert_all_data.py

输出目录结构:
    {OUT_BASE}/00342442_inclusive/
        trn_data_000.zst  ... trn_data_049.zst   (训练, 800 evt/chunk)
        val_data_000.zst  ... val_data_046.zst   (验证, 200 evt/chunk)
        tst_data_000.zst  ... tst_data_049.zst   (测试, 200 evt/chunk)
"""

import os
import io
import sys
import time
import gc
import glob
import argparse
import zstandard as zstd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from functools import partial
from multiprocessing.pool import ThreadPool

# ========== 路径配置 ==========
SRC_TRAIN = "/lzufs/user/guoqingxiang/DFEI_data/cached_data/training_dataset"
SRC_VAL   = "/lzufs/user/guoqingxiang/DFEI_data/cached_data/validation_dataset"
SRC_TEST  = "/lzufs/user/guoqingxiang/DFEI_data/truth_inclusive_10k_49193/test_dataset"

OUT_BASE  = "/lzufs/user/guoqingxiang/DFEI_data/converted"
SAMPLE    = "00342442_inclusive"  # 模拟原始 sample 名称

# Chunk 大小
TRN_EVENTS_PER_FILE = 800
VAL_EVENTS_PER_FILE = 200
TST_EVENTS_PER_FILE = 200

# 压缩级别 (1-22, 越高越小越慢)
COMPRESSION_LEVEL = 3

# 并行线程数
N_THREADS = 8


def compute_ft_labels(n_nodes, senders, receivers, lcag_classes):
    """从 LCAG edge labels 推导 node-level FT labels。"""
    ft = torch.ones(n_nodes, dtype=torch.long)  # 默认 background (1)
    non_zero_mask = lcag_classes != 0
    if non_zero_mask.any():
        b_nodes = torch.unique(torch.cat([senders[non_zero_mask], receivers[non_zero_mask]]))
        ft[b_nodes] = 2
    return ft


def compute_pv_association(nodes_array):
    """
    从节点特征的 PV 坐标 (columns 10-12) 创建 PV 节点和 track-PV 关联。
    使用坐标聚类来识别唯一 PV。
    """
    pv_positions = nodes_array[:, 10:13]
    # 四舍五入到 0.1mm 精度来聚类
    rounded = np.round(pv_positions, decimals=1)
    unique_pvs, inverse_idx = np.unique(rounded, axis=0, return_inverse=True)
    return torch.from_numpy(unique_pvs).float(), torch.from_numpy(inverse_idx).long()


def convert_single_event(input_path, target_path):
    """将一对 npy 文件转换为单个 HeteroData 对象。"""
    try:
        inp = np.load(input_path, allow_pickle=True).item()
        tgt = np.load(target_path, allow_pickle=True).item()
    except Exception as e:
        print(f"  ⚠️ 加载失败 {input_path}: {e}")
        return None

    n_nodes = inp['nodes'].shape[0]
    n_edges = inp['edges'].shape[0]

    # 节点特征 (前10列)
    track_x = torch.from_numpy(inp['nodes'][:, :10]).float()

    # 边
    senders = torch.from_numpy(inp['senders']).long()
    receivers = torch.from_numpy(inp['receivers']).long()
    edge_index = torch.stack([senders, receivers], dim=0)
    edge_feats = torch.from_numpy(inp['edges']).float()

    # LCAG 标签
    lcag_onehot = torch.from_numpy(tgt['edges']).float()
    lcag_classes = lcag_onehot.argmax(dim=1)

    # PV 节点与关联
    unique_pvs, pv_assignments = compute_pv_association(inp['nodes'])
    n_pvs = unique_pvs.shape[0]

    # Track→PV 边
    tr_senders = torch.arange(n_nodes)
    tr_pv_edge_index = torch.stack([tr_senders, pv_assignments], dim=0)
    pv_asso_y = torch.ones(n_nodes, 1)

    # FT 标签
    ft = compute_ft_labels(n_nodes, senders, receivers, lcag_classes)

    # 构建 HeteroData
    data = HeteroData()
    data['tracks'].x = track_x
    data['tracks'].ft = ft
    data['tracks'].frag = torch.zeros(n_nodes, dtype=torch.long)
    data['pvs'].x = unique_pvs
    data['pvs'].pos = unique_pvs
    data[('tracks', 'to', 'tracks')].edge_index = edge_index
    data[('tracks', 'to', 'tracks')].edges = edge_feats
    data[('tracks', 'to', 'tracks')].y = lcag_classes
    data[('tracks', 'to', 'pvs')].edge_index = tr_pv_edge_index
    data[('tracks', 'to', 'pvs')].edges = torch.zeros(n_nodes, 1)
    data[('tracks', 'to', 'pvs')].y = pv_asso_y
    data[('tracks', 'to', 'pvs')].filter = torch.ones(n_nodes, dtype=torch.bool)

    # === 粒子级 truth (论文数据 "Additional Truth Information for Evaluation") ===
    # 论文 truth 边是粒子对(双向), 只保留 sender<receiver 的单向边, 过滤后每条边
    # 恰好对应一个母粒子(truth_ids), 与论文原版 lca_truth_matrix 的假设一致。
    # 注意: 所有事件都必须设置这些属性(即使为空), 否则 DataLoader collate 会 KeyError。
    ts = np.asarray(inp.get('truth_senders', [])).ravel() if isinstance(inp, dict) else np.asarray([]).ravel()
    tr = np.asarray(inp.get('truth_receivers', [])).ravel() if isinstance(inp, dict) else np.asarray([]).ravel()
    if 'truth_y' in inp:
        ty = np.asarray(inp['truth_y'])
        ty_cls = ty.reshape(-1, 4).argmax(axis=1) if ty.size > 0 else np.zeros(0, dtype=np.int64)
    else:
        ty_cls = np.zeros(len(ts), dtype=np.int64)
    mask = ts < tr if ts.size > 0 else np.zeros(0, dtype=bool)
    data['truth_senders'] = torch.from_numpy(ts[mask]).long()
    data['truth_receivers'] = torch.from_numpy(tr[mask]).long()
    data['truth_y'] = torch.from_numpy(ty_cls[mask]).long()
    for src, dst in [('truth_ids', 'truth_moth_ids'), ('lca_chain', 'lca_chain'),
                     ('keys', 'final_keys'), ('truth_part_keys', 'truth_part_keys'),
                     ('truth_part_ids', 'truth_part_ids')]:
        arr = np.asarray(inp.get(src, [])).ravel() if isinstance(inp, dict) else np.asarray([]).ravel()
        data[dst] = torch.from_numpy(arr).long()

    # 全局特征 (从原始 npy 的 globals 字段)
    global_val = inp['globals']
    if global_val.ndim == 0:
        global_val = global_val.reshape(1)
    data['globals'].x = torch.from_numpy(global_val).float().reshape(1, -1)

    return data


def convert_event_wrapper(args):
    """用于多线程的包装函数。"""
    inp_path, tgt_path = args
    return convert_single_event(inp_path, tgt_path)


def save_chunk(data_list, out_path):
    """将 HeteroData 列表保存为 zstd 压缩文件。"""
    cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
    buffer = io.BytesIO()
    torch.save(data_list, buffer)
    compressed = cctx.compress(buffer.getvalue())
    with open(out_path, 'wb') as f:
        f.write(compressed)
    return len(compressed)


def batch_convert(name, src_dir, out_dir, events_per_file, prefix, n_threads=N_THREADS, max_events=None):
    """
    批量转换一个数据集。

    Args:
        name: 数据集名称（日志用）
        src_dir: 源 npy 目录
        out_dir: 输出目录
        events_per_file: 每个 chunk 文件的事件数
        prefix: 输出文件前缀 (trn_data_, val_data_, tst_data_)
        n_threads: 并行线程数
        max_events: 最多处理的事件数 (None=全部)
    """
    # 获取所有已配对的 event 编号
    inputs = sorted(glob.glob(os.path.join(src_dir, "input_*.npy")))
    event_nums = sorted(set(
        int(os.path.basename(f).replace("input_", "").replace(".npy", ""))
        for f in inputs
    ))

    # 测试模式下只取前 max_events 个事件
    if max_events is not None:
        event_nums = event_nums[:max_events]

    # 检查已有输出
    existing = set(glob.glob(os.path.join(out_dir, f"{prefix}*.zst")))
    if existing:
        print(f"  ⚠️  输出目录已有 {len(existing)} 个 {prefix} 文件，跳过已存在的...")
        # 简单处理：不跳过，重新生成
        for f in existing:
            os.remove(f)
        print(f"     已清除，重新生成")

    n_events = len(event_nums)
    n_chunks = (n_events + events_per_file - 1) // events_per_file
    print(f"  [{name}] {n_events} events → {n_chunks} chunk files ({events_per_file} evt/file)")

    total_compressed = 0
    t_start = time.time()

    for chunk_id in range(n_chunks):
        chunk_start = chunk_id * events_per_file
        chunk_end = min(chunk_start + events_per_file, n_events)
        chunk_nums = event_nums[chunk_start:chunk_end]

        # 收集文件路径
        file_pairs = []
        for num in chunk_nums:
            inp = os.path.join(src_dir, f"input_{num}.npy")
            tgt = os.path.join(src_dir, f"target_{num}.npy")
            if os.path.exists(inp) and os.path.exists(tgt):
                file_pairs.append((inp, tgt))

        # 并行转换
        chunk_data = []
        if n_threads > 1 and len(file_pairs) > 1:
            with ThreadPool(processes=min(n_threads, len(file_pairs))) as pool:
                results = pool.map(convert_event_wrapper, file_pairs)
            chunk_data = [r for r in results if r is not None]
        else:
            for pair in file_pairs:
                result = convert_single_event(*pair)
                if result is not None:
                    chunk_data.append(result)

        # 保存
        if chunk_data:
            out_path = os.path.join(out_dir, f"{prefix}{chunk_id:03d}.zst")
            compressed_size = save_chunk(chunk_data, out_path)
            total_compressed += compressed_size

        # 进度
        elapsed = time.time() - t_start
        rate = (chunk_id + 1) / elapsed if elapsed > 0 else 0
        eta = (n_chunks - chunk_id - 1) / rate if rate > 0 else 0
        mb_per_file = total_compressed / (chunk_id + 1) / 1024 / 1024 if chunk_id >= 0 else 0
        print(f"    chunk {chunk_id+1:3d}/{n_chunks} | "
              f"{len(chunk_data):3d} events | "
              f"{compressed_size/1024/1024:.1f} MB | "
              f"进度 {rate*60:.1f} chunk/min | "
              f"ETA {eta:.0f}s", end="\r")

        # 定期 GC
        if chunk_id % 5 == 0:
            gc.collect()

    elapsed = time.time() - t_start
    avg_mb = total_compressed / n_chunks / 1024 / 1024
    print(f"\n    ✅ 完成! {n_chunks} files, "
          f"{total_compressed/1024/1024/1024:.2f} GB, "
          f"平均 {avg_mb:.1f} MB/file, "
          f"耗时 {elapsed:.0f}s ({n_events/elapsed:.0f} evt/s)")


def validate_chunks(out_dir, prefix, expected_count=None):
    """验证转换后的 chunk 文件。"""
    files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}*.zst")))
    if expected_count and len(files) != expected_count:
        print(f"  ⚠️  文件数不匹配: 期望 {expected_count}, 实际 {len(files)}")

    dctx = zstd.ZstdDecompressor()
    total_events = 0
    for fpath in files[:3]:  # 只检查前3个
        with open(fpath, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                decompressed = reader.read()
                data_list = torch.load(io.BytesIO(decompressed), weights_only=False)
                total_events += len(data_list)
                for i, d in enumerate(data_list[:2]):
                    n_ft = len(torch.unique(d['tracks'].ft))
                    n_lcag = len(torch.unique(d[('tracks', 'to', 'tracks')].y))
                    print(f"    [{os.path.basename(fpath)} evt{i}]: "
                          f"tracks={d['tracks'].x.shape[0]}, "
                          f"pvs={d['pvs'].x.shape[0]}, "
                          f"FT={n_ft}cls, LCAG={n_lcag}cls")
    return total_events


def main():
    parser = argparse.ArgumentParser(description="批量转换 npy 数据为 zst chunk 格式")
    parser.add_argument("--threads", type=int, default=N_THREADS, help="并行线程数")
    parser.add_argument("--validate-only", action="store_true", help="仅验证已有的转换结果")
    parser.add_argument("--test", action="store_true", help="快速测试模式 (仅转换少量事件)")
    args = parser.parse_args()

    n_threads = args.threads
    print("=" * 60)
    print("完整数据集批量转换")
    print(f"源训练数据: {SRC_TRAIN}")
    print(f"源验证数据: {SRC_VAL}")
    print(f"源测试数据: {SRC_TEST}")
    print(f"输出目录:   {OUT_BASE}/{SAMPLE}/")
    print(f"并行线程:   {n_threads}")
    print("=" * 60)

    out_dir = os.path.join(OUT_BASE, SAMPLE)
    os.makedirs(out_dir, exist_ok=True)

    is_test = args.test
    if is_test:
        print("\n🔬 快速测试模式 - 仅转换少量事件")

    if args.validate_only:
        print("\n验证已有转换结果...")
        total = 0
        for prefix in ["trn_data_", "val_data_", "tst_data_"]:
            n = validate_chunks(out_dir, prefix)
            total += n
        print(f"\n共 {total} 个事件")
        return

    # 1. 训练数据
    print("\n[1/3] 训练数据转换")
    max_events = 4 if is_test else None
    batch_convert("训练", SRC_TRAIN, out_dir, TRN_EVENTS_PER_FILE, "trn_data_", n_threads, max_events=max_events)

    # 2. 验证数据
    print("\n[2/3] 验证数据转换")
    batch_convert("验证", SRC_VAL, out_dir, VAL_EVENTS_PER_FILE, "val_data_", n_threads, max_events=max_events)

    # 3. 测试数据
    print("\n[3/3] 测试数据转换")
    batch_convert("测试", SRC_TEST, out_dir, TST_EVENTS_PER_FILE, "tst_data_", n_threads, max_events=max_events)

    # 4. 最终验证
    print("\n" + "=" * 60)
    print("最终验证")
    print("=" * 60)
    for prefix, expected in [("trn_data_", None), ("val_data_", None), ("tst_data_", None)]:
        files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}*.zst")))
        total_size = sum(os.path.getsize(f) for f in files)
        print(f"  {prefix}*: {len(files)} files, {total_size/1024/1024/1024:.2f} GB")

    # 抽样验证
    print("\n抽样验证 (每类前2个chunk)...")
    validate_chunks(out_dir, "trn_data_")
    validate_chunks(out_dir, "val_data_")
    validate_chunks(out_dir, "tst_data_")

    print(f"\n{'='*60}")
    print(f"✅ 全部转换完成!")
    print(f"输出目录: {out_dir}")
    print(f"使用时将配置文件 data_dir 指向: {OUT_BASE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
