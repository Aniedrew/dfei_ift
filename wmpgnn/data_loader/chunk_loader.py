import gc
import glob
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from wmpgnn.data_loader.helper import *


def permute_event_order(evt, seed):
    """在单个事件内随机重排节点编号（轨道节点与 PV 节点各一个置换）。

    用途：检验模型/管线是否依赖输入数组的"排列顺序"。
    该置换是**一致的**：节点特征、逐节点标签、以及两张边表的端点索引同步重映射，
    因此图本身（连边关系 + 每条边/每个节点的属性）完全不变，只是编号被换了。
    纯消息传递 GNN 没有位置编码、也不读索引，所以理论上指标应当逐位不变；
    若指标明显变化，说明管线里有东西在吃"顺序"（例如并列时按序取第一个、top-k 截断）。

    ⚠️ 约定（2026-09-16 修正过一处错）：设新编号 j 承载旧编号 pi[j] 的节点，
       即 x_new = x_old[pi]。那么**旧边 (a,b) 在做完重编号后必须变成
       (pi_inv[a], pi_inv[b])** —— 不是 (pi[a], pi[b])！后者会把图接错线
       （等价于同时做了置换和"边反向映射"，图被改掉了），从而得到假阳性结论。

    seed 只影响置换本身，用于跑不同 seed 来得到"置换噪声带"。
    """
    n = int(evt['tracks'].x.shape[0])
    npv = int(evt['pvs'].x.shape[0])
    g = torch.Generator().manual_seed(int(seed))
    pi = torch.randperm(n, generator=g)
    piv = torch.randperm(npv, generator=g)
    inv = torch.empty_like(pi); inv[pi] = torch.arange(n)
    inv_v = torch.empty_like(piv); inv_v[piv] = torch.arange(npv)

    # 逐节点属性（x / ft / pid / part_keys ...）：首维与节点数相同的一律同步置换
    for store, perm, size in ((evt['tracks'], pi, n), (evt['pvs'], piv, npv)):
        for key in list(store.keys()):
            val = store[key]
            if torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == size:
                setattr(store, key, val[perm])

    # 轨道-轨道边：两端都是轨道索引 -> 用 pi 的逆
    tt = evt[('tracks', 'to', 'tracks')]
    tt.edge_index = inv[tt.edge_index]

    # 轨道-PV 边：第 0 行是轨道索引，第 1 行是 PV 索引
    tp = evt[('tracks', 'to', 'pvs')]
    ei = tp.edge_index.clone()
    ei[0] = inv[ei[0]]
    ei[1] = inv_v[ei[1]]
    tp.edge_index = ei
    return evt


class ChunkDataset(IterableDataset):
    # Loading a chunk of the dataset to cpu memory instead of all files
    def __init__(self, file_paths, configs, mode="train", n_chunks=32):
        super().__init__()
        self.file_paths = file_paths
        self.n_chunks = n_chunks
        self.mode = mode
        self.configs = configs
        cumulative_sizes = [0]
        total = 0
        file_index = []

        for i, key in enumerate(self.file_paths.keys()):
            total += len(self.file_paths[key])
            cumulative_sizes.append(total)
            file_index.append(self._generate_groups(n_chunks=self.n_chunks,
                                                    low=cumulative_sizes[i],
                                                    high=cumulative_sizes[i + 1]))
        self.chunk_index = torch.cat(file_index, dim=1).to(torch.long)
        self.files_per_chunk = self.chunk_index.shape[1]
        self.n_files = {}
        for sample in self.file_paths.keys():
            self.n_files[sample] = len(self.file_paths[sample])
        self.file_paths = list(chain.from_iterable(self.file_paths[sample] for sample in self.file_paths.keys()))

        self.seeds = torch.randint(0, 1000, (1000,))
        self.seed_tracker = 0

    @staticmethod
    def _generate_groups(n_chunks: int, low: int, high: int)-> torch.Tensor:
        groups = []
        available = np.arange(low, high)
        interval_size = high - low
        chunksize = int(np.ceil(interval_size / n_chunks))

        for i in range(n_chunks):
            if len(available) >= chunksize:
                indices = np.random.choice(len(available), size=chunksize, replace=False)
                group = available[indices]
                available = np.delete(available, indices)
            else:
                if len(available) > 0:
                    group = np.concatenate([
                        available,
                        np.random.choice(np.arange(low, high), size=chunksize - len(available), replace=True)
                    ])
                else:
                    group = np.random.choice(np.arange(low, high), size=chunksize, replace=True)
                available = np.array([])
            groups.append(group.tolist())
        return torch.tensor(groups)

    def _load_chunk(self, chunk_number, mode="loading"):
        import time
        nevnts = 800
        if self.mode == "validation" or self.mode == "test":
            nevnts = 200

        chunk = self.chunk_index[chunk_number]
        files = [self.file_paths[i] for i in chunk]
        desc = f"Loading {self.mode} chunk {chunk_number + 1}/{self.n_chunks} ({self.files_per_chunk} files, ~{self.files_per_chunk * nevnts} events)"
        # 诊断日志: 记录加载起点, 便于定位卡死发生在哪个 chunk
        print(f"[chunk] {time.strftime('%H:%M:%S')} start {desc}", flush=True)
        if mode == "loading":
            dataset = []
            load_dataset_part = partial(load_dataset, configs=self.configs, mode=mode)
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for fi, r in enumerate(tqdm(pool.imap(load_dataset_part, files), total=len(files), desc=desc, leave=False)):
                    dataset.extend(r)
                    print(f"[chunk] {time.strftime('%H:%M:%S')} file {fi + 1}/{len(files)} loaded ({len(r)} events)", flush=True)
            print(f"[chunk] {time.strftime('%H:%M:%S')} done chunk {chunk_number + 1} ({len(dataset)} events)", flush=True)
            return dataset
        elif mode == "weights":
            weights = {}
            load_dataset_part = partial(load_dataset, configs=self.configs, mode="weights_only")
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset_part, files), total=len(files), desc=desc, leave=False):
                    for key, value in r.items():
                        if key not in weights:
                            weights[key] = value
                        else:
                            weights[key] += value
            return weights
        else:
            raise NotImplementedError

    def __iter__(self):
        # assuming 4 num workers which each want to load a chunk
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_chunks
        else:
            worker_id = worker_info.id
            per_worker = int(np.ceil(self.n_chunks / worker_info.num_workers))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_chunks)

        for chunk_number in range(iter_start, iter_end):
            # for chunk_number in range(self.n_chunks):
            chunk_events = self._load_chunk(chunk_number)

            # create index
            idx = torch.arange(0, len(chunk_events))
            if self.mode == "train":
                idx = idx[torch.randperm(len(idx))]
            else:
                # val and test are shuffled
                g = torch.Generator()
                g.manual_seed(self.seeds[self.seed_tracker].item())
                self.seed_tracker += 1
                idx = idx[torch.randperm(len(idx), generator=g)]

            # Yield events in shuffled order
            perm_on = bool(self.configs.get("settings", {}).get("permute_graph_order", False))
            perm_seed = int(self.configs.get("evaluate", {}).get("permute_seed", 111))
            for i, i_idx in enumerate(idx):
                evt = chunk_events[i_idx]
                if perm_on:
                    # 顺序置换检验（见 permute_event_order 注释）
                    evt = permute_event_order(evt, perm_seed + i)
                yield evt

            del chunk_events
            gc.collect()

    def get_weights(self):
        weights = {}
        n_chunks = self.chunk_index.shape[0]
        for i in range(min(10, n_chunks)):
            weights[i] = self._load_chunk(i, mode="weights")
        return weights


class ChunkLoader(pl.LightningDataModule):
    # Data container for pytorch lightning module
    def __init__(self, configs, trn_dataset=None, val_dataset=None, tst_dataset=None, batch_size=None, num_workers=None):
        super().__init__()
        self.trn_dataset = trn_dataset
        self.val_dataset = val_dataset
        self.tst_dataset = tst_dataset
        self.configs = configs

        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            self.batch_size = self.configs["settings"]["batch_size"]
        if isinstance(num_workers, int):  # we need to be cautious to not load too much to cpu mem
            self.num_workers = num_workers
        else:
            self.num_workers = self.configs["settings"]["ncpu"] * 2

    def train_dataloader(self):
        if not isinstance(self.trn_dataset, type(None)):
            trn_loader = DataLoader(self.trn_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("trn_dataset must not be None")
        return trn_loader

    def val_dataloader(self):
        if not isinstance(self.val_dataset, type(None)):
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("val_dataset must not be None")
        return val_loader

    def test_dataloader(self):
        if not isinstance(self.tst_dataset, type(None)):
            tst_loader = DataLoader(self.tst_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=False, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("tst_dataset not must be None")
        return tst_loader


def get_trn_val_loaders(_configs) -> ChunkLoader:
    data_dir = _configs["settings"]["data_dir"]
    num_workers = _configs["settings"]["ncpu"] * 2
    nfiles = get_nfiles(_configs["settings"])

    """Training"""
    path_dict = {}
    for sample, files in nfiles.items():
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:files]

    # Number of chunks definition and safeguard for the files per chunk
    # Lower threshold (= fewer files per chunk) reduces per-chunk memory footprint.
    # Changed from 8 to 2 to support datasets with large per-file size (e.g. public LHCb data).
    min_files\
        = min(len(v) for v in path_dict.values() if len(v) > 0)
    total_files = sum(len(v) for v in path_dict.values())
    num_chunks = np.ceil(min_files / num_workers).astype(int)
    # Safeguard: increase chunks until files_per_chunk < threshold
    files_per_chunk_thr = 2
    while total_files / num_chunks >= files_per_chunk_thr:
        num_chunks += num_workers
    print(f"Number of chunks: {num_chunks}")
    print(f"Files per chunk: {np.ceil(total_files / num_chunks).astype(int)}")

    trn_dataset = ChunkDataset(path_dict, _configs, mode="train", n_chunks=num_chunks)

    """Validation"""
    path_dict = {}
    for sample, files  in nfiles.items():
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:5]
    val_dataset = ChunkDataset(path_dict, _configs, mode="validation", n_chunks=num_chunks)

    return ChunkLoader(_configs, trn_dataset=trn_dataset, val_dataset=val_dataset)


def get_tst_loader(_configs) -> ChunkLoader:
    data_dir = _configs["settings"]["data_dir"]
    nfiles = get_nfiles(_configs["evaluate"])

    """Testing"""
    path_dict = {"testing": []}
    for sample, files in nfiles.items():
        path_dict["testing"].append(sorted(glob.glob(f'{data_dir}/{sample}/tst_data_*'))[:files])
    path_dict["testing"] = sum(path_dict["testing"], [])
    # Each file is saved individually in a chunk
    batch_size = 32  # Reduced from 128 to avoid GPU OOM (10.57 GB limit, public data has ~150 tracks/evt)
    tst_dataset = ChunkDataset(path_dict, _configs, mode="test", n_chunks=len(path_dict["testing"]))

    return ChunkLoader(_configs, tst_dataset=tst_dataset, batch_size=batch_size, num_workers=2)
