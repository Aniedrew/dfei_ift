"""Learned Deterministic Annealing (DA) 聚类 —— 推理侧 track->PV 分簇。

背景 (2026-08-31 方向重启):
- 之前方案 (v40-42) 是"训练侧切子图 + Gumbel 硬分配", 结论: 训练侧切子图伤模型 (class1 掉 20pp),
  推理侧手工几何分簇无效 (-0.56pp)。
- 本模块实现 CMS 生产级 DA 聚类 (Chabanat 2003 / Rose 1990) 的确定性退火核心,
  亲和度由**可训练 MLP** (GNN latent 表征 -> 低维嵌入) 提供, 与手工几何距离解耦。

核心算法 (确定性退火, 无 Gumbel 采样):
1. 从单簇开始 (或给定初始中心), 温度 T 从 T_start 指数退到 T_end。
2. 每个温度下迭代: 软分配 p(i|k) = softmax(-d_ik / T), 更新中心 c_k。
3. 分裂 (deterministic annealing 的关键): 对每个簇中心做特征值分解,
   若簇在某个主方向的方差 > 分裂阈值 (对应 T 下降跨过相变点), 则分裂为两个中心。
   这样簇数自适应, 不需要预先指定 (PV finding 里 = PV 数)。
4. 返回最终分配 argmax_k p(i|k) (置信度低于 conf_thr 的归 -1 噪声组)。

用法 (推理侧):
    clu = DeterministicAnnealingClusterer(T_start=10.0, T_end=0.05, n_anneal=60,
                                          max_clusters=16, split_thr=1.0)
    labels = clu.fit_predict(X)          # X: [N, dim] 亲和度嵌入
"""
import math

import torch


class DeterministicAnnealingClusterer:
    def __init__(self, T_start=10.0, T_end=0.05, n_anneal=60,
                 max_clusters=16, split_thr=1.0, improve_fac=0.02, merge_fac=2.0,
                 em_iters=20, min_cluster_size=3, conf_thr=0.0, seed=None):
        """
        Args:
            T_start: 起始温度 (高 = 所有点一个簇).
            T_end:   终止温度 (低 = 硬分配).
            n_anneal: 温度步数 (指数退火).
            max_clusters: 簇数上限.
            split_thr: 临界温度判据系数 (vmax/T > split_thr 才进入分裂候选).
            improve_fac: 拟合改善判据 (分裂后加权最近距离² 须下降 > improve_fac, 默认 0.05).
            merge_fac: 合并后处理系数 (中心距离 < merge_fac * σ_hat 的簇合并;
                       σ_hat = 点到最近中心的 75 分位数尺度; 默认 2.0).
            em_iters: 每个温度下的软 EM 迭代次数.
            min_cluster_size: 分裂后最小簇成员数, 小于则撤销分裂.
            conf_thr: 最终分配置信度阈值, 低于则归 -1.
            seed: 随机种子 (分裂扰动).
        """
        self.T_start = T_start
        self.T_end = T_end
        self.n_anneal = n_anneal
        self.max_clusters = max_clusters
        self.split_thr = split_thr
        self.improve_fac = improve_fac
        self.merge_fac = merge_fac
        self.em_iters = em_iters
        self.min_cluster_size = min_cluster_size
        self.conf_thr = conf_thr
        self.seed = seed

    # ==== 温度退火: 指数序列 ====
    def _temperature_schedule(self):
        ts = []
        for i in range(self.n_anneal):
            frac = i / max(self.n_anneal - 1, 1)
            T = self.T_start * (self.T_end / self.T_start) ** frac
            ts.append(T)
        return ts

    @staticmethod
    def _soft_assign(X, centers, T):
        """软分配 p(i|k) = softmax(-d_ik / T), d_ik = ||x_i - c_k||^2."""
        # [N, K] = ||x||^2 - 2 x·c + ||c||^2
        d2 = (X.pow(2).sum(1, keepdim=True) - 2 * X @ centers.T
              + centers.pow(2).sum(1, keepdim=True).T)
        d2 = d2.clamp(min=0)
        P = torch.softmax(-d2 / max(T, 1e-9), dim=1)   # [N, K]
        return P

    def _update_centers(self, X, P):
        denom = P.sum(0).clamp(min=1e-9)
        return (P.T @ X) / denom.unsqueeze(1)          # [K, dim]

    def _split_centers(self, X, P, centers, T):
        """DA 分裂: 临界温度候选 + 拟合改善判据确认。

        候选: 簇主方向方差 λ_max / T > split_thr (温度跨过临界温度)。
        确认: 分裂后 (两中心 ±ε 扰动, 局部 EM 数轮) 加权最近距离² 改善 > improve_fac。
        """
        new_centers = []
        for k in range(centers.shape[0]):
            pk = P[:, k]                               # [N]
            if pk.sum() < self.min_cluster_size:
                new_centers.append(centers[k])
                continue
            # 加权协方差
            denom = pk.sum()
            mu = (pk.unsqueeze(1) * X).sum(0) / denom  # [dim]
            Xc = X - mu.unsqueeze(0)
            cov = (pk.unsqueeze(1) * Xc).T @ Xc / denom
            try:
                evals, evecs = torch.linalg.eigh(cov)  # 升序
                vmax = evals[-1]
            except Exception:
                vmax = torch.zeros((), device=cov.device)
                evecs = torch.eye(cov.shape[0], device=cov.device)

            if vmax / max(T, 1e-9) <= self.split_thr or vmax <= 1e-6:
                new_centers.append(centers[k])          # 未跨临界温度: 不分裂
                continue

            # ==== 拟合改善判据: 分裂后加权最近距离² 下降 > improve_fac 才接受 ====
            # (替代自由能: T 极低时 exp(-d²/T) 下溢导致自由能比较不稳定, 误拒合理分裂)
            # 注意: DA 对任意数据 (含单高斯) 在低温有过分裂倾向, 由 fit 末尾
            # _merge_close_centers 合并兜底 (CMS DA 同款后处理)。
            w = pk / denom                              # 簇内软权重 [N]
            def _obj(centers_k):
                d2k = (X.pow(2).sum(1, keepdim=True) - 2 * X @ centers_k.T
                       + centers_k.pow(2).sum(1, keepdim=True).T).clamp(min=0)
                return (w * d2k.min(dim=1).values).sum()
            obj_no = _obj(mu.unsqueeze(0))
            ev = evecs[:, -1]
            eps = math.sqrt(max(vmax.item(), 1e-9)) * 0.5
            cand = torch.stack([mu + eps * ev, mu - eps * ev])   # [2, dim]
            # 局部 EM: 仅候选两中心软分配, 收敛数轮
            for _ in range(5):
                cand = self._update_centers(X, self._soft_assign(X, cand, T))
            obj_sp = _obj(cand)
            if obj_sp < obj_no * (1.0 - self.improve_fac):
                # ==== 冗余检查: 新中心与已有中心过近 -> 同一数据簇被重复分裂, 拒绝 ====
                d_new = torch.cdist(cand, centers)
                scale = math.sqrt(max(vmax.item(), 1e-9))
                if d_new.min() > 0.1 * scale:
                    new_centers.append(cand[0].detach())
                    new_centers.append(cand[1].detach())
                else:
                    new_centers.append(centers[k])
            else:
                new_centers.append(centers[k])
        centers = torch.stack(new_centers)
        if centers.shape[0] > self.max_clusters:        # 上限保护: 截断
            centers = centers[:self.max_clusters]
        return centers

    def _merge_close_centers(self, X, centers):
        """合并后处理: 层次合并距离过近的簇 (DA 过分裂兜底)。

        阈值 = merge_fac * σ_hat; σ_hat = 点到最近中心距离的 75 分位数
        (比 MAD 更贴近真实簇尺度, 对过分裂更鲁棒)。默认 merge_fac=2.0:
        相距 < 2σ 的簇 (伪分裂) 合并。
        """
        if centers.shape[0] <= 1:
            return centers
        # 尺度: 点到最近中心的距离的 75 分位数
        d2 = (X.pow(2).sum(1, keepdim=True) - 2 * X @ centers.T
              + centers.pow(2).sum(1, keepdim=True).T).clamp(min=0)
        nn = d2.min(dim=1).values.sqrt()
        sigma = nn.quantile(0.75).clamp(min=1e-6)
        thr = self.merge_fac * sigma
        centers = centers.clone()
        merged = True
        while merged and centers.shape[0] > 1:
            merged = False
            d = torch.cdist(centers, centers)
            # 最近簇对
            i, j = torch.triu_indices(centers.shape[0], centers.shape[0], offset=1)
            d_ij = d[i, j]
            kmin = d_ij.argmin()
            if d_ij[kmin] > thr:
                break
            ci, cj = int(i[kmin]), int(j[kmin])
            # 合并: 加权平均
            n_i = torch.clamp(self._final_P[:, ci].sum() if hasattr(self, "_final_P") else torch.tensor(1.0), min=1e-9)
            n_j = torch.clamp(self._final_P[:, cj].sum() if hasattr(self, "_final_P") else torch.tensor(1.0), min=1e-9)
            newc = (n_i * centers[ci] + n_j * centers[cj]) / (n_i + n_j)
            keep = [k for k in range(centers.shape[0]) if k not in (ci, cj)]
            centers = torch.stack([centers[k] for k in keep] + [newc])
            merged = True
        return centers

    def fit(self, X):
        """输入 X [N, dim], 执行确定性退火, 保存 final P."""
        X = X.float()
        N = X.shape[0]
        if N == 0:
            self._final_P = torch.zeros((0, 0))
            return self
        g = torch.Generator().manual_seed(self.seed) if self.seed is not None else None

        # 初始: 单簇 (中心 = 全局均值)
        centers = X.mean(0, keepdim=True)
        # 退火
        for T in self._temperature_schedule():
            for _ in range(self.em_iters):
                P = self._soft_assign(X, centers, T)
                centers = self._update_centers(X, P)
            centers = self._split_centers(X, P, centers, T)
            if centers.shape[0] >= self.max_clusters:
                break
        # 合并后处理: 消除 DA 过分裂的伪簇
        centers = self._merge_close_centers(X, centers)
        self._final_P = self._soft_assign(X, centers, self.T_end)
        self._centers = centers
        return self

    def predict(self, conf_thr=None):
        """最终分配 [N], 置信度低于阈值 -> -1."""
        thr = self.conf_thr if conf_thr is None else conf_thr
        pmax, arg = self._final_P.max(dim=1)
        labels = arg.clone()
        if thr > 0:
            labels[pmax < thr] = -1
        return labels

    def fit_predict(self, X, conf_thr=None):
        self.fit(X)
        return self.predict(conf_thr)

    @property
    def n_clusters(self):
        return self._centers.shape[0]
