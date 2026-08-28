# 从 CERN EOS 迁移数据到兰州大学集群

## 为什么需要迁移

原作者预处理的 zst chunk 数据（**含 PID 信息**）存储在 CERN EOS：
```
/eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/
```

而我们当前从 Zenodo 下载的 npy 数据**不含 PID**。迁移后可以获得带 PID 的完整数据，模型性能会更好。

---

## 推荐方案：从 CERN lxplus 推到 LZU

在**您的本地终端**（不是在 Trae 里）SSH 到 CERN lxplus，然后执行以下操作。

### 步骤 1：登录 CERN lxplus 并查看数据

```bash
# 在您自己的终端中执行
ssh -XY username@lxplus.cern.ch
```

登录后，先查看数据结构和大小：

```bash
# 检查数据目录结构
EOS_MGM="root://eosuser.cern.ch"
eos $EOS_MGM ls /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/

# 查看各 sample 子目录
eos $EOS_MGM find /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/ --size | head -30

# 大概看看总大小
eos $EOS_MGM find /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/ --size | awk -F'size=' '{sum+=$2} END {print "Total: " sum/1024/1024/1024 " GB"}'
```

### 步骤 2：先压缩再传输

```bash
# 创建压缩归档（保持目录结构）
tar -czf DFEI_LHCb_data.tar.gz \
  -C /eos/user/y/yukaiz/DFEI_IFT/cached_data/ \
  DFEI_LHCb_LHCbcollision_normed_pt_data/

# 或者只传输需要的样本，例如只要 inclusive：
eos $EOS_MGM ls /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/ | grep inclusive
tar -czf inclusive_data.tar.gz \
  -C /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/ \
  00342442_inclusive 00342451_inclusive
```

### 步骤 3：传输到兰州大学

从 lxplus 推送到 LZU（**在 lxplus 上执行**）：

```bash
# 方案 A：scp 直接传输（简单，但大文件可能慢）
scp -c aes128-ctr DFEI_LHCb_data.tar.gz \
  guoqingxiang@lzulogin.hep.lzu.edu.cn:/lzufs/user/guoqingxiang/DFEI_data/

# 方案 B：rsync（支持断点续传）
rsync -avzP --progress -e "ssh -c aes128-ctr" \
  DFEI_LHCb_data.tar.gz \
  guoqingxiang@lzulogin.hep.lzu.edu.cn:/lzufs/user/guoqingxiang/DFEI_data/
```

### 步骤 4：在 LZU 上解压

（在 Trae 或 LZU 登录节点上执行）

```bash
# 解压
cd /lzufs/user/guoqingxiang/DFEI_data
tar -xzf DFEI_LHCb_data.tar.gz

# 更新 symlink
rm -f converted_LHCbcollision
ln -s DFEI_LHCb_data converted_LHCbcollision
```

---

## 备选方案

### 方案 2：xrdcp + grid proxy（如果 EOS 可认证）

如果您的 CERN 账号配置了 grid proxy，可以直接从 LZU 拉取：

```bash
# 在 LZU 上
export X509_USER_PROXY=/path/to/your/proxy
xrdcp -r \
  root://eosuser.cern.ch//eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/ \
  /lzufs/user/guoqingxiang/DFEI_data/CERN_data/
```

### 方案 3：分样本选择性迁移

CERN EOS 上的目录结构可能包含多个 sample：
- `00342442_inclusive` → 训练用 inclusive 数据
- `00342451_inclusive` → 更多训练数据
- `00342629_Bs_Jpsiphi` 等 → exclusive 样本

可以先只迁移 inclusive 样本（用得最多）。

### 方案 4：HTCondor 传输作业

在 CERN 提交一个 HTCondor 作业来传输数据：

```bash
# CERN 上的作业脚本 transfer.sh
#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
xrdcp -r root://eosuser.cern.ch//eos/user/y/yukaiz/... \
  root://lzu.eos.endpoint/.../ || scp -r ... guoqingxiang@lzulogin.hep.lzu.edu.cn:...
```

---

## 注意事项

| 问题 | 建议 |
|------|------|
| **传输中断** | 用 rsync（支持断点续传），或用 `tar` 分卷（`split -b 5G archive.tar.gz "part_"`） |
| **认证** | lxplus → LZU 需要 LZU 密码或 SSH key |
| **速度** | CERN → 中国约 200-500 Mbps，完整数据可能在几小时到一天 |
| **磁盘空间** | `/lzufs/user/USERNAME` 有 **21 TB** 配额，完全够用 |
| **时间估计** | 数据 ~50 GB，压缩后更小，约 30 分钟到 2 小时 |
