# CERN lxplus Agent 提示词 - 数据迁移任务

## 你的身份

你当前位于 **CERN lxplus 登录节点**（lxplus.cern.ch），以用户 `qingxian` 的身份操作。

## ⚠️ CERN 使用规则（必须遵守）

1. **严禁** 将文件放在 AFS 家目录 `/afs/cern.ch/user/q/qingxian/` 下。AFS 有严格的配额限制，且不适合大文件操作。
2. **中转目录**：所有需要暂存的文件统一放到 EOS 空间：
   ```
   /eos/user/q/qingxian/
   ```
3. **避免在登录节点上运行长时间、高负载任务**。大文件传输请在提交的作业（batch job）中完成，或用 `screen`/`tmux` 在后台运行。
4. **尊重他人资源**：不要占用过多 CPU/IO 资源。
5. **遵守 CERN 计算规则**，不要尝试绕过安全限制。

## 任务目标

将原作者预处理的 zst chunk 数据从 CERN EOS 拷贝到兰州大学计算集群。这些数据包含了**PID（粒子识别）信息**，对模型训练很重要。

### 源数据位置

```
/eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/
```

这是原作者的 EOS 空间，数据格式已经是 zst chunk 格式（我们的代码可以直接读取）。

### 目标位置

```
/lzufs/user/guoqingxiang/DFEI_data/CERN_data/
```

这是兰州大学集群的 Lustre 文件系统（大容量、适合存储数据）。

## 操作步骤

### 步骤 1：检查源数据

先查看数据目录结构和大小，不要直接盲目拷贝：

```bash
# 用 eos 命令查看目录结构
eos root://eosuser.cern.ch ls /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/

# 查看每个 sample 目录的大小
for dir in $(eos root://eosuser.cern.ch ls /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/); do
    size=$(eos root://eosuser.cern.ch find --size /eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/$dir 2>/dev/null | awk -F'size=' '{sum+=$2} END {print sum/1024/1024 " MB"}')
    echo "$dir: $size"
done
```

需要重点关注：
- 有哪些 sample 子目录（如 `00342442_inclusive` 等）
- 每个目录下有多少个 `trn_data_*`、`val_data_*`、`tst_data_*` 文件
- 文件是否包含 `pid` 字段（可以随机挑一个 zst 文件解压检查）

### 步骤 2：检查目标位置是否可访问

先确认兰州大学的服务器是否可达：

```bash
ssh -o ConnectTimeout=10 -o BatchMode=yes guoqingxiang@lzulogin.hep.lzu.edu.cn "echo 'LZU reachable'" 
```

如果 SSH 不通，可能需要密码认证——这种情况下不要卡住，停下來告诉用户。

### 步骤 3：选择传输方式

#### 方案 A：rsync（推荐，支持断点续传）

```bash
# 先在 CERN 上将数据拷贝到中转目录
mkdir -p /eos/user/q/qingxian/temp_transfer/
# 或者直接用 EOS cp（更快，不经过本地）
xrdcp -r root://eosuser.cern.ch//eos/user/y/yukaiz/DFEI_IFT/cached_data/DFEI_LHCb_LHCbcollision_normed_pt_data/ /eos/user/q/qingxian/temp_transfer/
```

然后从 CERN 推到 LZU：

```bash
# 如果 SSH key 已配置，可以直接 rsync
rsync -avzP --progress /eos/user/q/qingxian/temp_transfer/ guoqingxiang@lzulogin.hep.lzu.edu.cn:/lzufs/user/guoqingxiang/DFEI_data/CERN_data/
```

#### 方案 B：分步压缩传输（大文件推荐）

```bash
# 1. 先将数据压缩成一个 tar 包（在中转目录）
cd /eos/user/q/qingxian/
tar -czf DFEI_LHCb_data.tar.gz -C /eos/user/y/yukaiz/DFEI_IFT/cached_data/ DFEI_LHCb_LHCbcollision_normed_pt_data/

# 2. 用 scp 传输（单个大文件比海量小文件快）
scp -c aes128-ctr DFEI_LHCb_data.tar.gz guoqingxiang@lzulogin.hep.lzu.edu.cn:/lzufs/user/guoqingxiang/DFEI_data/
```

### 步骤 4：数据验证（LZU 侧）

传输完成后，在 LZU 上验证数据完整性：

```bash
# 如果用了 tar，则解压
cd /lzufs/user/guoqingxiang/DFEI_data
tar -xzf DFEI_LHCb_data.tar.gz

# 检查目录结构
ls CERN_data/
ls CERN_data/*/ | head -20

# 检查文件数
echo "trn: $(ls CERN_data/*/trn_data_* 2>/dev/null | wc -l)"
echo "val: $(ls CERN_data/*/val_data_* 2>/dev/null | wc -l)"
echo "tst: $(ls CERN_data/*/tst_data_* 2>/dev/null | wc -l)"

# 随机检查 zst 文件是否包含 PID 字段
python3 -c "
import zstandard as zstd, io, torch
with open('CERN_data/00342442_inclusive/trn_data_000.zst', 'rb') as f:
    data = torch.load(io.BytesIO(zstd.decompress(f.read())), weights_only=False)
print('Total events:', len(data))
print('Has pid:', hasattr(data[0]['tracks'], 'pid'))
print('Keys:', data[0]['tracks'].keys())
"
```

## 需要询问用户的问题

在开始任何操作前，先问清楚：

1. **是否已配置 SSH key？** 从 CERN 到 LZU 的 SSH 免密登录是否已设置？如果没有，需要先配置。
2. **需要传输哪些 sample？** 是全部传输还是只传 inclusive 训练数据？
3. **可接受的传输时间？** 数据量估计在 50-200 GB，传输可能需要几小时到一天。

## 可能遇到的问题

| 问题 | 处理方式 |
|------|---------|
| EOS 访问权限不足 | 原作者的 EOS 可能未公开，联系用户确认是否有读权限 |
| SSH 连接中断 | 使用 `screen` 或 `tmux` 运行传输命令，防止断连 |
| 传输速度慢 | 先 `tar` 打包再单文件传输，比传海量小文件快得多 |
| 磁盘空间不足 | EOS 空间 `/eos/user/q/qingxian/` 一般有 1TB，足够做中转 |
| LZU 目标目录不存在 | 先 SSH 到 LZU 创建目录 |

## 完成后需要告知用户的内容

1. 传输了哪些文件（列表）
2. 总数据量大小
3. 传输耗时
4. 校验结果（文件数、PID 是否存在）
5. 下一步建议（更新配置文件 `data_dir` → 新路径）
