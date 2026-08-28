## Hard Constraints
- GPU memory limit is 10GB, requiring careful batch size and model configuration
- Training must use the 'yukai_IFT' branch with the class weight bug fix (LCA__weights → LCA_weights)
- CERN official MC data (DFEI_IFT_20260702) has a known track-sorting issue that may impact DFEI performance

## Engineering Conventions
- Reconstruction pipeline uses greedy pruning (edge thresholding + sequential clustering) leading to ~30% chain loss
- Multi-task GNN architecture with shared backbone and separate heads for LCAG classification, node pruning, edge pruning, and PV association
- Training uses inverse frequency class weights for imbalanced LCAG categories (LCA weights: [0.25, 1024, 341, 1838])

## Lessons Learned
- Class weight misconfiguration (typo in config key) caused LCAG class 1 accuracy to drop to 0%, severely limiting PerfectReco rate
- Training speed is slow (30-50 min/epoch) due to GPU memory constraints and large event size (91-139 tracks/event)
- Hard threshold pruning leads to significant signal chain loss; top-k edge selection and Gumbel-Softmax node masking show promise for improvement
## 2026-08-24 方向调整（用户明确）
- **评估口径转向 trigger**：AllParticles 比 Perfect 更关键（trigger 场景）。class1/class2 区分是物理难题，优先级下调，不再作为紧迫优化目标。
- **待办：公开数据验证**：用论文公开数据集训练并与论文结果对比，验证 v38 提升真实可靠（需拿到公开数据集 + 配训练配置）。
- **PV 分簇方向重启**：之前的方法（hard argmax + Gumbel + 训练侧切子图）太原始且失败。已调研领域算法，选定方向：**学习式确定性退火（learned Deterministic Annealing）**——用可训练 MLP 提供亲和度 + 温度退火软分配（CMS 生产级 DA 聚类 + 我们的 GNN 亲和度的结合），参考 LHCb PV-finder (Akar) 与 CMS DA (Chabanat 2003)。
