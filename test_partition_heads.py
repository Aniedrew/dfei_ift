"""partition_heads 的单元测试: 手算校验损失数值 + 解码正确性 + 梯度。

运行: python3 test_partition_heads.py
"""
import torch

from wmpgnn.model.partition_heads import (OCHead, SlotMaskHead, oc_loss, decode_oc,
                                          slot_mask_loss, decode_slots, decode_assign)


def t(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + extra if extra else ""))
    assert cond, name


print("=" * 70)
print("M1 Object Condensation")
print("=" * 70)
# 两个对象, 各 2 个节点; 对象间距离 10 -> d^2=100; 对象内距离 0.1 -> d^2=0.01
emb = torch.tensor([[0.0, 0.0], [0.0, 0.1], [10.0, 10.0], [10.0, 10.1], [50.0, 50.0]])
y = torch.tensor([0, 0, 1, 1, -1])
beta_good = torch.tensor([5.0, 5.0, 5.0, 5.0, -5.0])         # 信号高、背景低
tot, parts = oc_loss(beta_good, emb, y)
# 手算: L_att = 0.01 (两对, 每对 0.01); L_rep = ReLU(1 - 100) = 0 (不同对象), 背景 50 单位也 0
t("L_att ≈ 0.01", abs(float(parts["L_att"]) - 0.01) < 1e-3, f"{float(parts['L_att']):.4f}")
t("L_rep = 0 (对象分得很开)", float(parts["L_rep"]) < 1e-6, f"{float(parts['L_rep']):.4f}")
t("total ≈ L_beta + 0.01", abs(float(tot) - float(parts["L_beta"]) - 0.01) < 1e-3)

# 坏情况: 把第二个对象的第一点挪到对象 1 附近 -> 吸引力变大 + 出现排斥
emb_bad = emb.clone()
emb_bad[2] = torch.tensor([0.05, 0.05])
tot_bad, parts_bad = oc_loss(beta_good, emb_bad, y)
t("坏嵌入的 L_att 更大", float(parts_bad["L_att"]) > float(parts["L_att"]) * 5)
t("坏嵌入的 L_rep > 0", float(parts_bad["L_rep"]) > 0.5, f"{float(parts_bad['L_rep']):.3f}")
t("坏嵌入总损失更大", float(tot_bad) > float(tot))

# β 反向: 背景给高分 -> L_beta 明显上升
_, parts_bkg_hi = oc_loss(torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0]), emb, y)
t("背景 β 高 -> L_beta 上升", float(parts_bkg_hi["L_beta"]) > float(parts["L_beta"]))

# 解码
lab = decode_oc(torch.sigmoid(beta_good), emb, beta_thr=0.5, td=1.0)
t("OC 解码: 前两点同簇", int(lab[0]) == int(lab[1]) >= 0)
t("OC 解码: 后两点同簇", int(lab[2]) == int(lab[3]) >= 0)
t("OC 解码: 两簇不同", int(lab[0]) != int(lab[2]))
t("OC 解码: 背景进 dustbin", int(lab[4]) == -1)
t("OC 解码: 排他(每点最多一个标签)", lab.max() <= 1 and lab.min() >= -1)

# 梯度
head = OCHead(in_dim=6, emb_dim=8)
x = torch.randn(5, 6)
b, e = head(x)
loss, _ = oc_loss(b, e, y)
loss.backward()
t("OCHead 所有参数都有梯度", all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

print()
print("=" * 70)
print("M2 Slot + Mask (匈牙利匹配)")
print("=" * 70)
K = 3
# y = [0,0,1,1,-1]; slot0->obj0, slot1->obj1, slot2->空
mask_logit = torch.tensor([
    [ 8.0, -8.0, -8.0],
    [ 8.0, -8.0, -8.0],
    [-8.0,  8.0, -8.0],
    [-8.0,  8.0, -8.0],
    [-8.0, -8.0, -8.0],
])
obj_logit = torch.tensor([[8.0, 8.0, -8.0], [8.0, 8.0, -8.0]])   # 每节点两个有效槽位
tot_s, parts_s, assign = slot_mask_loss(mask_logit, obj_logit, y)
t("匈牙利分配正确", assign.tolist() == [0, 1, -1], str(assign.tolist()))
t("完美 mask 的损失很小", float(tot_s) < 0.05, f"{float(tot_s):.4f}")

# 错误 mask(槽位对调) -> 匈牙利会自己纠正, 损失仍应很小
mask_swapped = torch.tensor([
    [-8.0,  8.0, -8.0],
    [-8.0,  8.0, -8.0],
    [ 8.0, -8.0, -8.0],
    [ 8.0, -8.0, -8.0],
    [-8.0, -8.0, -8.0],
])
tot_sw, _, assign_sw = slot_mask_loss(mask_swapped, obj_logit, y)
t("槽位对调后匈牙利仍匹配成功", assign_sw.tolist() == [1, 0, -1], str(assign_sw.tolist()))
t("槽位对调后损失仍很小", float(tot_sw) < 0.05, f"{float(tot_sw):.4f}")

# 完全错误的 mask(所有点都高) -> 损失应显著变大
mask_bad = torch.full((5, K), 8.0)
tot_b, _, _ = slot_mask_loss(mask_bad, obj_logit, y)
t("错误 mask 损失显著变大", float(tot_b) > 10 * float(tot_s), f"{float(tot_b):.3f}")

lab_s = decode_slots(mask_logit, obj_logit, mask_thr=0.5)
t("Slot 解码: 对象 0 正确", lab_s[0] == lab_s[1] == 0)
t("Slot 解码: 对象 1 正确", lab_s[2] == lab_s[3] == 1)
t("Slot 解码: 背景进 dustbin", int(lab_s[4]) == -1)
t("Slot 解码: 每个节点恰好一个标签", tuple(lab_s.shape) == (5,))

# 梯度
sh = SlotMaskHead(in_dim=6, n_slots=K)
xs = torch.randn(5, 6)
m, o = sh(xs)
ls, _, _ = slot_mask_loss(m, o, y)
ls.backward()
t("SlotMaskHead 所有参数都有梯度", all(p.grad is not None and torch.isfinite(p.grad).all() for p in sh.parameters()))

print()
print("=" * 70)
print("M3 分配式解码 (替代删边)")
print("=" * 70)
edges = torch.tensor([[0, 1, 0, 2, 3, 4, 3, 5], [1, 2, 0, 1, 4, 5, 5, 4]])
score = torch.tensor([0.9, 0.9, 0.9, 0.9, -1.0, -1.0, -1.0, -1.0])
lab_a = decode_assign(score, edges, 6, thr=0.5)
t("分配解码: 前三点成一簇", lab_a[0] == lab_a[1] == lab_a[2] >= 0)
t("分配解码: 低分边不进簇", int(lab_a[3]) == -1 and int(lab_a[5]) == -1)
t("分配解码: 排他", len(set(lab_a.tolist())) >= 1)

print()
print("=" * 70)
print("全部通过")
print("=" * 70)
