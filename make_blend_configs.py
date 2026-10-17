"""生成 blend 臂: 以 v654 或 v651 为基底, 自由叠加已验证有效的成分 (2026-10-16)。

已单独验证有效 (前沿残差 >0, 见 COMPREHENSIVE_EVAL / 对话记录):
  ctxprune   = context_prune 注意力上下文剪枝头   (v670, 残差 +6.91)  ← 本轮最大
  rankx2     = edge_rank_weight 10 -> 20 逐对排序  (v667, 残差 +4.99)  ← 第二
  crtau03    = chain_recall(tau=0.3) 温和链级损失  (v660, 召回旋钮)
  ep120      = epochs 60 -> 120                    (v675, clean +4.1pp)
  已证伪不用: seqrec/setpred 解码器(丢链, found 掉到 32%), geomcand 门控(负)

臂:
  blend_a       v654 + ctxprune + rankx2                       -> 1201
  blend_b       v651 + ctxprune + rankx2                       -> 1202
  blend_c       v651 + ctxprune + rankx2 + crtau03             -> 1203
  blend_d       v651 + ctxprune + rankx2 + rankx4(40)          -> 1204   (权重是否还要更大)
  faircombo     v651 + crtau03 + ep120 + patience60            -> 1205   (组合臂公平重跑, 去掉解码器)
"""
import re

GN_ANCHOR = "    extra_edge_dim: 18"
E_ANCHOR = "  chain_contrastive_weight: 0.5"
CTX = ["    context_prune:", "      mode: attn", "      heads: 4",
       "      hidden: 32", "      refine_edge: true"]
CR = ["  chain_recall_weight: 5.0", "  chain_recall_edge_weight: 5.0",
      "  chain_recall_thr: 0.9", "  chain_recall_tau: 0.3"]

BASE = {"v654": "config_files/train_CERN_0904_v654.yaml",
        "v651": "config_files/train_CERN_0904_v651.yaml"}

ARMS = [
    # tag,        base,  ctx,  rank_w, cr,  epochs, patience
    ("blend_a", "v654", True, 20.0, False, None, None),
    ("blend_b", "v651", True, 20.0, False, None, None),
    ("blend_c", "v651", True, 20.0, True, None, None),
    ("blend_d", "v651", True, 40.0, False, None, None),
    ("faircombo", "v651", False, None, True, 120, 60),
    # 四合一: 四个已验证成分全部叠加 (ctxprune +6.91 / rankx2 +4.99 / crtau03 +4.71 / ep120 +3.81)
    ("quad_v654", "v654", True, 20.0, True, 120, None),
    ("quad_v651", "v651", True, 20.0, True, 120, None),
]


def set_key(txt, key, value, anchor):
    """有该键则改值(避免 YAML 重复键), 无则插到 anchor 之后。"""
    pat = rf"^(\s*{re.escape(key)}:\s*)\S+\s*$"
    if re.search(pat, txt, re.M):
        return re.sub(pat, rf"\g<1>{value}", txt, count=1, flags=re.M)
    return txt.replace(anchor, anchor + f"\n  {key}: {value}", 1)


def main():
    for i, (tag, base, ctx, rank_w, cr, ep, pat) in enumerate(ARMS):
        ver = 1201 + i
        txt = open(BASE[base]).read()
        if ctx:
            assert GN_ANCHOR in txt
            assert "context_prune:" not in txt
            txt = txt.replace(GN_ANCHOR, GN_ANCHOR + "\n" + "\n".join(CTX), 1)
        if rank_w:
            assert E_ANCHOR in txt
            txt = set_key(txt, "edge_rank_weight", rank_w, E_ANCHOR)
        if cr:
            assert E_ANCHOR in txt
            for k, v in (("chain_recall_weight", 5.0), ("chain_recall_edge_weight", 5.0),
                         ("chain_recall_thr", 0.9), ("chain_recall_tau", 0.3)):
                txt = set_key(txt, k, v, E_ANCHOR)
        if ep:
            txt = re.sub(r"^(\s*epochs:\s*)\d+\s*$", rf"\g<1>{ep}", txt, count=1, flags=re.M)
        if pat:
            txt = re.sub(r"^(\s*early_stop_patience:\s*)\d+\s*$", rf"\g<1>{pat}", txt,
                         count=1, flags=re.M)
        # 三个"版本身份"键必须一起改 (2026-10-16 事故: 漏改 model 键导致产物写进别家目录)
        txt = re.sub(r"^(\s*over_write:\s*).*$", rf"\g<1>{tag}", txt, count=1, flags=re.M)
        txt = re.sub(r"^(\s*log_version:\s*)\d+\s*$", rf"\g<1>{ver}", txt, count=1, flags=re.M)
        txt = re.sub(r"^(\s*model:\s*)\d+\s*$", rf"\g<1>{ver}", txt, count=1, flags=re.M)
        assert txt.count(f"log_version: {ver}") == 1 and txt.count(f"model: {ver}") == 1
        assert txt.count(f"over_write: {tag}") == 1, tag
        if ctx:
            assert txt.count("context_prune:") == 1
        if rank_w:
            assert txt.count("edge_rank_weight:") == 1, f"{tag}: rank 键重复"
        if cr:
            assert txt.count("chain_recall_tau: 0.3") == 1, f"{tag}: cr 键重复"
        if ep:
            assert re.search(rf"^\s*epochs: {ep}\s*$", txt, re.M)
        if pat:
            assert re.search(rf"^\s*early_stop_patience: {pat}\s*$", txt, re.M)
        p = f"config_files/train_CERN_0904_{tag}.yaml"
        open(p, "w").write(txt)
        print(f"  {p}  base={base} ver={ver} ctx={ctx} rank={rank_w} cr={cr} ep={ep} pat={pat}")


if __name__ == "__main__":
    main()
