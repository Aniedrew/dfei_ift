"""生成"组合臂": 以 v651 为基底, 叠加三个已验证的正收益改动 (2026-10-15)。

设计 (2x2 归因):
  combo_all      = v651 + chain_recall(tau=0.3) + epochs 120 + seqrec 解码器   -> 1101
  combo_noseqrec = v651 + chain_recall(tau=0.3) + epochs 120                  -> 1102
  base_ep120     = v651 + epochs 120                                          -> 1103
  base_seqrec    = v651 + seqrec 解码器                                        -> 1104

基底为什么换 v651: v651 = v648 + line_graph 注意力 + 几何 pair-bias,
在"无条件整链恢复"上是最好的 (27.96%, 单B 32.11%), 且是 v648 的超集。

已证实的三个成分 (见 COMPREHENSIVE_EVAL_20261014.md):
  * chain_recall 温和档 (tau=0.3, v660): 移动工作点到更均衡处 (clean 22.6 / found 58.7)
  * 长训练 (v675 epochs 120): clean 18.26 -> 22.39 (+4.1pp)
  * seqrec 解码器 (v901): clean 18.26 -> 24.25 (+6.0pp), 且 v904 证明增益来自解码器本身
"""
import re

SRC = "config_files/train_CERN_0904_v651.yaml"
CR = ["  chain_recall_weight: 5.0", "  chain_recall_edge_weight: 5.0",
      "  chain_recall_thr: 0.9", "  chain_recall_tau: 0.3"]
SEQ = ["  seqrec_use: true", "  seqrec_metric: mix", "  seqrec_lam: 1.0",
       "  seqrec_lam_learn: true", "  seqrec_cut: 0.02",
       "  seqrec_cut_learn: false", "  seqrec_loss_weight: 1.0"]
ANCHOR = "  chain_contrastive_weight: 0.5"

ARMS = [
    ("combo_all", 1101, True, True),
    ("combo_noseqrec", 1102, True, False),
    ("base_ep120", 1103, False, False),
    ("base_seqrec", 1104, False, True),
]


def main():
    src = open(SRC).read().split("\n")
    for tag, ver, cr, seq in ARMS:
        out = []
        for ln in src:
            if re.match(r"\s*log_version:", ln):
                ln = re.sub(r":\s*\d+\s*$", f": {ver}", ln)
            elif re.match(r"\s*over_write:", ln):
                ln = re.sub(r":.*$", f": {tag}", ln)
            elif re.match(r"\s*epochs:", ln):
                ln = re.sub(r":\s*\d+\s*$", ": 120", ln)
            out.append(ln)
        txt = "\n".join(out)
        add = (CR if cr else []) + (SEQ if seq else [])
        assert ANCHOR in txt, "锚点行未找到"
        txt = txt.replace(ANCHOR, ANCHOR + "\n" + "\n".join(add), 1)
        # 复核: 关键键各出现一次
        for k in (["chain_recall_tau"] if cr else []) + (["seqrec_use"] if seq else []):
            assert txt.count(k) == 1, f"{tag}: {k} 出现 {txt.count(k)} 次"
        assert txt.count(f"log_version: {ver}") == 1 and txt.count(f"over_write: {tag}") == 1
        assert re.search(r"^\s*epochs: 120\s*$", txt, re.M), f"{tag}: epochs 未改为 120"
        p = f"config_files/train_CERN_0904_{tag}.yaml"
        open(p, "w").write(txt)
        print(f"  {p}  (v{ver}, chain_recall={cr}, seqrec={seq}, epochs=120)")


if __name__ == "__main__":
    main()
