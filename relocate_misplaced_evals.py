"""把"写错目录"的评估产物归位。

根因: 评估输出目录 = settings.model; 而作业启动时会把配置快照到 version_X/input_config.yaml,
之后改 master 配置不影响已提交作业。1101-1104 的快照里 settings.model=651(修 bug 之前生成),
所以它们的 signal_reco_df / event_reco_df / info_*_reco.txt 都写进了 version_651/。

做法: 用 master 配置建立 tag -> 正确版本 的映射, 扫所有 version_* 目录,
      若某文件的 tag 属于另一个版本, 就移动到那个版本目录。默认 dry-run。
用法: python3 relocate_misplaced_evals.py [--apply]
"""
import glob
import os
import re
import shutil
import sys

import yaml

APPLY = "--apply" in sys.argv


def tag_to_version():
    m = {}
    for f in glob.glob("config_files/train_CERN_0904_*.yaml"):
        try:
            c = yaml.safe_load(open(f))
        except Exception:
            continue
        ow = (c.get("evaluate") or {}).get("over_write")
        mv = (c.get("settings") or {}).get("model")
        if ow and mv is not None:
            m[str(ow)] = str(mv)
    return m


def main():
    t2v = tag_to_version()
    moved = 0
    for d in sorted(glob.glob("LHCb_logs/DFEI/version_*")):
        v = os.path.basename(d).replace("version_", "")
        for f in glob.glob(os.path.join(d, "*")):
            b = os.path.basename(f)
            if "_reco.txt" not in b and "reco_df_" not in b:
                continue
            m = re.search(r"__(.+?)(?:_reco)?\.(csv|txt)$", b)
            if not m:
                continue
            tag = m.group(1)
            want = t2v.get(tag)
            if not want or want == v:
                continue
            dst_dir = f"LHCb_logs/DFEI/version_{want}"
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, b)
            print(f"{'MOVE' if APPLY else 'would move'}: {d}/{b}\n            -> {dst}")
            if APPLY and not os.path.exists(dst):
                shutil.move(f, dst)
            moved += 1
    print(f"\n{'moved' if APPLY else 'would move'} {moved} file(s).  " +
          ("" if APPLY else "加 --apply 执行。"))


if __name__ == "__main__":
    main()
