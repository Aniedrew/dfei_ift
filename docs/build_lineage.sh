#!/bin/bash
# 重新生成「DFEI 版本家族谱」多页矢量 PDF
# 用法: bash docs/build_lineage.sh
# 依赖: graphviz(dot) + poppler(pdfunite)
cd "$(dirname "$0")" || exit 1

PAGES="overview main attn ablation public"
for f in $PAGES; do
  dot -Tpdf "fig_${f}.dot" -o "p_${f}.pdf" || { echo "render fig_${f} failed"; exit 1; }
done

pdfunite p_overview.pdf p_main.pdf p_attn.pdf p_ablation.pdf p_public.pdf version_lineage.pdf
rm -f p_overview.pdf p_main.pdf p_attn.pdf p_ablation.pdf p_public.pdf

echo "OK -> docs/version_lineage.pdf"
pdfinfo version_lineage.pdf | grep -E 'Pages|File size'
