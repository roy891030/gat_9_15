#!/bin/bash
set -euo pipefail

# 針對現有 artifacts_* 目錄產出評估指標、報告與可解釋性圖表。
# 使用方式：
#   bash post_process_all.sh [industry_csv]
# 若未提供 industry_csv，預設使用 unique_2019q3to2025q3.csv。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INDUSTRY_CSV="${1:-unique_2019q3to2025q3.csv}"
EXPERIMENTS_DIR="experiments"
mkdir -p "$EXPERIMENTS_DIR"

run_dmfm_suite() {
  local tag="$1"  # short / medium / long
  local artifact_dir="artifacts_${tag}"
  local weight="${artifact_dir}/dmfm_wei2022_best.pt"

  if [ ! -f "$weight" ]; then
    echo "⚠️ 找不到 ${artifact_dir} 的 DMFM 權重，跳過。"
    return
  fi

  echo "\n======================================"
  echo "後處理 DMFM (${tag})"
  echo "======================================"

  local out_root="${EXPERIMENTS_DIR}/exp_${tag}"
  mkdir -p "$out_root" "$out_root/plots" "$out_root/contexts" "$out_root/attention"

  python evaluate_metrics.py \
    --artifact_dir "$artifact_dir" \
    --weights "$weight" \
    --device cuda \
    --industry_csv "$INDUSTRY_CSV" \
    | tee "${out_root}/results.txt"

  python plot_reports.py \
    --artifact_dir "$artifact_dir" \
    --weights "$weight" \
    --out_dir "${out_root}/plots" \
    --device cuda \
    --industry_csv "$INDUSTRY_CSV"

  python analyze_contexts.py \
    --artifact_dir "$artifact_dir" \
    --model_path "$weight" \
    --output_dir "${out_root}/contexts" \
    --device cpu \
    --sample_days 20

  python visualize_factor_attention.py \
    --artifact_dir "$artifact_dir" \
    --weights "$weight" \
    --output_dir "${out_root}/attention" \
    --device cpu
}

run_gat_eval() {
  local artifact_dir="artifacts_${1}"
  local weight="${artifact_dir}/gat_regressor.pt"

  if [ ! -f "$weight" ]; then
    echo "⚠️ 找不到 ${artifact_dir} 的 GAT 權重，跳過。"
    return
  fi

  echo "\n======================================"
  echo "後處理 GAT (${1})"
  echo "======================================"

  python evaluate_metrics.py \
    --artifact_dir "$artifact_dir" \
    --weights "$weight" \
    --device cuda \
    --industry_csv "$INDUSTRY_CSV" \
    | tee "${EXPERIMENTS_DIR}/exp_${1}_gat_results.txt"
}

run_dmfm_suite short
run_dmfm_suite medium
run_dmfm_suite long
run_gat_eval short
run_gat_eval medium
run_gat_eval long

echo "\n✅ 後處理完成，輸出位於 ${EXPERIMENTS_DIR}/"
