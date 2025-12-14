#!/bin/bash
set -e

echo "======================================"
echo "執行核心實驗（6 個）"
echo "固定起始點：2019-09-16"
echo "======================================"

# 建立實驗資料夾
mkdir -p experiments

# 記錄開始時間
START_TIME=$(date +%s)

# ==================== 準備工作 ====================
echo ""
echo "======================================"
echo "階段 0: 準備 Artifacts"
echo "======================================"

# artifacts_short 已經存在，跳過

# Medium
if [ ! -f "artifacts_medium/Ft_tensor.pt" ]; then
    echo "[0/6] 建立 artifacts_medium..."
    python build_artifacts.py \
      --prices unique_2019q3to2025q3.csv \
      --artifact_dir artifacts_medium \
      --start_date 2019-09-16 \
      --end_date 2022-12-31 \
      --horizon 5
else
    echo "[0/6] artifacts_medium 已存在，跳過"
fi

# Long
if [ ! -f "artifacts_long/Ft_tensor.pt" ]; then
    echo "[0/6] 建立 artifacts_long..."
    python build_artifacts.py \
      --prices unique_2019q3to2025q3.csv \
      --artifact_dir artifacts_long \
      --start_date 2019-09-16 \
      --end_date 2025-09-12 \
      --horizon 5
else
    echo "[0/6] artifacts_long 已存在，跳過"
fi

# ==================== Short 系列 ====================
echo ""
echo "======================================"
echo "階段 1: Short (2019-09-16 ~ 2020-12-31)"
echo "======================================"

# EXP-S2: GAT Short
echo ""
echo "[1/6] EXP-S2: GAT Short"
python train_gat_fixed.py \
  --artifact_dir artifacts_short \
  --epochs 30 --lr 1e-3 --device cuda \
  --loss corr_mse_ind --alpha_mse 0.03 --lambda_var 0.1 \
  --hid 64 --heads 2 --patience 10 \
  --industry_csv unique_2019q3to2025q3.csv

python evaluate_metrics.py \
  --artifact_dir artifacts_short \
  --weights artifacts_short/gat_regressor.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv \
  | tee experiments/exp_s2_results.txt

echo "✓ EXP-S2 完成"

# EXP-S4: DMFM Short
echo ""
echo "[2/6] EXP-S4: DMFM Short"
python train_dmfm_wei2022.py \
  --artifact_dir artifacts_short \
  --epochs 50 --lr 1e-4 --device cuda \
  --hidden_dim 64 --heads 2 --dropout 0.1 \
  --lambda_attn 0.1 --lambda_ic 1.0 --patience 20

python evaluate_metrics.py \
  --artifact_dir artifacts_short \
  --weights artifacts_short/dmfm_wei2022_best.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv \
  | tee experiments/exp_s4_results.txt

echo "✓ EXP-S4 完成"

# ==================== Medium 系列 ====================
echo ""
echo "======================================"
echo "階段 2: Medium (2019-09-16 ~ 2022-12-31)"
echo "======================================"

# EXP-M2: GAT Medium
echo ""
echo "[3/6] EXP-M2: GAT Medium"
python train_gat_fixed.py \
  --artifact_dir artifacts_medium \
  --epochs 50 --lr 1e-3 --device cuda \
  --loss corr_mse_ind --alpha_mse 0.03 --lambda_var 0.1 \
  --hid 64 --heads 2 --patience 15 \
  --industry_csv unique_2019q3to2025q3.csv

python evaluate_metrics.py \
  --artifact_dir artifacts_medium \
  --weights artifacts_medium/gat_regressor.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv \
  | tee experiments/exp_m2_results.txt

echo "✓ EXP-M2 完成"

# EXP-M4: DMFM Medium
echo ""
echo "[4/6] EXP-M4: DMFM Medium"
python train_dmfm_wei2022.py \
  --artifact_dir artifacts_medium \
  --epochs 100 --lr 1e-4 --device cuda \
  --hidden_dim 64 --heads 2 --dropout 0.1 \
  --lambda_attn 0.1 --lambda_ic 1.0 --patience 30

python evaluate_metrics.py \
  --artifact_dir artifacts_medium \
  --weights artifacts_medium/dmfm_wei2022_best.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv \
  | tee experiments/exp_m4_results.txt

echo ""
echo "  生成 Medium 完整報告..."
python plot_reports.py \
  --artifact_dir artifacts_medium \
  --weights artifacts_medium/dmfm_wei2022_best.pt \
  --out_dir experiments/exp_m4_plots \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv

python analyze_contexts.py \
  --artifact_dir artifacts_medium \
  --model_path artifacts_medium/dmfm_wei2022_best.pt \
  --output_dir experiments/exp_m4_contexts \
  --device cpu \
  --sample_days 15

python visualize_factor_attention.py \
  --artifact_dir artifacts_medium \
  --model_path artifacts_medium/dmfm_wei2022_best.pt \
  --output_dir experiments/exp_m4_attention \
  --device cpu

echo "✓ EXP-M4 完成"

# ==================== Long 系列 ====================
echo ""
echo "======================================"
echo "階段 3: Long (2019-09-16 ~ 2025-09-12)"
echo "======================================"

# EXP-L2: GAT Long
echo ""
echo "[5/6] EXP-L2: GAT Long"
python train_gat_fixed.py \
  --artifact_dir artifacts_long \
  --epochs 50 --lr 1e-3 --device cuda \
  --loss corr_mse_ind --alpha_mse 0.03 --lambda_var 0.1 \
  --hid 64 --heads 2 --patience 15 \
  --industry_csv unique_2019q3to2025q3.csv

python evaluate_metrics.py \
  --artifact_dir artifacts_long \
  --weights artifacts_long/gat_regressor.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv \
  | tee experiments/exp_l2_results.txt

echo "✓ EXP-L2 完成"

# EXP-L4: DMFM Long
echo ""
echo "[6/6] EXP-L4: DMFM Long"
python train_dmfm_wei2022.py \
  --artifact_dir artifacts_long \
  --epochs 100 --lr 1e-4 --device cuda \
  --hidden_dim 64 --heads 2 --dropout 0.1 \
  --lambda_attn 0.1 --lambda_ic 1.0 --patience 30

python evaluate_metrics.py \
  --artifact_dir artifacts_long \
  --weights artifacts_long/dmfm_wei2022_best.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv \
  | tee experiments/exp_l4_results.txt

echo ""
echo "  生成 Long 完整報告..."
python plot_reports.py \
  --artifact_dir artifacts_long \
  --weights artifacts_long/dmfm_wei2022_best.pt \
  --out_dir experiments/exp_l4_plots \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv

python analyze_contexts.py \
  --artifact_dir artifacts_long \
  --model_path artifacts_long/dmfm_wei2022_best.pt \
  --output_dir experiments/exp_l4_contexts \
  --device cpu \
  --sample_days 20

python visualize_factor_attention.py \
  --artifact_dir artifacts_long \
  --model_path artifacts_long/dmfm_wei2022_best.pt \
  --output_dir experiments/exp_l4_attention \
  --device cpu \
  --top_k 20

echo "✓ EXP-L4 完成"

# ==================== 整理結果 ====================
echo ""
echo "======================================"
echo "整理實驗結果..."
echo "======================================"

# 提取關鍵指標函數
extract_ic() {
    if [ -f "$1" ]; then
        grep "測試集 IC:" "$1" | head -1 | awk '{print $3}' || echo "N/A"
    else
        echo "N/A"
    fi
}

extract_icir() {
    if [ -f "$1" ]; then
        grep "測試集 ICIR:" "$1" | head -1 | awk '{print $3}' || echo "N/A"
    else
        echo "N/A"
    fi
}

extract_sharpe() {
    if [ -f "$1" ]; then
        grep "Sharpe" "$1" | head -1 | awk '{print $3}' || echo "N/A"
    else
        echo "N/A"
    fi
}

# 建立摘要表格
cat > experiments/results_summary.txt << SUMMARY
====================================
核心實驗結果摘要
====================================

時間窗口：
- Short:  2019-09-16 ~ 2020-12-31 (T=319)
- Medium: 2019-09-16 ~ 2022-12-31 (T=~820)
- Long:   2019-09-16 ~ 2025-09-12 (T=~1460)

結果：
----------------------------------------

Short 系列：
  EXP-S2 (GAT):  IC=$(extract_ic experiments/exp_s2_results.txt), ICIR=$(extract_icir experiments/exp_s2_results.txt)
  EXP-S4 (DMFM): IC=$(extract_ic experiments/exp_s4_results.txt), ICIR=$(extract_icir experiments/exp_s4_results.txt)

Medium 系列：
  EXP-M2 (GAT):  IC=$(extract_ic experiments/exp_m2_results.txt), ICIR=$(extract_icir experiments/exp_m2_results.txt)
  EXP-M4 (DMFM): IC=$(extract_ic experiments/exp_m4_results.txt), ICIR=$(extract_icir experiments/exp_m4_results.txt)

Long 系列：
  EXP-L2 (GAT):  IC=$(extract_ic experiments/exp_l2_results.txt), ICIR=$(extract_icir experiments/exp_l2_results.txt)
  EXP-L4 (DMFM): IC=$(extract_ic experiments/exp_l4_results.txt), ICIR=$(extract_icir experiments/exp_l4_results.txt)

視覺化報告：
  - experiments/exp_m4_plots/
  - experiments/exp_m4_contexts/
  - experiments/exp_m4_attention/
  - experiments/exp_l4_plots/
  - experiments/exp_l4_contexts/
  - experiments/exp_l4_attention/

SUMMARY

# 顯示摘要
cat experiments/results_summary.txt

# 計算總時間
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "======================================"
echo "✅ 所有實驗完成！"
echo "======================================"
echo "總耗時: ${HOURS}h ${MINUTES}m"
echo ""
echo "生成的檔案："
echo "  - artifacts_short/"
echo "  - artifacts_medium/"
echo "  - artifacts_long/"
echo "  - experiments/exp_*_results.txt"
echo "  - experiments/exp_m4_plots/"
echo "  - experiments/exp_l4_plots/"
echo "  - experiments/results_summary.txt"
echo ""
