#!/bin/bash

# ============================================================
# 完整實驗執行腳本
# ============================================================

# 設定參數
DEVICE="cuda"
INDUSTRY_CSV="unique_2019q3to2025q3.csv"
BENCHMARK_CSV="GAT0050.csv"

# ============================================================
# 實驗 1: 短期資料（2019-2020）
# ============================================================
echo "====== 實驗 1: 短期資料（2019-2020） ======"

# 1.1 建立 Artifacts
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_short \
  --start_date 2019-09-16 \
  --end_date 2020-12-31 \
  --horizon 5

# 1.2 訓練 DMFM
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_short \
  --epochs 50 --lr 1e-3 --device $DEVICE \
  --loss dmfm \
  --alpha_mse 0.03 --lambda_var 0.1 --lambda_attn 0.05 \
  --industry_csv $INDUSTRY_CSV

# 1.3 評估
python evaluate_metrics.py \
  --artifact_dir gat_artifacts_short \
  --weights gat_artifacts_short/gat_regressor.pt \
  --device $DEVICE \
  --industry_csv $INDUSTRY_CSV \
  > results_short_metrics.txt

# 1.4 投組回測
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_short \
  --weights gat_artifacts_short/gat_regressor.pt \
  --device $DEVICE \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv $INDUSTRY_CSV \
  > results_short_portfolio.txt

# 1.5 生成圖表
python plot_reports.py \
  --artifact_dir gat_artifacts_short \
  --weights gat_artifacts_short/gat_regressor.pt \
  --device $DEVICE \
  --out_dir plots_short_vs_0050 \
  --top_pct 0.10 --rebalance_days 5 \
  --benchmark_csv $BENCHMARK_CSV

echo "短期資料實驗完成！"

# ============================================================
# 實驗 2: 中期資料（2019-2022）
# ============================================================
echo "====== 實驗 2: 中期資料（2019-2022） ======"

# 2.1 建立 Artifacts（2019-09-16 ~ 2022-12-31）
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_medium \
  --start_date 2019-09-16 \
  --end_date 2022-12-31 \
  --horizon 5

# 2.2 訓練 DMFM（與短/長期一致的 loss 與超參數）
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_medium \
  --epochs 50 --lr 1e-3 --device $DEVICE \
  --loss dmfm \
  --alpha_mse 0.03 --lambda_var 0.1 --lambda_attn 0.05 \
  --industry_csv $INDUSTRY_CSV

# 2.3 評估（保存指標到檔案）
python evaluate_metrics.py \
  --artifact_dir gat_artifacts_medium \
  --weights gat_artifacts_medium/gat_regressor.pt \
  --device $DEVICE \
  --industry_csv $INDUSTRY_CSV \
  > results_medium_metrics.txt

# 2.4 投組回測（同樣 top_pct=10%、每 5 日調整）
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_medium \
  --weights gat_artifacts_medium/gat_regressor.pt \
  --device $DEVICE \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv $INDUSTRY_CSV \
  > results_medium_portfolio.txt

# 2.5 生成與 0050 的比較圖
python plot_reports.py \
  --artifact_dir gat_artifacts_medium \
  --weights gat_artifacts_medium/gat_regressor.pt \
  --device $DEVICE \
  --out_dir plots_medium_vs_0050 \
  --top_pct 0.10 --rebalance_days 5 \
  --benchmark_csv $BENCHMARK_CSV

echo "中期資料實驗完成！"

# ============================================================
# 實驗 3: 長期資料（2019-2025）
# ============================================================
echo "====== 實驗 3: 長期資料（2019-2025） ======"

# 3.1 建立 Artifacts
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_long \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5

# 3.2 訓練 DMFM
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_long \
  --epochs 50 --lr 1e-3 --device $DEVICE \
  --loss dmfm \
  --alpha_mse 0.03 --lambda_var 0.1 --lambda_attn 0.05 \
  --industry_csv $INDUSTRY_CSV

# 3.3 評估
python evaluate_metrics.py \
  --artifact_dir gat_artifacts_long \
  --weights gat_artifacts_long/gat_regressor.pt \
  --device $DEVICE \
  --industry_csv $INDUSTRY_CSV \
  > results_long_metrics.txt

# 3.4 投組回測
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_long \
  --weights gat_artifacts_long/gat_regressor.pt \
  --device $DEVICE \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv $INDUSTRY_CSV \
  > results_long_portfolio.txt

# 3.5 生成圖表
python plot_reports.py \
  --artifact_dir gat_artifacts_long \
  --weights gat_artifacts_long/gat_regressor.pt \
  --device $DEVICE \
  --out_dir plots_long_vs_0050 \
  --top_pct 0.10 --rebalance_days 5 \
  --benchmark_csv $BENCHMARK_CSV

echo "長期資料實驗完成！"

# ============================================================
# 實驗 4: GATRegressor 對照（中期資料）
# ============================================================
echo "====== 實驗 4: GATRegressor 對照 ======"

# 4.1 訓練 GATRegressor
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_out_plus \
  --epochs 50 --lr 1e-3 --device $DEVICE \
  --loss corr_mse_ind \
  --alpha_mse 0.03 --lambda_var 0.1 \
  --industry_csv $INDUSTRY_CSV

# 4.2 評估
python evaluate_metrics.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device $DEVICE \
  --industry_csv $INDUSTRY_CSV \
  > results_gat_metrics.txt

# 4.3 投組回測
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device $DEVICE \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv $INDUSTRY_CSV \
  > results_gat_portfolio.txt

# 4.4 生成圖表
python plot_reports.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device $DEVICE \
  --out_dir plots_gat_vs_0050 \
  --top_pct 0.10 --rebalance_days 5 \
  --benchmark_csv $BENCHMARK_CSV

echo "GATRegressor 對照實驗完成！"

# ============================================================
# 完成
# ============================================================
echo "====== 所有實驗完成！ ======"
echo "結果檔案："
echo "  - results_short_metrics.txt"
echo "  - results_short_portfolio.txt"
echo "  - results_long_metrics.txt"
echo "  - results_long_portfolio.txt"
echo "  - results_gat_metrics.txt"
echo "  - results_gat_portfolio.txt"
echo ""
echo "圖表資料夾："
echo "  - plots_short_vs_0050/"
echo "  - plots_medium_vs_0050/"
echo "  - plots_long_vs_0050/"
echo "  - plots_gat_vs_0050/"