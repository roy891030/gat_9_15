#!/bin/bash

# ============================================================
# 在 RunPods 上重新訓練所有模型
# ============================================================

set -e  # 遇到錯誤立即停止

DEVICE="cuda"
INDUSTRY_CSV="unique_2019q3to2025q3.csv"
PRICES_CSV="unique_2019q3to2025q3.csv"
BENCHMARK_CSV="GAT0050.csv"

echo "============================================================"
echo "RunPods 完整實驗執行腳本"
echo "============================================================"
echo "裝置: $DEVICE"
echo "開始時間: $(date)"
echo "============================================================"

# ============================================================
# 實驗 1: 短期資料（2019-2020）- DMFM
# ============================================================
echo ""
echo "====== 實驗 1: 短期資料（2019-2020）- DMFM ======"
echo "開始時間: $(date)"

# 1.1 建立 Artifacts
echo "[1/5] 建立 artifacts..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_short \
  --start_date 2019-09-16 \
  --end_date 2020-12-31 \
  --horizon 5

# 1.2 訓練 DMFM
echo "[2/5] 訓練 DMFM 模型..."
nohup python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_short \
  --epochs 200 \
  --lr 1e-3 \
  --device $DEVICE \
  --hidden_dim 64 \
  --heads 2 \
  --patience 30 \
  > train_short.log 2>&1 &

SHORT_PID=$!
echo "訓練進程 PID: $SHORT_PID"
echo "等待訓練完成..."
wait $SHORT_PID

# 1.3 視覺化
echo "[3/5] 視覺化 Factor Attention..."
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts_short \
  --output_dir plots_short_attention \
  --device cpu

echo "[4/5] 分析階層式特徵..."
python analyze_contexts.py \
  --artifact_dir gat_artifacts_short \
  --output_dir plots_short_contexts \
  --device cpu

# 1.4 評估（如果支援）
echo "[5/5] 評估模型..."
if [ -f "evaluate_metrics.py" ]; then
    python evaluate_metrics.py \
      --artifact_dir gat_artifacts_short \
      --weights gat_artifacts_short/dmfm_wei2022_best.pt \
      --device $DEVICE \
      --industry_csv $INDUSTRY_CSV \
      > results_short_metrics.txt 2>&1 || echo "⚠ 評估可能不支援 DMFM_Wei2022"
fi

echo "✅ 短期資料實驗完成！ ($(date))"

# ============================================================
# 實驗 2: 中期資料（2019-2022）- DMFM
# ============================================================
echo ""
echo "====== 實驗 2: 中期資料（2019-2022）- DMFM ======"
echo "開始時間: $(date)"

# 2.1 建立 Artifacts
echo "[1/5] 建立 artifacts..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_medium \
  --start_date 2019-09-16 \
  --end_date 2022-12-31 \
  --horizon 5

# 2.2 訓練 DMFM
echo "[2/5] 訓練 DMFM 模型..."
nohup python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_medium \
  --epochs 200 \
  --lr 1e-3 \
  --device $DEVICE \
  --hidden_dim 64 \
  --heads 2 \
  --patience 30 \
  > train_medium.log 2>&1 &

MEDIUM_PID=$!
echo "訓練進程 PID: $MEDIUM_PID"
echo "等待訓練完成..."
wait $MEDIUM_PID

# 2.3 視覺化
echo "[3/5] 視覺化 Factor Attention..."
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts_medium \
  --output_dir plots_medium_attention \
  --device cpu

echo "[4/5] 分析階層式特徵..."
python analyze_contexts.py \
  --artifact_dir gat_artifacts_medium \
  --output_dir plots_medium_contexts \
  --device cpu

# 2.4 評估
echo "[5/5] 評估模型..."
if [ -f "evaluate_metrics.py" ]; then
    python evaluate_metrics.py \
      --artifact_dir gat_artifacts_medium \
      --weights gat_artifacts_medium/dmfm_wei2022_best.pt \
      --device $DEVICE \
      --industry_csv $INDUSTRY_CSV \
      > results_medium_metrics.txt 2>&1 || echo "⚠ 評估可能不支援 DMFM_Wei2022"
fi

echo "✅ 中期資料實驗完成！ ($(date))"

# ============================================================
# 實驗 3: 長期資料（2019-2025）- DMFM
# ============================================================
echo ""
echo "====== 實驗 3: 長期資料（2019-2025）- DMFM ======"
echo "開始時間: $(date)"

# 3.1 建立 Artifacts
echo "[1/5] 建立 artifacts..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_long \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5

# 3.2 訓練 DMFM
echo "[2/5] 訓練 DMFM 模型..."
nohup python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_long \
  --epochs 200 \
  --lr 1e-3 \
  --device $DEVICE \
  --hidden_dim 64 \
  --heads 2 \
  --patience 30 \
  > train_long.log 2>&1 &

LONG_PID=$!
echo "訓練進程 PID: $LONG_PID"
echo "等待訓練完成..."
wait $LONG_PID

# 3.3 視覺化
echo "[3/5] 視覺化 Factor Attention..."
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts_long \
  --output_dir plots_long_attention \
  --device cpu

echo "[4/5] 分析階層式特徵..."
python analyze_contexts.py \
  --artifact_dir gat_artifacts_long \
  --output_dir plots_long_contexts \
  --device cpu

# 3.4 評估
echo "[5/5] 評估模型..."
if [ -f "evaluate_metrics.py" ]; then
    python evaluate_metrics.py \
      --artifact_dir gat_artifacts_long \
      --weights gat_artifacts_long/dmfm_wei2022_best.pt \
      --device $DEVICE \
      --industry_csv $INDUSTRY_CSV \
      > results_long_metrics.txt 2>&1 || echo "⚠ 評估可能不支援 DMFM_Wei2022"
fi

echo "✅ 長期資料實驗完成！ ($(date))"

# ============================================================
# 實驗 4: GATRegressor 對照（中期資料）
# ============================================================
echo ""
echo "====== 實驗 4: GATRegressor 對照（舊版模型）======"
echo "開始時間: $(date)"

# 4.1 建立 Artifacts（使用舊版預處理）
echo "[1/4] 建立 artifacts（注意：使用新版 build_artifacts.py）..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_gat \
  --start_date 2019-09-16 \
  --end_date 2022-12-31 \
  --horizon 5

# 4.2 訓練 GATRegressor（使用舊版訓練腳本）
echo "[2/4] 訓練 GATRegressor 模型..."
if [ -f "train_gat_fixed.py" ]; then
    nohup python train_gat_fixed.py \
      --artifact_dir gat_artifacts_gat \
      --epochs 50 \
      --lr 1e-3 \
      --device $DEVICE \
      --loss corr_mse_ind \
      --alpha_mse 0.03 \
      --lambda_var 0.1 \
      --industry_csv $INDUSTRY_CSV \
      > train_gat.log 2>&1 &

    GAT_PID=$!
    echo "訓練進程 PID: $GAT_PID"
    echo "等待訓練完成..."
    wait $GAT_PID
else
    echo "⚠ 找不到 train_gat_fixed.py，跳過 GATRegressor 訓練"
fi

# 4.3 評估
echo "[3/4] 評估模型..."
if [ -f "evaluate_metrics.py" ] && [ -f "gat_artifacts_gat/gat_regressor.pt" ]; then
    python evaluate_metrics.py \
      --artifact_dir gat_artifacts_gat \
      --weights gat_artifacts_gat/gat_regressor.pt \
      --device $DEVICE \
      --industry_csv $INDUSTRY_CSV \
      > results_gat_metrics.txt 2>&1 || echo "⚠ 評估失敗"
fi

# 4.4 投組回測
echo "[4/4] 投資組合回測..."
if [ -f "evaluate_portfolio.py" ] && [ -f "gat_artifacts_gat/gat_regressor.pt" ]; then
    python evaluate_portfolio.py \
      --artifact_dir gat_artifacts_gat \
      --weights gat_artifacts_gat/gat_regressor.pt \
      --device $DEVICE \
      --top_pct 0.10 \
      --rebalance_days 5 \
      --industry_csv $INDUSTRY_CSV \
      > results_gat_portfolio.txt 2>&1 || echo "⚠ 回測失敗"
fi

echo "✅ GATRegressor 對照實驗完成！ ($(date))"

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "所有實驗完成！"
echo "結束時間: $(date)"
echo "============================================================"

echo ""
echo "生成的檔案："
echo ""
echo "📊 訓練日誌："
ls -lh train_*.log 2>/dev/null || echo "  (無)"

echo ""
echo "📁 Artifacts："
ls -d gat_artifacts_* 2>/dev/null || echo "  (無)"

echo ""
echo "📈 視覺化："
ls -d plots_* 2>/dev/null || echo "  (無)"

echo ""
echo "📋 評估結果："
ls -lh results_*.txt 2>/dev/null || echo "  (無)"

echo ""
echo "============================================================"
echo "查看訓練日誌："
echo "  tail -f train_short.log"
echo "  tail -f train_medium.log"
echo "  tail -f train_long.log"
echo "  tail -f train_gat.log"
echo ""
echo "查看 Factor Attention 分析："
echo "  cat plots_*_attention/factor_attention_summary.txt"
echo ""
echo "查看階層式特徵分析："
echo "  cat plots_*_contexts/context_analysis_summary.txt"
echo "============================================================"
