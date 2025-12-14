#!/bin/bash

# ============================================================
# 在 RunPods 上並行訓練所有模型（更快！）
# ============================================================

set -e

DEVICE="cuda"
INDUSTRY_CSV="unique_2019q3to2025q3.csv"
PRICES_CSV="unique_2019q3to2025q3.csv"

echo "============================================================"
echo "RunPods 並行訓練腳本"
echo "============================================================"
echo "⚠️  注意：此腳本會同時訓練多個模型，需要足夠的 GPU 記憶體"
echo "建議：至少 24GB VRAM（如 RTX 3090 或 A5000）"
echo "============================================================"
echo "開始時間: $(date)"
echo "============================================================"

# ============================================================
# Step 1: 建立所有 Artifacts（串行）
# ============================================================
echo ""
echo "====== Step 1: 建立所有 Artifacts ======"

echo "[1/4] 短期資料 (2019-2020)..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_short \
  --start_date 2019-09-16 \
  --end_date 2020-12-31 \
  --horizon 5

echo "[2/4] 中期資料 (2019-2022)..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_medium \
  --start_date 2019-09-16 \
  --end_date 2022-12-31 \
  --horizon 5

echo "[3/4] 長期資料 (2019-2025)..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_long \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5

echo "[4/4] GATRegressor 資料 (2019-2022)..."
python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir gat_artifacts_gat \
  --start_date 2019-09-16 \
  --end_date 2022-12-31 \
  --horizon 5

echo "✅ 所有 Artifacts 建立完成！"

# ============================================================
# Step 2: 並行訓練所有模型
# ============================================================
echo ""
echo "====== Step 2: 並行訓練所有模型 ======"
echo "⚠️  開始並行訓練，請監控 GPU 記憶體使用量"

# 訓練短期 DMFM
echo "[1/4] 啟動短期 DMFM 訓練..."
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
echo "  PID: $SHORT_PID"

# 等待 5 秒避免 GPU 爭搶
sleep 5

# 訓練中期 DMFM
echo "[2/4] 啟動中期 DMFM 訓練..."
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
echo "  PID: $MEDIUM_PID"

sleep 5

# 訓練長期 DMFM
echo "[3/4] 啟動長期 DMFM 訓練..."
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
echo "  PID: $LONG_PID"

sleep 5

# 訓練 GATRegressor（如果存在）
if [ -f "train_gat_fixed.py" ]; then
    echo "[4/4] 啟動 GATRegressor 訓練..."
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
    echo "  PID: $GAT_PID"
else
    echo "[4/4] ⚠ 找不到 train_gat_fixed.py，跳過"
    GAT_PID=""
fi

echo ""
echo "✅ 所有訓練已啟動（背景執行）"
echo ""
echo "訓練進程："
echo "  短期 DMFM:  PID $SHORT_PID  (log: train_short.log)"
echo "  中期 DMFM:  PID $MEDIUM_PID (log: train_medium.log)"
echo "  長期 DMFM:  PID $LONG_PID   (log: train_long.log)"
if [ -n "$GAT_PID" ]; then
    echo "  GATRegressor: PID $GAT_PID  (log: train_gat.log)"
fi

echo ""
echo "============================================================"
echo "監控訓練進度："
echo "============================================================"
echo "  tail -f train_short.log"
echo "  tail -f train_medium.log"
echo "  tail -f train_long.log"
echo "  tail -f train_gat.log"
echo ""
echo "監控 GPU 使用："
echo "  watch -n 1 nvidia-smi"
echo ""
echo "檢查訓練是否完成："
echo "  ps aux | grep train"
echo ""
echo "等待所有訓練完成："
echo "  wait $SHORT_PID $MEDIUM_PID $LONG_PID $GAT_PID"
echo "============================================================"

# 提供等待選項
read -p "是否等待所有訓練完成？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "等待訓練完成..."
    wait $SHORT_PID
    echo "✅ 短期 DMFM 完成"
    wait $MEDIUM_PID
    echo "✅ 中期 DMFM 完成"
    wait $LONG_PID
    echo "✅ 長期 DMFM 完成"
    if [ -n "$GAT_PID" ]; then
        wait $GAT_PID
        echo "✅ GATRegressor 完成"
    fi

    echo ""
    echo "============================================================"
    echo "所有訓練完成！開始後處理..."
    echo "============================================================"

    # 後處理：視覺化和評估
    bash post_process_all.sh
else
    echo ""
    echo "訓練在背景執行中..."
    echo "使用以下命令稍後查看結果："
    echo "  bash post_process_all.sh"
fi
