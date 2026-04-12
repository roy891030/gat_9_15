# RunPods 完整訓練指南

## 🚀 快速開始

### Step 1: 清理所有舊結果

```bash
cd /workspace/gat_9_15  # 或你的專案路徑

# 清理所有舊的結果、artifacts、圖表
bash clean_all_results.sh
```

**清理內容：**
- ✓ 停止所有正在運行的訓練進程
- ✓ 刪除所有 `gat_artifacts_*` 目錄
- ✓ 刪除所有 `results_*.txt` 檔案
- ✓ 刪除所有 `plots_*` 目錄
- ✓ 刪除所有 `train*.log` 檔案

---

### Step 2: 選擇訓練方式

#### 方式 A: 串行訓練（推薦，穩定）

```bash
bash run_all_models.sh
```

**特點：**
- ✅ 一次訓練一個模型（穩定）
- ✅ GPU 記憶體使用較低
- ✅ 適合 16GB VRAM（如 RTX 4090）
- ⏱️ 訓練時間較長（約 4-6 小時）

---

#### 方式 B: 並行訓練（更快，需要更多 VRAM）

```bash
bash run_all_models_parallel.sh
```

**特點：**
- ⚡ 同時訓練多個模型（更快）
- ⚠️ GPU 記憶體使用較高
- ✅ 適合 24GB+ VRAM（如 RTX 3090, A5000, A6000）
- ⏱️ 訓練時間較短（約 2-3 小時）

---

### Step 3: 監控訓練進度

#### 查看訓練日誌

```bash
# 查看所有訓練日誌
tail -f train_short.log     # 短期 DMFM
tail -f train_medium.log    # 中期 DMFM
tail -f train_long.log      # 長期 DMFM
tail -f train_gat.log       # GATRegressor

# 同時查看多個日誌
tail -f train_*.log
```

#### 監控 GPU 使用

```bash
# 實時監控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat（如果安裝）
watch -n 1 gpustat
```

#### 檢查訓練進程

```bash
# 查看正在運行的訓練
ps aux | grep train

# 查看特定進程
ps aux | grep train_dmfm_wei2022
ps aux | grep train_gat_fixed
```

---

### Step 4: 等待訓練完成

訓練會在背景執行，你可以：

1. **斷線後繼續訓練**（使用 nohup）
2. **關閉終端機**（訓練仍繼續）
3. **稍後重新連線**查看結果

#### 檢查訓練是否完成

```bash
# 查看訓練日誌最後幾行
tail -20 train_short.log
tail -20 train_medium.log
tail -20 train_long.log
tail -20 train_gat.log

# 尋找 "訓練完成" 或 "Early stopping"
grep "訓練完成\|Early stopping\|完成" train_*.log
```

#### 檢查模型檔案

```bash
# 查看是否生成模型檔案
ls -lh gat_artifacts_*/dmfm_wei2022_best.pt
ls -lh gat_artifacts_*/gat_regressor.pt
```

---

### Step 5: 視覺化和評估（訓練完成後）

```bash
# 執行後處理腳本
bash post_process_all.sh
```

**生成內容：**
- ✓ Factor Attention 分析（每個模型）
- ✓ 階層式特徵分析（每個模型）
- ✓ 總結報告（RESULTS_SUMMARY.md）

---

## 📊 訓練模型清單

| 模型 | 資料期間 | Artifacts 目錄 | 訓練日誌 | 預期時間 |
|------|---------|---------------|---------|---------|
| DMFM (短期) | 2019-2020 | gat_artifacts_short | train_short.log | 30-40 min |
| DMFM (中期) | 2019-2022 | gat_artifacts_medium | train_medium.log | 50-60 min |
| DMFM (長期) | 2019-2025 | gat_artifacts_long | train_long.log | 60-80 min |
| GATRegressor | 2019-2022 | gat_artifacts_gat | train_gat.log | 20-30 min |

**總計：** 約 2.5-3.5 小時（串行），1.5-2.5 小時（並行）

---

## 🔍 查看結果

### 訓練指標

```bash
# 查看訓練日誌中的最佳指標
grep "best\|Best" train_*.log

# 查看 Early Stopping
grep "Early stopping" train_*.log
```

### Factor Attention 分析

```bash
# 查看短期 DMFM 的 Factor Attention
cat plots_short_attention/factor_attention_summary.txt

# 查看中期 DMFM
cat plots_medium_attention/factor_attention_summary.txt

# 查看長期 DMFM
cat plots_long_attention/factor_attention_summary.txt
```

### 階層式特徵分析

```bash
# 查看短期 DMFM 的階層式特徵效果
cat plots_short_contexts/context_analysis_summary.txt

# 查看中期 DMFM
cat plots_medium_contexts/context_analysis_summary.txt

# 查看長期 DMFM
cat plots_long_contexts/context_analysis_summary.txt
```

### 總結報告

```bash
# 查看自動生成的總結報告
cat RESULTS_SUMMARY.md
```

---

## 📁 生成的檔案結構

```
gat_9_15/
├── 訓練日誌
│   ├── train_short.log
│   ├── train_medium.log
│   ├── train_long.log
│   └── train_gat.log
│
├── Artifacts（模型和資料）
│   ├── artifacts/{short,medium,long}/ # 依視窗分組
│   └── runs/                         # run_pipeline.py 統一產出的 metrics/backtest/plots
│
├── 範例輸出（只讀參考）
│   └── examples/
│       ├── artifacts/{covid_crash,rate_hike}/
│       └── plots/short|medium|long|covid_crash|rate_hike/{dmfm,gat}/
│
└── 總結報告（若有生成）
    └── RESULTS_SUMMARY.md
```

---

## ⚠️ 常見問題

### Q1: 訓練中斷怎麼辦？

**A:** 使用 nohup，訓練會在背景持續進行：

```bash
# 檢查訓練是否還在運行
ps aux | grep train

# 查看訓練日誌確認進度
tail -f train_short.log

# 如果真的中斷，重新啟動特定模型：
nohup python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_short \
  --epochs 200 \
  --lr 1e-3 \
  --device cuda \
  > train_short.log 2>&1 &
```

---

### Q2: GPU 記憶體不足怎麼辦？

**A:** 減少 batch size 或使用較小的 hidden_dim：

```bash
# 方法 1: 使用串行訓練（不要並行）
bash run_all_models.sh

# 方法 2: 修改模型參數（較小的 hidden_dim）
python train_dmfm_wei2022.py \
  --hidden_dim 32 \  # 預設 64
  --device cuda \
  ...
```

---

### Q3: 如何只訓練特定模型？

**A:** 手動執行特定步驟：

```bash
# 只訓練長期 DMFM
python build_artifacts.py \
  --artifact_dir gat_artifacts_long \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5

nohup python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_long \
  --epochs 200 \
  --device cuda \
  > train_long.log 2>&1 &

# 訓練完成後
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts_long \
  --output_dir plots_long_attention

python analyze_contexts.py \
  --artifact_dir gat_artifacts_long \
  --output_dir plots_long_contexts
```

---

### Q4: 如何下載結果到本地？

**A:** 使用 SCP 或 RunPods 的檔案管理：

```bash
# 打包所有結果
tar -czf results.tar.gz \
  plots_*/ \
  gat_artifacts_*/dmfm_wei2022_best.pt \
  gat_artifacts_*/train_log_wei2022.txt \
  train_*.log \
  RESULTS_SUMMARY.md

# 然後從 RunPods 介面下載 results.tar.gz
```

---

### Q5: 訓練完成後如何比較模型？

**A:** 查看總結報告和視覺化：

```bash
# 1. 查看總結報告
cat RESULTS_SUMMARY.md

# 2. 比較 Factor Attention（哪些特徵重要）
diff plots_short_attention/factor_attention_summary.txt \
     plots_long_attention/factor_attention_summary.txt

# 3. 比較訓練日誌（收斂速度、最佳 IC）
grep "Best\|best" train_*.log

# 4. 比較階層式特徵效果（變異數降低）
grep "變異數降低\|Variance reduction" plots_*/context_analysis_summary.txt
```

---

## 🎯 完整執行流程（總結）

```bash
# 1. 清理舊結果
bash clean_all_results.sh

# 2. 訓練所有模型（選一個）
bash run_all_models.sh              # 串行（穩定）
# 或
bash run_all_models_parallel.sh     # 並行（更快）

# 3. 監控訓練
tail -f train_*.log
watch -n 1 nvidia-smi

# 4. 訓練完成後，視覺化和評估
bash post_process_all.sh

# 5. 查看結果
cat RESULTS_SUMMARY.md
cat plots_*_attention/factor_attention_summary.txt
cat plots_*_contexts/context_analysis_summary.txt

# 6. 下載結果（可選）
tar -czf results.tar.gz plots_*/ *.md *.log
```

---

## 📞 需要幫助？

- 查看訓練日誌：`tail -f train_*.log`
- 查看 GPU 狀態：`nvidia-smi`
- 檢查進程：`ps aux | grep train`
- 查看錯誤：`grep -i error train_*.log`

---

祝訓練順利！🎉
