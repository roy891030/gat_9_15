# 深度多因子模型 (DMFM) - Graph Attention Networks 股票預測

基於 Wei et al. (2022) 論文的完整實作，使用階層式中性化 GAT 架構進行台股預測。

**作者：** Lo Yi (羅頤)
**學校：** National Yang Ming Chiao Tung University
**E-mail:** roy60404@gmail.com

---

## 🔄 統一執行入口（新版）

目前建議以 `run_pipeline.py` 為主要流程入口，支援：
- 單獨跑 `baseline / gat / dmfm`
- 跑 `short / medium / long` 任意時間窗
- `smoke` 與 `full` 兩種模式
- 只做後處理（`--skip_build --skip_train`）

完整指令請看：`docs/UNIFIED_WORKFLOW.md`

---

## 📁 專案結構（已整理）

```
gat_9_15/
├── 🔧 核心腳本
│   ├── build_artifacts.py              # [1] 資料預處理與特徵工程
│   ├── train_dmfm_wei2022.py           # [2] 訓練 DMFM（推薦）
│   ├── train_gat_fixed.py              # [3] 訓練簡化版 GAT/DMFM
│   ├── train_baselines.py              # [4] 線性/LSTM/XGBoost 對照組
│   ├── evaluate_metrics.py             # [5] 評估 IC, ICIR, MSE 等指標
│   ├── evaluate_portfolio.py           # [6] 投資組合回測
│   ├── visualize_factor_attention.py   # [7] 視覺化 Factor Attention
│   ├── analyze_contexts.py             # [8] 分析階層式特徵
│   └── plot_reports.py                 # [9] 生成完整報告與圖表
│
├── 🤖 模型定義
│   └── model_dmfm_wei2022.py           # DMFM 完整模型（對齊論文）
│
├── 🚀 執行腳本
│   ├── run_core_experiments.sh         # 一鍵執行核心 DMFM/GAT 實驗
│   ├── run_all_models.sh               # run_core_experiments 的包裝器
│   └── run_all_models_parallel.sh      # 並行版包裝（呼叫 archived 腳本）
│
├── 🧪 範例輸出 (examples/)
│   ├── artifacts/                      # 範例情境的 meta 與訓練日誌
│   │   ├── covid_crash/
│   │   └── rate_hike/
│   └── plots/                          # 整併的示例視覺化結果
│       ├── short/{dmfm,gat}/
│       ├── medium/{dmfm,gat}/
│       ├── long/{dmfm,gat}/
│       ├── covid_crash/{dmfm,gat}/
│       └── rate_hike/{dmfm,gat}/
│
├── 📚 文件 (docs/)
│   ├── PROJECT_OVERVIEW.md             # 專案總覽
│   ├── HIERARCHICAL_NEUTRALIZATION_EXPLAINED.md  # 階層式中性化詳解
│   ├── README_DMFM_Wei2022.md          # DMFM 技術文件
│   ├── RUNPODS_GUIDE.md                # RunPods 使用指南
│   ├── VENV_SETUP.md                   # 虛擬環境設置
│   ├── FUNCTIONAL_COMMAND_MAP.md       # 功能、指令與輸出對照
│   ├── CONFLICT_RESOLUTION.md          # 合併衝突處理指引
│   └── CHANGES_DMFM_Wei2022.md         # 變更記錄
│
├── 🛠️ 工具 (utils/)
│   ├── check_csv.sh                    # CSV 檢查工具
│   ├── fix_csv_columns.py              # CSV 修復工具
│   └── setup_env.sh                    # 環境設置腳本
│
└── 📦 歸檔 (archived/)
    ├── run_dmfm_wei2022.sh             # 舊版執行腳本
    ├── run_experiments.sh              # 舊版實驗腳本
    ├── run_all_models_parallel.sh      # 平行執行腳本
    └── clean_all_results.sh            # 清理腳本
```

> 需要快速檢視「功能 → 指令 → 主要參數/輸出」的對照表，可參考 `docs/FUNCTIONAL_COMMAND_MAP.md`。

### 🧭 先同步最新版以避免衝突
- 若本地分支尚未設定遠端追蹤，先建立連線並綁定上游：
  ```bash
  git remote add origin <repo_url>
  git fetch origin
  git branch --set-upstream-to=origin/main work  # 將當前分支綁定到遠端 main
  ```
- 每次修改前先拉取最新提交，避免與他人產生衝突：
  ```bash
  git pull --rebase
  ```

---

## 🚀 快速開始（推薦流程）

### Step 1: 建立資料 Artifacts

```bash
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv unique_2019q3to2025q3.csv \
  --artifact_dir gat_artifacts \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5
```

**輸出檔案：**
- `gat_artifacts/Ft_tensor.pt` - 特徵張量 [T, N, F]
- `gat_artifacts/yt_tensor.pt` - 標籤張量 [T, N]
- `gat_artifacts/industry_edge_index.pt` - 產業圖結構
- `gat_artifacts/universe_edge_index.pt` - 全市場圖結構
- `gat_artifacts/meta.pkl` - Metadata

---

### Step 2: 訓練 DMFM 模型

```bash
python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts \
  --epochs 200 \
  --lr 1e-4 \
  --device cuda \
  --hidden_dim 64 \
  --heads 2 \
  --patience 30
```

**輸出檔案：**
- `gat_artifacts/dmfm_wei2022_best.pt` - 最佳模型
- `gat_artifacts/dmfm_wei2022.pt` - 最終模型
- `gat_artifacts/train_log_wei2022.txt` - 訓練日誌

**預期結果：**
- Train IC: 0.02 ~ 0.08
- Test IC: 0.015 ~ 0.05
- Test ICIR: 0.2 ~ 0.8

---

### Step 3: 評估與視覺化

#### 3A. 評估指標

```bash
python evaluate_metrics.py \
  --artifact_dir gat_artifacts \
  --weights gat_artifacts/dmfm_wei2022_best.pt \
  --device cuda
```

**評估指標說明：**

| 指標 | 說明 | 良好範圍 |
|------|------|---------|
| **IC** | Information Coefficient（相關性） | 0.03 - 0.08 |
| **ICIR** | IC Information Ratio（穩定性） | 0.5 - 2.0 |
| **Rank IC** | Spearman 相關係數 | 0.03 - 0.08 |
| **MSE** | 均方誤差 | 越小越好 |
| **Dir Acc** | 方向準確率 | > 52% |

#### 3B. Factor Attention 視覺化

```bash
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts \
  --output_dir plots_attention \
  --top_k 15 \
  --device cpu
```

**輸出圖表：**
- `factor_attention_top_features.png` - Top 15 重要特徵
- `factor_attention_heatmap.png` - 特徵注意力熱力圖
- `factor_attention_timeseries.png` - Top 5 特徵時間序列
- `factor_attention_summary.txt` - 統計摘要

#### 3C. 階層式特徵分析

```bash
python analyze_contexts.py \
  --artifact_dir gat_artifacts \
  --output_dir plots_contexts \
  --sample_days 20 \
  --device cpu
```

**輸出圖表：**
- `context_distributions.png` - C, C_I, C_U 分布比較
- `variance_reduction.png` - 中性化後的變異數降低
- `influence_magnitude.png` - 產業/市場影響力大小
- `context_pca_projection.png` - 2D PCA 投影

#### 3D. 投資組合回測與完整報告

```bash
python plot_reports.py \
  --artifact_dir gat_artifacts \
  --weights gat_artifacts/dmfm_wei2022_best.pt \
  --benchmark_csv GAT0050.csv \
  --out_dir plots_reports \
  --top_pct 0.10 \
  --rebalance_days 5 \
  --device cuda
```

**輸出圖表：**
- `cum_returns.png` - 累積報酬（策略 vs 0050）
- `daily_ic.png` - Daily IC 時間序列
- `hitrate_by_month.png` - 月度命中率
- `pred_dispersion.png` - 預測離散度
- `ic_distribution.png` - IC 分布直方圖
- `attention_weights.png` - 注意力權重（DMFM 限定）

---

## 🔄 Baseline 對照模型（Linear / LSTM / XGBoost）

使用 `train_baselines.py` 可以快速訓練非圖神經網路的對照組，與 DMFM/GAT 做橫向比較。所有模型共用 `build_artifacts.py` 產出的資料。

**1) 線性回歸（Ridge）**

```bash
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model linear \
  --train_ratio 0.8
```

**2) XGBoost**

```bash
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model xgboost \
  --n_estimators 300 \
  --max_depth 6 \
  --learning_rate 0.05
```

**3) LSTM（使用 lookback 時序）**

```bash
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model lstm \
  --lookback 10 \
  --epochs 30 \
  --batch_size 256 \
  --device cuda
```

**輸出檔案（存放在 `artifact_dir`）：**

| 模型 | 權重/模型 | Scaler | 指標檔 |
|------|-----------|--------|--------|
| linear | `baseline_linear.pkl` | `baseline_linear_scaler.pkl` | `baseline_linear_metrics.json` |
| xgboost | `baseline_xgboost.json` | `baseline_xgboost_scaler.pkl` | `baseline_xgboost_metrics.json` |
| lstm | `baseline_lstm.pt` | - | `baseline_lstm_metrics.json` |

每個指標檔包含訓練/測試集的 MSE、IC、ICIR、方向準確率等，方便與 DMFM、GAT 作圖或表格比較。

---

## 📊 一鍵執行完整實驗

```bash
bash run_all_models.sh
```

**包含以下 4 個實驗：**

| 實驗 | 模型 | 時間範圍 | 資料量 | 預估時間 |
|------|------|---------|--------|---------|
| 1 | DMFM | 2019-2020 (短期) | 1.3年 | ~40分鐘 |
| 2 | DMFM | 2019-2022 (中期) | 3.3年 | ~60分鐘 |
| 3 | DMFM | 2019-2025 (長期) | 6年 | ~90分鐘 |
| 4 | GATRegressor | 2019-2022 (對照) | 3.3年 | ~25分鐘 |

**總時間：** 約 3.5-4 小時（RTX 5090）

**輸出結構：**
```
artifacts_short|medium|long/  # 依時間視窗儲存的訓練張量與權重
experiments/                  # run_core_experiments.sh 產出的指標/圖表
examples/                     # 已整理好的範例 artifacts 與 plots（只讀示例）
```

---

## 🔬 進階用法

### 調整超參數

```bash
python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts \
  --epochs 200 \
  --lr 5e-4 \              # 學習率（預設 1e-4）
  --hidden_dim 128 \       # 隱藏層維度（預設 64）
  --heads 4 \              # 注意力頭數（預設 2）
  --dropout 0.2 \          # Dropout（預設 0.1）
  --lambda_attn 0.1 \      # Attention loss 權重
  --lambda_ic 1.0 \        # IC loss 權重
  --patience 50 \          # Early stopping 耐心值
  --device cuda
```

### 不同預測視野實驗

```bash
# Horizon = 1 日（超短期）
python build_artifacts.py ... --horizon 1
python train_dmfm_wei2022.py --artifact_dir gat_artifacts_h1 ...

# Horizon = 10 日（中期）
python build_artifacts.py ... --horizon 10
python train_dmfm_wei2022.py --artifact_dir gat_artifacts_h10 ...

# Horizon = 20 日（長期）
python build_artifacts.py ... --horizon 20
python train_dmfm_wei2022.py --artifact_dir gat_artifacts_h20 ...
```

---

## 🏗️ DMFM 模型架構

### 完整架構圖（對齊 Wei et al. 2022 論文）

```
原始特徵 x [N, F=56]
    ↓
[BatchNorm]  ← 等價於截面 z-score 標準化
    ↓
[MLP Encoder: F → 64]
    ↓
編碼特徵 C [N, 64]  ← 第一種特徵
    ↓
┌─────────────────────────────────┐
│  產業中性化 (Industry Neutral)   │
└─────────────────────────────────┘
    ↓
[GAT on Industry Graph, concat=False]
    ↓
產業共同影響 H_I [N, 64]
    ↓
C - H_I = C_I [N, 64]  ← 第二種特徵（產業中性化）
    ↓
┌─────────────────────────────────┐
│  全市場中性化 (Universe Neutral) │
└─────────────────────────────────┘
    ↓
[GAT on Universe Graph, concat=False]
    ↓
全市場共同影響 H_U [N, 64]
    ↓
C_I - H_U = C_U [N, 64]  ← 第三種特徵（全市場中性化）
    ↓
┌─────────────────────────────────┐
│   階層式特徵拼接                 │
└─────────────────────────────────┘
    ↓
[Concatenate: C || C_I || C_U] [N, 192]
    ↓
[MLP Decoder: 192 → 1]
    ↓
深度因子 f [N, 1]
    ↓
┌─────────────────────────────────┐
│   Factor Attention（可解釋性）    │
└─────────────────────────────────┘
    ↓
注意力權重 α [N, F]
    ↓
估計因子 f̂ = F^T · α
```

### 關鍵設計決策

| 設計 | Wei et al. 2022 論文 | 我們的實作 | 狀態 |
|------|---------------------|-----------|------|
| GAT concat 模式 | concat=False（平均） | concat=False | ✅ |
| Universe GAT 輸入 | C_I（不是 C） | C_I | ✅ |
| BatchNorm | 截面標準化 | BatchNorm | ✅ |
| Loss 函數 | d + (1-IC) - b | λ_attn·d + λ_IC·(1-IC) - λ_b·b | ✅ |
| Factor return clip | 無 | Clip [-10, 10] | ✅ 數值穩定 |

---

## 🐛 常見問題與解決方案

### Q1: 訓練時 Test IC 變成 0.0？

**現象：**
```
Epoch  20 | Test IC: 0.0000 | Test ICIR: 0.0000 | Test FR: 0.0000
```

**原因：** 過濾 NaN 後圖結構太稀疏，GAT 輸出退化

**解決：** ✅ 已自動跳過邊數 < 10 (industry) 或 < 100 (universe) 的時間點

---

### Q2: Loss 震盪劇烈？

**現象：**
```
Epoch   5 | Loss: 8.93
Epoch  10 | Loss: -903.45
Epoch  15 | Loss: 378.12
```

**原因：**
1. 學習率太高
2. Factor return 不穩定（數值過大）

**解決：**
- ✅ 預設學習率降至 1e-4
- ✅ Factor return clip 到 [-10, 10]
- ✅ 降低 factor return 權重 (lambda_b=0.01)

---

### Q3: GPU 沒有被使用？

**現象：**
```
使用裝置: cpu
```

**原因：** 沒有指定 `--device cuda`

**解決：**
```bash
python train_dmfm_wei2022.py --device cuda ...

# 監控 GPU 使用
watch -n 1 nvidia-smi
```

---

### Q4: CUDA index out of bounds？

**現象：**
```
scatter gather kernel index out of bounds
```

**原因：** 過濾節點後 edge_index 沒有重新映射

**解決：** ✅ 已修正（使用 `filter_edge_index` 函數自動重新映射）

---

### Q5: 變數名稱衝突？

**現象：**
```
AttributeError: 'int' object has no attribute 'elu'
```

**原因：** `N, F = x.shape` 覆蓋了 `import torch.nn.functional as F`

**解決：** ✅ 已修正（改用 `num_features`）

---

## 📦 環境需求

### Python 套件

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse
pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm
```

或使用：
```bash
bash utils/setup_env.sh
```

### 硬體需求

| 配置 | 最低 | 推薦 |
|------|------|------|
| GPU | GTX 1080 (8GB) | RTX 4090 / 5090 (24GB+) |
| RAM | 16GB | 32GB+ |
| 儲存空間 | 10GB | 50GB+ |

---

## 📚 相關文件

- **專案總覽：** [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- **階層式中性化：** [docs/HIERARCHICAL_NEUTRALIZATION_EXPLAINED.md](docs/HIERARCHICAL_NEUTRALIZATION_EXPLAINED.md)
- **DMFM 技術細節：** [docs/README_DMFM_Wei2022.md](docs/README_DMFM_Wei2022.md)
- **RunPods 指南：** [docs/RUNPODS_GUIDE.md](docs/RUNPODS_GUIDE.md)
- **環境設置：** [docs/VENV_SETUP.md](docs/VENV_SETUP.md)

---

## 📖 參考文獻

1. Wei et al. (2022). "A Deep Multi-Factor Model for Stock Return Prediction"
2. Veličković et al. (2018). "Graph Attention Networks" (ICLR 2018)
3. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

---

## 📧 聯絡方式

**Lo Yi (羅頤)**
National Yang Ming Chiao Tung University
Graduate Institute of Information Management & Finance
E-mail: roy60404@gmail.com

---

**最後更新：** 2025-12-14
**版本：** v2.0 (重構後)
