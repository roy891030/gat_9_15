# Deep Multi-Factor GAT Model for Stock Prediction

本專案實作了一個基於 **Graph Attention Network (GAT)** 與 **Deep Multi-Factor Model (DMFM)** 的股票預測系統，結合多層特徵工程、產業圖關聯結構以及橫截面超額報酬預測。  
專案可自動完成資料處理、模型訓練、績效評估與投資組合回測。

---

## 專案架構

```
├── build_artifacts.py        # 建立特徵與標籤張量、產業圖與全市場圖
├── train_gat_fixed.py        # 訓練 GAT 或 DMFM 模型
├── evaluate_metrics.py       # 輸出 IC、ICIR、MSE、Dir Accuracy 等指標
├── evaluate_portfolio.py     # 以預測結果建立投組、進行回測與統計
├── plot_reports.py           # 生成與 0050 等基準的比較圖表
├── run_all_experiments.sh    # 一鍵執行短/中/長期實驗的自動化腳本
└── README.md
```

---

## 模型概述

### 1. Deep Multi-Factor Model (DMFM)
DMFM 為改良版 GAT 模型，結合：
- **Cross-sectional correlation loss (corr_mse_ind)**  
- **Factor variance regularization (λ_var)**  
- **Attention sparsity constraint (λ_attn)**  

以捕捉多層次因子間的非線性互動與橫截面穩定性。

### 2. GAT Regressor
以產業圖與全市場圖為結構，透過 GAT 聚合鄰近股票特徵，預測未來 N 日的報酬率。

---

## 執行流程

整體流程分為四個主要階段：

### 資料前處理與特徵構建
```bash
python build_artifacts.py   --prices unique_2019q3to2025q3.csv   --industry_csv unique_2019q3to2025q3.csv   --artifact_dir gat_artifacts_out_plus   --start_date 2019-09-16   --end_date 2025-09-12   --horizon 5
```
輸出：
- `Ft_tensor.pt`、`yt_tensor.pt`、`industry_edge_index.pt`、`universe_edge_index.pt`、`meta.pkl`

---

### 模型訓練
```bash
python train_gat_fixed.py   --artifact_dir gat_artifacts_out_plus   --epochs 50 --lr 1e-3 --device cuda   --loss dmfm   --alpha_mse 0.03 --lambda_var 0.1 --lambda_attn 0.05   --industry_csv unique_2019q3to2025q3.csv
```
輸出：`gat_regressor.pt`、`train_log.txt`

---

### 模型評估
```bash
python evaluate_metrics.py   --artifact_dir gat_artifacts_out_plus   --weights gat_artifacts_out_plus/gat_regressor.pt   --device cuda   --industry_csv unique_2019q3to2025q3.csv
```
| 指標 | 含義 | 評價準則 |
|------|------|-----------|
| IC | Information Coefficient (預測與真實報酬的相關性) | > 0.05 為可交易水準 |
| ICIR | IC 的穩定性 (均值 / 標準差) | > 0.5 穩定，> 1 表現極佳 |
| Dir Accuracy | 預測方向準確率 | > 55% 為具實用價值 |
| MSE / MAE | 預測誤差 | 越低越好 |

---

### 投資組合回測與視覺化
```bash
python evaluate_portfolio.py   --artifact_dir gat_artifacts_out_plus   --weights gat_artifacts_out_plus/gat_regressor.pt   --device cuda   --top_pct 0.10 --rebalance_days 5   --industry_csv unique_2019q3to2025q3.csv
```

```bash
python plot_reports.py   --artifact_dir gat_artifacts_out_plus   --weights gat_artifacts_out_plus/gat_regressor.pt   --device cuda   --out_dir plots_vs_0050   --top_pct 0.10 --rebalance_days 5   --benchmark_csv GAT0050.csv
```

---

## 一鍵執行完整實驗
```bash
bash run_all_experiments.sh
```
輸出：
```
results_short_metrics.txt
results_medium_metrics.txt
results_long_metrics.txt
results_gat_metrics.txt
plots_short_vs_0050/
plots_medium_vs_0050/
plots_long_vs_0050/
plots_gat_vs_0050/
```

---

## 參數說明

| 參數 | 說明 |
|------|------|
| `--artifact_dir` | 模型輸入與輸出資料夾 |
| `--epochs` | 訓練週期數 |
| `--lr` | 學習率 |
| `--loss` | 損失函數類型（`dmfm`、`corr_mse`、`corr_mse_ind`） |
| `--alpha_mse` | MSE 權重 |
| `--lambda_var` | 變異度正則化權重 |
| `--lambda_attn` | Attention 稀疏正則化權重 |
| `--top_pct` | 回測中每次持有的 top 百分比股票 |
| `--rebalance_days` | 投組調整間隔天數 |
| `--benchmark_csv` | 基準檔案路徑 (例：0050 ETF) |

---

## 評估指標範例解讀

| 指標 | 結果 | 評價 |
|------|------|------|
| IC (測試集) | 0.078 | ✅ 可用水準（0.05~0.10） |
| ICIR (測試集) | 0.94 | ✅ 高穩定性 |
| Dir Accuracy | 57% | ✅ 明顯優於隨機 |
| MSE | 0.0074 | ⚠ 稍高於天真模型，仍合理 |

---

## 注意事項

- 請勿將 `unique_2019q3to2025q3.csv` 或 `gat_artifacts_*` 上傳至 GitHub。
- 若需上傳大檔案，請使用 [Git LFS](https://git-lfs.github.com/)。
- 需安裝的主要套件包括：
  ```bash
  pip install torch numpy pandas matplotlib tqdm talib
  ```

---

## 作者

**Lo Yi (羅頤)**  
National Yang Ming Chiao Tung University  
Graduate Institute of Information Management & Finance  
E-mail: roy60404@gmail.com
