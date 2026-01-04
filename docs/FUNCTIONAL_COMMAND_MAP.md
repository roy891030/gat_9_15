# 功能與指令對照表

下列整理專案的主要功能、對應指令與常用參數，以及主要輸出與涵義，便於快速查找。指令預設均以 `python` 執行，除非特別標註。

## 1. 資料建置（Artifacts）
- **功能**：將原始價格/產業資料轉換為模型可用的特徵張量、標籤與圖結構。
- **指令**：`python build_artifacts.py`
- **常用參數**：
  - `--prices`（必填）：價格或因子 CSV。
  - `--industry_csv`：產業對照 CSV。
  - `--artifact_dir`：輸出資料夾。
  - `--start_date` / `--end_date`：日期區間。
  - `--horizon`：預測視野（天）。【F:README.md†L69-L84】【fe0e45†L1-L10】
- **輸出**：`Ft_tensor.pt`、`yt_tensor.pt`、圖結構（industry/universe edge）、`meta.pkl`。【F:README.md†L79-L84】

## 2. DMFM 訓練（Wei 2022 實作）
- **功能**：依論文架構訓練階層式中性化 DMFM。
- **指令**：`python train_dmfm_wei2022.py`
- **常用參數**：
  - `--artifact_dir`：輸入 artifacts 目錄。
  - `--epochs`、`--lr`、`--weight_decay`：訓練超參數。
  - `--hidden_dim`、`--heads`、`--dropout`：模型維度與正則。
  - `--lambda_attn`、`--lambda_ic`：損失權重。
  - `--patience`、`--train_ratio`：早停與時間切分。
  - `--device`：`cpu/cuda/mps/auto`。【6da12b†L1-L22】
- **輸出**：最佳/最終權重與訓練日誌（置於 artifact_dir）。【F:README.md†L101-L105】

## 3. 簡化版 GAT/DMFM 訓練
- **功能**：以 GATRegressor 或精簡 DMFM 結構訓練，支援多種 loss 與梯度裁剪。
- **指令**：`python train_gat_fixed.py`
- **常用參數**：
  - `--artifact_dir`、`--device`、`--epochs`、`--lr`、`--patience`。
  - `--hid`、`--heads`、`--dropout`、`--tanh_cap`：模型與輸出限制。
  - `--loss`：`mse/huber/corr_mse/corr_mse_ind`。
  - `--clip_grad`、`--huber_delta`、`--alpha_mse`、`--lambda_var`、`--lambda_attn`：穩定性與正則設定。【0f1b6c†L1-L31】
- **輸出**：權重與訓練記錄（存於 `artifact_dir`）。

## 4. Baseline 對照組（Linear / XGBoost / LSTM）
- **功能**：提供非圖模型基準以比較 DMFM/GAT 效果。
- **指令**：`python train_baselines.py`
- **常用參數**：
  - 通用：`--artifact_dir`、`--model {linear,lstm,xgboost}`、`--train_ratio`、`--device`、`--seed`。
  - LSTM：`--lookback`、`--epochs`、`--batch_size`、`--hidden_dim`、`--dropout`、`--lr`。
  - XGBoost：`--n_estimators`、`--max_depth`、`--learning_rate`、`--subsample`、`--colsample_bytree`。【56db31†L267-L288】
- **輸出**：各模型權重 + scaler（linear/xgboost）與 `baseline_<model>_metrics.json` 指標檔，含訓練/測試的 MSE、IC、ICIR、方向準確率等。【F:README.md†L225-L233】

## 5. 評估指標（IC/ICIR/MSE/DirAcc 等）
- **功能**：載入模型權重與 artifacts，輸出多項預測指標。
- **指令**：`python evaluate_metrics.py`
- **常用參數**：`--artifact_dir`、`--weights`、`--device`、`--tanh_cap`、`--industry_csv`、`--hid`、`--heads`。【6eab2f†L1-L14】
- **輸出**：終端列印與檔案（在 `artifact_dir` 或 `experiments/`）的指標結果，對應 README 中表格範圍。【F:README.md†L117-L133】

## 6. 階層式特徵分析
- **功能**：檢視 C/C_I/C_U 分布、變異數降低與 PCA 投影以解讀階層式中性化效果。
- **指令**：`python analyze_contexts.py`
- **常用參數**：`--artifact_dir`、`--model_path`、`--output_dir`、`--device`、`--sample_days`。【05ef32†L1-L5】
- **輸出**：多張圖表（分布、變異數、影響力、PCA）與摘要。【F:README.md†L150-L164】

## 7. 投資組合回測與完整報告
- **功能**：依模型預測生成選股、回測績效、IC 時序、命中率等完整報告，可含注意力圖。
- **指令**：`python plot_reports.py`
- **常用參數**：`--artifact_dir`、`--weights`、`--out_dir`、`--device`、`--tanh_cap`、`--hid`、`--heads`；策略設定 `--top_pct`、`--rebalance_days`；基準/產業 CSV 透過 `--benchmark_csv`、`--industry_csv`。【9a73d4†L1-L12】【F:README.md†L168-L185】
- **輸出**：回測曲線、IC/命中率/離散度等圖表與注意力權重視覺化，統一存於 `out_dir`。【F:README.md†L168-L185】

## 8. 一鍵核心實驗
- **功能**：批次執行短/中/長期 DMFM 與 GAT 對照四組實驗。
- **指令**：`bash run_all_models.sh`
- **輸出**：`artifacts_short|medium|long/`、`experiments/` 結果與 `examples/` 參考樣本，耗時估計 3.5~4 小時（RTX 5090）。【F:README.md†L237-L259】

## 9. 其他工具
- `evaluate_portfolio.py`：著重回測與注意力視覺化，參數與 `plot_reports.py` 接近。【3b3fdb†L500-L543】
- `analyze_contexts.py`、`diagnose_labels.py`、`diagnose_model.py`：資料與模型診斷；`utils/` 內有 CSV 修正、環境設置腳本。【F:README.md†L15-L55】

> 小技巧：若需不同預測視野，可在建檔時修改 `--horizon` 並重跑訓練；DMFM 超參數可用 README「進階用法」範例作為起點再微調。【F:README.md†L265-L295】
