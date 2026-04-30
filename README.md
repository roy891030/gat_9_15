# DMFM / GAT 台股預測

本專案保留單一主流程，主要入口是 `run_pipeline.py`。

## 主要入口

- `run_pipeline.py`: 統一跑 `baseline / gat / dmfm`
- `run_all_models.sh`: full run 包裝器
- `post_process_all.sh`: 只做評估、回測與圖表

## 核心腳本

- `build_artifacts.py`: 建立 `artifacts/<window>/`
- `train_dmfm_wei2022.py`: 訓練 DMFM
- `train_gat_fixed.py`: 訓練 GAT
- `train_baselines.py`: 訓練 `linear / xgboost / lstm`
- `evaluate_metrics.py`: 輸出 train / val / test 指標
- `evaluate_portfolio.py`: 輸出回測與圖表
- `visualize_factor_attention.py`: DMFM attention 視覺化
- `analyze_contexts.py`: DMFM context 分析
- `model_dmfm_wei2022.py`: 模型定義

## 快速開始

使用專案 venv：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --device cuda
```

只做 smoke test：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode smoke \
  --device cpu
```

只做後處理：

```bash
bash post_process_all.sh --device cuda --output_root runs_gpu_full
```

## 輸出結構

- `artifacts/<window>/`: 特徵、標籤、圖結構、模型權重
- `runs*/<window>/<model>/`: `train.log`、`metrics.json`、`portfolio.json`、`plots/`

## 主要評估口徑

- `IC`
- `DailyIC`
- `ICIR`
- `DirAcc`
- `MSE / RMSE / MAE`

## 備註

- 回測優先使用 `yraw_tensor.pt`
- `run_all_models_parallel.sh` 目前只是 `run_pipeline.py` 的別名包裝
- 專案已移除舊診斷腳本與重複文件，後續請以本 README 和程式 CLI 為準
