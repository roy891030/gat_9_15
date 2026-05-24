# DMFM / GAT 台股預測

本專案保留單一主流程，主要入口是 `run_pipeline.py`。

## 主要入口

- `run_pipeline.py`: 統一跑 `baseline` 與 factor ablation models
- `run_all_models.sh`: full run 包裝器
- `post_process_all.sh`: 只做評估、回測與圖表

## 核心腳本

- `build_artifacts.py`: 建立 `artifacts/<window>/`
- `train_dmfm_wei2022.py`: 訓練舊版完整 DMFM（保留給舊 checkpoint / standalone 實驗）
- `train_factor_variants.py`: 訓練 `mlp / gat_industry / gat_universe / gat_two_graph_no_neutral / dmfm_ind_neutral / dmfm_full`
- `train_gat_fixed.py`: 訓練舊版 GAT（保留給舊 checkpoint / standalone 實驗）
- `train_baselines.py`: 訓練 `linear / xgboost / lstm`
- `evaluate_metrics.py`: 輸出 train / val / test 指標
- `evaluate_portfolio.py`: 輸出回測與圖表
- `model_dmfm_wei2022.py`: 模型定義
- `unused_not_main_pipeline/`: 不在主要 pipeline 中使用的舊分析、視覺化與實驗性程式

## 快速開始

目前主要 pipeline 預設使用 static graph，並先建立一次 shared full artifact，再切出各 window view。

Dynamic weighted graph 程式仍保留，但不作為目前主流程預設。若明確指定 `--graph_modes dynamic`，會建立：

- Industry graph: 每日 rolling return correlation，同產業 top-k 加權邊
- Universe graph: 每日 rolling return correlation，全市場 top-k 加權邊
- Edge attribute: `[abs_corr, signed_corr]`

使用專案 venv：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts
```

只做 smoke test：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --windows all \
  --mode smoke \
  --device cpu \
  --rebuild_artifacts
```

只跑六個 factor ablation models：

```bash
.venv/bin/python run_pipeline.py \
  --models factor_variants \
  --windows all \
  --mode smoke \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_factor_smoke
```

只跑指定模型與指定時間窗：

```bash
.venv/bin/python run_pipeline.py \
  --models mlp,gat_industry,gat_universe,gat_two_graph_no_neutral,dmfm_ind_neutral,dmfm_full \
  --windows short,medium,long \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_factor_full
```

RunPod 上只跑六個 GAT/DMFM ablation models：

```bash
cd /workspace/gat_9_15
python run_pipeline.py \
  --models factor_variants \
  --windows short,medium,long \
  --mode full \
  --device cuda \
  --artifact_root artifacts_dynamic \
  --output_root runs_factor_dynamic \
  --rebuild_artifacts
```

RunPod 上用同一份 pipeline 統一比較 baseline / static graph / dynamic graph：

```bash
cd /workspace/gat_9_15
python run_pipeline.py \
  --models baseline,factor_variants \
  --windows short,medium,long \
  --mode full \
  --device cuda \
  --graph_modes static,dynamic \
  --baseline_graph_mode static \
  --artifact_root artifacts_unified \
  --output_root runs_unified \
  --rebuild_artifacts \
  --parallel_jobs 2 \
  --preload_gpu
```

- `--graph_modes static,dynamic` 會用同一組 `WINDOW_SPECS` 建立 `artifacts_unified/static/<window>` 與 `artifacts_unified/dynamic/<window>`，避免 static/dynamic 的時間窗口不一致。
- baseline 不使用 graph，預設只跑在 `static` 口徑下；若需要兩邊都跑，改成 `--baseline_graph_mode both`。
- RTX 4090 上建議先用 `--parallel_jobs 2`；若 VRAM 還有餘量再提高到 3。dynamic graph artifact 建構是 CPU/NumPy 工作，GPU 使用率低是正常的；GPU 主要用於 train/evaluation。

RunPod smoke test：

```bash
cd /workspace/gat_9_15
python run_pipeline.py \
  --models factor_variants \
  --windows short \
  --mode smoke \
  --device cuda \
  --artifact_root artifacts_dynamic_smoke \
  --output_root runs_factor_dynamic_smoke \
  --rebuild_artifacts
```

只做後處理：

```bash
bash post_process_all.sh --device cuda --output_root runs_gpu_full
```

## 輸出結構

- `artifacts/<window>/`: 特徵、標籤、static graph、dynamic weighted graph、模型權重
- `runs*/<window>/<model>/`: `train.log`、`metrics.json`、`portfolio.json`、`plots/`
- `--models all`: 預設跑 `baseline_linear / baseline_xgboost / baseline_lstm` 加上六個 factor ablation models
- 舊版 `gat` 與 `dmfm` 不再由 `run_pipeline.py` 執行；如需載入舊 checkpoint，evaluation 仍保留相容。

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
