# DMFM / GAT 台股預測

本專案保留單一主流程，主要入口是 `run_pipeline.py`。

## 主要入口

- `run_pipeline.py`: 統一跑 `baseline` 與 factor ablation models
- `run_all_models.sh`: full run 包裝器
- `post_process_all.sh`: 只做評估、回測與圖表

## 核心腳本

- `build_artifacts.py`: 建立 `artifacts/<window>/`
- `train_dmfm_wei2022.py`: 訓練舊版完整 DMFM
- `train_factor_variants.py`: 訓練 `mlp / gat_industry / gat_universe / gat_two_graph_no_neutral / dmfm_ind_neutral / dmfm_full`
- `train_gat_fixed.py`: 訓練 GAT
- `train_baselines.py`: 訓練 `linear / xgboost / lstm`
- `evaluate_metrics.py`: 輸出 train / val / test 指標
- `evaluate_portfolio.py`: 輸出回測與圖表
- `visualize_factor_attention.py`: DMFM attention 視覺化
- `analyze_contexts.py`: DMFM context 分析
- `model_dmfm_wei2022.py`: 模型定義

## 快速開始

預設會建立 dynamic weighted graph：

- Industry graph: 每日 rolling return correlation，同產業 top-k 加權邊
- Universe graph: 每日 rolling return correlation，全市場 top-k 加權邊
- Edge attribute: `[abs_corr, signed_corr]`
- 舊 static binary graph 仍保留作 fallback；如果要強制不用 dynamic graph，加入 `--no_dynamic_graphs`

使用專案 venv：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --graph_lookback 60 \
  --graph_min_obs 20 \
  --industry_top_k 20 \
  --universe_top_k 40
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
  --models baseline,gat,dmfm,factor_variants \
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
- 舊版 `gat` 與 `dmfm` 仍可用 `--models gat` 或 `--models dmfm` 單獨執行

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
