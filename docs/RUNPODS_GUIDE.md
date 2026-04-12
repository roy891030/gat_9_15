# RunPods / Remote GPU Guide

本專案目前的唯一正式入口是 `run_pipeline.py`。  
`run_all_models.sh` 與 `post_process_all.sh` 只是它的包裝，不要再使用舊的 `train_short.log`、`gat_artifacts_short`、`plots_short_*` 這套流程名稱。

## 1. 上傳到遠端前，哪些檔案不用帶

建議不要上傳這些本機冗餘內容：

- `.venv/`
- `name_env/`
- `__pycache__/`
- `runs/`
- `runs_smoke_validation/`
- `artifacts/`（除非你刻意要帶本機 artifacts）
- `.DS_Store`

可以保留但不是遠端訓練必要檔：

- `docs/SMOKE_TEST_REPORT.md`
- `DMFM_Model_Structure*`
- `GATRegressor_Structure*`
- `label_diagnosis.png`
- `archived/`

遠端真正需要的核心檔案：

- `run_pipeline.py`
- `build_artifacts.py`
- `train_baselines.py`
- `train_gat_fixed.py`
- `train_dmfm_wei2022.py`
- `evaluate_metrics.py`
- `evaluate_portfolio.py`
- `evaluate_baseline_portfolio.py`
- `model_dmfm_wei2022.py`
- `report_utils.py`
- `utils/setup_env.sh`
- `run_all_models.sh`
- `post_process_all.sh`
- `unique_2019q3to2025q3.csv`
- `GAT0050.csv`

## 2. 建議的遠端目錄結構

```text
gat_9_15/
├── *.py
├── docs/
├── utils/
├── unique_2019q3to2025q3.csv
├── GAT0050.csv
├── artifacts/                  # 執行後產生
│   ├── short/
│   ├── medium/
│   └── long/
└── runs/                       # 執行後產生
    └── <window>/<model>/
```

## 3. 環境安裝

```bash
cd /workspace/gat_9_15
bash utils/setup_env.sh
source .venv/bin/activate
```

確認 GPU：

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. 先做 smoke 測試

遠端第一次啟動時，建議先做一次 smoke：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode smoke \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_smoke_remote
```

## 5. 真實 full 測試

### 全部模型一起跑

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_gpu_full
```

或：

```bash
bash run_all_models.sh --device cuda --rebuild_artifacts --output_root runs_gpu_full
```

### 只跑圖模型

```bash
.venv/bin/python run_pipeline.py \
  --models gat,dmfm \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_gpu_graph
```

### 只跑 baseline

```bash
.venv/bin/python run_pipeline.py \
  --models baseline \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_gpu_baseline
```

## 6. 中斷後只做後處理

如果模型已經訓練完，不想重建資料也不想重訓：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --device cuda \
  --skip_build \
  --skip_train \
  --output_root runs_gpu_full
```

或：

```bash
bash post_process_all.sh --device cuda --output_root runs_gpu_full
```

## 7. 背景執行

```bash
nohup .venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_gpu_full \
  > remote_full.log 2>&1 &
```

## 8. 監控方式

看 GPU：

```bash
watch -n 1 nvidia-smi
```

看主流程 log：

```bash
tail -f remote_full.log
```

看單一模型輸出：

```bash
tail -f runs_gpu_full/short/dmfm/train.log
tail -f runs_gpu_full/medium/gat/metrics.log
tail -f runs_gpu_full/long/baseline_xgboost/backtest.log
```

看目前已產生哪些結果：

```bash
find runs_gpu_full -maxdepth 3 -type f | sort
```

## 9. 完成後要檢查什麼

主摘要：

```bash
cat runs_gpu_full/run_summary.json
```

單一模型輸出應至少包含：

- `train.log`
- `backtest.log`
- `metrics.json`
- `portfolio.json`
- `plots/`
- `metrics.log`（GAT / DMFM）

## 10. 清理舊結果

如果要清空遠端舊結果重新跑：

```bash
bash clean_all_results.sh
```

這會刪除：

- `artifacts/`
- `runs/`
- 舊版 `artifacts_*`
- 舊版 `gat_artifacts_*`
- 舊版 `plots_*`
- 舊版 `experiments`

## 11. 實務建議

- 遠端正式跑請直接用 `run_pipeline.py`。
- `run_all_models_parallel.sh` 現在只是同一條 pipeline 的包裝，不是舊版多進程訓練器。
- benchmark 只使用 `GAT0050.csv`，不要再帶 `GAT0050.xlsx`。
- 如果你只是要正式 GPU 測試，最穩定的命令就是：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --device cuda \
  --rebuild_artifacts \
  --output_root runs_gpu_full
```
