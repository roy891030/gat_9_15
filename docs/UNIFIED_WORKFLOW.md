# Unified Workflow (baseline / GAT / DMFM)

本專案已整合為單一入口：`run_pipeline.py`。

## 1) 先做 smoke 測試（本機）

### baseline（短中長）
```bash
python3 run_pipeline.py --models baseline --baseline_models linear,xgboost,lstm --windows all --mode smoke
```

### GAT（短中長）
```bash
python3 run_pipeline.py --models gat --windows all --mode smoke
```

### DMFM（短中長）
```bash
python3 run_pipeline.py --models dmfm --windows all --mode smoke
```

## 2) 單獨跑某個模型（full）

### baseline（短中長）
```bash
python3 run_pipeline.py --models baseline --baseline_models linear,xgboost,lstm --windows all --mode full --device cuda
```

### GAT（短中長）
```bash
python3 run_pipeline.py --models gat --windows all --mode full --device cuda
```

### DMFM（短中長）
```bash
python3 run_pipeline.py --models dmfm --windows all --mode full --device cuda
```

## 3) 全部一起跑（full）
```bash
python3 run_pipeline.py --models all --baseline_models linear,xgboost,lstm --windows all --mode full --device cuda
```

或使用相容腳本：
```bash
bash run_all_models.sh
```

## 4) 只做後處理（已有模型）
不重建資料、不重訓，只重新輸出：
- evaluation 指標
- 回測表現
- 指標與回測圖

```bash
python3 run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  --skip_build \
  --skip_train
```

或：
```bash
bash post_process_all.sh
```

## 5) 輸出結構

### Artifacts
- `artifacts/short/`
- `artifacts/medium/`
- `artifacts/long/`

### Results
- `runs/<window>/<model>/`
  - `train.log`
  - `metrics.log`（GAT/DMFM）
  - `evaluate.log`（baseline）
  - `backtest.log`
  - `plots/`
    - `daily_ic.png`
    - `pred_dispersion.png`
    - `hitrate_by_month.png`
    - `ic_distribution.png`
    - `cum_returns.png`
    - `metrics.json`（baseline）

- `runs/run_summary.json`：本次執行摘要。

## 6) 常用參數
- `--models baseline,gat,dmfm,all`
- `--baseline_models linear,xgboost,lstm`
- `--windows short,medium,long,all`
- `--mode smoke|full`
- `--device auto|cpu|cuda|mps`
- `--skip_build`
- `--skip_train`
- `--dry_run`（只印命令，不執行）

