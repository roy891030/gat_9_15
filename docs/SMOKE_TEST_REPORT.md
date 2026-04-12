# Smoke Test Report

執行時間：2026-04-12  
執行指令：

```bash
.venv/bin/python run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode smoke \
  --device cpu \
  --rebuild_artifacts \
  --output_root runs_smoke_validation
```

摘要檔：
- `runs_smoke_validation/run_summary.json`

## 結論

- smoke test 已完整跑完 `15/15` 組實驗。
- 未掃到 `Traceback`、`Error`、`Exception`、`No such file` 等流程錯誤。
- 三個視窗 `short / medium / long` 都成功完成：
  - `baseline_linear`
  - `baseline_xgboost`
  - `baseline_lstm`
  - `gat`
  - `dmfm`
- 每組輸出都已對齊到統一結構：
  - `train.log`
  - `backtest.log`
  - `metrics.json`
  - `portfolio.json`
  - `metrics.log`（僅 `gat` / `dmfm`）
- benchmark `GAT0050.csv` 在三個視窗都可正常參與回測。

## Benchmark 對齊

| Window | 對齊區間 | 對齊期數 |
|---|---|---:|
| short | `2020-09-30 ~ 2020-12-28` | 13 |
| medium | `2022-05-16 ~ 2022-12-29` | 33 |
| long | `2024-07-04 ~ 2025-09-04` | 58 |

註記：
- `long` 視窗原始 test rebalance 共有 59 次，但最後一次因為 `horizon=5` 需要後續交易日，benchmark 無法形成最後一筆 forward-5，因此實際對齊 58 期。這是資料邊界限制，不是流程錯誤。

## 輸出對齊檢查

| Window | Model | `metrics.json` | `portfolio.json` | `train.log` | `backtest.log` | `metrics.log` |
|---|---|---|---|---|---|---|
| short | baseline_linear | yes | yes | yes | yes | n/a |
| short | baseline_xgboost | yes | yes | yes | yes | n/a |
| short | baseline_lstm | yes | yes | yes | yes | n/a |
| short | gat | yes | yes | yes | yes | yes |
| short | dmfm | yes | yes | yes | yes | yes |
| medium | baseline_linear | yes | yes | yes | yes | n/a |
| medium | baseline_xgboost | yes | yes | yes | yes | n/a |
| medium | baseline_lstm | yes | yes | yes | yes | n/a |
| medium | gat | yes | yes | yes | yes | yes |
| medium | dmfm | yes | yes | yes | yes | yes |
| long | baseline_linear | yes | yes | yes | yes | n/a |
| long | baseline_xgboost | yes | yes | yes | yes | n/a |
| long | baseline_lstm | yes | yes | yes | yes | n/a |
| long | gat | yes | yes | yes | yes | yes |
| long | dmfm | yes | yes | yes | yes | yes |

## 指標摘要

### short

| Model | Test IC | Test ICIR | Strategy Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| baseline_linear | 0.0238 | 0.3778 | 0.6498 | 5.2428 | -0.0172 |
| baseline_xgboost | 0.0318 | 0.2646 | 0.6262 | 5.0714 | -0.0149 |
| baseline_lstm | 0.0545 | 0.7162 | 0.9338 | 6.9447 | -0.0040 |
| gat | 0.0297 | 0.3704 | 0.4709 | 4.2727 | -0.0154 |
| dmfm | 0.0467 | 0.5027 | 0.9205 | 6.3119 | -0.0120 |

### medium

| Model | Test IC | Test ICIR | Strategy Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| baseline_linear | 0.0181 | 0.1155 | 0.0342 | 0.1226 | -0.1609 |
| baseline_xgboost | 0.0389 | 0.3716 | 0.2058 | 0.7466 | -0.1529 |
| baseline_lstm | 0.0620 | 0.5882 | 0.3085 | 1.2349 | -0.1200 |
| gat | -0.0173 | -0.1792 | 0.2372 | 0.9555 | -0.1584 |
| dmfm | 0.0454 | 0.4384 | 0.2702 | 0.9505 | -0.1438 |

### long

| Model | Test IC | Test ICIR | Strategy Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| baseline_linear | 0.0356 | 0.3337 | 0.2021 | 0.6789 | -0.2172 |
| baseline_xgboost | 0.0741 | 0.7968 | 0.3536 | 1.2295 | -0.2076 |
| baseline_lstm | 0.0810 | 0.7012 | 0.3621 | 1.1863 | -0.2095 |
| gat | 0.0017 | 0.0514 | 0.0095 | 0.0375 | -0.2114 |
| dmfm | 0.0623 | 0.7757 | 0.3400 | 1.2290 | -0.1970 |

## 觀察

- smoke 流程本身已經證明目前的資料建置、train/val/test 切分、結構化評估輸出、回測輸出與 benchmark 對齊是通的。
- `baseline` 現在和 `gat / dmfm` 一樣都走 `train / val / test`，輸出 schema 也一致。
- `portfolio.json` 在所有模型上都已包含交易面統計欄位，例如：
  - `annual_return`
  - `annual_volatility`
  - `sharpe`
  - `hit_rate`
  - `total_return`
  - `cagr`
  - `max_drawdown`
- `dmfm` 三個視窗都成功輸出 attention 圖；baseline 與 GAT 的 `attention_weights` 欄位為空值，符合預期。
- CPU smoke 下最耗時的是 `medium / long` 的 `dmfm` 訓練，但最後皆能完成，沒有流程中斷。
