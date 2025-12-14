# 🚀 DMFM 快速上手指南

## 一、最快開始（3 行指令）

```bash
# 1. 建立資料
python build_artifacts.py --prices unique_2019q3to2025q3.csv --industry_csv unique_2019q3to2025q3.csv --artifact_dir gat_artifacts --start_date 2019-09-16 --end_date 2025-09-12 --horizon 5

# 2. 訓練模型
python train_dmfm_wei2022.py --artifact_dir gat_artifacts --epochs 200 --device cuda

# 3. 評估結果
python evaluate_metrics.py --artifact_dir gat_artifacts --weights gat_artifacts/dmfm_wei2022_best.pt --device cuda
```

---

## 二、一鍵執行所有實驗

```bash
bash run_all_models.sh
```

**包含：** 短期/中期/長期 DMFM + GATRegressor 對照
**時間：** 約 3.5-4 小時（RTX 5090）

---

## 三、核心檔案清單

| 檔案 | 用途 |
|------|------|
| `build_artifacts.py` | 資料預處理 |
| `train_dmfm_wei2022.py` | 訓練 DMFM（推薦） |
| `evaluate_metrics.py` | 評估 IC/ICIR |
| `visualize_factor_attention.py` | 視覺化注意力權重 |
| `analyze_contexts.py` | 分析階層式特徵 |
| `plot_reports.py` | 完整報告與回測 |

---

## 四、常用參數

### train_dmfm_wei2022.py

```bash
--artifact_dir    # Artifacts 資料夾
--epochs 200      # 訓練週期
--lr 1e-4         # 學習率（穩定版）
--device cuda     # 使用 GPU
--hidden_dim 64   # 隱藏層維度
--heads 2         # 注意力頭數
--patience 30     # Early stopping
```

### build_artifacts.py

```bash
--start_date 2019-09-16   # 開始日期
--end_date 2025-09-12     # 結束日期
--horizon 5               # 預測天數
```

---

## 五、評估指標標準

| 指標 | 良好範圍 | 說明 |
|------|---------|------|
| IC | 0.03 - 0.08 | 預測相關性 |
| ICIR | 0.5 - 2.0 | IC 穩定性 |
| Dir Acc | > 52% | 方向準確率 |

---

## 六、常見問題速查

| 問題 | 解決方法 |
|------|---------|
| GPU 沒用到 | 加 `--device cuda` |
| Loss 震盪 | 用預設 `lr=1e-4` |
| Test IC = 0 | 已自動修正（跳過稀疏圖） |

---

## 七、專案結構

```
gat_9_15/
├── 核心腳本 (*.py)       # 主要功能
├── docs/                 # 文件
├── utils/                # 工具腳本
└── archived/             # 舊版檔案
```

---

## 八、完整文件

- 詳細說明：[README.md](README.md)
- 階層式中性化：[docs/HIERARCHICAL_NEUTRALIZATION_EXPLAINED.md](docs/HIERARCHICAL_NEUTRALIZATION_EXPLAINED.md)
- RunPods 指南：[docs/RUNPODS_GUIDE.md](docs/RUNPODS_GUIDE.md)

---

**最後更新：** 2025-12-14
