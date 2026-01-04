# 📋 專案結構整理報告

**整理日期：** 2026-01-04  
**目的：** 把重複的輸出資料集中到單一目錄，讓核心程式與文件更容易瀏覽。

---

## ✅ 本次重點調整

1. **範例輸出集中化：** 將原本分散在根目錄的 `plots_*` 以及情境 artifacts 改為 `examples/plots/*` 與 `examples/artifacts/*`，避免視覺化結果佔滿根目錄。
2. **樹狀結構精簡：** 只保留核心腳本、模型、執行腳本與文件的主路徑，舊版腳本統一放在 `archived/`。
3. **文件同步：** README 與指南皆對齊新的範例輸出位置，減少重複描述。

---

## 📂 最新專案結構總覽

```
gat_9_15/
├── 🔧 核心腳本
│   ├── build_artifacts.py
│   ├── train_dmfm_wei2022.py
│   ├── train_gat_fixed.py
│   ├── train_baselines.py
│   ├── evaluate_metrics.py
│   ├── evaluate_portfolio.py
│   ├── visualize_factor_attention.py
│   ├── analyze_contexts.py
│   └── plot_reports.py
│
├── 🤖 模型定義
│   └── model_dmfm_wei2022.py
│
├── 🚀 執行腳本
│   ├── run_core_experiments.sh
│   ├── run_all_models.sh               # 包裝 run_core_experiments
│   └── run_all_models_parallel.sh      # 包裝 archived 並行腳本
│
├── 🧹 輔助腳本
│   ├── clean_all_results.sh            # 清理 artifacts/plots/log/彙總
│   └── post_process_all.sh             # 對現有 artifacts_* 生成評估與圖表
│
├── 🧪 範例輸出 (examples/)
│   ├── artifacts/
│   │   ├── covid_crash/
│   │   └── rate_hike/
│   └── plots/
│       ├── short/{dmfm,gat}/
│       ├── medium/{dmfm,gat}/
│       ├── long/{dmfm,gat}/
│       ├── covid_crash/{dmfm,gat}/
│       └── rate_hike/{dmfm,gat}/
│
├── 📚 文件 (docs/)
│   ├── PROJECT_OVERVIEW.md
│   ├── HIERARCHICAL_NEUTRALIZATION_EXPLAINED.md
│   ├── README_DMFM_Wei2022.md
│   ├── RUNPODS_GUIDE.md
│   ├── FUNCTIONAL_COMMAND_MAP.md
│   ├── CONFLICT_RESOLUTION.md
│   ├── VENV_SETUP.md
│   └── CHANGES_DMFM_Wei2022.md
│
├── 🛠️ 工具 (utils/)
│   ├── check_csv.sh
│   ├── fix_csv_columns.py
│   └── setup_env.sh
│
└── 📦 歸檔 (archived/)
    ├── run_dmfm_wei2022.sh
    ├── run_experiments.sh
    ├── run_all_models_parallel.sh
    └── clean_all_results.sh
```

---

## 🔍 瀏覽提示
- **想看範例圖表：** 前往 `examples/plots/`，依時間窗口（short/medium/long）或情境（covid_crash/rate_hike）分組，並以 `dmfm`、`gat` 區分模型。
- **想對照情境設定：** `examples/artifacts/` 保留疫情崩盤與升息期間的 meta 與訓練日誌，方便確認設定。
- **需要重跑：** 直接使用根目錄的核心腳本；範例輸出僅作參考，與新的訓練流程互不干擾。

