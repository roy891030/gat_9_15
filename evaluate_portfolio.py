# -*- coding: utf-8 -*-
# evaluate_portfolio.py
"""
生成視覺化報告（支援 DMFM 和 GATRegressor）

執行方式：
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device cuda \
  --benchmark_csv GAT0050.csv \
  --out_dir plots_dmfm \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv unique_2019q3to2025q3.csv

輸出檔案結構：
plots_dmfm/
├── daily_ic.png              # Daily IC time series
├── pred_dispersion.png       # Prediction cross-sectional std
├── hitrate_by_month.png      # Monthly directional accuracy
├── ic_distribution.png       # IC distribution histogram
├── cum_returns.png           # Cumulative returns comparison
└── attention_weights.png     # Feature attention weights (DMFM only)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

from checkpoint_utils import build_graph_model_from_checkpoint, detect_checkpoint_model_type
from graph_utils import graph_at, load_dynamic_graphs
from model_dmfm_wei2022 import DMFM_Wei2022 as DMFM, FactorGraphAblation, GATRegressor
from report_utils import (
    compound_forward,
    compute_return_stats,
    infer_summary_dir,
    parse_benchmark,
    parse_dates_list,
    save_json,
)
from train_gat_fixed import load_artifacts, time_split_indices_3
# ---------- Device Selection ----------
def pick_device(device_str: str) -> torch.device:
    """自動選擇可用的計算裝置"""
    s = (device_str or "").lower()
    if s in ("cpu", "cuda", "mps"):
        if s == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA 不可用，改用 CPU")
            return torch.device("cpu")
        if s == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print("[warn] MPS 不可用，改用 CPU")
                return torch.device("cpu")
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- Utilities ----------
def safe_corr(a, b):
    """安全計算 Pearson 相關係數"""
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < 1e-8 or sb < 1e-8:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan

def detect_model_type(weights_path, device="cpu"):
    """自動偵測模型類型"""
    model_type = detect_checkpoint_model_type(weights_path, map_location=device)
    if model_type not in {"dmfm", "gat", "factor_variant"}:
        raise ValueError(f"無法辨識模型類型: {weights_path}")
    return model_type


def forward_graph_model(
    model,
    x,
    edge_industry,
    edge_universe,
    industry_edge_attr=None,
    universe_edge_attr=None,
):
    out = model(x, edge_industry, edge_universe, industry_edge_attr, universe_edge_attr)
    if isinstance(out, tuple):
        pred = out[0]
        attn_weights = out[1] if len(out) > 1 else None
        contexts = out[2] if len(out) > 2 else None
        return pred, attn_weights, contexts
    return out, None, None


def load_industry_labels(industry_csv, stocks):
    """載入產業標籤"""
    if not industry_csv or not os.path.exists(industry_csv):
        return None
    
    try:
        df = pd.read_csv(industry_csv, dtype={"證券代碼": str, "證券代碼_純代碼": str})
    except Exception:
        return None

    # 股票代碼欄
    sid_col = None
    for c in ["證券代碼_純代碼", "證券代碼", "sid", "StockID", "stock_id"]:
        if c in df.columns:
            sid_col = c
            break

    # 產業欄
    ind_col = None
    for c in ["TEJ產業_名稱", "TEJ產業_代碼", "TSE產業_名稱", "industry_TSE", "Industry", "industry", "產業"]:
        if c in df.columns:
            ind_col = c
            break

    if sid_col is None or ind_col is None:
        return None

    df = df[[sid_col, ind_col]].dropna().drop_duplicates()
    df[sid_col] = df[sid_col].astype(str)
    m = dict(zip(df[sid_col], df[ind_col]))
    return [m.get(s, "UNK") for s in stocks]


# ---------- Main Report Generation ----------
def build_reports(artifact_dir, weights, out_dir,
                  device="auto", tanh_cap=0.2, hid=64, heads=2,
                  top_pct=0.10, rebalance_days=5,
                  benchmark_csv=None, industry_csv=None,
                  train_ratio=0.8, val_ratio=0.1):
    """
    生成完整的視覺化報告
    
    生成圖表：
    1. daily_ic.png - Daily IC time series
    2. pred_dispersion.png - Prediction cross-sectional std
    3. hitrate_by_month.png - Monthly directional accuracy
    4. cum_returns.png - Cumulative returns comparison
    5. attention_weights.png - Feature attention weights (DMFM only)
    6. ic_distribution.png - IC distribution histogram
    
    輸出檔案結構：
    plots_dmfm/
    ├── daily_ic.png              # Daily IC time series
    ├── pred_dispersion.png       # Prediction cross-sectional std
    ├── hitrate_by_month.png      # Monthly directional accuracy
    ├── ic_distribution.png       # IC distribution histogram
    ├── cum_returns.png           # Cumulative returns comparison
    └── attention_weights.png     # Feature attention weights (DMFM only)
    """
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(device)
    
    print("=" * 60)
    print("視覺化報告生成")
    print("=" * 60)
    print(f"使用裝置: {device}")
    print(f"輸出資料夾: {out_dir}")

    # 載入 artifacts
    Ft, yt, edge_industry, edge_universe, meta = load_artifacts(artifact_dir)
    Ft = Ft.to(device).float()
    yt = yt.to(device)
    edge_industry = edge_industry.to(device)
    edge_universe = edge_universe.to(device)
    dynamic_graphs = load_dynamic_graphs(artifact_dir, map_location="cpu")

    # 回測報酬優先使用原始 forward return，避免和去均值標籤混用
    y_backtest = yt
    raw_label_path = os.path.join(artifact_dir, "yraw_tensor.pt")
    if os.path.exists(raw_label_path):
        try:
            y_backtest = torch.load(raw_label_path, map_location=device).to(device)
        except Exception as e:
            print(f"[warn] 載入 yraw_tensor.pt 失敗，回退 yt: {e}")
            y_backtest = yt
    else:
        print("[warn] 找不到 yraw_tensor.pt，回測將使用去均值 yt（和基準口徑不同）")

    T, N, Fdim = Ft.shape
    horizon_k = int(meta.get("horizon", rebalance_days))
    dates_all = parse_dates_list(meta["dates"])
    _, val_idx, test_idx = time_split_indices_3(
        meta["dates"], train_ratio=train_ratio, val_ratio=val_ratio
    )
    
    stocks = meta.get("stocks", [str(i) for i in range(N)])
    feature_cols = meta.get("feature_cols", [])

    print(f"資料: T={T}, N={N}, F={Fdim}")
    if dynamic_graphs is not None:
        print(f"動態加權圖: 有 (edge_dim={dynamic_graphs.edge_dim})")
    else:
        print("動態加權圖: 無，使用 static binary fallback")
    print(f"驗證期: {len(val_idx)} 天 | 測試期: {len(test_idx)} 天")

    # 偵測並載入模型
    model, checkpoint, missing, unexpected = build_graph_model_from_checkpoint(
        weights,
        map_location=device,
        fallback_in_dim=Fdim,
        fallback_hid=hid,
        fallback_heads=heads,
        fallback_dropout=0.1,
        fallback_tanh_cap=tanh_cap,
    )
    model_type = checkpoint["model_type"]
    model = model.to(device)
    print(f"模型類型: {model_type.upper()}")
    print(
        f"Checkpoint: model_type={checkpoint['model_type']} "
        f"format_version={checkpoint['format_version']} legacy={checkpoint['is_legacy']}"
    )
    if model_type in {"dmfm", "factor_variant"} and (missing or unexpected):
        print(f"[warn] 權重鍵不完全匹配 (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    # 收集測試期數據
    print("\n收集測試期數據...")
    daily_ic = []
    daily_dir = []
    pred_std = []
    test_dates = []
    all_predictions = []
    all_labels = []
    all_attentions = [] if model_type in {"dmfm", "factor_variant"} else None
    
    with torch.no_grad():
        for t in test_idx:
            x = Ft[t]
            y = yt[t]
            mask = torch.isfinite(y)
            
            if mask.sum() == 0:
                daily_ic.append(np.nan)
                daily_dir.append(np.nan)
                pred_std.append(np.nan)
                test_dates.append(dates_all[t])
                continue
            
            x = torch.nan_to_num(x, nan=0.0)

            industry_ei_t, industry_attr_t = graph_at(
                dynamic_graphs, "industry", t, edge_industry, device=device
            )
            universe_ei_t, universe_attr_t = graph_at(
                dynamic_graphs, "universe", t, edge_universe, device=device
            )
            p, attn_weights, contexts = forward_graph_model(
                model,
                x,
                industry_ei_t,
                universe_ei_t,
                industry_attr_t,
                universe_attr_t,
            )
            if attn_weights is not None and all_attentions is not None:
                all_attentions.append(attn_weights[mask].detach().cpu().numpy())

            P = p[mask].detach().cpu().numpy().flatten()  # ← 加 .flatten()
            Y = y[mask].detach().cpu().numpy().flatten()  # ← 加 .flatten()

            all_predictions.append(P)
            all_labels.append(Y)
            
            daily_ic.append(safe_corr(P, Y))
            daily_dir.append(float((np.sign(P) == np.sign(Y)).mean()))
            pred_std.append(float(np.nanstd(P)))
            test_dates.append(dates_all[t])

    ser_ic = pd.Series(daily_ic, index=pd.to_datetime(test_dates))
    ser_dir = pd.Series(daily_dir, index=pd.to_datetime(test_dates))
    ser_psd = pd.Series(pred_std, index=pd.to_datetime(test_dates))

    # ============================================================
    # 圖表 1: Daily IC 時序圖
    # ============================================================
    print("生成圖表 1/6: Daily IC...")
    plt.figure(figsize=(12, 5))
    plt.plot(ser_ic.index, ser_ic.values, linewidth=1.5, alpha=0.7)
    m = np.nanmean(ser_ic.values)
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    if np.isfinite(m):
        plt.axhline(m, color='red', linestyle='--', linewidth=1.5, label=f'Mean IC = {m:.4f}')
    plt.title(f"Daily IC (Test Set) | Mean IC = {m:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("IC", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_ic.png"), dpi=150)
    plt.close()

    # ============================================================
    # 圖表 2: 預測離散度
    # ============================================================
    print("生成圖表 2/6: 預測離散度...")
    plt.figure(figsize=(12, 5))
    plt.plot(ser_psd.index, ser_psd.values, linewidth=1.5, alpha=0.7, color='orange')
    plt.title("Prediction Cross-Sectional Std (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Prediction Std", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_dispersion.png"), dpi=150)
    plt.close()

    # ============================================================
    # 圖表 3: 月度命中率
    # ============================================================
    print("生成圖表 3/6: 月度命中率...")
    hr_month = ser_dir.groupby(pd.Grouper(freq="ME")).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(hr_month.index, hr_month.values, marker='o', linewidth=2, markersize=6)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Random baseline (50%)')
    plt.title("Monthly Directional Accuracy (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Hit Rate", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hitrate_by_month.png"), dpi=150)
    plt.close()

    # ============================================================
    # 圖表 4: IC 分佈直方圖
    # ============================================================
    print("生成圖表 4/6: IC 分佈...")
    plt.figure(figsize=(10, 5))
    ic_values = ser_ic.dropna().values
    plt.hist(ic_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(m, color='red', linestyle='--', linewidth=2, label=f'Mean IC = {m:.4f}')
    plt.axvline(0, color='gray', linestyle='-', linewidth=1)
    plt.title("IC Distribution (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("IC Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ic_distribution.png"), dpi=150)
    plt.close()

    # ============================================================
    # 圖表 5: 累積報酬比較
    # ============================================================
    print("生成圖表 5/6: 累積報酬...")
    step = max(1, int(rebalance_days))
    take_idx = test_idx[::step] if len(test_idx) > 0 else []
    long_rets = []
    long_dates = []
    
    with torch.no_grad():
        for t in take_idx:
            x = Ft[t]
            y = yt[t]
            y_bt = y_backtest[t]
            mask = torch.isfinite(y) & torch.isfinite(y_bt)
            if mask.sum() == 0:
                continue
            
            x = torch.nan_to_num(x, nan=0.0)
            
            industry_ei_t, industry_attr_t = graph_at(
                dynamic_graphs, "industry", t, edge_industry, device=device
            )
            universe_ei_t, universe_attr_t = graph_at(
                dynamic_graphs, "universe", t, edge_universe, device=device
            )
            p, _, _ = forward_graph_model(
                model,
                x,
                industry_ei_t,
                universe_ei_t,
                industry_attr_t,
                universe_attr_t,
            )
            
            # ✅ 修正：統一加 flatten()
            P = p[mask].detach().cpu().numpy().flatten()
            Y = y_bt[mask].detach().cpu().numpy().flatten()
            
            # 取前 top_pct
            qh = np.nanquantile(P, 1.0 - float(top_pct))
            sel = (P >= qh)
            if sel.sum() == 0:
                continue
            
            # ✅ 修正：Y 已經是 1D，直接用 sel 即可
            long_rets.append(np.nanmean(Y[sel]))
            long_dates.append(dates_all[t])

    strat = pd.Series(long_rets, index=pd.to_datetime(long_dates)).sort_index()
    strat_cum = (1.0 + strat.fillna(0)).cumprod() - 1.0

    # 基準比較
    bmk_cum = None
    strat_cmp = None
    bmk_on = None
    if benchmark_csv and os.path.exists(benchmark_csv):
        try:
            bmk = parse_benchmark(benchmark_csv)
            fwd = compound_forward(bmk["ret"], steps=horizon_k)
            bmk_on = fwd.reindex(strat.index)
            mask = strat.notna() & bmk_on.notna()
            if int(mask.sum()) >= 2:
                strat_cmp = strat[mask]
                bmk_on = bmk_on[mask]
                bmk_cum = (1.0 + bmk_on).cumprod() - 1.0
                print(
                    f"基準對齊區間: {strat_cmp.index.min().date()} ~ {strat_cmp.index.max().date()} "
                    f"(n={len(strat_cmp)})"
                )
            else:
                print("[warn] 基準與策略重疊日期不足，略過基準比較")
        except Exception as e:
            print(f"[warn] 無法載入基準: {e}")

    plt.figure(figsize=(12, 6))
    if strat_cmp is not None:
        strat_plot = (1.0 + strat_cmp).cumprod() - 1.0
        plt.plot(
            strat_plot.index,
            strat_plot.values * 100,
            linewidth=2.5,
            label=f"Strategy (aligned, top {top_pct*100:.0f}%)",
            color="#2E86AB",
        )
    else:
        plt.plot(
            strat_cum.index,
            strat_cum.values * 100,
            linewidth=2.5,
            label=f"Strategy (top {top_pct*100:.0f}%)",
            color="#2E86AB",
        )
    if bmk_cum is not None:
        plt.plot(
            bmk_cum.index,
            bmk_cum.values * 100,
            linewidth=2.5,
            label="Benchmark (0050)",
            color="#A23B72",
            linestyle="--",
        )
    plt.title("Cumulative Returns Comparison (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cum_returns.png"), dpi=150)
    plt.close()

    strategy_stats = compute_return_stats(strat.values, step)
    aligned_strategy_stats = compute_return_stats(
        strat_cmp.values if strat_cmp is not None else [], step
    )
    benchmark_stats = (
        compute_return_stats(bmk_on.values, step)
        if bmk_cum is not None and bmk_on is not None
        else None
    )

    print(
        f"\n策略統計: n={strategy_stats['n']} | 年化={strategy_stats['annual_return']:.4f} "
        f"| 夏普={strategy_stats['sharpe']:.3f} | 勝率={strategy_stats['hit_rate']:.3f}"
    )

    if benchmark_stats is not None:
        print(
            f"基準統計: n={benchmark_stats['n']} | 年化={benchmark_stats['annual_return']:.4f} "
            f"| 夏普={benchmark_stats['sharpe']:.3f} | 勝率={benchmark_stats['hit_rate']:.3f}"
        )

    # ============================================================
    # 圖表 6: 注意力權重熱圖（僅 DMFM）
    # ============================================================
    top_feature_summary = []
    attention_plot_path = None
    if model_type in {"dmfm", "factor_variant"} and all_attentions and len(all_attentions) > 0 and len(feature_cols) > 0:
        print("生成圖表 6/6: 注意力權重...")
        
        # 平均所有測試期的注意力權重
        attn_avg = np.concatenate(all_attentions, axis=0).mean(axis=0)  # [F]
        
        # 只顯示前 30 個特徵
        n_show = min(30, len(feature_cols))
        top_indices = np.argsort(attn_avg)[-n_show:][::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(n_show), attn_avg[top_indices], color='steelblue')
        plt.yticks(range(n_show), [feature_cols[i] for i in top_indices])
        plt.xlabel("Average Attention Weight", fontsize=12)
        plt.title("Feature Attention Weights (Top 30)", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        attention_plot_path = os.path.join(out_dir, "attention_weights.png")
        plt.savefig(attention_plot_path, dpi=150)
        plt.close()
        
        print(f"\n注意力權重最高的前 5 個特徵：")
        for i, idx in enumerate(top_indices[:5]):
            print(f"  {i+1}. {feature_cols[idx]}: {attn_avg[idx]:.6f}")
            top_feature_summary.append(
                {
                    "rank": i + 1,
                    "feature": feature_cols[idx],
                    "attention_weight": float(attn_avg[idx]),
                }
            )

    summary_dir = infer_summary_dir(out_dir)
    plot_paths = {
        "daily_ic": os.path.join(out_dir, "daily_ic.png"),
        "pred_dispersion": os.path.join(out_dir, "pred_dispersion.png"),
        "hitrate_by_month": os.path.join(out_dir, "hitrate_by_month.png"),
        "ic_distribution": os.path.join(out_dir, "ic_distribution.png"),
        "cum_returns": os.path.join(out_dir, "cum_returns.png"),
        "attention_weights": attention_plot_path,
    }
    alignment = None
    if strat_cmp is not None and bmk_on is not None and len(strat_cmp) > 0:
        alignment = {
            "start": str(strat_cmp.index.min().date()),
            "end": str(strat_cmp.index.max().date()),
            "n_periods": int(len(strat_cmp)),
        }
    portfolio_payload = {
        "model_type": model_type,
        "artifact_dir": artifact_dir,
        "weights": weights,
        "model_config": checkpoint.get("config", {}),
        "dynamic_graphs": meta.get("dynamic_graphs"),
        "checkpoint_format_version": checkpoint.get("format_version"),
        "checkpoint_is_legacy": checkpoint.get("is_legacy"),
        "plots_dir": out_dir,
        "plots": plot_paths,
        "benchmark_path": benchmark_csv,
        "top_pct": float(top_pct),
        "rebalance_days": int(rebalance_days),
        "horizon": int(horizon_k),
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "val_days": len(val_idx),
            "test_days": len(test_idx),
        },
        "strategy": strategy_stats,
        "strategy_aligned": aligned_strategy_stats if strat_cmp is not None else None,
        "benchmark": benchmark_stats,
        "alignment": alignment,
        "top_attention_features": top_feature_summary,
    }
    if benchmark_stats is not None and strat_cmp is not None:
        compare_stats = aligned_strategy_stats
        portfolio_payload["excess"] = {
            "annual_return": compare_stats["annual_return"] - benchmark_stats["annual_return"],
            "total_return": compare_stats["total_return"] - benchmark_stats["total_return"],
            "sharpe": (
                compare_stats["sharpe"] - benchmark_stats["sharpe"]
                if compare_stats["sharpe"] is not None and benchmark_stats["sharpe"] is not None
                else None
            ),
        }
    save_json(summary_dir / "portfolio.json", portfolio_payload)
    
    print("\n" + "=" * 60)
    print(f"所有圖表已儲存至: {out_dir}")
    print(f"投組摘要已儲存至: {summary_dir / 'portfolio.json'}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="生成視覺化報告")
    
    # 基本參數
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus",
                    help="Artifacts 資料夾路徑")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt",
                    help="模型權重檔路徑")
    ap.add_argument("--out_dir", type=str, default="./plots_dmfm",
                    help="輸出圖表資料夾")
    ap.add_argument("--device", type=str, default="cuda",
                    help="計算裝置: cpu, cuda, mps, 或 auto")
    
    # 模型參數
    ap.add_argument("--tanh_cap", type=float, default=0.2,
                    help="輸出 tanh 限制範圍")
    ap.add_argument("--hid", type=int, default=64,
                    help="隱藏層維度")
    ap.add_argument("--heads", type=int, default=2,
                    help="GAT 注意力頭數")
    
    # 策略參數
    ap.add_argument("--top_pct", type=float, default=0.10,
                    help="選股百分比")
    ap.add_argument("--rebalance_days", type=int, default=5,
                    help="再平衡頻率（天數）")
    
    # 可選檔案
    ap.add_argument("--benchmark_csv", type=str, default=None,
                    help="基準 CSV 檔案（例如 GAT0050.csv）")
    ap.add_argument("--industry_csv", type=str, default=None,
                    help="產業對照表 CSV 檔案")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="train+val 的切分比例（test 從此比例之後開始）")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="在 train 區段中劃給 validation 的比例")
    
    args = ap.parse_args()

    build_reports(
        artifact_dir=args.artifact_dir,
        weights=args.weights,
        out_dir=args.out_dir,
        device=args.device,
        tanh_cap=args.tanh_cap,
        hid=args.hid,
        heads=args.heads,
        top_pct=args.top_pct,
        rebalance_days=args.rebalance_days,
        benchmark_csv=args.benchmark_csv,
        industry_csv=args.industry_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
