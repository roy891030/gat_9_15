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

from model_dmfm_wei2022 import DMFM_Wei2022 as DMFM, GATRegressor
from train_gat_fixed import load_artifacts, time_split_indices
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
EPS = 1e-8

def safe_corr(a, b):
    """安全計算 Pearson 相關係數"""
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < EPS or sb < EPS:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def parse_dates_list(str_dates):
    """解析日期字串列表"""
    return pd.to_datetime(pd.Series(str_dates), errors="coerce").tolist()


def detect_model_type(weights_path, device="cpu"):
    """自動偵測模型類型"""
    state_dict = torch.load(weights_path, map_location=device)
    dmfm_keys = ["encoder.0.weight", "gat_universe.lin_src.weight", "factor_attn.weight"]
    is_dmfm = any(key in state_dict for key in dmfm_keys)
    return "dmfm" if is_dmfm else "gat"


def parse_benchmark(benchmark_csv):
    """
    解析基準 CSV 檔案
    
    支援欄位：
    - 日期：['date', 'Date', '年月日']
    - 價格：['收盤價(元)', 'Close', 'Adj Close', '收盤價']
    - 日報酬：['報酬率％', 'Return', 'ret', 'pct_change']
    
    回傳：DataFrame(index=日期, columns=['ret']) 的日報酬率（小數）
    """
    df = pd.read_csv(benchmark_csv)
    
    # 日期欄
    dcol = None
    for c in ["date", "Date", "年月日"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        dcol = df.columns[0]
    
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    # 優先用現成報酬欄
    for rc in ["報酬率％", "Return", "ret", "pct_change", "RET", "ret1"]:
        if rc in df.columns:
            r = pd.to_numeric(df[rc], errors="coerce")
            if rc == "報酬率％":
                r = r / 100.0
            return pd.DataFrame({"ret": r.values}, index=df.index)

    # 否則用收盤價自己算
    pcol = None
    for c in ["收盤價(元)", "Close", "Adj Close", "收盤價", "close", "adj_close"]:
        if c in df.columns:
            pcol = c
            break
    
    if pcol is None:
        raise ValueError("benchmark_csv 找不到收盤或報酬欄位")
    
    px = pd.to_numeric(df[pcol], errors="coerce")
    ret = px.pct_change()
    return pd.DataFrame({"ret": ret.values}, index=df.index)


def compound_forward(ret_series: pd.Series, steps: int) -> pd.Series:
    """
    將日報酬轉成未來 k 日複利報酬
    
    公式：(1+r_t+1) * ... * (1+r_t+k) - 1
    """
    if steps <= 1:
        return ret_series.shift(-1)
    
    arr = (1.0 + ret_series.values)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    
    for i in range(0, len(arr) - steps):
        window = arr[i+1:i+1+steps]
        if np.any(~np.isfinite(window)):
            out[i] = np.nan
        else:
            out[i] = np.prod(window) - 1.0
    
    return pd.Series(out, index=ret_series.index)


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
                  benchmark_csv=None, industry_csv=None):
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

    T, N, Fdim = Ft.shape
    horizon_k = int(meta.get("horizon", rebalance_days))
    dates_all = parse_dates_list(meta["dates"])
    _, test_idx = time_split_indices(meta["dates"], 0.8)
    
    stocks = meta.get("stocks", [str(i) for i in range(N)])
    feature_cols = meta.get("feature_cols", [])

    print(f"資料: T={T}, N={N}, F={Fdim}")
    print(f"測試期: {len(test_idx)} 天")

    # 偵測並載入模型
    model_type = detect_model_type(weights, device=device)
    print(f"模型類型: {model_type.upper()}")
    
    if model_type == "dmfm":
        model = DMFM(
            in_dim=Fdim,
            hidden_dim=hid,  # ← 正確
            heads=heads,
            dropout=0.1,
            use_factor_attention=True
        ).to(device)

    else:
        model = GATRegressor(
            in_dim=Fdim,
            tanh_cap=tanh_cap
        ).to(device)
    
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # 收集測試期數據
    print("\n收集測試期數據...")
    daily_ic = []
    daily_dir = []
    pred_std = []
    test_dates = []
    all_predictions = []
    all_labels = []
    all_attentions = [] if model_type == "dmfm" else None
    
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
            
            # 前向傳播
            if model_type == "dmfm":
                p, attn_weights, contexts = model(x, edge_industry, edge_universe)
                # attn_weights 是 [N, F] 的張量
                if attn_weights is not None and all_attentions is not None:
                    all_attentions.append(attn_weights[mask].detach().cpu().numpy())
            else:
                p = model(x, edge_industry)
            
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
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            
            x = torch.nan_to_num(x, nan=0.0)
            
            if model_type == "dmfm":
                p, _, _ = model(x, edge_industry, edge_universe)
            else:
                p = model(x, edge_industry)
            
            # ✅ 修正：統一加 flatten()
            P = p[mask].detach().cpu().numpy().flatten()
            Y = y[mask].detach().cpu().numpy().flatten()
            
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
    if benchmark_csv and os.path.exists(benchmark_csv):
        try:
            bmk = parse_benchmark(benchmark_csv)
            fwd = compound_forward(bmk["ret"], steps=horizon_k)
            bmk_on = fwd.reindex(strat.index)
            bmk_cum = (1.0 + bmk_on.fillna(0)).cumprod() - 1.0
        except Exception as e:
            print(f"[warn] 無法載入基準: {e}")

    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum.index, strat_cum.values * 100, 
             linewidth=2.5, label=f"Strategy (top {top_pct*100:.0f}%)", color='#2E86AB')
    if bmk_cum is not None:
        plt.plot(bmk_cum.index, bmk_cum.values * 100, 
                 linewidth=2.5, label="Benchmark (0050)", color='#A23B72', linestyle='--')
    plt.title("Cumulative Returns Comparison (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cum_returns.png"), dpi=150)
    plt.close()

    # 輸出統計
    def stats(x):
        x = pd.Series(x).dropna()
        if x.empty:
            return dict(n=0, mean=np.nan, std=np.nan, ann=np.nan, sharpe=np.nan, hit=np.nan)
        freq = 252.0 / float(step)
        mu = x.mean()
        sd = x.std(ddof=1)
        sharpe = (mu / sd * np.sqrt(freq)) if (np.isfinite(sd) and sd > 0) else np.nan
        return dict(n=len(x), mean=mu, std=sd, ann=mu*freq, sharpe=sharpe, hit=(x>0).mean())
    
    s_strat = stats(strat.values)
    print(f"\n策略統計: n={s_strat['n']} | 年化={s_strat['ann']:.4f} | 夏普={s_strat['sharpe']:.3f} | 勝率={s_strat['hit']:.3f}")
    
    if bmk_cum is not None:
        s_bmk = stats(bmk_on.values)
        print(f"基準統計: n={s_bmk['n']} | 年化={s_bmk['ann']:.4f} | 夏普={s_bmk['sharpe']:.3f} | 勝率={s_bmk['hit']:.3f}")

    # ============================================================
    # 圖表 6: 注意力權重熱圖（僅 DMFM）
    # ============================================================
    if model_type == "dmfm" and all_attentions and len(all_attentions) > 0 and len(feature_cols) > 0:
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
        plt.savefig(os.path.join(out_dir, "attention_weights.png"), dpi=150)
        plt.close()
        
        print(f"\n注意力權重最高的前 5 個特徵：")
        for i, idx in enumerate(top_indices[:5]):
            print(f"  {i+1}. {feature_cols[idx]}: {attn_avg[idx]:.6f}")
    
    print("\n" + "=" * 60)
    print(f"所有圖表已儲存至: {out_dir}")
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
        industry_csv=args.industry_csv
    )


if __name__ == "__main__":
    main()