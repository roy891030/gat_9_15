"""
生成視覺化報告（支援 DMFM 和 GATRegressor）

執行方式：
python plot_reports.py \
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
matplotlib.use('Agg')  # Use non-interactive backend

from train_gat_fixed import GATRegressor, DMFM, load_artifacts, time_split_indices

# ---------- Device Selection ----------
def pick_device(device_str: str) -> torch.device:
    """Auto-select available computing device"""
    s = (device_str or "").lower()
    if s in ("cpu", "cuda", "mps"):
        if s == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, using CPU")
            return torch.device("cpu")
        if s == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print("[warn] MPS not available, using CPU")
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
    """Safely compute Pearson correlation coefficient"""
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < EPS or sb < EPS:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def parse_dates_list(str_dates):
    """Parse list of date strings"""
    return pd.to_datetime(pd.Series(str_dates), errors="coerce").tolist()


def detect_model_type(weights_path, device="cpu"):
    """Auto-detect model type (DMFM or GATRegressor)"""
    state_dict = torch.load(weights_path, map_location=device)
    dmfm_keys = ["encoder.0.weight", "gat_universe.lin_src.weight", "factor_attn.weight"]
    is_dmfm = any(key in state_dict for key in dmfm_keys)
    return "dmfm" if is_dmfm else "gat"


def parse_benchmark(benchmark_csv):
    """
    Parse benchmark CSV file
    
    Supported columns:
    - Date: ['date', 'Date', '年月日']
    - Price: ['收盤價(元)', 'Close', 'Adj Close', '收盤價']
    - Daily return: ['報酬率％', 'Return', 'ret', 'pct_change']
    
    Returns: DataFrame(index=date, columns=['ret']) with daily returns (decimal)
    """
    df = pd.read_csv(benchmark_csv)
    
    # Date column
    dcol = None
    for c in ["date", "Date", "年月日"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        dcol = df.columns[0]
    
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    # Priority: existing return column
    for rc in ["報酬率％", "Return", "ret", "pct_change", "RET", "ret1"]:
        if rc in df.columns:
            r = pd.to_numeric(df[rc], errors="coerce")
            if rc == "報酬率％":
                r = r / 100.0
            return pd.DataFrame({"ret": r.values}, index=df.index)

    # Otherwise compute from price
    pcol = None
    for c in ["收盤價(元)", "Close", "Adj Close", "收盤價", "close", "adj_close"]:
        if c in df.columns:
            pcol = c
            break
    
    if pcol is None:
        raise ValueError("benchmark_csv: cannot find price or return column")
    
    px = pd.to_numeric(df[pcol], errors="coerce")
    ret = px.pct_change()
    return pd.DataFrame({"ret": ret.values}, index=df.index)


def compound_forward(ret_series: pd.Series, steps: int) -> pd.Series:
    """
    Convert daily returns to forward k-day compound returns
    
    Formula: (1+r_t+1) * ... * (1+r_t+k) - 1
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
    """Load industry labels"""
    if not industry_csv or not os.path.exists(industry_csv):
        return None
    
    try:
        df = pd.read_csv(industry_csv, dtype={"證券代碼": str, "證券代碼_純代碼": str})
    except Exception:
        return None

    # Stock code column
    sid_col = None
    for c in ["證券代碼_純代碼", "證券代碼", "sid", "StockID", "stock_id"]:
        if c in df.columns:
            sid_col = c
            break

    # Industry column
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
    Generate comprehensive visualization reports
    
    Generated plots:
    1. daily_ic.png - Daily IC time series
    2. pred_dispersion.png - Prediction cross-sectional std
    3. hitrate_by_month.png - Monthly directional accuracy
    4. cum_returns.png - Cumulative returns comparison
    5. attention_weights.png - Feature attention weights (DMFM only)
    6. ic_distribution.png - IC distribution histogram
    
    Output structure:
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
    print("Visualization Report Generation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    # Load artifacts
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

    print(f"Data: T={T}, N={N}, F={Fdim}")
    print(f"Test period: {len(test_idx)} days")

    # Detect and load model
    model_type = detect_model_type(weights, device=device)
    print(f"Model type: {model_type.upper()}")
    
    if model_type == "dmfm":
        model = DMFM(
            in_dim=Fdim,
            hid=hid,
            heads=heads,
            tanh_cap=tanh_cap,
            use_factor_attention=True
        ).to(device)
    else:
        model = GATRegressor(
            in_dim=Fdim,
            tanh_cap=tanh_cap
        ).to(device)
    
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # Collect test period data
    print("\nCollecting test period data...")
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
            
            # Forward pass
            if model_type == "dmfm":
                p, _, attn = model(x, edge_industry, edge_universe)
                if attn is not None:
                    all_attentions.append(attn[mask].detach().cpu().numpy())
            else:
                p = model(x, edge_industry)
            
            P = p[mask].detach().cpu().numpy()
            Y = y[mask].detach().cpu().numpy()
            
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
    # Plot 1: Daily IC Time Series
    # ============================================================
    print("Generating plot 1/6: Daily IC...")
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
    # Plot 2: Prediction Dispersion (detect constant predictions)
    # ============================================================
    print("Generating plot 2/6: Prediction dispersion...")
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
    # Plot 3: Monthly Hit Rate
    # ============================================================
    print("Generating plot 3/6: Monthly hit rate...")
    hr_month = ser_dir.groupby(pd.Grouper(freq="M")).mean()
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
    # Plot 4: IC Distribution Histogram
    # ============================================================
    print("Generating plot 4/6: IC distribution...")
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
    # Plot 5: Cumulative Returns Comparison (Strategy vs Benchmark)
    # ============================================================
    print("Generating plot 5/6: Cumulative returns...")
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
            
            P = p[mask].detach().cpu().numpy()
            Y = y[mask].detach().cpu().numpy()
            
            # Select top top_pct
            qh = np.nanquantile(P, 1.0 - float(top_pct))
            sel = (P >= qh)
            if sel.sum() == 0:
                continue
            
            long_rets.append(np.nanmean(Y[sel]))
            long_dates.append(dates_all[t])

    strat = pd.Series(long_rets, index=pd.to_datetime(long_dates)).sort_index()
    strat_cum = (1.0 + strat.fillna(0)).cumprod() - 1.0

    # Benchmark comparison
    bmk_cum = None
    if benchmark_csv and os.path.exists(benchmark_csv):
        try:
            bmk = parse_benchmark(benchmark_csv)
            fwd = compound_forward(bmk["ret"], steps=horizon_k)
            bmk_on = fwd.reindex(strat.index)
            bmk_cum = (1.0 + bmk_on.fillna(0)).cumprod() - 1.0
        except Exception as e:
            print(f"[warn] Cannot load benchmark: {e}")

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

    # Output statistics
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
    print(f"\nStrategy stats: n={s_strat['n']} | ann_ret={s_strat['ann']:.4f} | sharpe={s_strat['sharpe']:.3f} | hitrate={s_strat['hit']:.3f}")
    
    if bmk_cum is not None:
        s_bmk = stats(bmk_on.values)
        print(f"Benchmark stats: n={s_bmk['n']} | ann_ret={s_bmk['ann']:.4f} | sharpe={s_bmk['sharpe']:.3f} | hitrate={s_bmk['hit']:.3f}")

    # ============================================================
    # Plot 6: Attention Weights Heatmap (DMFM only)
    # ============================================================
    if model_type == "dmfm" and all_attentions and len(all_attentions) > 0 and len(feature_cols) > 0:
        print("Generating plot 6/6: Attention weights heatmap...")
        
        # Average attention weights across all test periods
        attn_avg = np.concatenate(all_attentions, axis=0).mean(axis=0)  # [F]
        
        # Show only top 30 features (avoid overcrowding)
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
        
        print(f"\nTop 5 features by attention weight:")
        for i, idx in enumerate(top_indices[:5]):
            print(f"  {i+1}. {feature_cols[idx]}: {attn_avg[idx]:.6f}")
    
    print("\n" + "=" * 60)
    print(f"All plots saved to: {out_dir}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Generate visualization reports")
    
    # Basic parameters
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus",
                    help="Artifacts directory path")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt",
                    help="Model weights file path")
    ap.add_argument("--out_dir", type=str, default="./plots_dmfm",
                    help="Output plots directory")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Computing device: cpu, cuda, mps, or auto")
    
    # Model parameters
    ap.add_argument("--tanh_cap", type=float, default=0.2,
                    help="Output tanh cap range")
    ap.add_argument("--hid", type=int, default=64,
                    help="Hidden layer dimension")
    ap.add_argument("--heads", type=int, default=2,
                    help="GAT attention heads")
    
    # Strategy parameters
    ap.add_argument("--top_pct", type=float, default=0.10,
                    help="Stock selection percentage")
    ap.add_argument("--rebalance_days", type=int, default=5,
                    help="Rebalance frequency (days)")
    
    # Optional files
    ap.add_argument("--benchmark_csv", type=str, default=None,
                    help="Benchmark CSV file (e.g., GAT0050.csv)")
    ap.add_argument("--industry_csv", type=str, default=None,
                    help="Industry mapping CSV file")
    
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