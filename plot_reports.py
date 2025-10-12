"""
python plot_reports.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device auto \
  --benchmark_csv GAT0050.csv \
  --out_dir plots_bh \
  --top_pct 0.10 --rebalance_days 5

"""

# plot_reports.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_gat_fixed import GATRegressor, load_artifacts, time_split_indices

# ---------- device 選擇（與其它腳本一致） ----------
def pick_device(device_str: str) -> torch.device:
    s = (device_str or "").lower()
    if s in ("cpu", "cuda", "mps"):
        if s == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA 不可用，改用 CPU"); return torch.device("cpu")
        if s == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print("[warn] MPS 不可用，改用 CPU"); return torch.device("cpu")
        return torch.device(s)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ---------- 小工具 ----------
EPS = 1e-8

def safe_corr(a, b):
    if a.size < 3: return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < EPS or sb < EPS: return np.nan
    c = np.corrcoef(a, b)[0,1]
    return float(c) if np.isfinite(c) else np.nan

def parse_dates_list(str_dates):
    # meta["dates"] 是字串列表
    return pd.to_datetime(pd.Series(str_dates), errors="coerce").tolist()

def parse_benchmark(benchmark_csv):
    """
    嘗試多種常見欄位：
    - 日期：['date','Date','年月日']
    - 價格：['收盤價(元)','Close','Adj Close','收盤價']
    - 日報酬：['報酬率％','Return','ret','pct_change']
    回傳：DataFrame(index=日期, columns=['ret']) 的**日報酬率(小數)**。
    """
    df = pd.read_csv(benchmark_csv)
    # 日期欄
    dcol = None
    for c in ["date","Date","年月日"]:
        if c in df.columns: dcol = c; break
    if dcol is None:
        # 最後嘗試第一欄
        dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    # 優先用現成報酬欄
    for rc in ["報酬率％","Return","ret","pct_change","RET","ret1"]:
        if rc in df.columns:
            r = pd.to_numeric(df[rc], errors="coerce")
            if rc == "報酬率％": r = r / 100.0
            return pd.DataFrame({"ret": r.values}, index=df.index)

    # 否則用收盤價自己算
    pcol = None
    for c in ["收盤價(元)","Close","Adj Close","收盤價","close","adj_close"]:
        if c in df.columns: pcol = c; break
    if pcol is None:
        raise ValueError("benchmark_csv 找不到收盤或報酬欄位")
    px = pd.to_numeric(df[pcol], errors="coerce")
    ret = px.pct_change()
    return pd.DataFrame({"ret": ret.values}, index=df.index)

def compound_forward(ret_series: pd.Series, steps: int) -> pd.Series:
    """
    把「日報酬」轉成「未來 k 日複利報酬」： (1+r_t+1)...(1+r_t+k) - 1
    回傳對齊 t 的 forward-k 報酬（最後 k 天會是 NaN）。
    """
    if steps <= 1: return ret_series.shift(-1)  # 退一步：k=1
    arr = (1.0 + ret_series.values)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(0, len(arr) - steps):
        window = arr[i+1:i+1+steps]  # 未來 k 日
        if np.any(~np.isfinite(window)): 
            out[i] = np.nan
        else:
            out[i] = np.prod(window) - 1.0
    return pd.Series(out, index=ret_series.index)

# ---------- 主流程 ----------
def build_reports(artifact_dir, weights, out_dir,
                  device="auto", tanh_cap=0.2,
                  top_pct=0.10, rebalance_days=5,
                  benchmark_csv=None):
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(device)
    print("Using device:", device)

    # 讀取 artifacts 與模型
    Ft, yt, ei, meta = load_artifacts(artifact_dir)
    Ft = Ft.to(device).float()
    yt = yt.to(device)
    ei = ei.to(device)

    T, N, Fdim = Ft.shape
    horizon_k = int(meta.get("horizon", rebalance_days))
    dates_all = parse_dates_list(meta["dates"])
    _, test_idx = time_split_indices(meta["dates"], 0.8)

    # 測試期的逐日預測、IC、命中率、預測離散度
    model = GATRegressor(in_dim=Fdim, tanh_cap=tanh_cap).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    daily_ic = []
    daily_dir = []
    pred_std = []
    test_dates = []
    with torch.no_grad():
        for t in test_idx:
            x = Ft[t]; y = yt[t]
            mask = torch.isfinite(y)
            if mask.sum()==0: 
                daily_ic.append(np.nan); daily_dir.append(np.nan); pred_std.append(np.nan); test_dates.append(dates_all[t]); continue
            p = model(torch.nan_to_num(x, nan=0.0), ei)
            P = p[mask].detach().cpu().numpy()
            Y = y[mask].detach().cpu().numpy()
            daily_ic.append(safe_corr(P, Y))
            daily_dir.append(float((np.sign(P) == np.sign(Y)).mean()))
            pred_std.append(float(np.nanstd(P)))
            test_dates.append(dates_all[t])

    ser_ic  = pd.Series(daily_ic, index=pd.to_datetime(test_dates))
    ser_dir = pd.Series(daily_dir, index=pd.to_datetime(test_dates))
    ser_psd = pd.Series(pred_std, index=pd.to_datetime(test_dates))

    # ---- 產圖 1：Daily IC ----
    plt.figure(figsize=(10,4))
    plt.plot(ser_ic.index, ser_ic.values)
    m = np.nanmean(ser_ic.values)
    plt.axhline(0, lw=1)
    if np.isfinite(m): plt.axhline(m, ls="--", lw=1)
    ttl = f"Daily IC (test) | mean={m:.4f}" if np.isfinite(m) else "Daily IC (test)"
    plt.title(ttl)
    plt.xlabel("Date"); plt.ylabel("IC")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_ic.png"), dpi=150)
    plt.close()

    # ---- 產圖 2：預測離散度（判斷是否「常數預測」） ----
    plt.figure(figsize=(10,3.6))
    plt.plot(ser_psd.index, ser_psd.values)
    plt.title("Prediction cross-sectional std (test)")
    plt.xlabel("Date"); plt.ylabel("std of predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_dispersion.png"), dpi=150)
    plt.close()

    # ---- 產圖 3：月度方向命中率 ----
    hr_month = ser_dir.groupby(pd.Grouper(freq="M")).mean()
    plt.figure(figsize=(8,3.6))
    plt.plot(hr_month.index, hr_month.values)
    plt.axhline(0.5, ls="--", lw=1)
    plt.title("Monthly Directional Accuracy (test)")
    plt.xlabel("Month"); plt.ylabel("Hit ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hitrate_by_month.png"), dpi=150)
    plt.close()

    # ---- 產圖 4：累積報酬（策略 vs 0050）----
    # 先做策略（每 rebalance_days 再平衡，取前 top_pct）
    step = max(1, int(rebalance_days))
    take_idx = test_idx[::step] if len(test_idx)>0 else []
    long_rets = []
    long_dates = []
    with torch.no_grad():
        for t in take_idx:
            x = Ft[t]; y = yt[t]
            mask = torch.isfinite(y)
            if mask.sum()==0: 
                continue
            p = model(torch.nan_to_num(x, nan=0.0), ei)
            P = p[mask].detach().cpu().numpy()
            Y = y[mask].detach().cpu().numpy()
            # 取前 top_pct
            qh = np.nanquantile(P, 1.0 - float(top_pct))
            sel = (P >= qh)
            if sel.sum()==0: 
                continue
            long_rets.append(np.nanmean(Y[sel]))
            long_dates.append(dates_all[t])

    strat = pd.Series(long_rets, index=pd.to_datetime(long_dates)).sort_index()
    strat_cum = (1.0 + strat.fillna(0)).cumprod() - 1.0

    # 0050 基準（可選）
    if benchmark_csv and os.path.exists(benchmark_csv):
        bmk = parse_benchmark(benchmark_csv)
        # 將 0050 日報酬轉成「未來 horizon_k 日」報酬，並對齊策略的起始日
        fwd = compound_forward(bmk["ret"], steps=horizon_k)
        # 用策略的起始日期去取 0050 的 forward-k 報酬；取不到就 NaN
        bmk_on = fwd.reindex(strat.index)
        bmk_cum = (1.0 + bmk_on.fillna(0)).cumprod() - 1.0
    else:
        bmk_on = None; bmk_cum = None

    plt.figure(figsize=(10,4))
    plt.plot(strat_cum.index, strat_cum.values, label="Strategy (long top {:.0f}%)".format(top_pct*100))
    if bmk_cum is not None:
        plt.plot(bmk_cum.index, bmk_cum.values, label="Benchmark (0050, k-day)")
    plt.title("Cumulative Return (test)")
    plt.xlabel("Date"); plt.ylabel("Cumulative return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cum_returns.png"), dpi=150)
    plt.close()

    # 簡表
    def stats(x):
        x = pd.Series(x).dropna()
        if x.empty: 
            return dict(n=0, mean=np.nan, std=np.nan, ann=np.nan, sharpe=np.nan, hit=np.nan)
        freq = 252.0/float(step)
        mu = x.mean(); sd = x.std(ddof=1)
        return dict(n=len(x), mean=mu, std=sd, ann=mu*freq, sharpe=(mu/sd*np.sqrt(freq)) if (np.isfinite(sd) and sd>0) else np.nan, hit=(x>0).mean())
    s_strat = stats(strat.values)
    print("[Strategy] n={n} mean={mean:.6f} std={std:.6f} ann={ann:.4f} sharpe={sharpe:.3f} hit={hit:.3f}".format(**s_strat))
    if bmk_on is not None:
        s_bmk = stats(bmk_on.values)
        print("[Benchmark] n={n} mean={mean:.6f} std={std:.6f} ann={ann:.4f} sharpe={sharpe:.3f} hit={hit:.3f}".format(**s_bmk))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt")
    ap.add_argument("--out_dir", type=str, default="./plots_bh")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tanh_cap", type=float, default=0.2)
    ap.add_argument("--top_pct", type=float, default=0.10)
    ap.add_argument("--rebalance_days", type=int, default=5)
    ap.add_argument("--benchmark_csv", type=str, default="GAT0050.csv")
    args = ap.parse_args()

    build_reports(
        artifact_dir=args.artifact_dir,
        weights=args.weights,
        out_dir=args.out_dir,
        device=args.device,
        tanh_cap=args.tanh_cap,
        top_pct=args.top_pct,
        rebalance_days=args.rebalance_days,
        benchmark_csv=args.benchmark_csv
    )

if __name__ == "__main__":
    main()
