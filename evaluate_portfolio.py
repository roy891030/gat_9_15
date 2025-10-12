"""
不帶產業中立版確認結果，之後再加選項比較：
# 全市場、每 5 日、前 10% 做多
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device auto \
  --top_pct 0.10 --rebalance_days 5

  
產業中立的版本（各產業各自取前 10% 合併），而且先把預測做產業內 z-score：
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device auto \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv unique_2019q3to2025q3.csv \
  --industry_neutral --neutralize_pred_by_industry

"""


# evaluate_portfolio.py
import os
import argparse
import numpy as np
import torch
import pandas as pd

from train_gat_fixed import GATRegressor, load_artifacts, time_split_indices

# -------- device 解析（與 train/evaluate_metrics 一致）--------
def pick_device(device_str: str) -> torch.device:
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

# -------- 產業對照：支援 TEJ 欄位；讀不到就用 UNK --------
def load_industry_map(industry_csv, stocks):
    if not industry_csv or not os.path.exists(industry_csv):
        return {s: "UNK" for s in stocks}

    df = pd.read_csv(industry_csv, dtype={"證券代碼": str, "證券代碼_純代碼": str})
    sid_col = None
    for c in ["證券代碼_純代碼", "證券代碼", "sid", "StockID", "stock_id"]:
        if c in df.columns:
            sid_col = c; break

    ind_col = None
    for c in ["TEJ產業_名稱", "TEJ產業_代碼", "TSE產業_名稱", "industry_TSE", "Industry", "industry", "產業"]:
        if c in df.columns:
            ind_col = c; break

    if sid_col is None or ind_col is None:
        return {s: "UNK" for s in stocks}

    df = df[[sid_col, ind_col]].dropna().drop_duplicates()
    df[sid_col] = df[sid_col].astype(str)
    m = dict(zip(df[sid_col], df[ind_col]))
    return {s: m.get(s, "UNK") for s in stocks}

@torch.no_grad()
def daily_scores(model, Ft_t, ei, y_t):
    x = torch.nan_to_num(Ft_t, nan=0.0).float()  # 保守：轉 float32
    mask = torch.isfinite(y_t)
    p = model(x, ei)
    return p, y_t, mask

def portfolio_stats(series, rebal_days=5):
    arr = np.asarray(series, dtype=np.float64)
    freq = 252.0 / float(rebal_days)
    mean = np.nanmean(arr) if arr.size else np.nan
    std  = np.nanstd(arr, ddof=1) if arr.size > 1 else np.nan
    sharpe = (mean / std) * np.sqrt(freq) if (np.isfinite(mean) and np.isfinite(std) and std > 0) else np.nan
    ann_ret = mean * freq if np.isfinite(mean) else np.nan
    hit = np.mean(arr > 0) if arr.size else np.nan
    return dict(mean=mean, std=std, ann_ret=ann_ret, sharpe=sharpe, hitrate=hit, n=int(arr.size))

def zscore_in_groups(values, groups):
    # 對每個產業分組做 z-score；缺值/單一樣本組直接設 0
    out = np.zeros_like(values, dtype=np.float64)
    ug = np.unique(groups)
    for g in ug:
        idx = (groups == g)
        v = values[idx]
        if v.size < 2:
            out[idx] = 0.0
            continue
        m = np.nanmean(v)
        s = np.nanstd(v, ddof=1)
        if not np.isfinite(s) or s == 0:
            out[idx] = 0.0
        else:
            out[idx] = (v - m) / s
    return out

def run_eval(artifact_dir, weights, device="cpu", tanh_cap=0.2,
             top_pct=0.10, long_short=False, rebalance_days=5,
             industry_csv=None, industry_neutral=False,
             neutralize_pred_by_industry=False):
    device = pick_device(device)
    print("Using device:", device)

    Ft, yt, ei, meta = load_artifacts(artifact_dir)
    Ft = Ft.to(device).float()   # 與訓練一致
    yt = yt.to(device)
    ei = ei.to(device)

    T, N, Fdim = Ft.shape
    stocks = meta.get("stocks", [str(i) for i in range(N)])
    _, test_idx_full = time_split_indices(meta["dates"], 0.8)

    step = max(1, int(rebalance_days))
    test_idx = list(range(test_idx_full[0], test_idx_full[-1] + 1, step))

    model = GATRegressor(in_dim=Fdim, tanh_cap=tanh_cap).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    ind_map = load_industry_map(industry_csv, stocks) if (industry_csv or industry_neutral or neutralize_pred_by_industry) else None
    inds = np.array([ind_map.get(s, "UNK") for s in stocks]) if ind_map is not None else None

    long_rets, spread_rets = [], []
    q_hi_fn = lambda P: np.nanquantile(P, 1.0 - float(top_pct))
    q_lo_fn = lambda P: np.nanquantile(P, float(top_pct))

    for t in test_idx:
        p, y, mask = daily_scores(model, Ft[t], ei, yt[t])
        if mask.sum() == 0:
            continue
        P = p[mask].detach().cpu().numpy()
        Y = y[mask].detach().cpu().numpy()
        idxs = np.where(mask.cpu().numpy())[0]

        if neutralize_pred_by_industry and inds is not None:
            P = zscore_in_groups(P, inds[idxs])  # 以產業內 z-score 的預測排序

        if not industry_neutral or inds is None:
            # 全市場選股
            try:
                qh = q_hi_fn(P)
                long_sel = (P >= qh)
                if long_sel.sum() > 0:
                    long_rets.append(Y[long_sel].mean())
                if long_short:
                    ql = q_lo_fn(P)
                    short_sel = (P <= ql)
                    if short_sel.sum() > 0:
                        spread_rets.append(Y[long_sel].mean() - Y[short_sel].mean())
            except Exception:
                continue
        else:
            # 產業中立：各產業各自取前 top_pct
            these_inds = inds[idxs]
            chosen_long, chosen_short = [], []
            for g in np.unique(these_inds):
                ig = (these_inds == g)
                if ig.sum() < 5:
                    continue
                Pg, Yg = P[ig], Y[ig]
                try:
                    qh = q_hi_fn(Pg)
                    lg = (Pg >= qh)
                    if lg.sum() > 0:
                        chosen_long.append(Yg[lg])
                    if long_short:
                        ql = q_lo_fn(Pg)
                        sg = (Pg <= ql)
                        if sg.sum() > 0:
                            chosen_short.append(Yg[sg])
                except Exception:
                    continue
            if chosen_long:
                long_rets.append(np.concatenate(chosen_long).mean())
            if long_short and chosen_long and chosen_short:
                spread_rets.append(np.concatenate(chosen_long).mean() - np.concatenate(chosen_short).mean())

    out = {"long_only": portfolio_stats(long_rets, rebal_days=rebalance_days)}
    if long_short and len(spread_rets) > 0:
        out["long_short"] = portfolio_stats(spread_rets, rebal_days=rebalance_days)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tanh_cap", type=float, default=0.2)
    ap.add_argument("--top_pct", type=float, default=0.10)         # 前 10%
    ap.add_argument("--long_short", action="store_true")
    ap.add_argument("--rebalance_days", type=int, default=5)       # 每 5 個交易日再平衡
    ap.add_argument("--industry_csv", type=str, default=None)      # 可直接給 unique_2019q3to2025q3.csv
    ap.add_argument("--industry_neutral", action="store_true")     # 產業中立選股
    ap.add_argument("--neutralize_pred_by_industry", action="store_true")  # 先做預測的產業內 z-score
    args = ap.parse_args()

    res = run_eval(args.artifact_dir, args.weights, device=args.device,
                   tanh_cap=args.tanh_cap, top_pct=args.top_pct,
                   long_short=args.long_short, rebalance_days=args.rebalance_days,
                   industry_csv=args.industry_csv, industry_neutral=args.industry_neutral,
                   neutralize_pred_by_industry=args.neutralize_pred_by_industry)

    def fmt(d):
        return (f"mean={d['mean']:.6f} | std={d['std']:.6f} | "
                f"ann_ret={d['ann_ret']:.4f} | sharpe={d['sharpe']:.3f} | "
                f"hit={d['hitrate']:.3f} | n={d['n']}")

    print("Long-only (top decile):", fmt(res["long_only"]))
    if "long_short" in res:
        print("Long-Short (top-bottom):", fmt(res["long_short"]))

if __name__ == "__main__":
    main()
