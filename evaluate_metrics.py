"""
python evaluate_metrics.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device auto \
  --industry_csv unique_2019q3to2025q3.csv


"""


# evaluate_metrics.py
import os
import argparse
import numpy as np
import torch
import pandas as pd

from train_gat_fixed import GATRegressor, load_artifacts, time_split_indices

EPS = 1e-8

# ---- device auto 解析（與 train 保持一致）----
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
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_corr(a, b):
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < EPS or sb < EPS:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan

def load_industry_labels(industry_csv, stocks):
    """回傳與 stocks 同長度的產業標籤 list；讀不到就回 None。"""
    path = industry_csv if industry_csv else None
    if not path or not os.path.exists(path):
        return None
    try:
        # 代碼強制成字串，避免 1101 → 1101.0
        df = pd.read_csv(path, dtype={"證券代碼": str, "證券代碼_純代碼": str})
    except Exception:
        return None

    # 股票代碼欄
    sid_col = None
    for c in ["證券代碼_純代碼", "證券代碼", "sid", "StockID", "stock_id"]:
        if c in df.columns:
            sid_col = c
            break

    # 產業欄（把 TEJ 放進來）
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

def industry_residuals(P, Y, inds):
    """對每個產業去均值，回傳殘差序列。"""
    df = pd.DataFrame({"p": P, "y": Y, "g": inds})
    grp = df.groupby("g", sort=False, observed=True)
    pr = df["p"] - grp["p"].transform("mean")
    yr = df["y"] - grp["y"].transform("mean")
    return pr.values, yr.values

@torch.no_grad()
def eval_indices(model, Ft, yt, edge_index, indices, device="cpu", inds=None):
    model.eval()
    preds, truths = [], []
    daily_ic, daily_dir = [], []
    daily_ic_ind = []
    resP_all, resY_all = [], []

    inds_arr = np.array(inds) if inds is not None else None

    for t in indices:
        x = Ft[t].to(device).float()        # 重要：轉成 float32
        y = yt[t].to(device)
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            continue

        x = torch.nan_to_num(x, nan=0.0)
        p = model(x, edge_index.to(device))[mask]
        yy = y[mask]

        P = p.detach().cpu().numpy()
        Y = yy.detach().cpu().numpy()
        preds.append(P)
        truths.append(Y)

        c = safe_corr(P, Y)
        if c == c:
            daily_ic.append(c)
        daily_dir.append(float((np.sign(P) == np.sign(Y)).mean()))

        if inds_arr is not None:
            sel = np.where(mask.cpu().numpy())[0]
            inds_t = inds_arr[sel]
            Pn, Yn = industry_residuals(P, Y, inds_t)
            cn = safe_corr(Pn, Yn)
            if cn == cn:
                daily_ic_ind.append(cn)
            resP_all.append(Pn)
            resY_all.append(Yn)

    out = {
        "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan,
        "IC": np.nan, "DailyIC": np.nan, "DirAcc": np.nan,
        "IC_ind": np.nan, "DailyIC_ind": np.nan, "n": 0
    }

    if preds:
        P_all = np.concatenate(preds)
        Y_all = np.concatenate(truths)
        mse = float(((P_all - Y_all) ** 2).mean())
        rmse = float(np.sqrt(mse))
        mae = float(np.abs(P_all - Y_all).mean())
        ic = safe_corr(P_all, Y_all)
        out.update({
            "MSE": mse, "RMSE": rmse, "MAE": mae,
            "IC": ic if ic == ic else np.nan,
            "DailyIC": float(np.nanmean(daily_ic)) if daily_ic else np.nan,
            "DirAcc": float(np.mean(daily_dir)) if daily_dir else np.nan,
            "n": int(Y_all.size)
        })

        if inds_arr is not None and len(resP_all) > 0:
            Pn_all = np.concatenate(resP_all)
            Yn_all = np.concatenate(resY_all)
            ic_ind = safe_corr(Pn_all, Yn_all)
            out.update({
                "IC_ind": ic_ind if ic_ind == ic_ind else np.nan,
                "DailyIC_ind": float(np.nanmean(daily_ic_ind)) if daily_ic_ind else np.nan
            })

    return out

def eval_naive_zero(yt, indices):
    Ys = []
    for t in indices:
        y = yt[t]
        m = torch.isfinite(y)
        if m.sum():
            Ys.append(y[m].cpu().numpy())
    if not Ys:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0}
    Y = np.concatenate(Ys)
    var = float(np.var(Y))
    return {"MSE": var, "RMSE": float(np.sqrt(var)), "MAE": float(np.mean(np.abs(Y))), "n": int(Y.size)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--tanh_cap", type=float, default=0.2)
    ap.add_argument("--industry_csv", type=str, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print("Using device:", device)

    Ft, yt, ei, meta = load_artifacts(args.artifact_dir)
    Ft = Ft.to(device).float()    # 與訓練一致：float32
    yt = yt.to(device)
    ei = ei.to(device)
    T, N, Fdim = Ft.shape
    train_idx, test_idx = time_split_indices(meta["dates"], 0.8)

    model = GATRegressor(in_dim=Fdim, tanh_cap=args.tanh_cap).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    stocks = meta.get("stocks", [str(i) for i in range(N)])
    inds = load_industry_labels(args.industry_csv, stocks)

    print(f"Data: T={T}, N={N}, F={Fdim} | tanh_cap={args.tanh_cap} | "
          f"industry_labels={'OK' if inds is not None else 'NONE'}")

    tr = eval_indices(model, Ft, yt, ei, train_idx, device=device, inds=inds)
    te = eval_indices(model, Ft, yt, ei, test_idx, device=device, inds=inds)
    bz = eval_naive_zero(yt, test_idx)

    def fmt(d, keys):
        return " | ".join(f"{k}={d[k]:.6f}" for k in keys if d.get(k) == d.get(k))

    print("\nTrain:", fmt(tr, ["MSE", "RMSE", "MAE"]),
          f"| IC={tr['IC']:.4f} | DailyIC={tr['DailyIC']:.4f}"
          + (f" | IC_ind={tr['IC_ind']:.4f} | DailyIC_ind={tr['DailyIC_ind']:.4f}" if inds is not None else "")
          + f" | DirAcc={tr['DirAcc']:.4f} | n={tr['n']}")
    print("Test :", fmt(te, ["MSE", "RMSE", "MAE"]),
          f"| IC={te['IC']:.4f} | DailyIC={te['DailyIC']:.4f}"
          + (f" | IC_ind={te['IC_ind']:.4f} | DailyIC_ind={te['DailyIC_ind']:.4f}" if inds is not None else "")
          + f" | DirAcc={te['DirAcc']:.4f} | n={te['n']}")
    print("Naive:", fmt(bz, ["MSE", "RMSE", "MAE"]), f"| n={bz['n']}")

    if np.isfinite(bz["MSE"]) and np.isfinite(te["MSE"]):
        impr = 100.0 * (1.0 - te["MSE"] / bz["MSE"])
        print(f"\n相對天真基準的 MSE 改善：{impr:.2f}%")

if __name__ == "__main__":
    main()
