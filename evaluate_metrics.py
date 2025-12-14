# -*- coding: utf-8 -*-
# evaluate_metrics.py
"""
評估 GAT/DMFM 模型的預測指標

執行方式：
python evaluate_metrics.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device cuda \
  --industry_csv unique_2019q3to2025q3.csv

支援模型：
- GATRegressor（簡化版）
- DMFM（完整版，自動偵測）
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd

from model_dmfm_wei2022 import DMFM_Wei2022 as DMFM, GATRegressor
from train_gat_fixed import load_artifacts, time_split_indices

EPS = 1e-8

# -------- Device Selection --------
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
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_corr(a, b):
    """安全計算 Pearson 相關係數"""
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < EPS or sb < EPS:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def load_industry_labels(industry_csv, stocks):
    """
    載入股票產業對應關係
    
    回傳：
        list: 與 stocks 同長度的產業標籤列表，讀不到則回 None
    """
    path = industry_csv if industry_csv else None
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype={"證券代碼": str, "證券代碼_純代碼": str})
    except Exception:
        return None

    sid_col = None
    for c in ["證券代碼_純代碼", "證券代碼", "sid", "StockID", "stock_id"]:
        if c in df.columns:
            sid_col = c
            break

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
    """
    對每個產業去均值（產業中性化）
    """
    df = pd.DataFrame({"p": P, "y": Y, "g": inds})
    grp = df.groupby("g", sort=False, observed=True)
    pr = df["p"] - grp["p"].transform("mean")
    yr = df["y"] - grp["y"].transform("mean")
    return pr.values, yr.values


def detect_model_type(weights_path, in_dim, device="cpu"):
    """自動偵測模型類型"""
    state_dict = torch.load(weights_path, map_location=device)
    dmfm_keys = ["encoder.0.weight", "gat_universe.lin_src.weight", "factor_attn.weight"]
    is_dmfm = any(key in state_dict for key in dmfm_keys)
    if is_dmfm:
        print("偵測到 DMFM 模型")
        return "dmfm"
    else:
        print("偵測到 GATRegressor 模型")
        return "gat"


@torch.no_grad()
def eval_indices(model, Ft, yt, edge_industry, edge_universe, indices, 
                 device="cpu", inds=None, return_predictions=False):
    """
    評估模型在指定索引上的表現
    """
    model.eval()
    preds, truths = [], []
    daily_ic, daily_dir = [], []
    daily_ic_ind = []
    resP_all, resY_all = [], []
    all_attentions = []  # 儲存注意力權重（若 DMFM）

    inds_arr = np.array(inds) if inds is not None else None

    for t in indices:
        x = Ft[t].to(device).float()
        y = yt[t].to(device)
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            continue

        x = torch.nan_to_num(x, nan=0.0)
        
        if isinstance(model, DMFM):
            p, attn_weights, contexts = model(x, edge_industry.to(device), edge_universe.to(device))
            if attn_weights is not None:
                all_attentions.append(attn_weights[mask].detach().cpu().numpy())
        else:
            p = model(x, edge_industry.to(device))
        
        p = p[mask]
        yy = y[mask]

        # === 關鍵修正：強制壓成 1D array，避免 np.corrcoef 維度不匹配 ===
        P = p.detach().cpu().numpy().flatten()
        Y = yy.detach().cpu().numpy().flatten()

        preds.append(P)
        truths.append(Y)

        # 每日 IC
        c = safe_corr(P, Y)
        if c == c:
            daily_ic.append(c)
        
        # 方向準確率
        daily_dir.append(float((np.sign(P) == np.sign(Y)).mean()))

        # 產業中性 IC
        if inds_arr is not None:
            sel = np.where(mask.cpu().numpy())[0]
            inds_t = inds_arr[sel]
            Pn, Yn = industry_residuals(P, Y, inds_t)
            cn = safe_corr(Pn, Yn)
            if cn == cn:
                daily_ic_ind.append(cn)
            resP_all.append(Pn)
            resY_all.append(Yn)

    # 整理輸出指標
    out = {
        "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan,
        "IC": np.nan, "DailyIC": np.nan, "ICIR": np.nan,
        "DirAcc": np.nan,
        "IC_ind": np.nan, "DailyIC_ind": np.nan, "ICIR_ind": np.nan,
        "n": 0
    }

    if preds:
        P_all = np.concatenate(preds)
        Y_all = np.concatenate(truths)
        
        mse = float(((P_all - Y_all) ** 2).mean())
        rmse = float(np.sqrt(mse))
        mae = float(np.abs(P_all - Y_all).mean())
        
        ic = safe_corr(P_all, Y_all)
        
        icir = np.nan
        if daily_ic and len(daily_ic) > 1:
            ic_mean = np.mean(daily_ic)
            ic_std = np.std(daily_ic, ddof=1)
            if ic_std > EPS:
                icir = float(ic_mean / ic_std)
        
        out.update({
            "MSE": mse, "RMSE": rmse, "MAE": mae,
            "IC": ic if np.isfinite(ic) else np.nan,
            "DailyIC": float(np.nanmean(daily_ic)) if daily_ic else np.nan,
            "ICIR": icir,
            "DirAcc": float(np.mean(daily_dir)) if daily_dir else np.nan,
            "n": int(Y_all.size)
        })

        if inds_arr is not None and len(resP_all) > 0:
            Pn_all = np.concatenate(resP_all)
            Yn_all = np.concatenate(resY_all)
            ic_ind = safe_corr(Pn_all, Yn_all)
            
            icir_ind = np.nan
            if daily_ic_ind and len(daily_ic_ind) > 1:
                ic_ind_mean = np.mean(daily_ic_ind)
                ic_ind_std = np.std(daily_ic_ind, ddof=1)
                if ic_ind_std > EPS:
                    icir_ind = float(ic_ind_mean / ic_ind_std)
            
            out.update({
                "IC_ind": ic_ind if np.isfinite(ic_ind) else np.nan,
                "DailyIC_ind": float(np.nanmean(daily_ic_ind)) if daily_ic_ind else np.nan,
                "ICIR_ind": icir_ind
            })
    
    if return_predictions:
        out["predictions"] = preds
        out["truths"] = truths
        out["daily_ic_series"] = daily_ic
        if all_attentions:
            out["attention_weights"] = all_attentions
    
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


def print_metrics(name, metrics, has_industry=False):
    print(f"\n{'='*60}")
    print(f"{name} 指標")
    print(f"{'='*60}")
    print(f"樣本數: {metrics['n']:,}")
    print(f"\n預測誤差：")
    print(f"  MSE   : {metrics['MSE']:.6f}")
    print(f"  RMSE  : {metrics['RMSE']:.6f}")
    print(f"  MAE   : {metrics['MAE']:.6f}")
    
    print(f"\n相關性指標（越高越好）：")
    print(f"  IC        : {metrics['IC']:.4f}")
    print(f"  Daily IC  : {metrics['DailyIC']:.4f}")
    print(f"  ICIR      : {metrics['ICIR']:.4f}")
    
    if has_industry:
        print(f"\n產業中性指標：")
        print(f"  IC (ind)       : {metrics['IC_ind']:.4f}")
        print(f"  Daily IC (ind) : {metrics['DailyIC_ind']:.4f}")
        print(f"  ICIR (ind)     : {metrics['ICIR_ind']:.4f}")
    
    print(f"\n方向準確率：")
    print(f"  Dir Accuracy : {metrics['DirAcc']:.4f} ({metrics['DirAcc']*100:.2f}%)")


def main():
    ap = argparse.ArgumentParser(description="評估 GAT/DMFM 模型的預測指標")
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tanh_cap", type=float, default=0.2)
    ap.add_argument("--industry_csv", type=str, default=None)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    args = ap.parse_args()

    device = pick_device(args.device)
    
    print("=" * 60)
    print("模型評估腳本")
    print("=" * 60)
    print(f"使用裝置: {device}")

    Ft, yt, edge_industry, edge_universe, meta = load_artifacts(args.artifact_dir)
    Ft = Ft.to(device).float()
    yt = yt.to(device)
    edge_industry = edge_industry.to(device)
    edge_universe = edge_universe.to(device)
    
    T, N, Fdim = Ft.shape
    train_idx, test_idx = time_split_indices(meta["dates"], 0.8)

    model_type = detect_model_type(args.weights, Fdim, device=device)
    
    if model_type == "dmfm":
        model = DMFM(
            in_dim=Fdim,
            hidden_dim=args.hid,
            heads=args.heads,
            use_factor_attention=True
        ).to(device)
    else:
        model = GATRegressor(
            in_dim=Fdim,
            hid=args.hid,          # ← 新增這行
            heads=args.heads,      # ← 新增這行
            tanh_cap=args.tanh_cap
        ).to(device)
    
    model.load_state_dict(torch.load(args.weights, map_location=device))

    stocks = meta.get("stocks", [str(i) for i in range(N)])
    inds = load_industry_labels(args.industry_csv, stocks)
    has_industry = inds is not None

    print(f"資料: T={T}, N={N}, F={Fdim}")
    print(f"產業標籤: {'有 ✓' if has_industry else '無'}")
    print(f"訓練集: {len(train_idx)} 天 | 測試集: {len(test_idx)} 天")

    print("\n評估訓練集...")
    tr = eval_indices(model, Ft, yt, edge_industry, edge_universe, train_idx, 
                      device=device, inds=inds)

    print("評估測試集...")
    te = eval_indices(model, Ft, yt, edge_industry, edge_universe, test_idx, 
                      device=device, inds=inds, return_predictions=True)

    bz = eval_naive_zero(yt, test_idx)

    print_metrics("訓練集", tr, has_industry=has_industry)
    print_metrics("測試集", te, has_industry=has_industry)
    
    print(f"\n{'='*60}")
    print("天真基準（預測為 0）")
    print(f"{'='*60}")
    print(f"  MSE  : {bz['MSE']:.6f}")
    print(f"  RMSE : {bz['RMSE']:.6f}")
    print(f"  MAE  : {bz['MAE']:.6f}")

    if np.isfinite(bz["MSE"]) and np.isfinite(te["MSE"]):
        impr = 100.0 * (1.0 - te["MSE"] / bz["MSE"])
        print(f"\n{'='*60}")
        print(f"相對天真基準的 MSE 改善：{impr:.2f}%")
        print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print("模型效果總結")
    print(f"{'='*60}")
    
    test_ic = te['IC']
    if np.isfinite(test_ic):
        if test_ic > 0.10:
            grade = "優秀 ⭐⭐⭐"
        elif test_ic > 0.08:
            grade = "良好 ⭐⭐"
        elif test_ic > 0.05:
            grade = "可用 ⭐"
        elif test_ic > 0.03:
            grade = "偏弱"
        else:
            grade = "不佳"
        print(f"測試集 IC: {test_ic:.4f} ({grade})")
    
    test_icir = te['ICIR']
    if np.isfinite(test_icir):
        if test_icir > 2.0:
            grade = "極佳"
        elif test_icir > 1.5:
            grade = "優秀"
        elif test_icir > 1.0:
            grade = "良好"
        elif test_icir > 0.5:
            grade = "可用"
        else:
            grade = "偏弱"
        print(f"測試集 ICIR: {test_icir:.4f} ({grade})")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()