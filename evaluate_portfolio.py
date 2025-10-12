"""
投資組合回測評估

執行方式：

# 全市場選股（每 5 日再平衡，前 10% 做多）
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device cuda \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv unique_2019q3to2025q3.csv

# 產業中立版本（各產業各自取前 10%，並對預測做產業內 z-score）
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device cuda \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv unique_2019q3to2025q3.csv \
  --industry_neutral --neutralize_pred_by_industry

# 多空策略（前 10% 做多，後 10% 做空）
python evaluate_portfolio.py \
  --artifact_dir gat_artifacts_out_plus \
  --weights gat_artifacts_out_plus/gat_regressor.pt \
  --device cuda \
  --top_pct 0.10 --rebalance_days 5 \
  --industry_csv unique_2019q3to2025q3.csv \
  --long_short
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd

from train_gat_fixed import GATRegressor, DMFM, load_artifacts, time_split_indices

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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_industry_map(industry_csv, stocks):
    """
    載入股票產業對應關係
    
    回傳：
        dict: {stock_code: industry_name}
    """
    if not industry_csv or not os.path.exists(industry_csv):
        return {s: "UNK" for s in stocks}

    df = pd.read_csv(industry_csv, dtype={"證券代碼": str, "證券代碼_純代碼": str})
    
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
        return {s: "UNK" for s in stocks}

    df = df[[sid_col, ind_col]].dropna().drop_duplicates()
    df[sid_col] = df[sid_col].astype(str)
    m = dict(zip(df[sid_col], df[ind_col]))
    return {s: m.get(s, "UNK") for s in stocks}


def detect_model_type(weights_path, device="cpu"):
    """自動偵測模型類型"""
    state_dict = torch.load(weights_path, map_location=device)
    dmfm_keys = ["encoder.0.weight", "gat_universe.lin_src.weight", "factor_attn.weight"]
    is_dmfm = any(key in state_dict for key in dmfm_keys)
    return "dmfm" if is_dmfm else "gat"


@torch.no_grad()
def daily_scores(model, Ft_t, edge_industry, edge_universe, y_t):
    """
    計算當日預測分數
    
    參數：
        model: 模型實例
        Ft_t: [N, F] 當日特徵
        edge_industry: 產業圖邊索引
        edge_universe: 全市場圖邊索引（DMFM 用）
        y_t: [N] 當日標籤
    
    回傳：
        (predictions, labels, mask): 預測值、真實值、有效遮罩
    """
    x = torch.nan_to_num(Ft_t, nan=0.0).float()
    mask = torch.isfinite(y_t)
    
    # 根據模型類型選擇前向傳播
    if isinstance(model, DMFM):
        p, _, _ = model(x, edge_industry, edge_universe)
    else:
        p = model(x, edge_industry)
    
    return p, y_t, mask


def portfolio_stats(series, rebal_days=5):
    """
    計算投資組合統計指標
    
    參數：
        series: 報酬率序列
        rebal_days: 再平衡頻率（天數）
    
    回傳：
        dict: 包含均值、標準差、年化報酬、夏普比率、勝率等
    """
    arr = np.asarray(series, dtype=np.float64)
    freq = 252.0 / float(rebal_days)  # 年化係數
    
    mean = np.nanmean(arr) if arr.size else np.nan
    std = np.nanstd(arr, ddof=1) if arr.size > 1 else np.nan
    
    # 夏普比率（假設無風險利率為 0）
    sharpe = (mean / std) * np.sqrt(freq) if (np.isfinite(mean) and np.isfinite(std) and std > 0) else np.nan
    
    # 年化報酬率
    ann_ret = mean * freq if np.isfinite(mean) else np.nan
    
    # 勝率
    hit = np.mean(arr > 0) if arr.size else np.nan
    
    return dict(
        mean=mean, 
        std=std, 
        ann_ret=ann_ret, 
        sharpe=sharpe, 
        hitrate=hit, 
        n=int(arr.size)
    )


def zscore_in_groups(values, groups):
    """
    對每個產業分組做 z-score 標準化
    
    參數：
        values: 預測值數組
        groups: 產業標籤數組
    
    回傳：
        標準化後的預測值
    """
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


def run_eval(artifact_dir, weights, device="cpu", tanh_cap=0.2, hid=64, heads=2,
             top_pct=0.10, long_short=False, rebalance_days=5,
             industry_csv=None, industry_neutral=False,
             neutralize_pred_by_industry=False):
    """
    執行投資組合回測
    
    參數：
        artifact_dir: artifacts 資料夾路徑
        weights: 模型權重檔路徑
        device: 計算裝置
        tanh_cap: 輸出限制範圍
        hid: 隱藏層維度（需與訓練時一致）
        heads: 注意力頭數（需與訓練時一致）
        top_pct: 選股百分比（0.10 = 前 10%）
        long_short: 是否做多空策略
        rebalance_days: 再平衡頻率（天數）
        industry_csv: 產業對照表
        industry_neutral: 是否產業中立選股
        neutralize_pred_by_industry: 是否對預測做產業內標準化
    
    回傳：
        dict: 包含 long_only 和 long_short（若啟用）的統計指標
    """
    device = pick_device(device)
    print("=" * 60)
    print("投資組合回測")
    print("=" * 60)
    print(f"使用裝置: {device}")

    # 載入 artifacts（包含兩種圖）
    Ft, yt, edge_industry, edge_universe, meta = load_artifacts(artifact_dir)
    Ft = Ft.to(device).float()
    yt = yt.to(device)
    edge_industry = edge_industry.to(device)
    edge_universe = edge_universe.to(device)

    T, N, Fdim = Ft.shape
    stocks = meta.get("stocks", [str(i) for i in range(N)])
    _, test_idx_full = time_split_indices(meta["dates"], 0.8)

    # 再平衡日期索引
    step = max(1, int(rebalance_days))
    test_idx = list(range(test_idx_full[0], test_idx_full[-1] + 1, step))

    print(f"資料: T={T}, N={N}, F={Fdim}")
    print(f"測試期: {len(test_idx)} 個再平衡日（每 {rebalance_days} 天）")

    # 自動偵測並載入模型
    model_type = detect_model_type(weights, device=device)
    
    if model_type == "dmfm":
        print("模型類型: DMFM")
        model = DMFM(
            in_dim=Fdim,
            hid=hid,
            heads=heads,
            tanh_cap=tanh_cap,
            use_factor_attention=True
        ).to(device)
    else:
        print("模型類型: GATRegressor")
        model = GATRegressor(
            in_dim=Fdim,
            tanh_cap=tanh_cap
        ).to(device)
    
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # 載入產業資訊
    ind_map = load_industry_map(industry_csv, stocks) if (industry_csv or industry_neutral or neutralize_pred_by_industry) else None
    inds = np.array([ind_map.get(s, "UNK") for s in stocks]) if ind_map is not None else None

    # 策略設定
    print(f"\n策略設定:")
    print(f"  選股百分比: {top_pct*100:.1f}%")
    print(f"  再平衡頻率: {rebalance_days} 天")
    print(f"  產業中立: {'是' if industry_neutral else '否'}")
    print(f"  預測標準化: {'產業內' if neutralize_pred_by_industry else '全市場'}")
    print(f"  多空策略: {'是' if long_short else '否'}")
    print("=" * 60)

    # 回測循環
    long_rets, spread_rets = [], []
    q_hi_fn = lambda P: np.nanquantile(P, 1.0 - float(top_pct))
    q_lo_fn = lambda P: np.nanquantile(P, float(top_pct))

    for t in test_idx:
        p, y, mask = daily_scores(model, Ft[t], edge_industry, edge_universe, yt[t])
        if mask.sum() == 0:
            continue
        
        P = p[mask].detach().cpu().numpy()
        Y = y[mask].detach().cpu().numpy()
        idxs = np.where(mask.cpu().numpy())[0]

        # 預測值標準化（若啟用）
        if neutralize_pred_by_industry and inds is not None:
            P = zscore_in_groups(P, inds[idxs])

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
                if ig.sum() < 5:  # 產業股票數太少則跳過
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

    # 計算統計指標
    out = {"long_only": portfolio_stats(long_rets, rebal_days=rebalance_days)}
    if long_short and len(spread_rets) > 0:
        out["long_short"] = portfolio_stats(spread_rets, rebal_days=rebalance_days)
    
    return out


def print_portfolio_results(results, strategy_name):
    """格式化輸出投資組合結果"""
    print(f"\n{'='*60}")
    print(f"{strategy_name}")
    print(f"{'='*60}")
    
    stats = results
    print(f"樣本數: {stats['n']}")
    print(f"\n報酬統計：")
    print(f"  平均報酬（每期）: {stats['mean']:.6f} ({stats['mean']*100:.4f}%)")
    print(f"  報酬標準差: {stats['std']:.6f}")
    print(f"  年化報酬率: {stats['ann_ret']:.4f} ({stats['ann_ret']*100:.2f}%)")
    print(f"\n風險調整報酬：")
    print(f"  夏普比率: {stats['sharpe']:.4f}")
    
    # 夏普比率評級
    sharpe = stats['sharpe']
    if np.isfinite(sharpe):
        if sharpe > 2.0:
            grade = "優秀 ⭐⭐⭐"
        elif sharpe > 1.5:
            grade = "良好 ⭐⭐"
        elif sharpe > 1.0:
            grade = "可用 ⭐"
        elif sharpe > 0.5:
            grade = "偏弱"
        else:
            grade = "不佳"
        print(f"  評級: {grade}")
    
    print(f"\n交易統計：")
    print(f"  勝率: {stats['hitrate']:.4f} ({stats['hitrate']*100:.2f}%)")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser(description="投資組合回測評估")
    
    # 基本參數
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus",
                    help="Artifacts 資料夾路徑")
    ap.add_argument("--weights", type=str, default="./gat_artifacts_out_plus/gat_regressor.pt",
                    help="模型權重檔路徑")
    ap.add_argument("--device", type=str, default="cuda",
                    help="計算裝置: cpu, cuda, mps, 或 auto")
    ap.add_argument("--industry_csv", type=str, default=None,
                    help="產業對照表 CSV 檔案")
    
    # 模型參數
    ap.add_argument("--tanh_cap", type=float, default=0.2,
                    help="輸出 tanh 限制範圍（需與訓練時一致）")
    ap.add_argument("--hid", type=int, default=64,
                    help="隱藏層維度（需與訓練時一致）")
    ap.add_argument("--heads", type=int, default=2,
                    help="GAT 注意力頭數（需與訓練時一致）")
    
    # 策略參數
    ap.add_argument("--top_pct", type=float, default=0.10,
                    help="選股百分比（0.10 = 前 10%）")
    ap.add_argument("--rebalance_days", type=int, default=5,
                    help="再平衡頻率（天數）")
    ap.add_argument("--long_short", action="store_true",
                    help="啟用多空策略（前 top_pct 做多，後 top_pct 做空）")
    ap.add_argument("--industry_neutral", action="store_true",
                    help="產業中立選股（各產業分別選股）")
    ap.add_argument("--neutralize_pred_by_industry", action="store_true",
                    help="對預測值做產業內標準化")
    
    args = ap.parse_args()

    # 執行回測
    results = run_eval(
        artifact_dir=args.artifact_dir,
        weights=args.weights,
        device=args.device,
        tanh_cap=args.tanh_cap,
        hid=args.hid,
        heads=args.heads,
        top_pct=args.top_pct,
        long_short=args.long_short,
        rebalance_days=args.rebalance_days,
        industry_csv=args.industry_csv,
        industry_neutral=args.industry_neutral,
        neutralize_pred_by_industry=args.neutralize_pred_by_industry
    )

    # 輸出結果
    print_portfolio_results(results["long_only"], "做多策略（Long Only）")
    
    if "long_short" in results:
        print_portfolio_results(results["long_short"], "多空策略（Long-Short）")


if __name__ == "__main__":
    main()