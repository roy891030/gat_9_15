# -*- coding: utf-8 -*-
"""
建立 GAT 模型所需的 artifacts（從單一 CSV 檔，且該檔已包含產業欄位）。

輸入參數 (Inputs):
  --prices:        股票/財務資料的 CSV 檔，例如 unique_2019q3to2025q3.csv
  --artifact_dir:  輸出資料夾名稱，預設為 gat_artifacts_out_plus
  --start_date/--end_date: 資料時間範圍，格式為 YYYY-MM-DD，包含起訖日期
  --horizon:       預測視野 k（forward-k 日報酬率作為標籤），預設值為 5

輸出檔案 (Outputs, 會存放在 artifact_dir 資料夾中):
  Ft_tensor.pt            # float16 格式，[T, N, F]
                          # 特徵張量：時間 × 股票數 × 特徵數
  yt_tensor.pt            # float32 格式，[T, N]
                          # 標籤張量：時間 × 股票數（對應 horizon 報酬率）
  industry_edge_index.pt  # long 格式，[2, E]
                          # 圖結構的邊索引，表示產業內股票的關聯
  meta.pkl                # 中繼資訊，紀錄資料時間範圍、股票數量、特徵數等
"""




'''
如何執行：
# 1) 進入專案並啟用 venv
cd /path/to/your/project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools pandas numpy torch

# 2) 產出 artifacts（用您的唯一股池主檔；產業也直接用同一檔）
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv unique_2019q3to2025q3.csv \
  --artifact_dir gat_artifacts_out_plus \
  --start_date 2019-07-01 \
  --end_date   2025-09-30 \
  --horizon 5


執行完成後，gat_artifacts_out_plus/ 會有四個檔。
接著就可以直接用您現成的訓練與評估腳本：

# 訓練
python train_gat_fixed.py --artifact_dir gat_artifacts_out_plus --epochs 50 --lr 1e-3 --device auto

# 指標
python evaluate_metrics.py --artifact_dir gat_artifacts_out_plus --industry_csv unique_2019q3to2025q3.csv

# 投組（與 0050 比較）
python evaluate_portfolio.py --artifact_dir gat_artifacts_out_plus \
  --industry_csv unique_2019q3to2025q3.csv --benchmark_csv GAT0050.csv \
  --rebalance_days 5 --top_pct 0.10 --long_short false

# 圖表
python plot_reports.py --artifact_dir gat_artifacts_out_plus \
  --industry_csv unique_2019q3to2025q3.csv --benchmark_csv GAT0050.csv \
  --out_dir plots_bh
'''
import argparse, os, pickle
from typing import List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import torch

# ---------------- Args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--industry_csv", default=None)  # 不合併，只做存在性檢查
    ap.add_argument("--artifact_dir", default="gat_artifacts_out_plus")
    ap.add_argument("--start_date", default="2019-07-01")
    ap.add_argument("--end_date",   default="2025-09-30")
    ap.add_argument("--horizon", type=int, default=5)
    return ap.parse_args()

# ---------------- Column maps (ZH/EN) ----------------
COLMAPS: List[Dict[str, str]] = [
    {
        "date": "年月日",
        "code": "證券代碼",
        "open": "開盤價(元)",
        "high": "最高價(元)",
        "low" : "最低價(元)",
        "close":"收盤價(元)",
        "volk": "成交量(千股)",
        "turnk":"成交值(千元)",
        "retpct":"報酬率％",
        "industry": "TEJ產業_名稱",
        "industry_code": "TEJ產業_代碼",
        "mktcap_m": "市值(百萬元)",
        "shares_k": "流通在外股數(千股)",
        "pb_raw": "股價淨值比-TEJ",
        "ps_raw": "股價營收比-TEJ",
    },
    {
        "date": "date",
        "code": "code",
        "open": "Open",
        "high": "High",
        "low" : "Low",
        "close":"Close",
        "volk": "Volume_k",
        "turnk":"Turnover_k",
        "retpct":"ReturnPct",
        "industry": "industry",
        "industry_code": "industry_code",
        "mktcap_m": "MktCap_M",
        "shares_k": "Shares_k",
        "pb_raw": "PB_T",
        "ps_raw": "PS_T",
    },
]

def choose_colmap(df: pd.DataFrame) -> Dict[str, str]:
    for m in COLMAPS:
        if m["date"] in df.columns and m["code"] in df.columns and m["close"] in df.columns:
            return m
    raise ValueError("找不到基本欄位（日期/代碼/收盤）；請檢查 CSV 欄名。")

# ---------------- Utils ----------------
def to_num(s): return pd.to_numeric(s, errors="coerce")
def to_dt (s): return pd.to_datetime(s, errors="coerce")
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

def xsec_zscore(A: np.ndarray):
    # A: [T,N] (NaN allowed) → 每日截面 z-score；NaN 留著，最後再填 0
    mu = np.nanmean(A, axis=1, keepdims=True)
    sd = np.nanstd (A, axis=1, keepdims=True)
    Z = (A - mu) / np.where(sd==0, np.nan, sd)
    return Z

def rolling_apply(series: pd.Series, window: int, func):
    return series.rolling(window, min_periods=window).apply(func, raw=True)

# TA-Lib (optional)
try:
    import talib
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

# ---------------- Feature engineering per stock ----------------
def compute_features_one(g: pd.DataFrame, cm: Dict[str,str]) -> pd.DataFrame:
    g = g.sort_values(cm["date"]).copy()
    close = to_num(g[cm["close"]]); high = to_num(g[cm["high"]]); low  = to_num(g[cm["low"]])
    openp = to_num(g[cm["open"]]);  volk  = to_num(g[cm["volk"]])

    # 基本報酬
    ret1  = close.pct_change(1);  ret3  = close.pct_change(3)
    ret5  = close.pct_change(5);  ret10 = close.pct_change(10); ret20 = close.pct_change(20)

    # SMA 比例
    sma = lambda w: close.rolling(w, min_periods=w).mean()
    px_over_sma_5  = close/sma(5)  - 1.0
    px_over_sma_10 = close/sma(10) - 1.0
    px_over_sma_20 = close/sma(20) - 1.0
    px_over_sma_60 = close/sma(60) - 1.0

    # 報酬波動（標準差）
    std_ret_5  = ret1.rolling(5 , min_periods=5 ).std()
    std_ret_10 = ret1.rolling(10, min_periods=10).std()
    std_ret_20 = ret1.rolling(20, min_periods=20).std()
    std_ret_60 = ret1.rolling(60, min_periods=60).std()

    # 量能相對均值
    vol_over = lambda w: (volk / volk.rolling(w, min_periods=w).mean()) - 1.0
    vol_over_ma_5  = vol_over(5)
    vol_over_ma_10 = vol_over(10)
    vol_over_ma_20 = vol_over(20)
    vol_over_ma_60 = vol_over(60)

    # 動能差（短-長）
    mom_diff_10 = ret10 - ret1
    mom_diff_20 = ret20 - ret1

    # RSI / STOCH / MACD / ATR
    if HAS_TALIB:
        rsi_14 = pd.Series(talib.RSI(close.values, timeperiod=14), index=g.index)
        k, d = talib.STOCH(high.values, low.values, close.values, fastk_period=14, slowk_period=3, slowd_period=3)
        stoch_k_14 = pd.Series(k, index=g.index); stoch_d_3 = pd.Series(d, index=g.index)
        macd, macd_signal, macd_hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = pd.Series(macd, index=g.index); macd_signal = pd.Series(macd_signal, index=g.index); macd_hist = pd.Series(macd_hist, index=g.index)
        tr = talib.TRANGE(high.values, low.values, close.values)
        atr_14 = pd.Series(talib.SMA(tr, timeperiod=14), index=g.index)
    else:
        delta = close.diff()
        up = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rsi_14 = 100 - (100/(1 + (up/down)))
        ll14 = low.rolling(14, min_periods=14).min()
        hh14 = high.rolling(14, min_periods=14).max()
        stoch_k_14 = 100*(close - ll14) / (hh14 - ll14)
        stoch_d_3  = stoch_k_14.rolling(3, min_periods=3).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()

    # 上/下漲日平均量（近20）
    up_mask = (ret1 > 0).astype(float); down_mask = (ret1 < 0).astype(float)
    up_vol_20   = (volk * up_mask).rolling(20, min_periods=20).mean()
    down_vol_20 = (volk * down_mask).rolling(20, min_periods=20).mean()

    # 偏度/峰度（近20；以報酬計）
    skew_20 = rolling_apply(ret1, 20, lambda x: pd.Series(x).skew())
    kurt_20 = rolling_apply(ret1, 20, lambda x: pd.Series(x).kurt())

    # 近20日最大回撤（以收盤價）
    roll_max_20_price = close.rolling(20, min_periods=20).max()
    mdd_20 = (close/roll_max_20_price - 1.0)

    # beta_60 與 idio_vol_60（簡化版）
    ret1_center = ret1 - ret1.rolling(60, min_periods=60).mean()
    beta_60 = ret1.rolling(60, min_periods=60).cov(ret1_center) / ret1.rolling(60, min_periods=60).var()
    resid_60 = ret1 - (beta_60 * ret1_center)
    idio_vol_60 = resid_60.rolling(60, min_periods=60).std()

    # Amihud illiquidity（|ret|/value；優先用成交流水）
    turn_col = cm.get("turnk")
    if turn_col and turn_col in g.columns:
        val = to_num(g[turn_col])
    else:
        val = volk * close
    amihud = (ret1.abs() / val.replace(0, np.nan))
    amihud_5  = amihud.rolling(5 , min_periods=5 ).mean()
    amihud_20 = amihud.rolling(20, min_periods=20).mean()

    # 與區間高低點的相對位置（近20）
    hh20 = high.rolling(20, min_periods=20).max(); ll20 = low.rolling(20, min_periods=20).min()
    pct_to_high_20 = (close - hh20) / hh20
    pct_to_low_20  = (close - ll20) / ll20

    # 價格 z-score（近20、60）
    zscore_close_20 = (close - close.rolling(20, min_periods=20).mean()) / close.rolling(20, min_periods=20).std()
    zscore_close_60 = (close - close.rolling(60, min_periods=60).mean()) / close.rolling(60, min_periods=60).std()

    # reversal
    rev_1  = -ret1
    rev_5  = -(close.pct_change(5))
    rev_10 = -(close.pct_change(10))

    # rolling 極值與陽線比例
    roll_max_5  = close.rolling(5 , min_periods=5 ).max()
    roll_min_5  = close.rolling(5 , min_periods=5 ).min()
    roll_max_10 = close.rolling(10, min_periods=10).max()
    roll_min_10 = close.rolling(10, min_periods=10).min()
    roll_max_20 = roll_max_20_price
    roll_min_20 = close.rolling(20, min_periods=20).min()
    roll_max_60 = close.rolling(60, min_periods=60).max()
    roll_min_60 = close.rolling(60, min_periods=60).min()

    pct_pos = lambda w: (ret1 > 0).rolling(w, min_periods=w).mean()
    pct_pos_5  = pct_pos(5)
    pct_pos_10 = pct_pos(10)
    pct_pos_20 = pct_pos(20)
    pct_pos_60 = pct_pos(60)

    out = pd.DataFrame({
        "ret_1": ret1, "px_over_sma_5": px_over_sma_5, "px_over_sma_10": px_over_sma_10,
        "px_over_sma_20": px_over_sma_20, "std_ret_5": std_ret_5, "std_ret_10": std_ret_10,
        "std_ret_20": std_ret_20, "vol_over_ma_5": vol_over_ma_5, "vol_over_ma_10": vol_over_ma_10,
        "vol_over_ma_20": vol_over_ma_20, "ret_3": ret3, "ret_5": ret5, "ret_10": ret10,
        "ret_20": ret20, "mom_diff_10": mom_diff_10, "mom_diff_20": mom_diff_20,
        "px_over_sma_60": px_over_sma_60, "rsi_14": rsi_14, "stoch_k_14": stoch_k_14,
        "stoch_d_3": stoch_d_3, "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "std_ret_60": std_ret_60, "up_vol_20": up_vol_20, "down_vol_20": down_vol_20,
        "skew_20": skew_20, "kurt_20": kurt_20, "atr_14": atr_14, "mdd_20": mdd_20,
        "beta_60": beta_60, "idio_vol_60": idio_vol_60, "vol_over_ma_60": vol_over_ma_60,
        "amihud_5": amihud_5, "amihud_20": amihud_20, "pct_to_high_20": pct_to_high_20,
        "pct_to_low_20": pct_to_low_20, "zscore_close_20": zscore_close_20, "zscore_close_60": zscore_close_60,
        "rev_1": rev_1, "rev_5": rev_5, "rev_10": rev_10, "roll_max_5": roll_max_5, "roll_min_5": roll_min_5,
        "pct_pos_5": pct_pos_5, "roll_max_10": roll_max_10, "roll_min_10": roll_min_10, "pct_pos_10": pct_pos_10,
        "roll_max_20": roll_max_20, "roll_min_20": roll_min_20, "pct_pos_20": pct_pos_20,
        "roll_max_60": roll_max_60, "roll_min_60": roll_min_60, "pct_pos_60": pct_pos_60,
    }, index=g.index)
    return out

# ---------------- Build Ft/yt ----------------
def build_features_and_label(df: pd.DataFrame, cm: Dict[str,str], horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    # 轉型
    for k in ["open","high","low","close","volk","turnk","mktcap_m","shares_k","pb_raw","ps_raw"]:
        col = cm.get(k)
        if col in df.columns:
            df[col] = to_num(df[col])
    df[cm["date"]] = to_dt(df[cm["date"]])
    df = df.sort_values([cm["code"], cm["date"]]).copy()

    # 統一 pb/ps 英文欄位名（若存在）
    if cm["pb_raw"] in df.columns: df["pb"] = df[cm["pb_raw"]]
    if cm["ps_raw"] in df.columns: df["ps"] = df[cm["ps_raw"]]

    # 每檔計算因子
    feats_list = [compute_features_one(g, cm) for _, g in df.groupby(cm["code"], sort=False)]
    feats = pd.concat(feats_list, axis=0).sort_index()
    df = pd.concat([df, feats], axis=1)

    # Label：未來 k 日報酬 → 每日截面去均值
    fwd = df.groupby(cm["code"])[cm["close"]].shift(-horizon)
    df["fwd_ret_k"] = (fwd - df[cm["close"]]) / df[cm["close"]]
    df["yt"] = df.groupby(cm["date"])["fwd_ret_k"].transform(lambda s: s - np.nanmean(s.values))

    # 特徵清單（存在才保留）
    wanted = [
        "ret_1","px_over_sma_5","px_over_sma_10","px_over_sma_20","std_ret_5","std_ret_10","std_ret_20",
        "vol_over_ma_5","vol_over_ma_10","vol_over_ma_20","ret_3","ret_5","ret_10","ret_20",
        "mom_diff_10","mom_diff_20","px_over_sma_60","rsi_14","stoch_k_14","stoch_d_3",
        "macd","macd_signal","macd_hist","std_ret_60","up_vol_20","down_vol_20","skew_20","kurt_20",
        "atr_14","mdd_20","beta_60","idio_vol_60","vol_over_ma_60","amihud_5","amihud_20",
        "pct_to_high_20","pct_to_low_20","zscore_close_20","zscore_close_60",
        "rev_1","rev_5","rev_10","roll_max_5","roll_min_5","pct_pos_5","roll_max_10","roll_min_10",
        "pct_pos_10","roll_max_20","roll_min_20","pct_pos_20","roll_max_60","roll_min_60","pct_pos_60",
        "pb","ps"
    ]
    feature_cols = [c for c in wanted if c in df.columns]
    return df, feature_cols

def to_tensors(df: pd.DataFrame, cm: Dict[str,str], feature_cols: List[str], start_date, end_date):
    df = df[(df[cm["date"]] >= pd.to_datetime(start_date)) & (df[cm["date"]] <= pd.to_datetime(end_date))].copy()
    df = df.drop_duplicates(subset=[cm["date"], cm["code"]])

    dates = pd.Index(sorted(df[cm["date"]].unique()))
    stocks = pd.Index(sorted(df[cm["code"]].astype(str).unique()))
    T, N, F = len(dates), len(stocks), len(feature_cols)

    df["t_idx"] = pd.Categorical(df[cm["date"]], categories=dates, ordered=True).codes
    df["s_idx"] = pd.Categorical(df[cm["code"]].astype(str), categories=stocks, ordered=True).codes

    Ft = np.full((T, N, F), np.nan, dtype=np.float32)
    for k, feat in enumerate(feature_cols):
        piv = df.pivot_table(index="t_idx", columns="s_idx", values=feat, aggfunc="first")
        A = np.full((T, N), np.nan, dtype=np.float32)
        if not piv.empty:
            A[piv.index.values[:,None], piv.columns.values[None,:]] = piv.values
        Z = xsec_zscore(A)
        Ft[:,:,k] = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    piv_y = df.pivot_table(index="t_idx", columns="s_idx", values="yt", aggfunc="first")
    Y = np.full((T, N), np.nan, dtype=np.float32)
    if not piv_y.empty:
        Y[piv_y.index.values[:,None], piv_y.columns.values[None,:]] = piv_y.values

    return torch.from_numpy(Ft).to(torch.float16), torch.from_numpy(Y).to(torch.float32), [str(d) for d in dates], list(stocks)

# ---------------- Industry graph ----------------
def build_industry_edges(df, cm, stocks):
    ind_col = cm["industry"] if cm["industry"] in df.columns else None
    if ind_col is None:
        inds = pd.Series(index=pd.Index(stocks), data="ALL")
    else:
        tmp = df[[cm["code"], ind_col]].dropna()
        tmp[cm["code"]] = tmp[cm["code"]].astype(str)
        first = tmp.drop_duplicates(subset=[cm["code"]], keep="first").set_index(cm["code"])[ind_col]
        inds = first.reindex(pd.Index(stocks)).fillna("UNK")

    N = len(stocks)
    adj = np.zeros((N, N), dtype=np.uint8)
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(stocks):
        groups.setdefault(inds.loc[s], []).append(i)
    for idx in groups.values():
        idx = np.array(idx, dtype=int)
        if len(idx): adj[np.ix_(idx, idx)] = 1
    np.fill_diagonal(adj, 1)
    rows, cols = np.where(adj == 1)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

# ---------------- Main ----------------
def main():
    args = parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)

    df = pd.read_csv(args.prices)
    cm = choose_colmap(df)
    df[cm["date"]] = to_dt(df[cm["date"]])

    if args.industry_csv and os.path.abspath(args.industry_csv) != os.path.abspath(args.prices):
        _ = pd.read_csv(args.industry_csv)  # 只驗存在

    df_b, feature_cols = build_features_and_label(df, cm, args.horizon)
    Ft_t, Y_t, dates, stocks = to_tensors(df_b, cm, feature_cols, args.start_date, args.end_date)
    edge_index = build_industry_edges(df_b, cm, stocks)

    torch.save(Ft_t, os.path.join(args.artifact_dir, "Ft_tensor.pt"))
    torch.save(Y_t , os.path.join(args.artifact_dir, "yt_tensor.pt"))
    torch.save(edge_index, os.path.join(args.artifact_dir, "industry_edge_index.pt"))
    meta = {"feature_cols": feature_cols, "dates": dates, "stocks": stocks,
            "horizon": args.horizon, "column_map": cm}
    with open(os.path.join(args.artifact_dir, "meta.pkl"), "wb") as f: pickle.dump(meta, f)

    print("Saved:")
    print(" Ft_tensor.pt :", tuple(Ft_t.shape), Ft_t.dtype)
    print(" yt_tensor.pt  :", tuple(Y_t.shape),  Y_t.dtype)
    print(" industry_edge_index.pt:", tuple(edge_index.shape))
    print(" features (n={}):".format(len(feature_cols)))
    for c in feature_cols: print("  -", c)

if __name__ == "__main__":
    main()


"""
Ft=(1460, 771, 56), yt=(1460, 771), edges=(2, 27923)
"""