# -*- coding: utf-8 -*-
# build_artifacts.py
"""
建立訓練/評估共用的 artifacts。

輸入：
- `--prices`: 股票主資料 CSV
- `--industry_csv`: 產業欄位來源 CSV；通常可與 `--prices` 相同
- `--artifact_dir`: 輸出目錄
- `--start_date` / `--end_date`: 資料區間
- `--horizon`: forward-k 報酬標籤

輸出：
- `Ft_tensor.pt`
- `yt_tensor.pt`
- `yraw_tensor.pt`
- `industry_edge_index.pt`
- `universe_edge_index.pt`
- `dynamic_*_edge_indices.pt`
- `dynamic_*_edge_attrs.pt`
- `meta.pkl`
"""
import argparse, json, os, pickle, time
from contextlib import contextmanager
from typing import List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import torch

from graph_utils import (
    DYNAMIC_GRAPH_META,
    INDUSTRY_EDGE_ATTRS,
    INDUSTRY_EDGE_INDICES,
    UNIVERSE_EDGE_ATTRS,
    UNIVERSE_EDGE_INDICES,
)

# ---------------- Args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--industry_csv", default=None)  # 不合併，只做存在性檢查
    ap.add_argument("--artifact_dir", default="gat_artifacts_out_plus")
    ap.add_argument("--start_date", default="2019-07-01")
    ap.add_argument("--end_date",   default="2025-09-30")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--no_dynamic_graphs", action="store_true",
                    help="Disable dynamic weighted graph artifacts")
    ap.add_argument("--graph_lookback", type=int, default=60,
                    help="Rolling lookback days for graph correlation weights")
    ap.add_argument("--graph_min_obs", type=int, default=20,
                    help="Minimum valid observations for rolling graph correlations")
    ap.add_argument("--industry_top_k", type=int, default=20,
                    help="Max non-self same-industry neighbors per target in dynamic industry graph")
    ap.add_argument("--universe_top_k", type=int, default=40,
                    help="Max non-self market neighbors per target in dynamic universe graph")
    ap.add_argument("--cache_parquet", action="store_true",
                    help="For CSV inputs, create/use a sibling parquet cache when possible")
    return ap.parse_args()


@contextmanager
def timed_stage(name: str):
    t0 = time.perf_counter()
    print(f"[timing] stage={name} started_at={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    try:
        yield
    finally:
        print(f"[timing] stage={name} duration_sec={time.perf_counter() - t0:.3f}")


def read_prices_table(path: str, cache_parquet: bool = False) -> pd.DataFrame:
    src = os.fspath(path)
    ext = os.path.splitext(src)[1].lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(src)

    if not cache_parquet:
        return pd.read_csv(src)

    cache_path = os.path.splitext(src)[0] + ".parquet"
    try:
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(src):
            print(f"[cache] read parquet cache: {cache_path}")
            return pd.read_parquet(cache_path)
    except Exception as e:
        print(f"[warn] parquet cache read skipped: {e}")

    df = pd.read_csv(src)
    try:
        print(f"[cache] write parquet cache: {cache_path}")
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"[warn] parquet cache write skipped: {e}")
    return df

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

    # Label：未來 k 日報酬
    fwd = df.groupby(cm["code"])[cm["close"]].shift(-horizon)
    df["fwd_ret_k"] = (fwd - df[cm["close"]]) / df[cm["close"]]

    # ✅ 關鍵修正：每日截面去均值（市場中性）
    df["yt"] = df.groupby(cm["date"])["fwd_ret_k"].transform(lambda x: x - x.mean())
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

        # ✅ 關鍵修正：每日截面標準化（跨股票）
        A_zscore = xsec_zscore(A)  # 這個函數你已經有了（第 67 行）
        Ft[:,:,k] = np.nan_to_num(A_zscore, nan=0.0, posinf=0.0, neginf=0.0)

    piv_y = df.pivot_table(index="t_idx", columns="s_idx", values="yt", aggfunc="first")
    Y = np.full((T, N), np.nan, dtype=np.float32)
    if not piv_y.empty:
        Y[piv_y.index.values[:,None], piv_y.columns.values[None,:]] = piv_y.values

    # 原始 forward return（未做每日去均值），供回測/基準比較使用
    piv_yraw = df.pivot_table(index="t_idx", columns="s_idx", values="fwd_ret_k", aggfunc="first")
    Y_raw = np.full((T, N), np.nan, dtype=np.float32)
    if not piv_yraw.empty:
        Y_raw[piv_yraw.index.values[:,None], piv_yraw.columns.values[None,:]] = piv_yraw.values

    return (
        torch.from_numpy(Ft).to(torch.float32),
        torch.from_numpy(Y).to(torch.float32),
        torch.from_numpy(Y_raw).to(torch.float32),
        [str(d) for d in dates],
        list(stocks),
    )
# ---------------- Graph Construction ----------------
def load_industry_series(df, cm, stocks):
    ind_col = cm["industry"] if cm["industry"] in df.columns else None
    if ind_col is None:
        return pd.Series(index=pd.Index(stocks), data="ALL")

    tmp = df[[cm["code"], ind_col]].dropna()
    tmp[cm["code"]] = tmp[cm["code"]].astype(str)
    first = tmp.drop_duplicates(subset=[cm["code"]], keep="first").set_index(cm["code"])[ind_col]
    return first.reindex(pd.Index(stocks)).fillna("UNK")


def build_industry_edges(df, cm, stocks):
    """
    建立產業圖（Industry Graph）的邊索引
    
    原理：
    - 同產業股票之間建立連接（無向圖）
    - 每個股票也連接自己（自環）
    
    參數：
        df: 包含產業資訊的 DataFrame
        cm: 欄位對應字典
        stocks: 股票代碼列表
    
    回傳：
        torch.Tensor [2, E]: 邊索引，格式為 [source_nodes, target_nodes]
    """
    inds = load_industry_series(df, cm, stocks)

    N = len(stocks)
    adj = np.zeros((N, N), dtype=np.uint8)
    
    # 將股票按產業分組
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(stocks):
        groups.setdefault(inds.loc[s], []).append(i)
    
    # 同產業內的股票互相連接
    for idx in groups.values():
        idx = np.array(idx, dtype=int)
        if len(idx): 
            adj[np.ix_(idx, idx)] = 1  # 建立該產業內所有股票的全連接
    
    # 加入自環（每個節點連接自己）
    np.fill_diagonal(adj, 1)
    
    # 轉換為 PyG 格式的邊索引 [2, num_edges]
    rows, cols = np.where(adj == 1)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def build_universe_edges(stocks):
    """
    建立全市場圖（Universe Graph）的邊索引
    
    原理：
    - 所有股票之間建立全連接（完全圖）
    - 用於學習跨產業的市場共同影響
    - 對應論文中的 Universe-level relationships
    
    參數：
        stocks: 股票代碼列表
    
    回傳：
        torch.Tensor [2, E]: 邊索引，E = N * N（全連接）
        
    注意：
    - 771 檔股票會產生 594,441 條邊，記憶體需求較高
    - 若記憶體不足，可考慮使用稀疏連接或 K-近鄰圖
    """
    N = len(stocks)
    
    # 方法：建立全連接鄰接矩陣（包含自環）
    adj = np.ones((N, N), dtype=np.uint8)  # 所有位置皆為 1
    
    # 轉換為邊索引格式
    rows, cols = np.where(adj == 1)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    
    return edge_index


def build_return_matrix(df, cm, dates, stocks):
    """Align daily one-day returns to [T, N] for dynamic graph weights."""
    d = df[(df[cm["date"]] >= pd.to_datetime(min(dates))) & (df[cm["date"]] <= pd.to_datetime(max(dates)))].copy()
    d = d.drop_duplicates(subset=[cm["date"], cm["code"]])
    d["t_idx"] = pd.Categorical(d[cm["date"]], categories=pd.to_datetime(dates), ordered=True).codes
    d["s_idx"] = pd.Categorical(d[cm["code"]].astype(str), categories=pd.Index(stocks), ordered=True).codes
    piv = d.pivot_table(index="t_idx", columns="s_idx", values="ret_1", aggfunc="first")
    R = np.full((len(dates), len(stocks)), np.nan, dtype=np.float32)
    if not piv.empty:
        R[piv.index.values[:, None], piv.columns.values[None, :]] = piv.values
    return R


def rolling_corr_matrix(returns, t, lookback, min_obs):
    start = max(0, t - lookback + 1)
    W = returns[start:t + 1].astype(np.float32, copy=False)
    valid = np.isfinite(W)
    counts = valid.sum(axis=0)
    enough = counts >= min_obs
    if enough.sum() < 2:
        return np.eye(W.shape[1], dtype=np.float32)

    mu = np.nanmean(W, axis=0)
    sd = np.nanstd(W, axis=0)
    Z = (W - mu) / sd
    Z[:, (~enough) | (~np.isfinite(sd)) | (sd <= 1e-12)] = 0.0
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    denom = np.maximum(np.minimum.outer(counts, counts) - 1, 1).astype(np.float32)
    corr = (Z.T @ Z) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    return corr


def topk_edges_from_corr(corr, candidate_mask, top_k):
    N = corr.shape[0]
    abs_corr = np.abs(corr)
    src_all, dst_all, attrs = [], [], []
    for dst in range(N):
        src_all.append(dst)
        dst_all.append(dst)
        attrs.append((1.0, 1.0))

        candidates = np.flatnonzero(candidate_mask[:, dst])
        candidates = candidates[candidates != dst]
        if candidates.size == 0 or top_k <= 0:
            continue

        scores = abs_corr[candidates, dst]
        keep = scores > 1e-6
        candidates = candidates[keep]
        scores = scores[keep]
        if candidates.size == 0:
            continue

        if candidates.size > top_k:
            top_pos = np.argpartition(scores, -top_k)[-top_k:]
            candidates = candidates[top_pos]
            scores = scores[top_pos]
        order = np.argsort(scores)[::-1]
        candidates = candidates[order]

        for src in candidates:
            signed = float(corr[src, dst])
            src_all.append(int(src))
            dst_all.append(int(dst))
            attrs.append((abs(signed), signed))

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def build_dynamic_weighted_graphs(df, cm, dates, stocks, lookback, min_obs, industry_top_k, universe_top_k):
    returns = build_return_matrix(df, cm, dates, stocks)
    inds = load_industry_series(df, cm, stocks)
    ind_values = inds.reindex(pd.Index(stocks)).astype(str).to_numpy()
    same_industry = ind_values[:, None] == ind_values[None, :]
    universe_candidates = np.ones((len(stocks), len(stocks)), dtype=bool)

    industry_edge_indices, industry_edge_attrs = [], []
    universe_edge_indices, universe_edge_attrs = [], []
    for t in range(len(dates)):
        corr = rolling_corr_matrix(returns, t, lookback=lookback, min_obs=min_obs)
        ind_ei, ind_attr = topk_edges_from_corr(corr, same_industry, industry_top_k)
        uni_ei, uni_attr = topk_edges_from_corr(corr, universe_candidates, universe_top_k)
        industry_edge_indices.append(ind_ei)
        industry_edge_attrs.append(ind_attr)
        universe_edge_indices.append(uni_ei)
        universe_edge_attrs.append(uni_attr)

    return industry_edge_indices, industry_edge_attrs, universe_edge_indices, universe_edge_attrs


# ---------------- Main ----------------
def main():
    total_t0 = time.perf_counter()
    args = parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)

    # 讀取資料
    with timed_stage("read_prices"):
        df = read_prices_table(args.prices, cache_parquet=args.cache_parquet)
    cm = choose_colmap(df)
    df[cm["date"]] = to_dt(df[cm["date"]])

    # 驗證產業檔案存在（若有指定）
    with timed_stage("validate_industry_csv"):
        if args.industry_csv and os.path.abspath(args.industry_csv) != os.path.abspath(args.prices):
            _ = read_prices_table(args.industry_csv, cache_parquet=args.cache_parquet)  # 只驗存在

    # 建立特徵與標籤
    with timed_stage("build_features_and_label"):
        df_b, feature_cols = build_features_and_label(df, cm, args.horizon)
    
    # 轉換為張量
    with timed_stage("to_tensors"):
        Ft_t, Y_t, Y_raw_t, dates, stocks = to_tensors(df_b, cm, feature_cols, args.start_date, args.end_date)
    
    # 建立兩種圖結構
    with timed_stage("build_static_graphs"):
        industry_edge_index = build_industry_edges(df_b, cm, stocks)      # 產業圖
        universe_edge_index = build_universe_edges(stocks)                # 全市場圖
    dynamic_graph_stats = None
    if not args.no_dynamic_graphs:
        print("建立 dynamic weighted graphs...")
        with timed_stage("build_dynamic_weighted_graphs"):
            (
                dynamic_industry_edge_indices,
                dynamic_industry_edge_attrs,
                dynamic_universe_edge_indices,
                dynamic_universe_edge_attrs,
            ) = build_dynamic_weighted_graphs(
                df_b,
                cm,
                dates,
                stocks,
                lookback=args.graph_lookback,
                min_obs=args.graph_min_obs,
                industry_top_k=args.industry_top_k,
                universe_top_k=args.universe_top_k,
            )
        dynamic_graph_stats = {
            "enabled": True,
            "edge_dim": 2,
            "edge_attr": ["abs_corr", "signed_corr"],
            "lookback": args.graph_lookback,
            "min_obs": args.graph_min_obs,
            "industry_top_k": args.industry_top_k,
            "universe_top_k": args.universe_top_k,
            "avg_industry_edges": float(np.mean([ei.shape[1] for ei in dynamic_industry_edge_indices])),
            "avg_universe_edges": float(np.mean([ei.shape[1] for ei in dynamic_universe_edge_indices])),
            "max_industry_edges": int(max(ei.shape[1] for ei in dynamic_industry_edge_indices)),
            "max_universe_edges": int(max(ei.shape[1] for ei in dynamic_universe_edge_indices)),
        }
    else:
        dynamic_industry_edge_indices = dynamic_industry_edge_attrs = None
        dynamic_universe_edge_indices = dynamic_universe_edge_attrs = None
        dynamic_graph_stats = {"enabled": False}

    # 儲存所有 artifacts
    with timed_stage("save_artifacts"):
        torch.save(Ft_t, os.path.join(args.artifact_dir, "Ft_tensor.pt"))
        torch.save(Y_t , os.path.join(args.artifact_dir, "yt_tensor.pt"))
        torch.save(Y_raw_t, os.path.join(args.artifact_dir, "yraw_tensor.pt"))
        torch.save(industry_edge_index, os.path.join(args.artifact_dir, "industry_edge_index.pt"))
        torch.save(universe_edge_index, os.path.join(args.artifact_dir, "universe_edge_index.pt"))  # 新增
        if dynamic_graph_stats.get("enabled"):
            torch.save(dynamic_industry_edge_indices, os.path.join(args.artifact_dir, INDUSTRY_EDGE_INDICES))
            torch.save(dynamic_industry_edge_attrs, os.path.join(args.artifact_dir, INDUSTRY_EDGE_ATTRS))
            torch.save(dynamic_universe_edge_indices, os.path.join(args.artifact_dir, UNIVERSE_EDGE_INDICES))
            torch.save(dynamic_universe_edge_attrs, os.path.join(args.artifact_dir, UNIVERSE_EDGE_ATTRS))
            with open(os.path.join(args.artifact_dir, DYNAMIC_GRAPH_META), "w", encoding="utf-8") as f:
                json.dump(dynamic_graph_stats, f, ensure_ascii=False, indent=2)
        else:
            for stale_name in (
                INDUSTRY_EDGE_INDICES,
                INDUSTRY_EDGE_ATTRS,
                UNIVERSE_EDGE_INDICES,
                UNIVERSE_EDGE_ATTRS,
                DYNAMIC_GRAPH_META,
            ):
                stale_path = os.path.join(args.artifact_dir, stale_name)
                if os.path.exists(stale_path):
                    os.remove(stale_path)

        # 儲存 metadata
        meta = {
            "feature_cols": feature_cols,
            "dates": dates,
            "stocks": stocks,
            "horizon": args.horizon,
            "column_map": cm,
            "raw_label_col": "fwd_ret_k",
            "num_stocks": len(stocks),                          # 新增：股票數量
            "num_features": len(feature_cols),                  # 新增：特徵數量
            "num_industry_edges": industry_edge_index.shape[1], # 新增：產業圖邊數
            "num_universe_edges": universe_edge_index.shape[1], # 新增：全市場圖邊數
            "dynamic_graphs": dynamic_graph_stats,
            "cache_parquet": bool(args.cache_parquet),
            "source_prices": args.prices,
        }
        with open(os.path.join(args.artifact_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    # 輸出摘要資訊
    print("=" * 60)
    print("Artifacts 建立完成！")
    print("=" * 60)
    print("\n儲存檔案：")
    print(f" ✓ Ft_tensor.pt            : {tuple(Ft_t.shape)}, {Ft_t.dtype}")
    print(f" ✓ yt_tensor.pt            : {tuple(Y_t.shape)}, {Y_t.dtype}")
    print(f" ✓ yraw_tensor.pt          : {tuple(Y_raw_t.shape)}, {Y_raw_t.dtype}")
    print(f" ✓ industry_edge_index.pt  : {tuple(industry_edge_index.shape)}")
    print(f" ✓ universe_edge_index.pt  : {tuple(universe_edge_index.shape)} (新增)")
    if dynamic_graph_stats.get("enabled"):
        print(f" ✓ {INDUSTRY_EDGE_INDICES}: {len(dynamic_industry_edge_indices)} days")
        print(f" ✓ {INDUSTRY_EDGE_ATTRS}  : edge_dim=2")
        print(f" ✓ {UNIVERSE_EDGE_INDICES}: {len(dynamic_universe_edge_indices)} days")
        print(f" ✓ {UNIVERSE_EDGE_ATTRS}  : edge_dim=2")
    print(f" ✓ meta.pkl                : 包含 {len(meta)} 項 metadata")
    
    print("\n資料統計：")
    print(f" - 時間範圍: {dates[0]} ~ {dates[-1]} ({len(dates)} 個交易日)")
    print(f" - 股票數量: {len(stocks)}")
    print(f" - 特徵數量: {len(feature_cols)}")
    print(f" - 預測視野: {args.horizon} 日")
    
    print("\n圖結構統計：")
    print(f" - 產業圖邊數: {industry_edge_index.shape[1]:,}")
    print(f" - 全市場圖邊數: {universe_edge_index.shape[1]:,}")
    if dynamic_graph_stats.get("enabled"):
        print(f" - 動態產業圖平均邊數/日: {dynamic_graph_stats['avg_industry_edges']:.1f}")
        print(f" - 動態全市場圖平均邊數/日: {dynamic_graph_stats['avg_universe_edges']:.1f}")
    print(f" - 平均每檔股票的產業內連接數: {industry_edge_index.shape[1] / len(stocks):.1f}")
    print(f" - 平均每檔股票的全市場連接數: {universe_edge_index.shape[1] / len(stocks):.1f}")
    
    # 記憶體使用估算
    industry_mem_mb = industry_edge_index.element_size() * industry_edge_index.nelement() / 1024 / 1024
    universe_mem_mb = universe_edge_index.element_size() * universe_edge_index.nelement() / 1024 / 1024
    total_mem_mb = Ft_t.element_size() * Ft_t.nelement() / 1024 / 1024 + \
                   Y_t.element_size() * Y_t.nelement() / 1024 / 1024 + \
                   Y_raw_t.element_size() * Y_raw_t.nelement() / 1024 / 1024 + \
                   industry_mem_mb + universe_mem_mb
    
    print("\n記憶體使用估算：")
    print(f" - 特徵張量 (Ft): {Ft_t.element_size() * Ft_t.nelement() / 1024 / 1024:.1f} MB")
    print(f" - 標籤張量 (yt): {Y_t.element_size() * Y_t.nelement() / 1024 / 1024:.1f} MB")
    print(f" - 原始標籤 (yraw): {Y_raw_t.element_size() * Y_raw_t.nelement() / 1024 / 1024:.1f} MB")
    print(f" - 產業圖: {industry_mem_mb:.1f} MB")
    print(f" - 全市場圖: {universe_mem_mb:.1f} MB")
    print(f" - 總計約: {total_mem_mb:.1f} MB")
    print(f" - Artifact build total time: {time.perf_counter() - total_t0:.1f}s")
    
    print("\n特徵列表 (n={}):".format(len(feature_cols)))
    # 按類別分組顯示特徵
    feature_groups = {
        "動量類": [f for f in feature_cols if any(x in f for x in ["ret_", "mom_", "rev_"])],
        "移動平均": [f for f in feature_cols if "sma" in f],
        "波動率": [f for f in feature_cols if "std_" in f or "atr" in f or "idio" in f],
        "成交量": [f for f in feature_cols if "vol_" in f or "vol" in f],
        "技術指標": [f for f in feature_cols if any(x in f for x in ["rsi", "stoch", "macd"])],
        "統計特性": [f for f in feature_cols if any(x in f for x in ["skew", "kurt", "zscore", "beta"])],
        "價位相關": [f for f in feature_cols if any(x in f for x in ["roll_", "pct_to", "pct_pos", "mdd"])],
        "流動性": [f for f in feature_cols if "amihud" in f],
        "估值": [f for f in feature_cols if f in ["pb", "ps"]],
    }
    
    for group_name, feats in feature_groups.items():
        if feats:
            print(f"\n  [{group_name}] ({len(feats)} 個)")
            for f in feats:
                print(f"    - {f}")
    
    print("\n" + "=" * 60)
    print("接下來可以執行訓練：")
    print(f"  python train_gat_fixed.py --artifact_dir {args.artifact_dir} --epochs 50 --lr 1e-3")
    print("=" * 60)

if __name__ == "__main__":
    main()


"""
執行結果範例：
============================================================
Artifacts 建立完成！
============================================================

儲存檔案：
 ✓ Ft_tensor.pt            : (1460, 771, 56), torch.float16
 ✓ yt_tensor.pt            : (1460, 771), torch.float32
 ✓ industry_edge_index.pt  : (2, 27923)
 ✓ universe_edge_index.pt  : (2, 594441) (新增)
 ✓ meta.pkl                : 包含 9 項 metadata

資料統計：
 - 時間範圍: 2019-07-01 ~ 2025-09-30 (1460 個交易日)
 - 股票數量: 771
 - 特徵數量: 56
 - 預測視野: 5 日

圖結構統計：
 - 產業圖邊數: 27,923
 - 全市場圖邊數: 594,441
 - 平均每檔股票的產業內連接數: 36.2
 - 平均每檔股票的全市場連接數: 771.0

記憶體使用估算：
 - 特徵張量 (Ft): 126.1 MB
 - 標籤張量 (yt): 4.5 MB
 - 產業圖: 0.2 MB
 - 全市場圖: 4.5 MB
 - 總計約: 135.3 MB

特徵列表 (n=56):

  [動量類] (9 個)
    - ret_1
    - ret_3
    - ret_5
    - ret_10
    - ret_20
    - mom_diff_10
    - mom_diff_20
    - rev_1
    - rev_5
    - rev_10

  [移動平均] (4 個)
    - px_over_sma_5
    - px_over_sma_10
    - px_over_sma_20
    - px_over_sma_60

  ... (以下類推)

============================================================
接下來可以執行訓練：
  python train_gat_fixed.py --artifact_dir gat_artifacts_out_plus --epochs 50 --lr 1e-3
============================================================
"""
