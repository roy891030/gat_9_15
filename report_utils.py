import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def parse_dates_list(str_dates: Sequence[str]) -> list[pd.Timestamp]:
    return pd.to_datetime(pd.Series(str_dates), errors="coerce").tolist()


def parse_datetime_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    dt = pd.to_datetime(series, format="%Y/%m/%d", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(series[missing], format="%Y-%m-%d", errors="coerce")
        dt.loc[missing] = dt2
        missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(series[missing], errors="coerce")
        dt.loc[missing] = dt2
    return dt


def load_dataframe(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    if "git-lfs.github.com/spec" in first_line:
        raise ValueError(f"{path} is a Git LFS pointer; fetch real data first")
    return pd.read_csv(path)


def parse_benchmark(path: str) -> pd.DataFrame:
    df = load_dataframe(path)

    date_col = None
    for col in ["date", "Date", "年月日"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = parse_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    for ret_col in ["報酬率％", "Return", "ret", "pct_change", "RET", "ret1"]:
        if ret_col in df.columns:
            ret = pd.to_numeric(df[ret_col], errors="coerce")
            if ret_col == "報酬率％":
                ret = ret / 100.0
            return pd.DataFrame({"ret": ret.values}, index=df.index)

    price_col = None
    for col in ["收盤價(元)", "Close", "Adj Close", "收盤價", "close", "adj_close"]:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        raise ValueError(f"{path}: cannot find price or return column")

    px = pd.to_numeric(df[price_col], errors="coerce")
    ret = px.pct_change()
    return pd.DataFrame({"ret": ret.values}, index=df.index)


def compound_forward(ret_series: pd.Series, steps: int) -> pd.Series:
    if steps <= 1:
        return ret_series.shift(-1)

    arr = 1.0 + ret_series.values
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for idx in range(0, len(arr) - steps):
        window = arr[idx + 1 : idx + 1 + steps]
        if np.any(~np.isfinite(window)):
            out[idx] = np.nan
        else:
            out[idx] = np.prod(window) - 1.0
    return pd.Series(out, index=ret_series.index)


def compute_return_stats(returns: pd.Series | np.ndarray | list[float], step: int) -> dict[str, float | int | None]:
    series = pd.Series(returns, dtype="float64").dropna()
    if series.empty:
        return {
            "n": 0,
            "mean_return": None,
            "std_return": None,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "hit_rate": None,
            "total_return": None,
            "cagr": None,
            "max_drawdown": None,
        }

    freq = 252.0 / float(max(step, 1))
    mean_return = float(series.mean())
    std_return = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    annual_return = mean_return * freq
    annual_volatility = std_return * np.sqrt(freq) if np.isfinite(std_return) else np.nan
    sharpe = (
        mean_return / std_return * np.sqrt(freq)
        if np.isfinite(std_return) and std_return > 0
        else np.nan
    )

    cumulative = (1.0 + series).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    if cumulative.iloc[-1] > 0:
        cagr = float(cumulative.iloc[-1] ** (freq / len(series)) - 1.0)
    else:
        cagr = np.nan
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    return {
        "n": int(len(series)),
        "mean_return": mean_return,
        "std_return": std_return,
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility) if np.isfinite(annual_volatility) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "hit_rate": float((series > 0).mean()),
        "total_return": total_return,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "max_drawdown": max_drawdown,
    }


def infer_summary_dir(out_dir: str) -> Path:
    path = Path(out_dir)
    return path.parent if path.name == "plots" else path


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def save_json(path: str | Path, payload: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, ensure_ascii=False, indent=2)
