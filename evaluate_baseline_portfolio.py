# -*- coding: utf-8 -*-
"""
Evaluate baseline models with prediction metrics + backtest plots.

Supported baselines:
- linear
- xgboost
- lstm
"""

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from report_utils import (
    compound_forward,
    compute_return_stats,
    infer_summary_dir,
    parse_benchmark,
    parse_dates_list,
    save_json,
)
from train_baselines import LSTMRegressor, pick_device, safe_corr
from train_gat_fixed import load_artifacts, time_split_indices_3

matplotlib.use("Agg")


@dataclass
class EvalMetrics:
    n: int
    mse: float
    rmse: float
    mae: float
    ic: float
    daily_ic: float
    icir: float
    dir_acc: float


def compute_metrics(
    preds: List[np.ndarray], truths: List[np.ndarray], daily_ic_series: List[float], daily_dir_series: List[float]
) -> EvalMetrics:
    if not preds:
        return EvalMetrics(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    p_all = np.concatenate(preds)
    y_all = np.concatenate(truths)
    mse = float(np.mean((p_all - y_all) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p_all - y_all)))
    ic = safe_corr(p_all, y_all)

    valid_daily_ic = [x for x in daily_ic_series if np.isfinite(x)]
    daily_ic = float(np.mean(valid_daily_ic)) if valid_daily_ic else np.nan
    if len(valid_daily_ic) > 1:
        ic_std = float(np.std(valid_daily_ic, ddof=1))
        icir = float(daily_ic / (ic_std + 1e-8)) if np.isfinite(ic_std) else np.nan
    else:
        icir = np.nan

    valid_daily_dir = [x for x in daily_dir_series if np.isfinite(x)]
    dir_acc = float(np.mean(valid_daily_dir)) if valid_daily_dir else np.nan

    return EvalMetrics(
        n=int(y_all.size),
        mse=mse,
        rmse=rmse,
        mae=mae,
        ic=float(ic) if np.isfinite(ic) else np.nan,
        daily_ic=daily_ic,
        icir=icir,
        dir_acc=dir_acc,
    )


def load_linear_or_xgboost(artifact_dir: str, model_name: str):
    if model_name == "linear":
        with open(os.path.join(artifact_dir, "baseline_linear.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(artifact_dir, "baseline_linear_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(os.path.join(artifact_dir, "baseline_xgboost.json"))
        with open(os.path.join(artifact_dir, "baseline_xgboost_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    raise ValueError(f"Unsupported non-LSTM model: {model_name}")


def resolve_model_artifacts(artifact_dir: str, model_name: str) -> Tuple[str, Optional[str]]:
    if model_name == "linear":
        return (
            os.path.join(artifact_dir, "baseline_linear.pkl"),
            os.path.join(artifact_dir, "baseline_linear_scaler.pkl"),
        )
    if model_name == "xgboost":
        return (
            os.path.join(artifact_dir, "baseline_xgboost.json"),
            os.path.join(artifact_dir, "baseline_xgboost_scaler.pkl"),
        )
    if model_name == "lstm":
        return (os.path.join(artifact_dir, "baseline_lstm.pt"), None)
    raise ValueError(f"Unsupported baseline model: {model_name}")


def infer_lstm_lookback(artifact_dir: str, fallback: int) -> int:
    metrics_path = os.path.join(artifact_dir, "baseline_lstm_metrics.json")
    if not os.path.exists(metrics_path):
        return fallback
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        lookback = obj.get("lookback")
        if isinstance(lookback, int) and lookback > 1:
            return lookback
    except Exception:
        return fallback
    return fallback


def predict_day_non_lstm(
    model,
    scaler,
    x_day: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    x_valid = x_day[mask].cpu().numpy()
    if x_valid.size == 0:
        return np.empty(0, dtype=np.float32)
    x_scaled = scaler.transform(x_valid)
    pred = model.predict(x_scaled)
    return np.asarray(pred, dtype=np.float32).flatten()


@torch.no_grad()
def predict_day_lstm(
    model: LSTMRegressor,
    ft: torch.Tensor,
    t: int,
    mask: torch.Tensor,
    lookback: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    if t < lookback - 1:
        return None
    valid_idx = torch.where(mask)[0]
    if valid_idx.numel() == 0:
        return np.empty(0, dtype=np.float32)

    x_seq = ft[t - lookback + 1 : t + 1]  # [L, N, F]
    x_seq = torch.nan_to_num(x_seq, nan=0.0).float()
    seq_batch = x_seq[:, valid_idx, :].permute(1, 0, 2).to(device)  # [M, L, F]
    pred = model(seq_batch).detach().cpu().numpy().flatten()
    return pred.astype(np.float32)


def evaluate_and_plot(
    artifact_dir: str,
    model_name: str,
    out_dir: str,
    benchmark_csv: Optional[str],
    top_pct: float,
    rebalance_days: int,
    device: torch.device,
    lookback: Optional[int],
    hidden_dim: int,
    dropout: float,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)

    ft, yt, _, _, meta = load_artifacts(artifact_dir)
    ft = ft.float()
    yt = yt
    y_backtest = yt
    raw_label_path = os.path.join(artifact_dir, "yraw_tensor.pt")
    if os.path.exists(raw_label_path):
        try:
            y_backtest = torch.load(raw_label_path)
        except Exception as e:
            print(f"[warn] failed to load yraw_tensor.pt, fallback to yt: {e}")
            y_backtest = yt
    else:
        print("[warn] yraw_tensor.pt not found; backtest uses demeaned yt")
    dates = parse_dates_list(meta["dates"])
    _, val_idx, test_idx = time_split_indices_3(
        meta["dates"], train_ratio=train_ratio, val_ratio=val_ratio
    )
    horizon_k = int(meta.get("horizon", rebalance_days))

    print("=" * 60)
    print(f"Baseline evaluate: {model_name}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Validation days: {len(val_idx)}")
    print(f"Test days: {len(test_idx)}")
    print("=" * 60)

    model = None
    scaler = None
    lstm_lookback = None

    weights_path, scaler_path = resolve_model_artifacts(artifact_dir, model_name)

    if model_name in {"linear", "xgboost"}:
        model, scaler = load_linear_or_xgboost(artifact_dir, model_name)
    elif model_name == "lstm":
        lstm_lookback = infer_lstm_lookback(artifact_dir, fallback=lookback or 5)
        model = LSTMRegressor(input_dim=ft.shape[-1], hidden_dim=hidden_dim, dropout=dropout).to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    preds_all: List[np.ndarray] = []
    truths_all: List[np.ndarray] = []
    daily_ic: List[float] = []
    daily_dir: List[float] = []
    pred_std: List[float] = []
    test_dates: List[pd.Timestamp] = []
    day_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for t in test_idx:
        y_t = yt[t]
        y_bt_t = y_backtest[t]
        mask = torch.isfinite(y_t) & torch.isfinite(y_bt_t)
        if mask.sum() == 0:
            continue

        x_t = torch.nan_to_num(ft[t], nan=0.0)
        y_np = y_t[mask].cpu().numpy().flatten()
        y_bt_np = y_bt_t[mask].cpu().numpy().flatten()

        if model_name in {"linear", "xgboost"}:
            p_np = predict_day_non_lstm(model, scaler, x_t, mask)
        else:
            p_np = predict_day_lstm(model, ft, t, mask, lstm_lookback, device)
            if p_np is None:
                continue

        if p_np.size == 0:
            continue

        preds_all.append(p_np)
        truths_all.append(y_np)
        day_cache[t] = (p_np, y_bt_np)
        test_dates.append(dates[t])

        ic = safe_corr(p_np, y_np)
        daily_ic.append(float(ic) if np.isfinite(ic) else np.nan)
        daily_dir.append(float((np.sign(p_np) == np.sign(y_np)).mean()))
        pred_std.append(float(np.nanstd(p_np)))

    metrics = compute_metrics(preds_all, truths_all, daily_ic, daily_dir)
    metrics_dict = asdict(metrics)

    # ---- Plot 1: daily_ic ----
    ser_ic = pd.Series(daily_ic, index=pd.to_datetime(test_dates))
    mean_ic = np.nanmean(ser_ic.values) if len(ser_ic) else np.nan
    plt.figure(figsize=(12, 5))
    plt.plot(ser_ic.index, ser_ic.values, linewidth=1.5, alpha=0.7)
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    if np.isfinite(mean_ic):
        plt.axhline(mean_ic, color="red", linestyle="--", linewidth=1.5, label=f"Mean IC = {mean_ic:.4f}")
    plt.title(f"Daily IC ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.grid(True, alpha=0.3)
    if np.isfinite(mean_ic):
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_ic.png"), dpi=150)
    plt.close()

    # ---- Plot 2: pred_dispersion ----
    ser_psd = pd.Series(pred_std, index=pd.to_datetime(test_dates))
    plt.figure(figsize=(12, 5))
    plt.plot(ser_psd.index, ser_psd.values, linewidth=1.5, alpha=0.7, color="orange")
    plt.title(f"Prediction Dispersion ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Prediction Std")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_dispersion.png"), dpi=150)
    plt.close()

    # ---- Plot 3: hitrate_by_month ----
    ser_dir = pd.Series(daily_dir, index=pd.to_datetime(test_dates))
    hr_month = ser_dir.groupby(pd.Grouper(freq="ME")).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(hr_month.index, hr_month.values, marker="o", linewidth=2, markersize=6)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random baseline (50%)")
    plt.title(f"Monthly Directional Accuracy ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Hit Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hitrate_by_month.png"), dpi=150)
    plt.close()

    # ---- Plot 4: ic_distribution ----
    ic_values = ser_ic.dropna().values
    plt.figure(figsize=(10, 5))
    plt.hist(ic_values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    if np.isfinite(mean_ic):
        plt.axvline(mean_ic, color="red", linestyle="--", linewidth=2, label=f"Mean IC = {mean_ic:.4f}")
    plt.axvline(0, color="gray", linestyle="-", linewidth=1)
    plt.title(f"IC Distribution ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("IC")
    plt.ylabel("Frequency")
    if np.isfinite(mean_ic):
        plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ic_distribution.png"), dpi=150)
    plt.close()

    # ---- Plot 5: cumulative returns ----
    step = max(1, int(rebalance_days))
    take_idx = test_idx[::step]
    long_rets: List[float] = []
    long_dates: List[pd.Timestamp] = []
    for t in take_idx:
        if t not in day_cache:
            continue
        p, y = day_cache[t]
        qh = np.nanquantile(p, 1.0 - float(top_pct))
        sel = p >= qh
        if sel.sum() == 0:
            continue
        long_rets.append(float(np.nanmean(y[sel])))
        long_dates.append(dates[t])

    strat = pd.Series(long_rets, index=pd.to_datetime(long_dates)).sort_index()
    strat_cum = (1.0 + strat.fillna(0.0)).cumprod() - 1.0

    bmk_cum = None
    strat_cmp = None
    bmk_on = None
    if benchmark_csv and os.path.exists(benchmark_csv):
        try:
            bmk = parse_benchmark(benchmark_csv)
            fwd = compound_forward(bmk["ret"], steps=horizon_k)
            bmk_on = fwd.reindex(strat.index)
            mask = strat.notna() & bmk_on.notna()
            if int(mask.sum()) >= 2:
                strat_cmp = strat[mask]
                bmk_on = bmk_on[mask]
                bmk_cum = (1.0 + bmk_on).cumprod() - 1.0
                print(
                    f"Benchmark aligned range: {strat_cmp.index.min().date()} ~ {strat_cmp.index.max().date()} "
                    f"(n={len(strat_cmp)})"
                )
            else:
                print("[warn] benchmark overlap too short; skip benchmark comparison")
        except Exception as e:
            print(f"[warn] benchmark parse failed: {e}")

    plt.figure(figsize=(12, 6))
    if strat_cmp is not None:
        strat_plot = (1.0 + strat_cmp).cumprod() - 1.0
        plt.plot(
            strat_plot.index,
            strat_plot.values * 100,
            linewidth=2.5,
            label=f"{model_name} strategy (aligned, top {top_pct*100:.0f}%)",
            color="#2E86AB",
        )
    else:
        plt.plot(
            strat_cum.index,
            strat_cum.values * 100,
            linewidth=2.5,
            label=f"{model_name} strategy (top {top_pct*100:.0f}%)",
            color="#2E86AB",
        )
    if bmk_cum is not None:
        plt.plot(
            bmk_cum.index,
            bmk_cum.values * 100,
            linewidth=2.5,
            label="Benchmark (0050)",
            color="#A23B72",
            linestyle="--",
        )
    plt.title(f"Cumulative Returns ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cum_returns.png"), dpi=150)
    plt.close()

    strategy_stats = compute_return_stats(strat.values, step)
    aligned_strategy_stats = compute_return_stats(
        strat_cmp.values if strat_cmp is not None else [], step
    )
    benchmark_stats = (
        compute_return_stats(bmk_on.values, step)
        if bmk_cum is not None and bmk_on is not None
        else None
    )

    summary_dir = infer_summary_dir(out_dir)
    plot_paths = {
        "daily_ic": os.path.join(out_dir, "daily_ic.png"),
        "pred_dispersion": os.path.join(out_dir, "pred_dispersion.png"),
        "hitrate_by_month": os.path.join(out_dir, "hitrate_by_month.png"),
        "ic_distribution": os.path.join(out_dir, "ic_distribution.png"),
        "cum_returns": os.path.join(out_dir, "cum_returns.png"),
        "attention_weights": None,
    }
    portfolio_payload = {
        "model_type": model_name,
        "artifact_dir": artifact_dir,
        "weights": weights_path,
        "scaler": scaler_path,
        "plots_dir": out_dir,
        "plots": plot_paths,
        "benchmark_path": benchmark_csv,
        "top_pct": float(top_pct),
        "rebalance_days": int(rebalance_days),
        "horizon": int(horizon_k),
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "val_days": len(val_idx),
            "test_days": len(test_idx),
        },
        "strategy": strategy_stats,
        "strategy_aligned": aligned_strategy_stats if strat_cmp is not None else None,
        "benchmark": benchmark_stats,
        "alignment": (
            {
                "start": str(strat_cmp.index.min().date()),
                "end": str(strat_cmp.index.max().date()),
                "n_periods": int(len(strat_cmp)),
            }
            if strat_cmp is not None and len(strat_cmp) > 0
            else None
        ),
        "top_attention_features": [],
        "lstm_lookback": lstm_lookback,
    }
    if benchmark_stats is not None and strat_cmp is not None:
        portfolio_payload["excess"] = {
            "annual_return": aligned_strategy_stats["annual_return"] - benchmark_stats["annual_return"],
            "total_return": aligned_strategy_stats["total_return"] - benchmark_stats["total_return"],
            "sharpe": (
                aligned_strategy_stats["sharpe"] - benchmark_stats["sharpe"]
                if aligned_strategy_stats["sharpe"] is not None and benchmark_stats["sharpe"] is not None
                else None
            ),
        }

    save_json(summary_dir / "portfolio.json", portfolio_payload)

    print("=" * 60)
    print(f"Done baseline evaluate: {model_name}")
    print(f"Portfolio: {summary_dir / 'portfolio.json'}")
    print(f"Plots: {out_dir}")
    print("=" * 60)

    return {
        "model_type": model_name,
        "artifact_dir": artifact_dir,
        "out_dir": out_dir,
        "metrics": metrics_dict,
        "lstm_lookback": lstm_lookback,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate baseline model metrics + backtest + plots")
    ap.add_argument("--artifact_dir", required=True, help="Artifact directory")
    ap.add_argument("--model", required=True, choices=["linear", "xgboost", "lstm"])
    ap.add_argument("--out_dir", required=True, help="Output directory for metrics and plots")
    ap.add_argument("--benchmark_csv", default=None, help="Benchmark CSV (optional)")
    ap.add_argument("--top_pct", type=float, default=0.10, help="Top percentile for long-only portfolio")
    ap.add_argument("--rebalance_days", type=int, default=5, help="Rebalance frequency")
    ap.add_argument("--device", default="auto", help="cpu/cuda/mps/auto")
    ap.add_argument("--lookback", type=int, default=None, help="LSTM lookback override")
    ap.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dim")
    ap.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio within train")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    evaluate_and_plot(
        artifact_dir=args.artifact_dir,
        model_name=args.model,
        out_dir=args.out_dir,
        benchmark_csv=args.benchmark_csv,
        top_pct=args.top_pct,
        rebalance_days=args.rebalance_days,
        device=device,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
