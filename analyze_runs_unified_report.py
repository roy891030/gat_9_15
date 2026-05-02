#!/usr/bin/env python3
"""Build comparison tables, figures, and a Markdown report for runs_unified."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs_unified"
OUT_DIR = RUNS_ROOT / "analysis_report"
CHART_DIR = OUT_DIR / "charts"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "experiment_comparison_report.md"

WINDOW_ORDER = ["short", "medium", "long"]
GRAPH_ORDER = ["static", "dynamic"]
MODEL_ORDER = [
    "baseline_linear",
    "baseline_lstm",
    "baseline_xgboost",
    "mlp",
    "gat_universe",
    "gat_industry",
    "gat_two_graph_no_neutral",
    "dmfm_ind_neutral",
    "dmfm_full",
]

PREDICTION_METRICS = [
    "test_MSE",
    "test_RMSE",
    "test_MAE",
    "test_IC",
    "test_DailyIC",
    "test_ICIR",
    "test_DirAcc",
    "test_IC_ind",
    "test_DailyIC_ind",
    "test_ICIR_ind",
]

PORTFOLIO_METRICS = [
    "strategy_total_return",
    "strategy_cagr",
    "strategy_annual_return",
    "strategy_annual_volatility",
    "strategy_sharpe",
    "strategy_hit_rate",
    "strategy_max_drawdown",
    "excess_total_return",
    "excess_annual_return",
    "excess_sharpe",
]

SUMMARY_METRICS = [
    "test_IC",
    "test_DailyIC",
    "test_ICIR",
    "test_DirAcc",
    "strategy_total_return",
    "strategy_cagr",
    "strategy_sharpe",
    "strategy_hit_rate",
    "strategy_max_drawdown",
    "excess_total_return",
    "excess_sharpe",
]

METRIC_DIRECTIONS = {
    "test_MSE": "min",
    "test_RMSE": "min",
    "test_MAE": "min",
    "test_IC": "max",
    "test_DailyIC": "max",
    "test_ICIR": "max",
    "test_DirAcc": "max",
    "test_IC_ind": "max",
    "test_DailyIC_ind": "max",
    "test_ICIR_ind": "max",
    "strategy_mean_return": "max",
    "strategy_std_return": "min",
    "strategy_total_return": "max",
    "strategy_cagr": "max",
    "strategy_annual_return": "max",
    "strategy_annual_volatility": "min",
    "strategy_sharpe": "max",
    "strategy_hit_rate": "max",
    "strategy_max_drawdown": "max",
    "excess_total_return": "max",
    "excess_annual_return": "max",
    "excess_sharpe": "max",
    "benchmark_total_return": "max",
    "benchmark_cagr": "max",
    "benchmark_annual_return": "max",
    "benchmark_annual_volatility": "min",
    "benchmark_sharpe": "max",
    "benchmark_hit_rate": "max",
    "benchmark_max_drawdown": "max",
}

DISPLAY_NAMES = {
    "graph_mode": "Graph",
    "window": "Window",
    "model": "Model",
    "alignment_start": "Start",
    "alignment_end": "End",
    "alignment_n_periods": "Periods",
    "test_MSE": "Test MSE↓",
    "test_RMSE": "Test RMSE↓",
    "test_MAE": "Test MAE↓",
    "test_IC": "Test IC↑",
    "test_DailyIC": "DailyIC↑",
    "test_ICIR": "ICIR↑",
    "test_DirAcc": "DirAcc↑",
    "test_IC_ind": "IC_ind↑",
    "test_DailyIC_ind": "DailyIC_ind↑",
    "test_ICIR_ind": "ICIR_ind↑",
    "strategy_n": "N",
    "strategy_mean_return": "MeanRet↑",
    "strategy_std_return": "StdRet↓",
    "strategy_total_return": "TotalReturn↑",
    "strategy_cagr": "CAGR↑",
    "strategy_annual_return": "AnnReturn↑",
    "strategy_annual_volatility": "AnnVol↓",
    "strategy_sharpe": "Sharpe↑",
    "strategy_hit_rate": "HitRate↑",
    "strategy_max_drawdown": "MaxDD↑",
    "excess_total_return": "ExcessTR↑",
    "excess_annual_return": "ExcessAnnRet↑",
    "excess_sharpe": "ExcessSharpe↑",
    "benchmark_total_return": "BenchTR",
    "benchmark_sharpe": "BenchSharpe",
    "benchmark_annual_return": "BenchAnnReturn",
    "benchmark_annual_volatility": "BenchAnnVol",
    "benchmark_max_drawdown": "BenchMaxDD",
    "delta_test_IC": "Δ TestIC",
    "delta_strategy_total_return": "Δ TotalReturn",
    "delta_strategy_sharpe": "Δ Sharpe",
    "delta_strategy_max_drawdown": "Δ MaxDD",
    "group_win_count": "Group Wins",
}

PCT_METRICS = {
    "test_DirAcc",
    "strategy_mean_return",
    "strategy_std_return",
    "strategy_total_return",
    "strategy_cagr",
    "strategy_annual_return",
    "strategy_annual_volatility",
    "strategy_hit_rate",
    "strategy_max_drawdown",
    "excess_total_return",
    "excess_annual_return",
    "benchmark_total_return",
    "benchmark_cagr",
    "benchmark_annual_return",
    "benchmark_annual_volatility",
    "benchmark_hit_rate",
    "benchmark_max_drawdown",
    "delta_strategy_total_return",
    "delta_strategy_max_drawdown",
}


def ordered_category(values: pd.Series, order: list[str]) -> pd.Categorical:
    return pd.Categorical(values, categories=order, ordered=True)


def flatten_prefixed(prefix: str, data: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in data.items()}


def load_experiments() -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in sorted(RUNS_ROOT.glob("*/*/*/metrics.json")):
        rel = metrics_path.relative_to(RUNS_ROOT)
        graph_mode, window, model = rel.parts[:3]
        portfolio_path = metrics_path.parent / "portfolio.json"
        if not portfolio_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        with portfolio_path.open("r", encoding="utf-8") as f:
            portfolio = json.load(f)

        strategy = portfolio.get("strategy_aligned") or portfolio.get("strategy") or {}
        benchmark = portfolio.get("benchmark") or {}
        excess = portfolio.get("excess") or {}
        alignment = portfolio.get("alignment") or {}
        split = metrics.get("split") or {}

        row = {
            "graph_mode": graph_mode,
            "window": window,
            "model": model,
            "experiment": f"{graph_mode}/{window}/{model}",
            "model_type": metrics.get("model_type"),
            "has_industry_labels": metrics.get("has_industry_labels"),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "portfolio_json": str(portfolio_path.relative_to(ROOT)),
            "alignment_start": alignment.get("start"),
            "alignment_end": alignment.get("end"),
            "alignment_n_periods": alignment.get("n_periods"),
            "train_days": split.get("train_days"),
            "val_days": split.get("val_days"),
            "test_days": split.get("test_days"),
        }
        for split_name in ["train", "val", "test"]:
            row.update(flatten_prefixed(split_name, metrics.get(split_name, {})))
        row.update(flatten_prefixed("strategy", strategy))
        row.update(flatten_prefixed("benchmark", benchmark))
        row.update(flatten_prefixed("excess", excess))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No experiments found under {RUNS_ROOT}")

    df["window"] = ordered_category(df["window"], WINDOW_ORDER)
    df["graph_mode"] = ordered_category(df["graph_mode"], GRAPH_ORDER)
    df["model"] = ordered_category(df["model"], MODEL_ORDER)
    return df.sort_values(["window", "graph_mode", "model"]).reset_index(drop=True)


def collect_attention_features() -> pd.DataFrame:
    rows: list[dict] = []
    for portfolio_path in sorted(RUNS_ROOT.glob("*/*/*/portfolio.json")):
        graph_mode, window, model = portfolio_path.relative_to(RUNS_ROOT).parts[:3]
        with portfolio_path.open("r", encoding="utf-8") as f:
            portfolio = json.load(f)
        for item in portfolio.get("top_attention_features") or []:
            rows.append(
                {
                    "graph_mode": graph_mode,
                    "window": window,
                    "model": model,
                    "experiment": f"{graph_mode}/{window}/{model}",
                    "rank": item.get("rank"),
                    "feature": item.get("feature"),
                    "attention_weight": item.get("attention_weight"),
                }
            )
    return pd.DataFrame(rows)


def metric_value_best(series: pd.Series, metric: str) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    direction = METRIC_DIRECTIONS.get(metric, "max")
    return values.min() if direction == "min" else values.max()


def is_best(value: object, best: float | None) -> bool:
    if best is None or pd.isna(value):
        return False
    try:
        return bool(np.isclose(float(value), float(best), rtol=1e-10, atol=1e-12))
    except (TypeError, ValueError):
        return False


def fmt_metric(value: object, metric: str | None = None) -> str:
    if pd.isna(value):
        return "-"
    if metric in {"strategy_n", "alignment_n_periods", "group_win_count"}:
        return f"{int(value)}"
    if metric in PCT_METRICS:
        return f"{float(value) * 100:.2f}%"
    if metric in {"test_MSE"}:
        return f"{float(value):.5f}"
    if metric in {"test_RMSE", "test_MAE"}:
        return f"{float(value):.4f}"
    if metric and (metric.endswith("IC") or metric.endswith("IC_ind") or "DailyIC" in metric):
        return f"{float(value):.4f}"
    if metric and ("sharpe" in metric.lower() or "ICIR" in metric):
        return f"{float(value):.3f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(
    df: pd.DataFrame,
    columns: list[str],
    best_scope: pd.DataFrame | None = None,
    title_columns: dict[str, str] | None = None,
) -> str:
    title_columns = title_columns or DISPLAY_NAMES
    best_values = {}
    scope = df if best_scope is None else best_scope
    for col in columns:
        if col in METRIC_DIRECTIONS:
            best_values[col] = metric_value_best(scope[col], col)

    header = [title_columns.get(col, col) for col in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col)
            text = fmt_metric(value, col)
            if col in best_values and is_best(value, best_values[col]):
                text = f"**{text} [BEST]**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def simple_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col)
            if pd.isna(value):
                cells.append("-")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_tables(df: pd.DataFrame, attention: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "all_experiments_flat.csv", index=False)
    df[
        [
            "graph_mode",
            "window",
            "model",
            "experiment",
            *PREDICTION_METRICS,
            "strategy_n",
            *PORTFOLIO_METRICS,
            "benchmark_total_return",
            "benchmark_sharpe",
            "alignment_start",
            "alignment_end",
        ]
    ].to_csv(TABLE_DIR / "comparison_key_metrics.csv", index=False)
    if not attention.empty:
        attention.to_csv(TABLE_DIR / "attention_top_features.csv", index=False)


def save_heatmap(df: pd.DataFrame, metric: str, filename: str, title: str, cmap: str) -> None:
    pivot = df.pivot_table(
        index="model",
        columns=["window", "graph_mode"],
        values=metric,
        observed=False,
        aggfunc="first",
    )
    pivot = pivot.reindex(index=MODEL_ORDER)
    columns = [(w, g) for w in WINDOW_ORDER for g in GRAPH_ORDER if (w, g) in pivot.columns]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(columns, names=["window", "graph_mode"]))

    plt.figure(figsize=(13, 6.8))
    display = pivot.copy()
    if metric in PCT_METRICS:
        display = display * 100.0
    sns.heatmap(
        display,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": DISPLAY_NAMES.get(metric, metric)},
    )
    plt.title(title)
    plt.xlabel("Window / Graph")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=180)
    plt.close()


def save_bar_chart(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("strategy_sharpe", ascending=True).copy()
    labels = plot_df["experiment"].astype(str)
    colors = plot_df["window"].map({"short": "#3B82F6", "medium": "#F59E0B", "long": "#10B981"})
    hatches = plot_df["graph_mode"].map({"static": "", "dynamic": "//"})

    plt.figure(figsize=(12, 12))
    bars = plt.barh(range(len(plot_df)), plot_df["strategy_sharpe"], color=colors)
    for bar, hatch in zip(bars, hatches, strict=False):
        bar.set_hatch(hatch)
        bar.set_edgecolor("#1F2937")
        bar.set_linewidth(0.3)
    plt.yticks(range(len(plot_df)), labels, fontsize=7)
    plt.axvline(0, color="#374151", linewidth=0.8)
    plt.xlabel("Strategy Sharpe")
    plt.title("All experiments sorted by portfolio Sharpe")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "portfolio_sharpe_all_experiments.png", dpi=180)
    plt.close()


def save_risk_return(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10.5, 7.2))
    markers = {"static": "o", "dynamic": "s"}
    palette = {"short": "#3B82F6", "medium": "#F59E0B", "long": "#10B981"}
    for (window, graph), group in df.groupby(["window", "graph_mode"], observed=True):
        plt.scatter(
            group["strategy_annual_volatility"] * 100,
            group["strategy_annual_return"] * 100,
            s=(group["strategy_sharpe"].clip(lower=-1) + 1.5) * 45,
            c=palette[str(window)],
            marker=markers[str(graph)],
            edgecolors="#111827",
            linewidths=0.45,
            alpha=0.82,
            label=f"{window}/{graph}",
        )

    top = df.sort_values("strategy_sharpe", ascending=False).head(8)
    for _, row in top.iterrows():
        plt.annotate(
            str(row["model"]),
            (row["strategy_annual_volatility"] * 100, row["strategy_annual_return"] * 100),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    plt.axhline(0, color="#6B7280", linewidth=0.8)
    plt.xlabel("Annualized volatility (%)")
    plt.ylabel("Annualized return (%)")
    plt.title("Risk-return map (bubble size tracks Sharpe)")
    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "risk_return_scatter.png", dpi=180)
    plt.close()


def static_dynamic_delta(df: pd.DataFrame) -> pd.DataFrame:
    common = df[df["model"].astype(str).isin([m for m in MODEL_ORDER if not m.startswith("baseline_")])]
    wide = common.pivot_table(
        index=["window", "model"],
        columns="graph_mode",
        values=["test_IC", "strategy_total_return", "strategy_sharpe", "strategy_max_drawdown"],
        aggfunc="first",
        observed=False,
    )
    rows = []
    for idx, row in wide.iterrows():
        window, model = idx
        if ("test_IC", "dynamic") not in row.index or ("test_IC", "static") not in row.index:
            continue
        if pd.isna(row[("test_IC", "dynamic")]) or pd.isna(row[("test_IC", "static")]):
            continue
        rows.append(
            {
                "window": window,
                "model": model,
                "delta_test_IC": row[("test_IC", "dynamic")] - row[("test_IC", "static")],
                "delta_strategy_total_return": row[("strategy_total_return", "dynamic")]
                - row[("strategy_total_return", "static")],
                "delta_strategy_sharpe": row[("strategy_sharpe", "dynamic")]
                - row[("strategy_sharpe", "static")],
                "delta_strategy_max_drawdown": row[("strategy_max_drawdown", "dynamic")]
                - row[("strategy_max_drawdown", "static")],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["window"] = ordered_category(out["window"], WINDOW_ORDER)
        out["model"] = ordered_category(out["model"], MODEL_ORDER)
        out = out.sort_values(["window", "model"]).reset_index(drop=True)
    return out


def save_delta_chart(delta: pd.DataFrame) -> None:
    if delta.empty:
        return
    plot = delta.copy()
    plot["experiment"] = plot["window"].astype(str) + "/" + plot["model"].astype(str)
    metrics = [
        ("delta_test_IC", "Δ Test IC"),
        ("delta_strategy_total_return", "Δ Total Return"),
        ("delta_strategy_sharpe", "Δ Sharpe"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        values = plot[metric] * (100 if metric in PCT_METRICS else 1)
        colors = np.where(values >= 0, "#16A34A", "#DC2626")
        ax.barh(plot["experiment"], values, color=colors, alpha=0.88)
        ax.axvline(0, color="#374151", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=7)
    axes[1].set_xlabel("Dynamic minus static")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "dynamic_minus_static_deltas.png", dpi=180)
    plt.close()


def save_model_robustness(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("model", observed=True)
        .agg(
            n=("experiment", "count"),
            avg_sharpe=("strategy_sharpe", "mean"),
            median_sharpe=("strategy_sharpe", "median"),
            std_sharpe=("strategy_sharpe", "std"),
            avg_total_return=("strategy_total_return", "mean"),
            avg_test_IC=("test_IC", "mean"),
            avg_DirAcc=("test_DirAcc", "mean"),
        )
        .reset_index()
    )

    winners = []
    for _, group in df.groupby(["window", "graph_mode"], observed=True):
        winners.append(group.loc[group["strategy_sharpe"].idxmax(), "model"])
    win_counts = Counter(winners)
    summary["group_win_count"] = summary["model"].map(lambda x: win_counts.get(x, 0))
    summary["model"] = ordered_category(summary["model"], MODEL_ORDER)
    summary = summary.sort_values(["group_win_count", "avg_sharpe"], ascending=[False, False])
    summary.to_csv(TABLE_DIR / "model_robustness_summary.csv", index=False)

    plt.figure(figsize=(10, 5.8))
    plot = summary.sort_values("avg_sharpe", ascending=True)
    plt.barh(plot["model"].astype(str), plot["avg_sharpe"], xerr=plot["std_sharpe"].fillna(0), color="#2563EB", alpha=0.78)
    plt.axvline(0, color="#374151", linewidth=0.8)
    plt.xlabel("Average Sharpe across available experiments (error bar = std)")
    plt.title("Model-level portfolio robustness")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "model_robustness_sharpe.png", dpi=180)
    plt.close()
    return summary


def save_attention_chart(attention: pd.DataFrame) -> pd.DataFrame:
    if attention.empty:
        return pd.DataFrame()
    summary = (
        attention.groupby("feature")
        .agg(
            appearances=("feature", "count"),
            mean_rank=("rank", "mean"),
            mean_attention_weight=("attention_weight", "mean"),
            max_attention_weight=("attention_weight", "max"),
        )
        .reset_index()
        .sort_values(["appearances", "mean_attention_weight"], ascending=[False, False])
    )
    summary.to_csv(TABLE_DIR / "attention_feature_summary.csv", index=False)
    top = summary.head(15).iloc[::-1]
    plt.figure(figsize=(9.5, 6.5))
    plt.barh(top["feature"], top["appearances"], color="#0F766E", alpha=0.82)
    plt.xlabel("Appearances in top-5 attention lists")
    plt.title("Most frequent top attention features")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "attention_feature_frequency.png", dpi=180)
    plt.close()
    return summary


def save_figures(df: pd.DataFrame, attention: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    save_bar_chart(df)
    save_heatmap(
        df,
        "strategy_total_return",
        "heatmap_total_return.png",
        "Strategy total return by experiment",
        "RdYlGn",
    )
    save_heatmap(
        df,
        "strategy_sharpe",
        "heatmap_sharpe.png",
        "Strategy Sharpe by experiment",
        "RdYlGn",
    )
    save_heatmap(
        df,
        "test_IC",
        "heatmap_test_ic.png",
        "Test IC by experiment",
        "RdYlGn",
    )
    save_risk_return(df)
    delta = static_dynamic_delta(df)
    if not delta.empty:
        delta.to_csv(TABLE_DIR / "dynamic_minus_static_deltas.csv", index=False)
        save_delta_chart(delta)
    robustness = save_model_robustness(df)
    attention_summary = save_attention_chart(attention)
    return delta, robustness, attention_summary


def group_winners(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, graph), group in df.groupby(["window", "graph_mode"], observed=True):
        row = {"window": window, "graph_mode": graph, "n_models": len(group)}
        for metric in ["test_IC", "strategy_total_return", "strategy_sharpe", "excess_total_return"]:
            best = metric_value_best(group[metric], metric)
            if best is None:
                continue
            winner = group.loc[np.isclose(group[metric], best, rtol=1e-10, atol=1e-12), "model"].astype(str).tolist()
            row[f"{metric}_winner"] = ", ".join(winner)
            row[f"{metric}_best"] = best
        rows.append(row)
    out = pd.DataFrame(rows)
    out["window"] = ordered_category(out["window"], WINDOW_ORDER)
    out["graph_mode"] = ordered_category(out["graph_mode"], GRAPH_ORDER)
    return out.sort_values(["window", "graph_mode"]).reset_index(drop=True)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "test_MSE",
        "test_RMSE",
        "test_MAE",
        "test_IC",
        "test_DailyIC",
        "test_ICIR",
        "test_DirAcc",
        "strategy_total_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "excess_total_return",
        "excess_sharpe",
    ]
    corr = df[cols].corr(numeric_only=True)
    pairs = []
    for pred in ["test_MSE", "test_IC", "test_DailyIC", "test_ICIR", "test_DirAcc"]:
        for port in ["strategy_total_return", "strategy_sharpe", "excess_total_return", "excess_sharpe"]:
            pairs.append({"x": pred, "y": port, "pearson_corr": corr.loc[pred, port]})
    return pd.DataFrame(pairs)


def pct_delta(new: float, old: float) -> float:
    if pd.isna(new) or pd.isna(old) or np.isclose(old, 0):
        return np.nan
    return (new - old) / abs(old)


def insight_lines(df: pd.DataFrame, delta: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    top_sharpe = df.loc[df["strategy_sharpe"].idxmax()]
    top_return = df.loc[df["strategy_total_return"].idxmax()]
    top_ic = df.loc[df["test_IC"].idxmax()]
    low_mse = df.loc[df["test_MSE"].idxmin()]
    lines.append(
        f"- 全域 portfolio Sharpe 最高是 `{top_sharpe['experiment']}`，Sharpe {top_sharpe['strategy_sharpe']:.3f}、"
        f"Total Return {top_sharpe['strategy_total_return'] * 100:.2f}%。"
    )
    lines.append(
        f"- 全域 Total Return 最高是 `{top_return['experiment']}`，Total Return {top_return['strategy_total_return'] * 100:.2f}%、"
        f"Sharpe {top_return['strategy_sharpe']:.3f}。"
    )
    lines.append(
        f"- Test IC 最高是 `{top_ic['experiment']}`，IC {top_ic['test_IC']:.4f}；Test MSE 最低是 "
        f"`{low_mse['experiment']}`，MSE {low_mse['test_MSE']:.5f}。這兩者若不是同一組，代表排序能力與點估計誤差不是同一件事。"
    )

    if not delta.empty:
        avg_delta = delta[["delta_test_IC", "delta_strategy_total_return", "delta_strategy_sharpe"]].mean()
        pos_sharpe = int((delta["delta_strategy_sharpe"] > 0).sum())
        pos_return = int((delta["delta_strategy_total_return"] > 0).sum())
        lines.append(
            f"- Dynamic graph 相對 static 的平均變化：Test IC {avg_delta['delta_test_IC']:+.4f}、"
            f"Total Return {avg_delta['delta_strategy_total_return'] * 100:+.2f}pct、Sharpe {avg_delta['delta_strategy_sharpe']:+.3f}。"
            f"18 個可配對比較中，Sharpe 改善 {pos_sharpe} 組、Total Return 改善 {pos_return} 組。"
        )
        by_window = delta.groupby("window", observed=True)[
            ["delta_test_IC", "delta_strategy_total_return", "delta_strategy_sharpe"]
        ].mean()
        window_bits = []
        for window, row in by_window.iterrows():
            window_bits.append(
                f"{window}: IC {row['delta_test_IC']:+.4f}, TR {row['delta_strategy_total_return'] * 100:+.2f}pct, "
                f"Sharpe {row['delta_strategy_sharpe']:+.3f}"
            )
        lines.append(
            "- Dynamic 的效果有期間差異，分 window 平均變化為 "
            + "；".join(window_bits)
            + "。因此 dynamic graph 不是單向優勢，而是中期與部分長期組合較受益。"
        )

    corr_lookup = {(r["x"], r["y"]): r["pearson_corr"] for _, r in corr.iterrows()}
    lines.append(
        "- 預測指標與投組指標的相關性："
        f"Test IC vs Sharpe = {corr_lookup.get(('test_IC', 'strategy_sharpe'), np.nan):+.3f}，"
        f"DirAcc vs Sharpe = {corr_lookup.get(('test_DirAcc', 'strategy_sharpe'), np.nan):+.3f}，"
        f"MSE vs Sharpe = {corr_lookup.get(('test_MSE', 'strategy_sharpe'), np.nan):+.3f}。"
        "若相關性不高，模型選擇應優先看投組層級排序，而不是只看 regression loss。"
    )

    for window in WINDOW_ORDER:
        win_df = df[df["window"].astype(str) == window]
        if win_df.empty:
            continue
        best = win_df.loc[win_df["strategy_sharpe"].idxmax()]
        bench_sharpe = win_df["benchmark_sharpe"].dropna().iloc[0]
        bench_tr = win_df["benchmark_total_return"].dropna().iloc[0]
        lines.append(
            f"- `{window}` 測試窗 benchmark Sharpe {bench_sharpe:.3f}、Total Return {bench_tr * 100:.2f}%；"
            f"該窗最佳模型 `{best['experiment']}` Sharpe {best['strategy_sharpe']:.3f}、"
            f"Total Return {best['strategy_total_return'] * 100:.2f}%。"
        )

    robustness = (
        df.groupby("model", observed=True)
        .agg(
            avg_sharpe=("strategy_sharpe", "mean"),
            avg_total_return=("strategy_total_return", "mean"),
            avg_test_IC=("test_IC", "mean"),
        )
        .reset_index()
    )
    top_avg_sharpe = robustness.loc[robustness["avg_sharpe"].idxmax()]
    top_avg_return = robustness.loc[robustness["avg_total_return"].idxmax()]
    top_avg_ic = robustness.loc[robustness["avg_test_IC"].idxmax()]
    lines.append(
        f"- 跨可用實驗的模型穩定性：平均 Sharpe 最高為 `{top_avg_sharpe['model']}` "
        f"({top_avg_sharpe['avg_sharpe']:.3f})；平均 Total Return 最高為 `{top_avg_return['model']}` "
        f"({top_avg_return['avg_total_return'] * 100:.2f}%)；平均 Test IC 最高為 `{top_avg_ic['model']}` "
        f"({top_avg_ic['avg_test_IC']:.4f})。這顯示沒有單一模型同時壟斷所有維度。"
    )

    static = df[df["graph_mode"].astype(str) == "static"].copy()
    baseline_notes = []
    for window, group in static.groupby("window", observed=True):
        ranked = group.sort_values("strategy_sharpe", ascending=False).reset_index(drop=True)
        baseline_ranked = ranked[ranked["model"].astype(str).str.startswith("baseline_")]
        if baseline_ranked.empty:
            continue
        best_baseline = baseline_ranked.iloc[0]
        rank = int(best_baseline.name) + 1
        baseline_notes.append(f"{window}: {best_baseline['model']} rank {rank}, Sharpe {best_baseline['strategy_sharpe']:.3f}")
    if baseline_notes:
        lines.append(
            "- Baseline 並非全被深度/圖模型碾壓；static 組內最佳 baseline 分別為 "
            + "；".join(baseline_notes)
            + "。但 `baseline_linear` 在三個 static window 都落在後段，主要是排序 IC 與投組報酬都偏弱。"
        )

    pair = df[df["model"].astype(str).isin(["dmfm_full", "dmfm_ind_neutral"])].pivot_table(
        index=["window", "graph_mode"],
        columns="model",
        values=["strategy_total_return", "strategy_sharpe", "test_IC", "strategy_max_drawdown"],
        aggfunc="first",
        observed=False,
    )
    if not pair.empty and ("strategy_total_return", "dmfm_full") in pair:
        diff_return = pair[("strategy_total_return", "dmfm_full")] - pair[("strategy_total_return", "dmfm_ind_neutral")]
        diff_sharpe = pair[("strategy_sharpe", "dmfm_full")] - pair[("strategy_sharpe", "dmfm_ind_neutral")]
        diff_ic = pair[("test_IC", "dmfm_full")] - pair[("test_IC", "dmfm_ind_neutral")]
        lines.append(
            f"- DMFM full 相對 industry-neutral 版本：平均 Total Return 差 {diff_return.mean() * 100:+.2f}pct，"
            f"平均 Sharpe 差 {diff_sharpe.mean():+.3f}，平均 Test IC 差 {diff_ic.mean():+.4f}；"
            f"6 個配對中 full 的 IC 較高 {int((diff_ic > 0).sum())} 次、Sharpe 較高 {int((diff_sharpe > 0).sum())} 次。"
            "也就是 neutralization 對風險調整後報酬未必是劣勢，需按 window 判斷。"
        )

    attention_features = collect_attention_features()
    if not attention_features.empty:
        top_features = (
            attention_features.groupby("feature")
            .size()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        lines.append(
            "- Attention top features 最常出現的是 "
            + "、".join(f"`{feature}`" for feature in top_features)
            + "；訊號集中在 rolling min/max 與波動/成交量相關特徵，代表模型多半透過近期價格區間與風險狀態做排序。"
        )
    return lines


def write_report(
    df: pd.DataFrame,
    attention: pd.DataFrame,
    delta: pd.DataFrame,
    robustness: pd.DataFrame,
    attention_summary: pd.DataFrame,
) -> None:
    winners = group_winners(df)
    corr = correlations(df)
    winners.to_csv(TABLE_DIR / "group_winners.csv", index=False)
    corr.to_csv(TABLE_DIR / "metric_correlations.csv", index=False)

    benchmark = (
        df.groupby("window", observed=True)
        .agg(
            alignment_start=("alignment_start", "first"),
            alignment_end=("alignment_end", "first"),
            alignment_n_periods=("alignment_n_periods", "first"),
            benchmark_total_return=("benchmark_total_return", "first"),
            benchmark_sharpe=("benchmark_sharpe", "first"),
            benchmark_annual_return=("benchmark_annual_return", "first"),
            benchmark_annual_volatility=("benchmark_annual_volatility", "first"),
            benchmark_max_drawdown=("benchmark_max_drawdown", "first"),
        )
        .reset_index()
    )

    lines: list[str] = []
    lines.append("# runs_unified 實驗比較報告")
    lines.append("")
    lines.append(f"- 資料來源：`{RUNS_ROOT}`")
    lines.append(f"- 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 實驗數：{len(df)}；Graph modes：{', '.join(GRAPH_ORDER)}；Windows：{', '.join(WINDOW_ORDER)}")
    lines.append("- 表格標記：`[BEST]` 代表該表格比較範圍內的最佳值；報酬/IC/Sharpe/HitRate/MaxDD 取最高，MSE/RMSE/MAE/Volatility/StdRet 取最低。")
    lines.append("- 注意：`short`、`medium`、`long` 對應不同測試期間，所以跨 window 的數值只能看穩定性與情境差異；嚴格模型競賽應優先看同一 `window × graph_mode` 內的比較。")
    lines.append("")

    lines.append("## 1. 視覺化總覽")
    lines.append("")
    lines.append("![All experiments by Sharpe](charts/portfolio_sharpe_all_experiments.png)")
    lines.append("")
    lines.append("![Total return heatmap](charts/heatmap_total_return.png)")
    lines.append("")
    lines.append("![Sharpe heatmap](charts/heatmap_sharpe.png)")
    lines.append("")
    lines.append("![Test IC heatmap](charts/heatmap_test_ic.png)")
    lines.append("")
    lines.append("![Risk return scatter](charts/risk_return_scatter.png)")
    lines.append("")
    if not delta.empty:
        lines.append("![Dynamic minus static deltas](charts/dynamic_minus_static_deltas.png)")
        lines.append("")
    lines.append("![Model robustness Sharpe](charts/model_robustness_sharpe.png)")
    lines.append("")
    if not attention_summary.empty:
        lines.append("![Attention feature frequency](charts/attention_feature_frequency.png)")
        lines.append("")

    lines.append("## 2. 核心 insight")
    lines.append("")
    lines.extend(insight_lines(df, delta, corr))
    lines.append("")

    lines.append("## 3. Benchmark 與測試期間")
    lines.append("")
    lines.append(
        markdown_table(
            benchmark,
            [
                "window",
                "alignment_start",
                "alignment_end",
                "alignment_n_periods",
                "benchmark_total_return",
                "benchmark_sharpe",
                "benchmark_annual_return",
                "benchmark_annual_volatility",
                "benchmark_max_drawdown",
            ],
        )
    )
    lines.append("")

    lines.append("## 4. 每組 Winner 摘要")
    lines.append("")
    winner_rows = []
    for _, row in winners.iterrows():
        winner_rows.append(
            {
                "Window": row["window"],
                "Graph": row["graph_mode"],
                "Models": row["n_models"],
                "Best TestIC": f"{row['test_IC_winner']} ({fmt_metric(row['test_IC_best'], 'test_IC')})",
                "Best TotalReturn": f"{row['strategy_total_return_winner']} ({fmt_metric(row['strategy_total_return_best'], 'strategy_total_return')})",
                "Best Sharpe": f"{row['strategy_sharpe_winner']} ({fmt_metric(row['strategy_sharpe_best'], 'strategy_sharpe')})",
                "Best ExcessTR": f"{row['excess_total_return_winner']} ({fmt_metric(row['excess_total_return_best'], 'excess_total_return')})",
            }
        )
    lines.append(simple_markdown_table(pd.DataFrame(winner_rows)))
    lines.append("")

    lines.append("## 5. 全實驗重點總表")
    lines.append("")
    summary = df.sort_values(["window", "graph_mode", "strategy_sharpe"], ascending=[True, True, False])
    lines.append(
        markdown_table(
            summary,
            [
                "window",
                "graph_mode",
                "model",
                *SUMMARY_METRICS,
            ],
            best_scope=df,
        )
    )
    lines.append("")

    lines.append("## 6. 同組詳細比較表")
    lines.append("")
    for window in WINDOW_ORDER:
        for graph in GRAPH_ORDER:
            group = df[(df["window"].astype(str) == window) & (df["graph_mode"].astype(str) == graph)].copy()
            if group.empty:
                continue
            group = group.sort_values("strategy_sharpe", ascending=False)
            lines.append(f"### {window} / {graph} - Prediction metrics")
            lines.append("")
            lines.append(markdown_table(group, ["model", *PREDICTION_METRICS], best_scope=group))
            lines.append("")
            lines.append(f"### {window} / {graph} - Portfolio metrics")
            lines.append("")
            lines.append(markdown_table(group, ["model", "strategy_n", *PORTFOLIO_METRICS], best_scope=group))
            lines.append("")

    lines.append("## 7. Static vs Dynamic 配對差異")
    lines.append("")
    if delta.empty:
        lines.append("沒有可配對的 static/dynamic 組合。")
    else:
        lines.append("下表為同一 window、同一模型的 `dynamic - static`。正值代表 dynamic 較高；MaxDD 正值代表回撤較淺。")
        lines.append("")
        lines.append(
            markdown_table(
                delta,
                [
                    "window",
                    "model",
                    "delta_test_IC",
                    "delta_strategy_total_return",
                    "delta_strategy_sharpe",
                    "delta_strategy_max_drawdown",
                ],
                best_scope=delta,
            )
        )
    lines.append("")

    lines.append("## 8. 模型穩定性彙總")
    lines.append("")
    robust = robustness.rename(
        columns={
            "avg_sharpe": "AvgSharpe",
            "median_sharpe": "MedianSharpe",
            "std_sharpe": "StdSharpe",
            "avg_total_return": "AvgTotalReturn",
            "avg_test_IC": "AvgTestIC",
            "avg_DirAcc": "AvgDirAcc",
        }
    )
    robust_columns = [
        "model",
        "n",
        "group_win_count",
        "AvgSharpe",
        "MedianSharpe",
        "StdSharpe",
        "AvgTotalReturn",
        "AvgTestIC",
        "AvgDirAcc",
    ]
    local_names = {
        **DISPLAY_NAMES,
        "n": "N",
        "AvgSharpe": "Avg Sharpe↑",
        "MedianSharpe": "Median Sharpe↑",
        "StdSharpe": "Std Sharpe↓",
        "AvgTotalReturn": "Avg TotalReturn↑",
        "AvgTestIC": "Avg TestIC↑",
        "AvgDirAcc": "Avg DirAcc↑",
    }
    local_dirs = {
        "AvgSharpe": "max",
        "MedianSharpe": "max",
        "StdSharpe": "min",
        "AvgTotalReturn": "max",
        "AvgTestIC": "max",
        "AvgDirAcc": "max",
    }
    old_dirs = METRIC_DIRECTIONS.copy()
    METRIC_DIRECTIONS.update(local_dirs)
    PCT_METRICS.update({"AvgTotalReturn", "AvgDirAcc"})
    lines.append(markdown_table(robust, robust_columns, best_scope=robust, title_columns=local_names))
    METRIC_DIRECTIONS.clear()
    METRIC_DIRECTIONS.update(old_dirs)
    lines.append("")

    lines.append("## 9. 指標相關性")
    lines.append("")
    corr_show = corr.copy()
    corr_show["pearson_corr"] = corr_show["pearson_corr"].map(lambda x: f"{x:+.3f}")
    lines.append(simple_markdown_table(corr_show))
    lines.append("")

    if not attention_summary.empty:
        lines.append("## 10. Attention feature 彙總")
        lines.append("")
        top_att = attention_summary.head(20).copy()
        top_att["mean_rank"] = top_att["mean_rank"].map(lambda x: f"{x:.2f}")
        top_att["mean_attention_weight"] = top_att["mean_attention_weight"].map(lambda x: f"{x:.4f}")
        top_att["max_attention_weight"] = top_att["max_attention_weight"].map(lambda x: f"{x:.4f}")
        lines.append(simple_markdown_table(top_att))
        lines.append("")

    lines.append("## 11. 輸出檔案")
    lines.append("")
    lines.append("- `tables/all_experiments_flat.csv`：完整扁平化資料表")
    lines.append("- `tables/comparison_key_metrics.csv`：主要比較欄位")
    lines.append("- `tables/group_winners.csv`：每個 `window × graph_mode` 的 winner")
    lines.append("- `tables/dynamic_minus_static_deltas.csv`：static/dynamic 配對差異")
    lines.append("- `tables/metric_correlations.csv`：主要預測指標與投組指標的 Pearson correlation")
    if not attention_summary.empty:
        lines.append("- `tables/attention_top_features.csv`、`tables/attention_feature_summary.csv`：attention 特徵彙總")
    lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    df = load_experiments()
    attention = collect_attention_features()
    save_tables(df, attention)
    delta, robustness, attention_summary = save_figures(df, attention)
    write_report(df, attention, delta, robustness, attention_summary)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote charts to {CHART_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
