# -*- coding: utf-8 -*-
"""
Unified experiment runner for baseline / GAT / DMFM.

Goals:
1) Run each model family independently across short/medium/long windows.
2) Run everything together with one command.
3) Support smoke mode for local checks and full mode for remote GPU runs.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


WINDOW_SPECS: Dict[str, Dict[str, str]] = {
    "short": {"start": "2019-09-16", "end": "2020-12-31"},
    "medium": {"start": "2019-09-16", "end": "2022-12-31"},
    "long": {"start": "2019-09-16", "end": "2025-09-12"},
}


@dataclass
class GATConfig:
    epochs: int
    lr: float
    patience: int
    hid: int = 64
    heads: int = 2
    loss: str = "corr_mse_ind"
    alpha_mse: float = 0.03
    lambda_var: float = 0.1


@dataclass
class DMFMConfig:
    epochs: int
    lr: float
    patience: int
    hidden_dim: int = 64
    heads: int = 2
    dropout: float = 0.1
    lambda_attn: float = 0.1
    lambda_ic: float = 1.0


@dataclass
class BaselineConfig:
    lstm_epochs: int
    lstm_lookback: int
    lstm_batch_size: int
    xgb_estimators: int
    xgb_max_depth: int
    xgb_lr: float


SMOKE_GAT = GATConfig(epochs=2, lr=1e-3, patience=2)
SMOKE_DMFM = DMFMConfig(epochs=2, lr=1e-4, patience=2)
SMOKE_BASELINE = BaselineConfig(
    lstm_epochs=2,
    lstm_lookback=5,
    lstm_batch_size=256,
    xgb_estimators=20,
    xgb_max_depth=3,
    xgb_lr=0.1,
)

FULL_GAT_BY_WINDOW = {
    "short": GATConfig(epochs=30, lr=1e-3, patience=10),
    "medium": GATConfig(epochs=50, lr=1e-3, patience=15),
    "long": GATConfig(epochs=50, lr=1e-3, patience=15),
}
FULL_DMFM_BY_WINDOW = {
    "short": DMFMConfig(epochs=50, lr=1e-4, patience=20),
    "medium": DMFMConfig(epochs=100, lr=1e-4, patience=30),
    "long": DMFMConfig(epochs=100, lr=1e-4, patience=30),
}
FULL_BASELINE = BaselineConfig(
    lstm_epochs=30,
    lstm_lookback=10,
    lstm_batch_size=512,
    xgb_estimators=300,
    xgb_max_depth=6,
    xgb_lr=0.05,
)


def parse_csv_list(raw: str, allowed: Sequence[str], allow_all: bool = True) -> List[str]:
    parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty selection")
    if allow_all and "all" in parts:
        return list(allowed)
    invalid = [x for x in parts if x not in allowed]
    if invalid:
        raise ValueError(f"invalid choices: {invalid}, allowed={list(allowed)}")
    # preserve order, remove duplicates
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool = False):
    cmd_str = " ".join(cmd)
    print(f"\n$ {cmd_str}")
    print(f"  log -> {log_path}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}\nSee log: {log_path}")


def ensure_artifact(
    py: str,
    window: str,
    spec: Dict[str, str],
    args,
    artifact_dir: Path,
    summary: Dict[str, object],
):
    ft_path = artifact_dir / "Ft_tensor.pt"
    if args.skip_build:
        if not ft_path.exists() and args.dry_run:
            print(f"[skip-build][dry-run] artifact not found yet (expected in real run): {artifact_dir}")
            return
        if not ft_path.exists():
            raise FileNotFoundError(f"--skip-build used but missing artifacts: {artifact_dir}")
        print(f"[skip-build] use existing: {artifact_dir}")
        return

    if ft_path.exists() and not args.rebuild_artifacts:
        print(f"[build] reuse existing artifacts: {artifact_dir}")
        return

    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_root) / "_logs" / window / "build_artifacts.log"
    cmd = [
        py,
        "build_artifacts.py",
        "--prices",
        args.prices,
        "--industry_csv",
        args.industry_csv,
        "--artifact_dir",
        str(artifact_dir),
        "--start_date",
        spec["start"],
        "--end_date",
        spec["end"],
        "--horizon",
        str(args.horizon),
    ]
    run_cmd(cmd, log_path=log_path, dry_run=args.dry_run)
    summary["artifacts"][window] = str(artifact_dir)


def run_baseline(
    py: str,
    window: str,
    artifact_dir: Path,
    baseline_model: str,
    args,
    summary: Dict[str, object],
):
    out_dir = Path(args.output_root) / window / f"baseline_{baseline_model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = SMOKE_BASELINE if args.mode == "smoke" else FULL_BASELINE

    if not args.skip_train:
        cmd = [
            py,
            "train_baselines.py",
            "--artifact_dir",
            str(artifact_dir),
            "--model",
            baseline_model,
            "--train_ratio",
            str(args.train_ratio),
            "--device",
            args.device,
        ]
        if baseline_model == "xgboost":
            cmd.extend(
                [
                    "--n_estimators",
                    str(base_cfg.xgb_estimators),
                    "--max_depth",
                    str(base_cfg.xgb_max_depth),
                    "--learning_rate",
                    str(base_cfg.xgb_lr),
                ]
            )
        if baseline_model == "lstm":
            cmd.extend(
                [
                    "--lookback",
                    str(base_cfg.lstm_lookback),
                    "--epochs",
                    str(base_cfg.lstm_epochs),
                    "--batch_size",
                    str(base_cfg.lstm_batch_size),
                ]
            )
        run_cmd(cmd, log_path=out_dir / "train.log", dry_run=args.dry_run)
    else:
        print(f"[skip-train] baseline_{baseline_model} {window}")

    cmd_eval = [
        py,
        "evaluate_baseline_portfolio.py",
        "--artifact_dir",
        str(artifact_dir),
        "--model",
        baseline_model,
        "--out_dir",
        str(out_dir / "plots"),
        "--benchmark_csv",
        args.benchmark_csv,
        "--top_pct",
        str(args.top_pct),
        "--rebalance_days",
        str(args.rebalance_days),
        "--device",
        args.device,
        "--train_ratio",
        str(args.train_ratio),
    ]
    run_cmd(cmd_eval, log_path=out_dir / "evaluate.log", dry_run=args.dry_run)

    summary["runs"].append(
        {
            "window": window,
            "model": f"baseline_{baseline_model}",
            "artifact_dir": str(artifact_dir),
            "output_dir": str(out_dir),
        }
    )


def run_gat(py: str, window: str, artifact_dir: Path, args, summary: Dict[str, object]):
    out_dir = Path(args.output_root) / window / "gat"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SMOKE_GAT if args.mode == "smoke" else FULL_GAT_BY_WINDOW[window]

    if not args.skip_train:
        cmd_train = [
            py,
            "train_gat_fixed.py",
            "--artifact_dir",
            str(artifact_dir),
            "--epochs",
            str(cfg.epochs),
            "--lr",
            str(cfg.lr),
            "--device",
            args.device,
            "--loss",
            cfg.loss,
            "--alpha_mse",
            str(cfg.alpha_mse),
            "--lambda_var",
            str(cfg.lambda_var),
            "--hid",
            str(cfg.hid),
            "--heads",
            str(cfg.heads),
            "--patience",
            str(cfg.patience),
            "--industry_csv",
            args.industry_csv,
        ]
        run_cmd(cmd_train, log_path=out_dir / "train.log", dry_run=args.dry_run)
    else:
        print(f"[skip-train] gat {window}")

    gat_weight = artifact_dir / "gat_regressor.pt"
    cmd_metrics = [
        py,
        "evaluate_metrics.py",
        "--artifact_dir",
        str(artifact_dir),
        "--weights",
        str(gat_weight),
        "--device",
        args.device,
        "--industry_csv",
        args.industry_csv,
    ]
    run_cmd(cmd_metrics, log_path=out_dir / "metrics.log", dry_run=args.dry_run)

    cmd_backtest = [
        py,
        "evaluate_portfolio.py",
        "--artifact_dir",
        str(artifact_dir),
        "--weights",
        str(gat_weight),
        "--out_dir",
        str(out_dir / "plots"),
        "--benchmark_csv",
        args.benchmark_csv,
        "--top_pct",
        str(args.top_pct),
        "--rebalance_days",
        str(args.rebalance_days),
        "--device",
        args.device,
        "--industry_csv",
        args.industry_csv,
    ]
    run_cmd(cmd_backtest, log_path=out_dir / "backtest.log", dry_run=args.dry_run)

    summary["runs"].append(
        {
            "window": window,
            "model": "gat",
            "artifact_dir": str(artifact_dir),
            "output_dir": str(out_dir),
        }
    )


def run_dmfm(py: str, window: str, artifact_dir: Path, args, summary: Dict[str, object]):
    out_dir = Path(args.output_root) / window / "dmfm"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SMOKE_DMFM if args.mode == "smoke" else FULL_DMFM_BY_WINDOW[window]

    if not args.skip_train:
        cmd_train = [
            py,
            "train_dmfm_wei2022.py",
            "--artifact_dir",
            str(artifact_dir),
            "--epochs",
            str(cfg.epochs),
            "--lr",
            str(cfg.lr),
            "--device",
            args.device,
            "--hidden_dim",
            str(cfg.hidden_dim),
            "--heads",
            str(cfg.heads),
            "--dropout",
            str(cfg.dropout),
            "--lambda_attn",
            str(cfg.lambda_attn),
            "--lambda_ic",
            str(cfg.lambda_ic),
            "--patience",
            str(cfg.patience),
            "--train_ratio",
            str(args.train_ratio),
        ]
        run_cmd(cmd_train, log_path=out_dir / "train.log", dry_run=args.dry_run)
    else:
        print(f"[skip-train] dmfm {window}")

    dmfm_weight = artifact_dir / "dmfm_wei2022_best.pt"
    cmd_metrics = [
        py,
        "evaluate_metrics.py",
        "--artifact_dir",
        str(artifact_dir),
        "--weights",
        str(dmfm_weight),
        "--device",
        args.device,
        "--industry_csv",
        args.industry_csv,
    ]
    run_cmd(cmd_metrics, log_path=out_dir / "metrics.log", dry_run=args.dry_run)

    cmd_backtest = [
        py,
        "evaluate_portfolio.py",
        "--artifact_dir",
        str(artifact_dir),
        "--weights",
        str(dmfm_weight),
        "--out_dir",
        str(out_dir / "plots"),
        "--benchmark_csv",
        args.benchmark_csv,
        "--top_pct",
        str(args.top_pct),
        "--rebalance_days",
        str(args.rebalance_days),
        "--device",
        args.device,
        "--industry_csv",
        args.industry_csv,
    ]
    run_cmd(cmd_backtest, log_path=out_dir / "backtest.log", dry_run=args.dry_run)

    if args.extra_analysis:
        cmd_attn = [
            py,
            "visualize_factor_attention.py",
            "--artifact_dir",
            str(artifact_dir),
            "--weights",
            str(dmfm_weight),
            "--output_dir",
            str(out_dir / "attention"),
            "--device",
            "cpu" if args.device == "cuda" else args.device,
        ]
        run_cmd(cmd_attn, log_path=out_dir / "attention.log", dry_run=args.dry_run)

        cmd_ctx = [
            py,
            "analyze_contexts.py",
            "--artifact_dir",
            str(artifact_dir),
            "--model_path",
            str(dmfm_weight),
            "--output_dir",
            str(out_dir / "contexts"),
            "--device",
            "cpu",
            "--sample_days",
            str(10 if args.mode == "smoke" else 20),
        ]
        run_cmd(cmd_ctx, log_path=out_dir / "contexts.log", dry_run=args.dry_run)

    summary["runs"].append(
        {
            "window": window,
            "model": "dmfm",
            "artifact_dir": str(artifact_dir),
            "output_dir": str(out_dir),
            "extra_analysis": args.extra_analysis,
        }
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Unified experiment pipeline")
    ap.add_argument(
        "--models",
        default="all",
        help="comma list: baseline,gat,dmfm,all",
    )
    ap.add_argument(
        "--baseline_models",
        default="linear",
        help="comma list when baseline is selected: linear,xgboost,lstm",
    )
    ap.add_argument(
        "--windows",
        default="all",
        help="comma list: short,medium,long,all",
    )
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--prices", default="unique_2019q3to2025q3.csv")
    ap.add_argument("--industry_csv", default="unique_2019q3to2025q3.csv")
    ap.add_argument("--benchmark_csv", default="GAT0050.csv")
    ap.add_argument("--artifact_root", default="artifacts")
    ap.add_argument("--output_root", default="runs")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top_pct", type=float, default=0.10)
    ap.add_argument("--rebalance_days", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--rebuild_artifacts", action="store_true")
    ap.add_argument("--extra_analysis", action="store_true", help="Run DMFM attention/context analysis")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    py = sys.executable
    models = parse_csv_list(args.models, ["baseline", "gat", "dmfm"])
    windows = parse_csv_list(args.windows, ["short", "medium", "long"])
    baseline_models = parse_csv_list(args.baseline_models, ["linear", "xgboost", "lstm"])

    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "models": models,
        "baseline_models": baseline_models,
        "windows": windows,
        "artifacts": {},
        "runs": [],
        "dry_run": args.dry_run,
    }

    print("=" * 72)
    print("Unified Pipeline")
    print("=" * 72)
    print(f"mode={args.mode} models={models} windows={windows} device={args.device}")
    print(f"artifact_root={artifact_root} output_root={output_root}")
    print(f"skip_build={args.skip_build} skip_train={args.skip_train} dry_run={args.dry_run}")

    for window in windows:
        spec = WINDOW_SPECS[window]
        artifact_dir = artifact_root / window
        print(f"\n{'-' * 72}")
        print(f"Window: {window} ({spec['start']} ~ {spec['end']})")
        print(f"{'-' * 72}")

        ensure_artifact(py, window, spec, args, artifact_dir, summary)

        if "baseline" in models:
            for bm in baseline_models:
                run_baseline(py, window, artifact_dir, bm, args, summary)

        if "gat" in models:
            run_gat(py, window, artifact_dir, args, summary)

        if "dmfm" in models:
            run_dmfm(py, window, artifact_dir, args, summary)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path = output_root / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Pipeline finished")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
