# -*- coding: utf-8 -*-
"""
Unified experiment runner for baseline and factor graph ablation models.

Goals:
1) Run each model family independently across short/medium/long windows.
2) Run everything together with one command.
3) Support smoke mode for local checks and full mode for remote GPU runs.
"""

import argparse
import copy
import concurrent.futures
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

from graph_utils import dynamic_graph_files_exist


WINDOW_SPECS: Dict[str, Dict[str, str]] = {
    "short": {"start": "2019-09-16", "end": "2020-12-31"},
    "medium": {"start": "2019-09-16", "end": "2022-12-31"},
    "long": {"start": "2019-09-16", "end": "2025-09-12"},
}

FACTOR_MODELS = [
    "mlp",
    "gat_industry",
    "gat_universe",
    "gat_two_graph_no_neutral",
    "dmfm_ind_neutral",
    "dmfm_full",
]
DEFAULT_MODELS = ["baseline", *FACTOR_MODELS]


@dataclass
class DMFMConfig:
    epochs: int
    lr: float
    patience: int
    hidden_dim: int = 64
    heads: int = 2
    dropout: float = 0.1
    lambda_attn: float = 0.05
    lambda_ic: float = 1.0


@dataclass
class BaselineConfig:
    lstm_epochs: int
    lstm_lookback: int
    lstm_batch_size: int
    xgb_estimators: int
    xgb_max_depth: int
    xgb_lr: float


SMOKE_DMFM = DMFMConfig(epochs=2, lr=1e-4, patience=2)
SMOKE_BASELINE = BaselineConfig(
    lstm_epochs=2,
    lstm_lookback=5,
    lstm_batch_size=256,
    xgb_estimators=20,
    xgb_max_depth=3,
    xgb_lr=0.1,
)

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


def parse_model_list(raw: str) -> List[str]:
    parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty model selection")
    if "all" in parts:
        return list(DEFAULT_MODELS)

    expanded = []
    for part in parts:
        if part == "factor_variants":
            expanded.extend(FACTOR_MODELS)
        else:
            expanded.append(part)

    allowed = ["baseline", "factor_variants", *FACTOR_MODELS]
    invalid = [x for x in expanded if x not in allowed or x == "factor_variants"]
    if invalid:
        raise ValueError(f"invalid models: {invalid}, allowed={allowed + ['all']}")

    seen = set()
    out = []
    for model in expanded:
        if model not in seen:
            seen.add(model)
            out.append(model)
    return out


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool = False) -> Optional[float]:
    cmd_str = " ".join(cmd)
    print(f"\n$ {cmd_str}")
    print(f"  log -> {log_path}")
    if dry_run:
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        print(f"[timing] command_started_at={started_at.isoformat(timespec='seconds')}", file=logf)
        print(f"[timing] command={cmd_str}", file=logf)
        print("-" * 72, file=logf)
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
        duration_sec = time.perf_counter() - t0
        finished_at = datetime.now()
        print("-" * 72, file=logf)
        print(f"[timing] command_finished_at={finished_at.isoformat(timespec='seconds')}", file=logf)
        print(f"[timing] command_duration_sec={duration_sec:.3f}", file=logf)
        print(f"[timing] command_returncode={proc.returncode}", file=logf)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}\nSee log: {log_path}")
    print(f"  finished in {duration_sec:.1f}s")
    return duration_sec


def graph_mode_from_args(args) -> str:
    explicit = getattr(args, "graph_mode", None)
    if explicit:
        return explicit
    return "static" if args.no_dynamic_graphs else "dynamic"


def append_run(summary: Dict[str, object], args, record: Dict[str, object]):
    record.setdefault("graph_mode", graph_mode_from_args(args))
    summary["runs"].append(record)


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
        if not args.no_dynamic_graphs and not dynamic_graph_files_exist(str(artifact_dir)):
            if args.dry_run:
                print(f"[skip-build][dry-run] dynamic graph artifacts not found yet: {artifact_dir}")
            else:
                raise FileNotFoundError(
                    "--skip-build used but dynamic weighted graph artifacts are missing. "
                    "Rebuild artifacts or pass --no_dynamic_graphs for static fallback."
                )
        print(f"[skip-build] use existing: {artifact_dir}")
        return

    if ft_path.exists() and not args.rebuild_artifacts:
        if not args.no_dynamic_graphs and not dynamic_graph_files_exist(str(artifact_dir)):
            print(f"[build] existing artifacts missing dynamic weighted graphs; rebuilding: {artifact_dir}")
        else:
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
    if args.no_dynamic_graphs:
        cmd.append("--no_dynamic_graphs")
    else:
        cmd.extend(
            [
                "--graph_lookback",
                str(args.graph_lookback),
                "--graph_min_obs",
                str(args.graph_min_obs),
                "--industry_top_k",
                str(args.industry_top_k),
                "--universe_top_k",
                str(args.universe_top_k),
            ]
        )
    if args.cache_parquet:
        cmd.append("--cache_parquet")
    run_cmd(cmd, log_path=log_path, dry_run=args.dry_run)
    graph_mode = getattr(args, "graph_mode", None)
    if graph_mode:
        summary["artifacts"].setdefault(graph_mode, {})[window] = str(artifact_dir)
    else:
        summary["artifacts"][window] = str(artifact_dir)


def date_key(value) -> str:
    return str(value)[:10]


def merged_window_spec(windows: Sequence[str]) -> Dict[str, str]:
    specs = [WINDOW_SPECS[w] for w in windows]
    return {
        "start": min(spec["start"] for spec in specs),
        "end": max(spec["end"] for spec in specs),
    }


def copy_or_link(src: Path, dst: Path, dry_run: bool = False):
    if dry_run:
        print(f"[dry-run] link/copy {src} -> {dst}")
        return
    if not src.exists():
        raise FileNotFoundError(f"Expected artifact not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def slice_dynamic_list(src: Path, dst: Path, indices: List[int], dry_run: bool = False):
    if dry_run:
        print(f"[dry-run] slice dynamic graph {src} -> {dst} ({len(indices)} days)")
        return
    data = torch.load(src, map_location="cpu")
    torch.save([data[i] for i in indices], dst)


def ensure_window_view_artifact(
    full_artifact_dir: Path,
    window_artifact_dir: Path,
    window: str,
    spec: Dict[str, str],
    args,
):
    """Create a lightweight per-window artifact by slicing tensors from one full artifact."""
    full_meta_path = full_artifact_dir / "meta.pkl"
    view_meta_path = window_artifact_dir / "meta.pkl"
    ft_path = window_artifact_dir / "Ft_tensor.pt"

    if (
        ft_path.exists()
        and view_meta_path.exists()
        and not args.rebuild_artifacts
        and not args.skip_build
    ):
        print(f"[build] reuse window artifact view: {window_artifact_dir}")
        return

    if args.skip_build:
        if not ft_path.exists():
            raise FileNotFoundError(f"--skip-build used but missing window artifact view: {window_artifact_dir}")
        print(f"[skip-build] use existing window artifact view: {window_artifact_dir}")
        return

    if args.dry_run:
        print(
            f"[dry-run] create window artifact view {window}: "
            f"{spec['start']} ~ {spec['end']} from {full_artifact_dir}"
        )
        return

    if not full_meta_path.exists():
        raise FileNotFoundError(f"Missing full artifact metadata: {full_meta_path}")

    started = time.perf_counter()
    window_artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(full_meta_path, "rb") as f:
        meta = pickle.load(f)

    dates = meta.get("dates", [])
    indices = [
        i
        for i, d in enumerate(dates)
        if spec["start"] <= date_key(d) <= spec["end"]
    ]
    if not indices:
        raise ValueError(
            f"No dates found for window={window} range={spec['start']}~{spec['end']} "
            f"in full artifact {full_artifact_dir}"
        )

    for name in ("Ft_tensor.pt", "yt_tensor.pt", "yraw_tensor.pt"):
        tensor = torch.load(full_artifact_dir / name, map_location="cpu")
        torch.save(tensor[indices], window_artifact_dir / name)

    for name in ("industry_edge_index.pt", "universe_edge_index.pt"):
        copy_or_link(full_artifact_dir / name, window_artifact_dir / name, dry_run=args.dry_run)

    dynamic_names = (
        "dynamic_industry_edge_indices.pt",
        "dynamic_industry_edge_attrs.pt",
        "dynamic_universe_edge_indices.pt",
        "dynamic_universe_edge_attrs.pt",
    )
    has_dynamic = all((full_artifact_dir / name).exists() for name in dynamic_names)
    if has_dynamic:
        for name in dynamic_names:
            slice_dynamic_list(full_artifact_dir / name, window_artifact_dir / name, indices, dry_run=args.dry_run)
        copy_or_link(full_artifact_dir / "dynamic_graph_meta.json", window_artifact_dir / "dynamic_graph_meta.json")
    else:
        for name in (*dynamic_names, "dynamic_graph_meta.json"):
            stale = window_artifact_dir / name
            if stale.exists():
                stale.unlink()

    view_meta = dict(meta)
    view_meta["dates"] = [dates[i] for i in indices]
    view_meta["window"] = window
    view_meta["window_start"] = spec["start"]
    view_meta["window_end"] = spec["end"]
    view_meta["source_artifact_dir"] = str(full_artifact_dir)
    view_meta["shared_artifact_view"] = True
    view_meta["num_days"] = len(indices)
    if not has_dynamic:
        view_meta["dynamic_graphs"] = {"enabled": False}
    with open(view_meta_path, "wb") as f:
        pickle.dump(view_meta, f)

    print(
        f"[build] window artifact view ready: {window_artifact_dir} "
        f"({len(indices)} days, {time.perf_counter() - started:.1f}s)"
    )


def ensure_shared_artifacts(
    py: str,
    windows: Sequence[str],
    graph_mode: Optional[str],
    args,
    summary: Dict[str, object],
) -> Dict[str, Path]:
    current_args = mode_args(args, graph_mode)
    full_spec = merged_window_spec(windows)
    full_dir = artifact_dir_for_mode(args, graph_mode, args.shared_artifact_name)
    ensure_artifact(py, args.shared_artifact_name, full_spec, current_args, full_dir, summary)
    out: Dict[str, Path] = {}
    for window in windows:
        spec = WINDOW_SPECS[window]
        view_dir = artifact_dir_for_mode(args, graph_mode, window)
        ensure_window_view_artifact(full_dir, view_dir, window, spec, current_args)
        out[window] = view_dir
        mode_label = graph_mode_from_args(current_args)
        if graph_mode:
            summary["artifacts"].setdefault(mode_label, {})[window] = str(view_dir)
            summary["artifacts"].setdefault(mode_label, {})[args.shared_artifact_name] = str(full_dir)
        else:
            summary["artifacts"][window] = str(view_dir)
            summary["artifacts"][args.shared_artifact_name] = str(full_dir)
    return out


def copy_json_artifact(src: Path, dst: Path, dry_run: bool = False):
    if dry_run:
        print(f"[dry-run] copy {src} -> {dst}")
        return
    if not src.exists():
        raise FileNotFoundError(f"Expected JSON artifact not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


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
            "--val_ratio",
            str(args.val_ratio),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
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

    baseline_metrics_src = artifact_dir / f"baseline_{baseline_model}_metrics.json"
    baseline_metrics_dst = out_dir / "metrics.json"
    copy_json_artifact(baseline_metrics_src, baseline_metrics_dst, dry_run=args.dry_run)

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
        "--val_ratio",
        str(args.val_ratio),
    ]
    run_cmd(cmd_eval, log_path=out_dir / "backtest.log", dry_run=args.dry_run)

    append_run(
        summary,
        args,
        {
            "window": window,
            "model": f"baseline_{baseline_model}",
            "artifact_dir": str(artifact_dir),
            "output_dir": str(out_dir),
            "metrics_json": str(baseline_metrics_dst),
            "portfolio_json": str(out_dir / "portfolio.json"),
            "plots_dir": str(out_dir / "plots"),
        },
    )


def run_factor_variant(py: str, window: str, artifact_dir: Path, variant: str, args, summary: Dict[str, object]):
    out_dir = Path(args.output_root) / window / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SMOKE_DMFM if args.mode == "smoke" else FULL_DMFM_BY_WINDOW[window]

    if not args.skip_train:
        cmd_train = [
            py,
            "train_factor_variants.py",
            "--artifact_dir",
            str(artifact_dir),
            "--variant",
            variant,
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
            "--lambda_b",
            str(args.lambda_b),
            "--patience",
            str(cfg.patience),
            "--train_ratio",
            str(args.train_ratio),
            "--val_ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ]
        if args.preload_gpu:
            cmd_train.append("--preload_gpu")
        run_cmd(cmd_train, log_path=out_dir / "train.log", dry_run=args.dry_run)
    else:
        print(f"[skip-train] {variant} {window}")

    weight = artifact_dir / f"{variant}_best.pt"
    cmd_metrics = [
        py,
        "evaluate_metrics.py",
        "--artifact_dir",
        str(artifact_dir),
        "--weights",
        str(weight),
        "--device",
        args.device,
        "--industry_csv",
        args.industry_csv,
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--out_json",
        str(out_dir / "metrics.json"),
    ]
    run_cmd(cmd_metrics, log_path=out_dir / "metrics.log", dry_run=args.dry_run)

    cmd_backtest = [
        py,
        "evaluate_portfolio.py",
        "--artifact_dir",
        str(artifact_dir),
        "--weights",
        str(weight),
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
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
    ]
    run_cmd(cmd_backtest, log_path=out_dir / "backtest.log", dry_run=args.dry_run)

    append_run(
        summary,
        args,
        {
            "window": window,
            "model": variant,
            "artifact_dir": str(artifact_dir),
            "output_dir": str(out_dir),
            "metrics_json": str(out_dir / "metrics.json"),
            "portfolio_json": str(out_dir / "portfolio.json"),
            "plots_dir": str(out_dir / "plots"),
        },
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Unified experiment pipeline")
    ap.add_argument(
        "--models",
        default="all",
        help=(
            "comma list: baseline,factor_variants,mlp,gat_industry,gat_universe,gat_two_graph_no_neutral,"
            "dmfm_ind_neutral,dmfm_full,all"
        ),
    )
    ap.add_argument(
        "--baseline_models",
        default="linear,xgboost,lstm",
        help="comma list when baseline is selected: linear,xgboost,lstm",
    )
    ap.add_argument(
        "--windows",
        default="all",
        help="comma list: short,medium,long,all",
    )
    ap.add_argument(
        "--graph_modes",
        default="static",
        help=(
            "comma list: static,dynamic. When set, the same windows/models are run under "
            "separate artifact/output subfolders so static and dynamic graph results share one summary."
        ),
    )
    ap.add_argument(
        "--baseline_graph_mode",
        default="static",
        choices=["static", "dynamic", "both"],
        help=(
            "When --graph_modes is set and baseline is selected, choose where to run baselines. "
            "Use static by default because baselines do not consume graph edges."
        ),
    )
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--prices", default="unique_2019q3to2025q3.csv")
    ap.add_argument("--industry_csv", default="unique_2019q3to2025q3.csv")
    ap.add_argument("--benchmark_csv", default="GAT0050.csv")
    ap.add_argument("--artifact_root", default="artifacts")
    ap.add_argument("--output_root", default="runs")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--no_dynamic_graphs", action="store_true", help="Use static binary graph artifacts")
    ap.add_argument("--graph_lookback", type=int, default=60)
    ap.add_argument("--graph_min_obs", type=int, default=20)
    ap.add_argument("--industry_top_k", type=int, default=20)
    ap.add_argument("--universe_top_k", type=int, default=40)
    ap.add_argument("--top_pct", type=float, default=0.10)
    ap.add_argument("--rebalance_days", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--lambda_b", type=float, default=0.01, help="Factor return loss weight for DMFM/factor variants")
    ap.add_argument("--seed", type=int, default=42, help="Random seed passed to training subprocesses")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--parallel_jobs",
        type=int,
        default=1,
        help="Number of model subprocesses to run concurrently after artifacts are built.",
    )
    ap.add_argument(
        "--preload_gpu",
        action="store_true",
        help="For graph neural models, move Ft/yt tensors to CUDA once at startup to reduce per-day transfer overhead.",
    )
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--rebuild_artifacts", action="store_true")
    ap.add_argument(
        "--separate_artifacts",
        action="store_true",
        help="Build each selected window independently. Default builds one shared full artifact and slices window views.",
    )
    ap.add_argument(
        "--shared_artifact_name",
        default="_full",
        help="Folder name for the shared full-range artifact when --separate_artifacts is not used.",
    )
    ap.add_argument(
        "--cache_parquet",
        action="store_true",
        help="Ask build_artifacts.py to create/use a parquet cache when reading CSV prices.",
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def parse_graph_modes(args) -> List[Optional[str]]:
    if not args.graph_modes.strip():
        return [None]
    modes = parse_csv_list(args.graph_modes, ["static", "dynamic"], allow_all=False)
    if args.no_dynamic_graphs and "dynamic" in modes:
        print("[warn] --graph_modes overrides --no_dynamic_graphs; dynamic mode will still build dynamic graph artifacts")
    return modes


def mode_args(args, graph_mode: Optional[str]):
    if graph_mode is None:
        return args
    out = copy.copy(args)
    out.graph_mode = graph_mode
    out.no_dynamic_graphs = graph_mode == "static"
    out.output_root = str(Path(args.output_root) / graph_mode)
    return out


def artifact_dir_for_mode(args, graph_mode: Optional[str], window: str) -> Path:
    root = Path(args.artifact_root)
    if graph_mode is None:
        return root / window
    return root / graph_mode / window


def baseline_enabled_for_mode(args, graph_mode: Optional[str]) -> bool:
    if graph_mode is None or not args.graph_modes.strip():
        return True
    return args.baseline_graph_mode == "both" or args.baseline_graph_mode == graph_mode


def run_tasks(tasks: List[Callable[[], None]], parallel_jobs: int, dry_run: bool = False):
    if parallel_jobs <= 1 or len(tasks) <= 1 or dry_run:
        for task in tasks:
            task()
        return

    print(f"\n[parallel] running {len(tasks)} model tasks with parallel_jobs={parallel_jobs}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        futures = [pool.submit(task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    py = sys.executable
    models = parse_model_list(args.models)
    windows = parse_csv_list(args.windows, ["short", "medium", "long"])
    baseline_models = parse_csv_list(args.baseline_models, ["linear", "xgboost", "lstm"])
    graph_modes = parse_graph_modes(args)
    if (
        args.graph_modes.strip()
        and args.baseline_graph_mode != "both"
        and args.baseline_graph_mode not in graph_modes
    ):
        fallback_mode = graph_modes[0]
        print(
            f"[warn] baseline_graph_mode={args.baseline_graph_mode!r} is not selected in "
            f"graph_modes={graph_modes}; baselines will run under {fallback_mode!r}."
        )
        args.baseline_graph_mode = fallback_mode

    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "models": models,
        "baseline_models": baseline_models,
        "windows": windows,
        "graph_modes": [m if m is not None else graph_mode_from_args(args) for m in graph_modes],
        "baseline_graph_mode": args.baseline_graph_mode if args.graph_modes.strip() else None,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "lambda_b": args.lambda_b,
        "seed": args.seed,
        "dynamic_graphs": not args.no_dynamic_graphs if not args.graph_modes.strip() else ("dynamic" in graph_modes),
        "graph_lookback": args.graph_lookback,
        "graph_min_obs": args.graph_min_obs,
        "industry_top_k": args.industry_top_k,
        "universe_top_k": args.universe_top_k,
        "parallel_jobs": args.parallel_jobs,
        "preload_gpu": args.preload_gpu,
        "separate_artifacts": args.separate_artifacts,
        "shared_artifact_name": args.shared_artifact_name,
        "cache_parquet": args.cache_parquet,
        "artifacts": {},
        "runs": [],
        "dry_run": args.dry_run,
    }

    print("=" * 72)
    print("Unified Pipeline")
    print("=" * 72)
    print(f"mode={args.mode} models={models} windows={windows} device={args.device}")
    print(f"graph_modes={summary['graph_modes']} baseline_graph_mode={summary['baseline_graph_mode']}")
    print(f"artifact_root={artifact_root} output_root={output_root}")
    print(
        f"dynamic_graphs={summary['dynamic_graphs']} "
        f"lookback={args.graph_lookback} min_obs={args.graph_min_obs} "
        f"industry_top_k={args.industry_top_k} universe_top_k={args.universe_top_k}"
    )
    print(
        f"parallel_jobs={args.parallel_jobs} preload_gpu={args.preload_gpu} "
        f"skip_build={args.skip_build} skip_train={args.skip_train} dry_run={args.dry_run}"
    )
    print(
        f"artifact_strategy={'separate' if args.separate_artifacts else 'shared_full_then_window_views'} "
        f"shared_artifact_name={args.shared_artifact_name} cache_parquet={args.cache_parquet}"
    )

    tasks: List[Callable[[], None]] = []
    for graph_mode in graph_modes:
        current_args = mode_args(args, graph_mode)
        mode_label = graph_mode_from_args(current_args)
        window_artifacts: Dict[str, Path] = {}
        if not args.separate_artifacts:
            print(f"\n{'-' * 72}")
            full_spec = merged_window_spec(windows)
            print(
                f"Shared artifact build ({full_spec['start']} ~ {full_spec['end']}) "
                f"| graph_mode={mode_label}"
            )
            print(f"{'-' * 72}")
            window_artifacts = ensure_shared_artifacts(py, windows, graph_mode, args, summary)

        for window in windows:
            spec = WINDOW_SPECS[window]
            current_args = mode_args(args, graph_mode)
            artifact_dir = (
                artifact_dir_for_mode(args, graph_mode, window)
                if args.separate_artifacts
                else window_artifacts[window]
            )
            print(f"\n{'-' * 72}")
            print(f"Window: {window} ({spec['start']} ~ {spec['end']}) | graph_mode={mode_label}")
            print(f"{'-' * 72}")

            if args.separate_artifacts:
                ensure_artifact(py, window, spec, current_args, artifact_dir, summary)

            if "baseline" in models and baseline_enabled_for_mode(args, graph_mode):
                for bm in baseline_models:
                    tasks.append(
                        lambda window=window, artifact_dir=artifact_dir, bm=bm, current_args=current_args:
                        run_baseline(py, window, artifact_dir, bm, current_args, summary)
                    )

            for variant in FACTOR_MODELS:
                if variant in models:
                    tasks.append(
                        lambda window=window, artifact_dir=artifact_dir, variant=variant, current_args=current_args:
                        run_factor_variant(py, window, artifact_dir, variant, current_args, summary)
                    )

    run_tasks(tasks, args.parallel_jobs, dry_run=args.dry_run)

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
