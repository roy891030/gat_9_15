# -*- coding: utf-8 -*-
"""
Train controlled factor model variants for MLP/GAT/DMFM ablation.
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.optim as optim

from checkpoint_utils import save_torch_checkpoint
from graph_utils import graph_at, load_dynamic_graphs
from model_dmfm_wei2022 import FACTOR_VARIANTS, FactorGraphAblation
from train_dmfm_wei2022 import (
    compute_loss,
    evaluate,
    filter_edge_index,
    pick_device,
    time_split_indices_3,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Train controlled factor graph variants")
    ap.add_argument("--artifact_dir", default="artifacts/short", help="Artifacts folder")
    ap.add_argument("--variant", choices=FACTOR_VARIANTS, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lambda_attn", type=float, default=0.05)
    ap.add_argument("--lambda_ic", type=float, default=1.0)
    ap.add_argument("--lambda_b", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_factor_attention", action="store_true")
    ap.add_argument(
        "--preload_gpu",
        action="store_true",
        help="Move Ft/yt tensors to CUDA once at startup to reduce per-day host->device copies.",
    )
    return ap.parse_args()


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, optimizer, Ft, yt, industry_ei, universe_ei, dynamic_graphs,
                    train_indices, lambda_attn, lambda_ic, lambda_b, device):
    model.train()
    metrics = {"loss": 0.0, "d_attn": 0.0, "b_factor": 0.0, "ic": 0.0}
    valid_steps = 0

    for t in train_indices:
        x_t = Ft[t].to(device)
        y_t = yt[t].to(device)
        mask = torch.isfinite(y_t)
        if mask.sum() < 3:
            continue

        x_t = x_t[mask]
        y_t = y_t[mask]
        industry_ei_filtered, industry_attr_filtered = graph_at(
            dynamic_graphs, "industry", t, industry_ei, mask=mask, device=device
        )
        universe_ei_filtered, universe_attr_filtered = graph_at(
            dynamic_graphs, "universe", t, universe_ei, mask=mask, device=device
        )

        if model.variant != "mlp":
            if industry_ei_filtered.shape[1] < 10 or universe_ei_filtered.shape[1] < 100:
                continue

        deep_factor, attn_weights, contexts = model(
            x_t,
            industry_ei_filtered,
            universe_ei_filtered,
            industry_attr_filtered,
            universe_attr_filtered,
        )
        f_hat = model.interpret_factor(x_t, attn_weights, x_norm=contexts.get("x_norm"))
        loss, step_metrics = compute_loss(deep_factor, f_hat, y_t, lambda_attn, lambda_ic, lambda_b)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[warn] skip t={t}: NaN/Inf loss")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key in metrics:
            if key in step_metrics:
                metrics[key] += step_metrics[key]
        valid_steps += 1

    if valid_steps > 0:
        for key in metrics:
            metrics[key] /= valid_steps
    return metrics


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = pick_device(args.device)

    print("=" * 60)
    print("Factor Variant Training")
    print("=" * 60)
    print(f"variant: {args.variant}")
    print(f"device: {device}")
    print(f"artifact_dir: {args.artifact_dir}")

    Ft = torch.load(os.path.join(args.artifact_dir, "Ft_tensor.pt"))
    yt = torch.load(os.path.join(args.artifact_dir, "yt_tensor.pt"))
    industry_ei = torch.load(os.path.join(args.artifact_dir, "industry_edge_index.pt")).to(device)
    universe_ei = torch.load(os.path.join(args.artifact_dir, "universe_edge_index.pt")).to(device)
    dynamic_graphs = load_dynamic_graphs(args.artifact_dir, map_location="cpu")
    with open(os.path.join(args.artifact_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    T, N, F = Ft.shape
    if args.preload_gpu and device.type == "cuda":
        print("preload_gpu=True: moving Ft/yt tensors to CUDA")
        Ft = Ft.to(device, non_blocking=True)
        yt = yt.to(device, non_blocking=True)
    elif args.preload_gpu:
        print(f"preload_gpu=True ignored for device={device}")

    train_idx, val_idx, test_idx = time_split_indices_3(
        meta.get("dates", list(range(T))),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"data: T={T}, N={N}, F={F}")
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"industry_edges={industry_ei.shape[1]:,} universe_edges={universe_ei.shape[1]:,}")
    if dynamic_graphs is not None:
        print(f"dynamic_weighted_graphs=True edge_dim={dynamic_graphs.edge_dim}")
    else:
        print("dynamic_weighted_graphs=False (static binary fallback)")

    model = FactorGraphAblation(
        in_dim=F,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        edge_dim=dynamic_graphs.edge_dim if dynamic_graphs is not None else None,
        variant=args.variant,
        use_factor_attention=not args.no_factor_attention,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={total_params:,}")

    config = {
        "in_dim": F,
        "num_features": F,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "edge_dim": dynamic_graphs.edge_dim if dynamic_graphs is not None else None,
        "variant": args.variant,
        "use_factor_attention": not args.no_factor_attention,
    }
    metadata = {
        "artifact_dir": args.artifact_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "weight_decay": args.weight_decay,
        "lambda_attn": args.lambda_attn,
        "lambda_ic": args.lambda_ic,
        "lambda_b": args.lambda_b,
        "seed": args.seed,
        "preload_gpu": bool(args.preload_gpu),
    }

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    monitor_indices = val_idx if len(val_idx) > 0 else test_idx
    monitor_name = "Val" if len(val_idx) > 0 else "Test"
    best_icir = -float("inf")
    best_state = None
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Start Training")
    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            Ft,
            yt,
            industry_ei,
            universe_ei,
            dynamic_graphs,
            train_idx,
            args.lambda_attn,
            args.lambda_ic,
            args.lambda_b,
            device,
        )

        if epoch % 5 == 0 or epoch == args.epochs:
            eval_metrics = evaluate(model, Ft, yt, industry_ei, universe_ei, monitor_indices, device, dynamic_graphs)
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
                f"Train IC: {train_metrics['ic']:.4f} | "
                f"{monitor_name} IC: {eval_metrics['mean_ic']:.4f} | "
                f"{monitor_name} ICIR: {eval_metrics['icir']:.4f} | "
                f"{monitor_name} FR: {eval_metrics['cumulative_factor_return']:.4f}"
            )

            if eval_metrics["icir"] > best_icir:
                best_icir = eval_metrics["icir"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                save_torch_checkpoint(
                    os.path.join(args.artifact_dir, f"{args.variant}_best.pt"),
                    model_type="factor_variant",
                    model_or_state_dict=best_state,
                    config=config,
                    metadata=metadata,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (best {monitor_name} ICIR={best_icir:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval = evaluate(model, Ft, yt, industry_ei, universe_ei, test_idx, device, dynamic_graphs)
    print("\nFinal Test:")
    print(f"  Mean IC: {final_eval['mean_ic']:.4f}")
    print(f"  ICIR: {final_eval['icir']:.4f}")
    print(f"  Cumulative Factor Return: {final_eval['cumulative_factor_return']:.4f}")

    final_path = os.path.join(args.artifact_dir, f"{args.variant}.pt")
    save_torch_checkpoint(
        final_path,
        model_type="factor_variant",
        model_or_state_dict=model,
        config=config,
        metadata=metadata,
    )
    print(f"model saved: {final_path}")


if __name__ == "__main__":
    main()
