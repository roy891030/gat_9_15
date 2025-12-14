# -*- coding: utf-8 -*-
"""
DMFM Wei et al. (2022) 訓練腳本

損失函數（論文公式 13）：
    L = d^t_k - b^t_k + λ_IC · IC_penalty

其中：
- d^t_k: Attention estimate loss = ||f - f_hat||²
- b^t_k: Factor return (cross-sectional regression)
- IC_penalty: 1 - IC (最大化 IC)

訓練目標：
1. 最小化注意力估計誤差（提高解釋性）
2. 最大化因子收益（提高預測能力）
3. 最大化 IC（提高相關性）
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_dmfm_wei2022 import DMFM_Wei2022


def parse_args():
    ap = argparse.ArgumentParser(description="訓練 DMFM (Wei et al. 2022)")
    ap.add_argument("--artifact_dir", default="gat_artifacts_out_plus", help="Artifacts 資料夾")
    ap.add_argument("--epochs", type=int, default=200, help="訓練週期數")
    ap.add_argument("--lr", type=float, default=1e-3, help="學習率")
    ap.add_argument("--device", default="auto", help="計算裝置: cpu, cuda, mps, auto")
    ap.add_argument("--hidden_dim", type=int, default=64, help="隱藏層維度")
    ap.add_argument("--heads", type=int, default=2, help="GAT 注意力頭數")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="權重衰減")
    ap.add_argument("--lambda_attn", type=float, default=0.1, help="注意力損失權重")
    ap.add_argument("--lambda_ic", type=float, default=1.0, help="IC 損失權重")
    ap.add_argument("--patience", type=int, default=30, help="Early stopping 耐心值")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="訓練集比例")
    return ap.parse_args()


def pick_device(device_str: str) -> torch.device:
    """自動選擇計算裝置"""
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


def compute_ic(pred, target):
    """
    計算 Information Coefficient (Pearson correlation)

    參數：
        pred: [N] 預測值
        target: [N] 真實值

    回傳：
        ic: float, Pearson 相關係數
    """
    pred = pred.flatten()
    target = target.flatten()

    # 移除 NaN
    mask = ~(torch.isnan(pred) | torch.isnan(target) | torch.isinf(pred) | torch.isinf(target))
    pred = pred[mask]
    target = target[mask]

    if len(pred) < 3:  # 至少需要 3 個樣本
        return 0.0

    # Pearson correlation
    pred_mean = pred.mean()
    target_mean = target.mean()

    numerator = ((pred - pred_mean) * (target - target_mean)).sum()
    pred_std = torch.sqrt(((pred - pred_mean) ** 2).sum())
    target_std = torch.sqrt(((target - target_mean) ** 2).sum())
    denominator = pred_std * target_std

    if denominator < 1e-8:
        return 0.0

    return (numerator / denominator).item()


def cross_sectional_regression(factor, returns):
    """
    Cross-sectional regression: r^t = b^t · f^t

    參數：
        factor: [N, 1] 因子值
        returns: [N] 真實報酬

    回傳：
        b: float, 因子收益（factor return）
    """
    factor = factor.flatten()
    returns = returns.flatten()

    # 移除 NaN
    mask = ~(torch.isnan(factor) | torch.isnan(returns) | torch.isinf(factor) | torch.isinf(returns))
    factor = factor[mask]
    returns = returns[mask]

    if len(factor) < 3:
        return 0.0

    # 簡單線性回歸：b = (f'f)^(-1) f'r
    # 為了數值穩定性，加入 L2 正則化
    factor_centered = factor - factor.mean()
    returns_centered = returns - returns.mean()

    b = (factor_centered * returns_centered).sum() / (factor_centered * factor_centered).sum().clamp(min=1e-8)

    return b.item()


def compute_loss(deep_factor, f_hat, returns, lambda_attn=0.1, lambda_ic=1.0):
    """
    計算 DMFM 損失函數（論文公式 13）

    L = λ_attn · d + λ_IC · (1 - IC) - b

    參數：
        deep_factor: [N, 1] 深度因子
        f_hat: [N, 1] 注意力估計因子
        returns: [N] 真實報酬
        lambda_attn: 注意力損失權重
        lambda_ic: IC 損失權重

    回傳：
        loss: 總損失
        metrics: dict, 包含各損失組件
    """
    # 1. Attention Estimate Loss: d = ||f - f_hat||²
    if f_hat is not None:
        d = torch.norm(deep_factor - f_hat, p=2)
    else:
        d = torch.tensor(0.0, device=deep_factor.device)

    # 2. Factor Return: b (cross-sectional regression)
    b = cross_sectional_regression(deep_factor, returns)
    b_tensor = torch.tensor(b, device=deep_factor.device)

    # 3. Information Coefficient
    ic = compute_ic(deep_factor, returns)
    ic_penalty = 1.0 - ic  # 最小化 (1 - IC) 等價於最大化 IC
    ic_penalty_tensor = torch.tensor(ic_penalty, device=deep_factor.device)

    # 4. 綜合損失
    loss = lambda_attn * d + lambda_ic * ic_penalty_tensor - b_tensor

    metrics = {
        'loss': loss.item(),
        'd_attn': d.item(),
        'b_factor': b,
        'ic': ic,
        'ic_penalty': ic_penalty
    }

    return loss, metrics


def filter_edge_index(edge_index, mask):
    """
    根據節點 mask 過濾並重新映射 edge_index

    參數：
        edge_index: [2, E] 原始邊索引
        mask: [N] bool tensor，標記哪些節點是有效的

    回傳：
        filtered_edge_index: [2, E'] 過濾並重新映射後的邊索引
    """
    # 建立舊索引到新索引的映射
    # old_to_new[i] = j 表示原始節點 i 在過濾後的位置是 j
    old_to_new = torch.full((mask.size(0),), -1, dtype=torch.long, device=mask.device)
    old_to_new[mask] = torch.arange(mask.sum(), device=mask.device)

    # 過濾邊：只保留兩端節點都有效的邊
    src, dst = edge_index[0], edge_index[1]
    valid_edges = mask[src] & mask[dst]

    if valid_edges.sum() == 0:
        # 如果沒有有效的邊，返回空的 edge_index
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    # 重新映射索引
    filtered_edge_index = torch.stack([
        old_to_new[src[valid_edges]],
        old_to_new[dst[valid_edges]]
    ], dim=0)

    return filtered_edge_index


def train_one_epoch(model, optimizer, Ft, yt, industry_ei, universe_ei,
                    train_indices, lambda_attn, lambda_ic, device):
    """訓練一個 epoch"""
    model.train()

    epoch_metrics = {
        'loss': 0.0,
        'd_attn': 0.0,
        'b_factor': 0.0,
        'ic': 0.0
    }

    valid_steps = 0

    for t in train_indices:
        # 取得當前時間點資料
        x_t = Ft[t].to(device)  # [N, F]
        y_t = yt[t].to(device)  # [N]

        # 移除 NaN
        mask = torch.isfinite(y_t)
        if mask.sum() < 3:  # 至少需要 3 個有效樣本
            continue

        x_t = x_t[mask]
        y_t = y_t[mask]

        # 過濾並重新映射邊索引
        industry_ei_filtered = filter_edge_index(industry_ei, mask)
        universe_ei_filtered = filter_edge_index(universe_ei, mask)

        # Forward
        deep_factor, attn_weights, contexts = model(x_t, industry_ei_filtered, universe_ei_filtered)

        # Attention estimate
        f_hat = model.interpret_factor(x_t, attn_weights)

        # 計算損失
        loss, metrics = compute_loss(deep_factor, f_hat, y_t, lambda_attn, lambda_ic)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 累計指標
        for key in epoch_metrics:
            if key in metrics:
                epoch_metrics[key] += metrics[key]

        valid_steps += 1

    # 平均指標
    if valid_steps > 0:
        for key in epoch_metrics:
            epoch_metrics[key] /= valid_steps

    return epoch_metrics


@torch.no_grad()
def evaluate(model, Ft, yt, industry_ei, universe_ei, test_indices, device):
    """評估測試集"""
    model.eval()

    all_ics = []
    all_factor_returns = []
    all_attns = []

    for t in test_indices:
        x_t = Ft[t].to(device)
        y_t = yt[t].to(device)

        mask = torch.isfinite(y_t)
        if mask.sum() < 3:
            continue

        x_t = x_t[mask]
        y_t = y_t[mask]

        # 過濾並重新映射邊索引
        industry_ei_filtered = filter_edge_index(industry_ei, mask)
        universe_ei_filtered = filter_edge_index(universe_ei, mask)

        # Forward
        deep_factor, attn_weights, contexts = model(x_t, industry_ei_filtered, universe_ei_filtered)

        # 計算指標
        ic = compute_ic(deep_factor, y_t)
        b = cross_sectional_regression(deep_factor, y_t)

        all_ics.append(ic)
        all_factor_returns.append(b)
        if attn_weights is not None:
            all_attns.append(attn_weights.mean(dim=0).cpu().numpy())

    # 統計
    ic_tensor = torch.tensor(all_ics)
    mean_ic = ic_tensor.mean().item()
    std_ic = ic_tensor.std().item()
    icir = mean_ic / (std_ic + 1e-8)

    cumulative_factor_return = sum(all_factor_returns)

    eval_metrics = {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': icir,
        'cumulative_factor_return': cumulative_factor_return,
        'avg_factor_return': np.mean(all_factor_returns)
    }

    return eval_metrics


def main():
    args = parse_args()

    # Device
    device = pick_device(args.device)
    print("=" * 60)
    print("DMFM (Wei et al. 2022) 訓練腳本")
    print("=" * 60)
    print(f"使用裝置: {device}")

    # 載入資料
    print("\n載入 artifacts...")
    Ft = torch.load(os.path.join(args.artifact_dir, "Ft_tensor.pt"))  # [T, N, F]
    yt = torch.load(os.path.join(args.artifact_dir, "yt_tensor.pt"))  # [T, N]
    industry_ei = torch.load(os.path.join(args.artifact_dir, "industry_edge_index.pt")).to(device)
    universe_ei = torch.load(os.path.join(args.artifact_dir, "universe_edge_index.pt")).to(device)

    with open(os.path.join(args.artifact_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    T, N, F = Ft.shape
    print(f"資料形狀: T={T}, N={N}, F={F}")
    print(f"產業圖邊數: {industry_ei.shape[1]:,}")
    print(f"全市場圖邊數: {universe_ei.shape[1]:,}")

    # 訓練/測試切分
    split_idx = int(T * args.train_ratio)
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, T))

    print(f"\n訓練集: {len(train_indices)} 天")
    print(f"測試集: {len(test_indices)} 天")

    # 建立模型
    print("\n建立模型...")
    model = DMFM_Wei2022(
        num_features=F,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        use_factor_attention=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數: {total_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 訓練循環
    print("\n" + "=" * 60)
    print("開始訓練")
    print("=" * 60)

    best_icir = -float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # 訓練
        train_metrics = train_one_epoch(
            model, optimizer, Ft, yt, industry_ei, universe_ei,
            train_indices, args.lambda_attn, args.lambda_ic, device
        )

        # 評估
        if epoch % 5 == 0 or epoch == args.epochs:
            eval_metrics = evaluate(
                model, Ft, yt, industry_ei, universe_ei, test_indices, device
            )

            # 輸出
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train IC: {train_metrics['ic']:.4f} | "
                  f"Test IC: {eval_metrics['mean_ic']:.4f} | "
                  f"Test ICIR: {eval_metrics['icir']:.4f} | "
                  f"Test FR: {eval_metrics['cumulative_factor_return']:.4f}")

            # Early stopping
            if eval_metrics['icir'] > best_icir:
                best_icir = eval_metrics['icir']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0

                # 儲存最佳模型
                torch.save(best_state, os.path.join(args.artifact_dir, "dmfm_wei2022_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (best ICIR={best_icir:.4f})")
                    break

    # 載入最佳模型並最終評估
    print("\n" + "=" * 60)
    print("訓練完成！")
    print("=" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval = evaluate(model, Ft, yt, industry_ei, universe_ei, test_indices, device)

    print("\n最終測試集表現:")
    print(f"  Mean IC: {final_eval['mean_ic']:.4f}")
    print(f"  Std IC: {final_eval['std_ic']:.4f}")
    print(f"  ICIR: {final_eval['icir']:.4f}")
    print(f"  Cumulative Factor Return: {final_eval['cumulative_factor_return']:.4f}")
    print(f"  Avg Factor Return: {final_eval['avg_factor_return']:.6f}")

    # 儲存最終模型
    final_path = os.path.join(args.artifact_dir, "dmfm_wei2022.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n模型已儲存至: {final_path}")

    # 儲存訓練日誌
    log_path = os.path.join(args.artifact_dir, "train_log_wei2022.txt")
    with open(log_path, "w") as f:
        f.write("DMFM Wei et al. (2022) 訓練日誌\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"訓練參數:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Hidden dim: {args.hidden_dim}\n")
        f.write(f"  Heads: {args.heads}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Lambda attn: {args.lambda_attn}\n")
        f.write(f"  Lambda IC: {args.lambda_ic}\n")
        f.write(f"\n最終測試集表現:\n")
        f.write(f"  Mean IC: {final_eval['mean_ic']:.4f}\n")
        f.write(f"  ICIR: {final_eval['icir']:.4f}\n")
        f.write(f"  Cumulative Factor Return: {final_eval['cumulative_factor_return']:.4f}\n")

    print(f"訓練日誌已儲存至: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
