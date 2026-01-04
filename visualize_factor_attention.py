import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import torch

from model_dmfm_wei2022 import DMFM_Wei2022
from train_gat_fixed import load_artifacts


# ---------------- Device Helper ----------------
def get_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- Attention Extraction ----------------
def collect_attention(
    Ft_tensor: torch.Tensor,
    industry_edge_index: torch.Tensor,
    universe_edge_index: torch.Tensor,
    model: DMFM_Wei2022,
    device: torch.device,
) -> torch.Tensor:
    """Return daily mean attention importance with shape [T, F]."""
    model.eval()
    industry_edge_index = industry_edge_index.to(device)
    universe_edge_index = universe_edge_index.to(device)

    daily_means: List[torch.Tensor] = []
    with torch.no_grad():
        for t in range(Ft_tensor.shape[0]):
            x = Ft_tensor[t].to(device)
            _, attn_weights, _ = model(x, industry_edge_index, universe_edge_index)
            if attn_weights is None:
                raise RuntimeError("Model was initialized without factor attention enabled.")
            daily_means.append(attn_weights.mean(dim=0).cpu())
    return torch.stack(daily_means, dim=0)


# ---------------- Plotters ----------------
def plot_top_features(
    feature_names: List[str],
    importance: torch.Tensor,
    top_k: int,
    output_path: str,
):
    values, indices = torch.topk(importance, k=min(top_k, importance.numel()))
    names = [feature_names[i] for i in indices.tolist()]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], values.tolist()[::-1], color="#2a9d8f")
    plt.xlabel("平均注意力權重")
    plt.title(f"Top {len(names)} Factor Attention Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap(
    feature_names: List[str],
    daily_importance: torch.Tensor,
    top_indices: List[int],
    output_path: str,
):
    data = daily_importance[:, top_indices].numpy()
    selected_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(12, 6))
    im = plt.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="注意力權重 (平均截面)")
    plt.yticks(range(0, data.shape[0], max(1, data.shape[0] // 10)))
    plt.xticks(range(len(selected_names)), selected_names, rotation=45, ha="right")
    plt.title("每日注意力權重熱力圖 (Top Features)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_timeseries(
    feature_names: List[str],
    dates: List[str],
    daily_importance: torch.Tensor,
    top_indices: List[int],
    output_path: str,
):
    plt.figure(figsize=(12, 6))
    for idx in top_indices:
        plt.plot(dates, daily_importance[:, idx].numpy(), label=feature_names[idx])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("日期")
    plt.ylabel("平均注意力權重")
    plt.title("Top 特徵注意力權重時間序列")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="視覺化 DMFM Factor Attention")
    ap.add_argument("--artifact_dir", required=True, help="artifacts 目錄")
    ap.add_argument(
        "--weights",
        default=None,
        help="模型權重路徑 (預設 artifact_dir/dmfm_wei2022_best.pt 或 dmfm_wei2022.pt)",
    )
    ap.add_argument(
        "--output_dir",
        default=None,
        help="輸出圖表目錄 (預設 artifact_dir/plots_attention)",
    )
    ap.add_argument("--device", default=None, help="裝置 (cuda/mps/cpu)，預設自動偵測")
    ap.add_argument("--top_k", type=int, default=15, help="輸出排名前幾的特徵")
    ap.add_argument("--hidden_dim", type=int, default=64, help="模型隱藏維度，需與訓練時一致")
    ap.add_argument("--heads", type=int, default=2, help="GAT heads，需與訓練時一致")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout 率，需與訓練時一致")
    return ap.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    Ft, yt, industry_ei, universe_ei, meta = load_artifacts(args.artifact_dir)
    feature_names: List[str] = list(meta.get("feature_cols", meta.get("feature_names", [])))
    dates: List[str] = list(meta.get("dates", []))

    output_dir = args.output_dir or os.path.join(args.artifact_dir, "plots_attention")
    os.makedirs(output_dir, exist_ok=True)

    # 推斷權重檔
    if args.weights:
        weight_path = args.weights
    else:
        candidates = [
            os.path.join(args.artifact_dir, "dmfm_wei2022_best.pt"),
            os.path.join(args.artifact_dir, "dmfm_wei2022.pt"),
        ]
        weight_path = next((p for p in candidates if os.path.exists(p)), None)
        if weight_path is None:
            raise FileNotFoundError("未找到模型權重，請使用 --weights 指定")

    # 建立模型並載入權重
    model = DMFM_Wei2022(
        num_features=Ft.shape[-1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        use_factor_attention=True,
    ).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)

    daily_importance = collect_attention(Ft, industry_ei, universe_ei, model, device)
    overall_importance = daily_importance.mean(dim=0)

    top_k = min(args.top_k, overall_importance.numel())
    top_indices = torch.topk(overall_importance, k=top_k).indices.tolist()

    # 產出圖表
    plot_top_features(
        feature_names=feature_names,
        importance=overall_importance,
        top_k=top_k,
        output_path=os.path.join(output_dir, "factor_attention_top_features.png"),
    )

    plot_heatmap(
        feature_names=feature_names,
        daily_importance=daily_importance,
        top_indices=top_indices,
        output_path=os.path.join(output_dir, "factor_attention_heatmap.png"),
    )

    if dates:
        plot_timeseries(
            feature_names=feature_names,
            dates=dates,
            daily_importance=daily_importance,
            top_indices=top_indices[: min(5, len(top_indices))],
            output_path=os.path.join(output_dir, "factor_attention_timeseries.png"),
        )

    # 摘要檔
    summary_path = os.path.join(output_dir, "factor_attention_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Factor Attention 摘要\n")
        f.write(f"模型權重: {weight_path}\n")
        f.write(f"輸出目錄: {output_dir}\n")
        f.write("Top 特徵 (平均注意力權重):\n")
        for rank, idx in enumerate(top_indices, 1):
            f.write(f"{rank:02d}. {feature_names[idx]}\t{overall_importance[idx]:.6f}\n")

    print("完成 Factor Attention 視覺化：")
    print(f"- Top 特徵圖: {os.path.join(output_dir, 'factor_attention_top_features.png')}")
    print(f"- 熱力圖: {os.path.join(output_dir, 'factor_attention_heatmap.png')}")
    if dates:
        print(f"- 時間序列圖: {os.path.join(output_dir, 'factor_attention_timeseries.png')}")
    print(f"- 摘要: {summary_path}")


if __name__ == "__main__":
    main()
