# -*- coding: utf-8 -*-
"""
Factor Attention 視覺化工具

分析 DMFM 的 Factor Attention 模組，
了解哪些原始特徵對深度因子最重要。
"""

import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_dmfm_wei2022 import DMFM_Wei2022

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    ap = argparse.ArgumentParser(description="視覺化 Factor Attention")
    ap.add_argument("--artifact_dir", default="gat_artifacts_out_plus", help="Artifacts 資料夾")
    ap.add_argument("--model_path", default=None, help="模型檔案路徑（預設為 artifact_dir/dmfm_wei2022_best.pt）")
    ap.add_argument("--output_dir", default="plots_attention", help="輸出圖表資料夾")
    ap.add_argument("--device", default="cpu", help="計算裝置")
    ap.add_argument("--top_k", type=int, default=15, help="顯示前 K 個重要特徵")
    return ap.parse_args()


def visualize_attention(args):
    """視覺化 Factor Attention 權重"""

    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 metadata
    with open(os.path.join(args.artifact_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    F = len(meta['feature_cols'])
    feature_names = meta['feature_cols']

    # 載入模型
    model = DMFM_Wei2022(num_features=F, hidden_dim=64, heads=2)
    model_path = args.model_path or os.path.join(args.artifact_dir, "dmfm_wei2022_best.pt")

    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案 {model_path}")
        print("請先訓練模型：python train_dmfm_wei2022.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    # 載入測試資料
    Ft = torch.load(os.path.join(args.artifact_dir, "Ft_tensor.pt"))
    industry_ei = torch.load(os.path.join(args.artifact_dir, "industry_edge_index.pt"))
    universe_ei = torch.load(os.path.join(args.artifact_dir, "universe_edge_index.pt"))

    T, N, _ = Ft.shape
    split_idx = int(T * 0.8)
    Ft_test = Ft[split_idx:]

    print(f"分析測試期 {len(Ft_test)} 天的 Factor Attention...")

    # 計算平均注意力權重
    all_attns = []

    with torch.no_grad():
        for t in range(len(Ft_test)):
            x_t = Ft_test[t]
            _, attn_weights, _ = model(x_t, industry_ei, universe_ei)

            if attn_weights is not None:
                # 計算特徵重要性（平均所有股票）
                importance = model.get_attention_importance(x_t, attn_weights)
                all_attns.append(importance.numpy())

    # 平均所有時間點
    avg_attn = np.mean(all_attns, axis=0)  # [F]
    std_attn = np.std(all_attns, axis=0)  # [F]

    # ==================== 圖表 1: 特徵重要性柱狀圖 ====================
    fig, ax = plt.subplots(figsize=(14, 8))

    indices = np.argsort(avg_attn)[::-1]  # 降序排列
    top_k = min(args.top_k, len(indices))

    y_pos = np.arange(top_k)
    ax.barh(y_pos, avg_attn[indices[:top_k]], xerr=std_attn[indices[:top_k]],
            alpha=0.8, color='steelblue', edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[:top_k]])
    ax.set_xlabel('Average Attention Weight', fontsize=12)
    ax.set_title(f'Top {top_k} Most Important Features (Factor Attention)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "factor_attention_top_features.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/factor_attention_top_features.png")
    plt.close()

    # ==================== 圖表 2: 所有特徵的注意力權重 ====================
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(avg_attn))
    colors = ['crimson' if i < top_k else 'lightgray' for i in range(len(avg_attn))]

    ax.barh(y_pos, avg_attn[indices], color=[colors[i] for i in range(len(avg_attn))],
            alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel('Average Attention Weight', fontsize=12)
    ax.set_title('All Features Ranked by Attention Weight', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "factor_attention_all_features.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/factor_attention_all_features.png")
    plt.close()

    # ==================== 圖表 3: 注意力權重時間序列（Top 5） ====================
    fig, ax = plt.subplots(figsize=(14, 6))

    top_5_idx = indices[:5]
    time_steps = np.arange(len(all_attns))

    for i, feat_idx in enumerate(top_5_idx):
        attn_series = [a[feat_idx] for a in all_attns]
        ax.plot(time_steps, attn_series, label=feature_names[feat_idx], linewidth=2, alpha=0.8)

    ax.set_xlabel('Time Step (Test Period)', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Attention Weight Evolution for Top 5 Features', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "factor_attention_timeseries.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/factor_attention_timeseries.png")
    plt.close()

    # ==================== 圖表 4: 注意力權重熱力圖（時間 x Top 20 特徵） ====================
    fig, ax = plt.subplots(figsize=(14, 10))

    top_20_idx = indices[:20]
    heatmap_data = np.array(all_attns)[:, top_20_idx].T  # [20, T]

    sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'},
                yticklabels=[feature_names[i] for i in top_20_idx],
                xticklabels=False)

    ax.set_xlabel('Time Step (Test Period)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Attention Weight Heatmap (Top 20 Features)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "factor_attention_heatmap.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/factor_attention_heatmap.png")
    plt.close()

    # ==================== 圖表 5: 特徵重要性分布（餅圖） ====================
    fig, ax = plt.subplots(figsize=(10, 8))

    top_10_weights = avg_attn[indices[:10]]
    other_weight = avg_attn[indices[10:]].sum()

    labels = [feature_names[i] for i in indices[:10]] + ['Others']
    sizes = list(top_10_weights) + [other_weight]
    colors = plt.cm.Set3(np.linspace(0, 1, 11))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 10})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Feature Importance Distribution (Top 10 + Others)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "factor_attention_pie.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/factor_attention_pie.png")
    plt.close()

    # ==================== 輸出統計摘要 ====================
    print("\n" + "=" * 60)
    print("Factor Attention 分析摘要")
    print("=" * 60)

    print(f"\nTop {top_k} Most Important Features:")
    for rank, feat_idx in enumerate(indices[:top_k], 1):
        print(f"  {rank:2d}. {feature_names[feat_idx]:25s}: {avg_attn[feat_idx]:.6f} ± {std_attn[feat_idx]:.6f}")

    print(f"\nTop 10 特徵的總權重: {avg_attn[indices[:10]].sum():.4f} ({avg_attn[indices[:10]].sum()/avg_attn.sum()*100:.1f}%)")
    print(f"Top 20 特徵的總權重: {avg_attn[indices[:20]].sum():.4f} ({avg_attn[indices[:20]].sum()/avg_attn.sum()*100:.1f}%)")

    print(f"\n注意力權重統計:")
    print(f"  Mean: {avg_attn.mean():.6f}")
    print(f"  Std: {avg_attn.std():.6f}")
    print(f"  Max: {avg_attn.max():.6f} ({feature_names[avg_attn.argmax()]})")
    print(f"  Min: {avg_attn.min():.6f} ({feature_names[avg_attn.argmin()]})")

    # 儲存統計結果
    summary_path = os.path.join(args.output_dir, "factor_attention_summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write("Factor Attention 分析摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Top {top_k} Most Important Features:\n")
        for rank, feat_idx in enumerate(indices[:top_k], 1):
            f.write(f"  {rank:2d}. {feature_names[feat_idx]:25s}: {avg_attn[feat_idx]:.6f} ± {std_attn[feat_idx]:.6f}\n")
        f.write(f"\nTop 10 特徵總權重: {avg_attn[indices[:10]].sum():.4f}\n")
        f.write(f"Top 20 特徵總權重: {avg_attn[indices[:20]].sum():.4f}\n")

    print(f"\n摘要已儲存至: {summary_path}")
    print("=" * 60)


def main():
    args = parse_args()
    visualize_attention(args)
    print("\n✅ Factor Attention 視覺化完成！")


if __name__ == "__main__":
    main()
