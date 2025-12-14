# -*- coding: utf-8 -*-
# analyze_contexts.py
"""
階層式特徵分析工具

分析 DMFM 的三種特徵：
1. C: 原始編碼特徵
2. C_I: 產業中性化特徵（C - H_I）
3. C_U: 全市場中性化特徵（C_I - H_U）

目的：
- 了解產業中性化和全市場中性化的效果
- 分析兩種影響（H_I, H_U）的大小和分布
- 驗證階層式中性化是否有效降低變異
"""

import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_dmfm_wei2022 import DMFM_Wei2022

# 設定視覺化樣式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def parse_args():
    ap = argparse.ArgumentParser(description="分析階層式特徵 (C, C_I, C_U)")
    ap.add_argument("--artifact_dir", default="gat_artifacts_out_plus", help="Artifacts 資料夾")
    ap.add_argument("--model_path", default=None, help="模型檔案路徑")
    ap.add_argument("--output_dir", default="plots_contexts", help="輸出圖表資料夾")
    ap.add_argument("--device", default="cpu", help="計算裝置")
    ap.add_argument("--sample_days", type=int, default=10, help="分析的樣本天數")
    return ap.parse_args()


def analyze_contexts(args):
    """分析三種特徵的差異"""

    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 metadata
    with open(os.path.join(args.artifact_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    F = len(meta['feature_cols'])

    # 載入模型
    model = DMFM_Wei2022(num_features=F, hidden_dim=64, heads=2)
    model_path = args.model_path or os.path.join(args.artifact_dir, "dmfm_wei2022_best.pt")

    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案 {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    # 載入資料
    Ft = torch.load(os.path.join(args.artifact_dir, "Ft_tensor.pt"))
    industry_ei = torch.load(os.path.join(args.artifact_dir, "industry_edge_index.pt"))
    universe_ei = torch.load(os.path.join(args.artifact_dir, "universe_edge_index.pt"))

    T, N, _ = Ft.shape
    split_idx = int(T * 0.8)
    Ft_test = Ft[split_idx:]

    print(f"分析測試期的 {min(args.sample_days, len(Ft_test))} 天資料...")

    # 收集多天的 contexts
    all_C = []
    all_C_I = []
    all_C_U = []
    all_H_I = []
    all_H_U = []

    with torch.no_grad():
        for t in range(min(args.sample_days, len(Ft_test))):
            x_t = Ft_test[t]
            _, _, contexts = model(x_t, industry_ei, universe_ei)

            all_C.append(contexts['C'].numpy())
            all_C_I.append(contexts['C_I'].numpy())
            all_C_U.append(contexts['C_U'].numpy())
            all_H_I.append(contexts['H_I'].numpy())
            all_H_U.append(contexts['H_U'].numpy())

    # 拼接所有天的資料 [days * N, hidden_dim]
    C = np.concatenate(all_C, axis=0)
    C_I = np.concatenate(all_C_I, axis=0)
    C_U = np.concatenate(all_C_U, axis=0)
    H_I = np.concatenate(all_H_I, axis=0)
    H_U = np.concatenate(all_H_U, axis=0)

    print(f"資料形狀: C={C.shape}, C_I={C_I.shape}, C_U={C_U.shape}")

    # ==================== 圖表 1: 特徵分布比較 ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 隨機選擇 5 個維度來視覺化
    sample_dims = np.random.choice(C.shape[1], size=min(5, C.shape[1]), replace=False)

    for dim in sample_dims:
        axes[0].hist(C[:, dim], bins=50, alpha=0.5, label=f'Dim {dim}')
        axes[1].hist(C_I[:, dim], bins=50, alpha=0.5, label=f'Dim {dim}')
        axes[2].hist(C_U[:, dim], bins=50, alpha=0.5, label=f'Dim {dim}')

    axes[0].set_title('Original Context (C)', fontweight='bold')
    axes[1].set_title('Industry-Neutralized (C_I)', fontweight='bold')
    axes[2].set_title('Universe-Neutralized (C_U)', fontweight='bold')

    for ax in axes:
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "context_distributions.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/context_distributions.png")
    plt.close()

    # ==================== 圖表 2: 變異數降低分析 ====================
    var_C = np.var(C, axis=0)
    var_C_I = np.var(C_I, axis=0)
    var_C_U = np.var(C_U, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(var_C))
    width = 0.25

    ax.bar(x - width, var_C, width, label='C (Original)', alpha=0.8, color='steelblue')
    ax.bar(x, var_C_I, width, label='C_I (Ind. Neutral)', alpha=0.8, color='orange')
    ax.bar(x + width, var_C_U, width, label='C_U (Univ. Neutral)', alpha=0.8, color='green')

    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Variance Reduction through Hierarchical Neutralization', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "variance_reduction.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/variance_reduction.png")
    plt.close()

    # ==================== 圖表 3: 變異數降低百分比 ====================
    var_reduction_I = (var_C - var_C_I) / var_C * 100  # %
    var_reduction_U = (var_C_I - var_C_U) / var_C_I * 100  # %

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(np.arange(len(var_reduction_I)), var_reduction_I, alpha=0.8, color='coral')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Variance Reduction (%)')
    axes[0].set_title('Industry Neutralization Effect (C → C_I)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(np.arange(len(var_reduction_U)), var_reduction_U, alpha=0.8, color='seagreen')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Variance Reduction (%)')
    axes[1].set_title('Universe Neutralization Effect (C_I → C_U)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "variance_reduction_percentage.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/variance_reduction_percentage.png")
    plt.close()

    # ==================== 圖表 4: 影響力大小分布 ====================
    H_I_norm = np.linalg.norm(H_I, axis=1)  # [N]
    H_U_norm = np.linalg.norm(H_U, axis=1)  # [N]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(H_I_norm, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=H_I_norm.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={H_I_norm.mean():.3f}')
    axes[0].set_title('Industry Influence Magnitude (||H_I||)', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('||H_I||')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(H_U_norm, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=H_U_norm.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={H_U_norm.mean():.3f}')
    axes[1].set_title('Universe Influence Magnitude (||H_U||)', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('||H_U||')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "influence_magnitude.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/influence_magnitude.png")
    plt.close()

    # ==================== 圖表 5: 特徵空間的 2D 投影 (PCA) ====================
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    # 對三種特徵分別做 PCA
    C_2d = pca.fit_transform(C)
    C_I_2d = pca.fit_transform(C_I)
    C_U_2d = pca.fit_transform(C_U)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 隨機採樣 1000 個點以加快繪圖
    sample_size = min(1000, C_2d.shape[0])
    sample_idx = np.random.choice(C_2d.shape[0], size=sample_size, replace=False)

    axes[0].scatter(C_2d[sample_idx, 0], C_2d[sample_idx, 1], alpha=0.3, s=10, color='steelblue')
    axes[0].set_title('Original Context (C)', fontweight='bold')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    axes[1].scatter(C_I_2d[sample_idx, 0], C_I_2d[sample_idx, 1], alpha=0.3, s=10, color='orange')
    axes[1].set_title('Industry-Neutralized (C_I)', fontweight='bold')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    axes[2].scatter(C_U_2d[sample_idx, 0], C_U_2d[sample_idx, 1], alpha=0.3, s=10, color='green')
    axes[2].set_title('Universe-Neutralized (C_U)', fontweight='bold')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "context_pca_projection.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/context_pca_projection.png")
    plt.close()

    # ==================== 圖表 6: 影響力相對大小 ====================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(H_I_norm, H_U_norm, alpha=0.3, s=20, color='purple')
    ax.plot([0, H_I_norm.max()], [0, H_I_norm.max()], 'r--', linewidth=2, label='H_I = H_U')
    ax.set_xlabel('Industry Influence (||H_I||)', fontsize=12)
    ax.set_ylabel('Universe Influence (||H_U||)', fontsize=12)
    ax.set_title('Comparison of Industry vs Universe Influence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "influence_comparison.png"), dpi=300)
    print(f"✓ 已儲存: {args.output_dir}/influence_comparison.png")
    plt.close()

    # ==================== 統計摘要 ====================
    print("\n" + "=" * 60)
    print("階層式特徵分析摘要")
    print("=" * 60)

    print("\n變異數統計:")
    print(f"  C (原始):             Mean={var_C.mean():.4f}, Std={var_C.std():.4f}")
    print(f"  C_I (產業中性):       Mean={var_C_I.mean():.4f}, Std={var_C_I.std():.4f}")
    print(f"  C_U (全市場中性):     Mean={var_C_U.mean():.4f}, Std={var_C_U.std():.4f}")

    print("\n變異數降低效果:")
    print(f"  產業中性化 (C → C_I): {(1 - var_C_I.mean()/var_C.mean())*100:.2f}%")
    print(f"  全市場中性化 (C_I → C_U): {(1 - var_C_U.mean()/var_C_I.mean())*100:.2f}%")
    print(f"  總體降低 (C → C_U):   {(1 - var_C_U.mean()/var_C.mean())*100:.2f}%")

    print("\n影響力統計:")
    print(f"  產業影響 (H_I):")
    print(f"    Mean magnitude: {H_I_norm.mean():.4f}")
    print(f"    Std magnitude:  {H_I_norm.std():.4f}")
    print(f"    Max magnitude:  {H_I_norm.max():.4f}")

    print(f"\n  全市場影響 (H_U):")
    print(f"    Mean magnitude: {H_U_norm.mean():.4f}")
    print(f"    Std magnitude:  {H_U_norm.std():.4f}")
    print(f"    Max magnitude:  {H_U_norm.max():.4f}")

    print(f"\n  影響力比值 (H_I / H_U):")
    ratio = H_I_norm / (H_U_norm + 1e-8)
    print(f"    Mean: {ratio.mean():.4f}")
    print(f"    Median: {np.median(ratio):.4f}")

    # 儲存統計結果
    summary_path = os.path.join(args.output_dir, "context_analysis_summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write("階層式特徵分析摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write("變異數統計:\n")
        f.write(f"  C (原始):             Mean={var_C.mean():.4f}, Std={var_C.std():.4f}\n")
        f.write(f"  C_I (產業中性):       Mean={var_C_I.mean():.4f}, Std={var_C_I.std():.4f}\n")
        f.write(f"  C_U (全市場中性):     Mean={var_C_U.mean():.4f}, Std={var_C_U.std():.4f}\n")
        f.write(f"\n變異數降低效果:\n")
        f.write(f"  產業中性化: {(1 - var_C_I.mean()/var_C.mean())*100:.2f}%\n")
        f.write(f"  全市場中性化: {(1 - var_C_U.mean()/var_C_I.mean())*100:.2f}%\n")
        f.write(f"  總體降低: {(1 - var_C_U.mean()/var_C.mean())*100:.2f}%\n")

    print(f"\n摘要已儲存至: {summary_path}")
    print("=" * 60)


def main():
    args = parse_args()
    analyze_contexts(args)
    print("\n✅ 階層式特徵分析完成！")


if __name__ == "__main__":
    main()
