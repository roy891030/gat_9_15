#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""診斷標籤分布"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 載入標籤
yt = torch.load("artifacts_short/yt_tensor.pt").numpy()  # [T, N]

print("=" * 60)
print("標籤診斷")
print("=" * 60)

# 全局統計
valid_y = yt[~np.isnan(yt)]
print(f"\n全局統計：")
print(f"  總樣本數: {len(valid_y):,}")
print(f"  均值: {valid_y.mean():.6f}")
print(f"  標準差: {valid_y.std():.6f}")
print(f"  最小值: {valid_y.min():.6f}")
print(f"  最大值: {valid_y.max():.6f}")
print(f"  中位數: {np.median(valid_y):.6f}")

# 每日截面統計
print(f"\n每日截面統計：")
daily_means = np.nanmean(yt, axis=1)
daily_stds = np.nanstd(yt, axis=1)

print(f"  截面均值的均值: {daily_means.mean():.6f}")
print(f"  截面均值的標準差: {daily_means.std():.6f}")
print(f"  截面標準差的均值: {daily_stds.mean():.6f}")  # ← 關鍵！
print(f"  截面標準差的標準差: {daily_stds.std():.6f}")

# 警告檢查
if daily_stds.mean() < 0.01:
    print(f"\n⚠️ 警告：截面標準差太小 ({daily_stds.mean():.6f})")
    print("   這會導致模型無法學習！")
    print("   建議：使用截面去均值的標籤")

# 繪製分布圖
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 全局分布
axes[0, 0].hist(valid_y, bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Overall Label Distribution')
axes[0, 0].set_xlabel('Return')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)

# 2. 截面均值時序圖
axes[0, 1].plot(daily_means, linewidth=1)
axes[0, 1].set_title('Daily Cross-Sectional Mean')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Mean Return')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)

# 3. 截面標準差時序圖
axes[1, 0].plot(daily_stds, linewidth=1, color='orange')
axes[1, 0].set_title('Daily Cross-Sectional Std')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Std Return')
axes[1, 0].axhline(0.01, color='red', linestyle='--', linewidth=1, label='Warning threshold')
axes[1, 0].legend()

# 4. 隨機選一天的截面分布
random_day = np.random.randint(0, yt.shape[0])
day_data = yt[random_day]
day_data = day_data[~np.isnan(day_data)]

axes[1, 1].hist(day_data, bins=50, alpha=0.7, edgecolor='black', color='green')
axes[1, 1].set_title(f'Day {random_day} Cross-Sectional Distribution')
axes[1, 1].set_xlabel('Return')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].text(0.05, 0.95, f'Mean: {day_data.mean():.4f}\nStd: {day_data.std():.4f}',
                transform=axes[1, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('label_diagnosis.png', dpi=150)
print(f"\n圖表已儲存: label_diagnosis.png")