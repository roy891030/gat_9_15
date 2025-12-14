#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""診斷模型輸出"""

import torch
from train_gat_fixed import load_artifacts
from model_dmfm_wei2022 import GATRegressor

# 載入資料
Ft, yt, edge_industry, edge_universe, meta = load_artifacts("artifacts_short")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Ft = Ft.to(device).float()
edge_industry = edge_industry.to(device)

# 載入模型
model = GATRegressor(in_dim=56, hid=64, heads=2, tanh_cap=0.2).to(device)
model.load_state_dict(torch.load("artifacts_short/gat_regressor.pt", map_location=device))
model.eval()

print("=" * 60)
print("模型輸出診斷")
print("=" * 60)

# 隨機選 5 天
test_idx = list(range(int(len(Ft) * 0.8), len(Ft)))
sample_days = test_idx[:5]

with torch.no_grad():
    for t in sample_days:
        x = Ft[t]
        pred = model(x, edge_industry).cpu().numpy()
        
        print(f"\nDay {t}:")
        print(f"  預測值統計：")
        print(f"    均值: {pred.mean():.6f}")
        print(f"    標準差: {pred.std():.6f}")  # ← 關鍵！
        print(f"    最小值: {pred.min():.6f}")
        print(f"    最大值: {pred.max():.6f}")
        print(f"    唯一值數量: {len(set(pred.round(6)))}")
        
        if pred.std() < 1e-6:
            print(f"  ⚠️ 警告：預測值幾乎是常數！")

print("\n" + "=" * 60)