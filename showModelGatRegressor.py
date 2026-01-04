# visualize_gat.py
# 生成 GATRegressor 結構圖（和 DMFM PDF 一模一樣風格！）

import torch
from torchviz import make_dot
from model_dmfm_wei2022 import GATRegressor

# 參數設定（和 DMFM 一致，讓圖風格統一）
N = 8                    # 股票數，8 最清楚（太大會擠）
F = 56                   # 特徵數

# 模擬輸入（只用 industry graph 就夠）
x = torch.randn(N, F)
edge_index = torch.randint(0, N, (2, 100))  # 隨機邊

# 建立模型（用你訓練時的參數，hid=64, heads=2）
model = GATRegressor(
    in_dim=F,
    hid=64,     # 你的預設值
    heads=2,
    dropout=0.1,
    tanh_cap=1.0
)

# 前向傳播
pred = model(x, edge_index)

# 生成計算圖
dot = make_dot(
    pred,
    params=dict(model.named_parameters()),
    show_attrs=True,   # 顯示張量形狀
    show_saved=True
)

# 儲存為 PDF（最清楚，和 DMFM 一樣）
dot.format = 'pdf'
dot.render("GATRegressor_Structure")  # 生成 GATRegressor_Structure.pdf

print("GATRegressor 結構圖已生成：GATRegressor_Structure.pdf")
print("風格和 DMFM_Model_Structure.pdf 完全一致！")
