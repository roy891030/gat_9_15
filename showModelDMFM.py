# showModelDMFM.py
import torch
from torchviz import make_dot
from model_dmfm_wei2022 import DMFM_Wei2022

# 模擬輸入（N 可以小一點，避免圖太大）
N = 10  # 小數量股票，圖才清楚
F = 56

x = torch.randn(N, F)

# 模擬邊索引（隨機，但數量合理）
industry_edge_index = torch.randint(0, N, (2, 50))
universe_edge_index = torch.randint(0, N, (2, 200))

# 建立模型
model = DMFM_Wei2022(
    num_features=F,
    hidden_dim=64,
    heads=2,
    dropout=0.1,
    use_factor_attention=True
)

# 前向傳播（只取 deep_factor 作為輸出節點）
deep_factor, attn_weights, contexts = model(x, industry_edge_index, universe_edge_index)

# 生成計算圖
dot = make_dot(
    deep_factor,
    params=dict(model.named_parameters()),
    show_attrs=True,   # 顯示形狀
    show_saved=True    # 顯示中間變數
)

# 儲存為 PDF（最推薦，超清楚）
dot.format = 'pdf'
dot.render("DMFM_Model_Structure")  # 生成 DMFM_Model_Structure.pdf

# 也可存 PNG
# dot.format = 'png'
# dot.render("DMFM_Model_Structure")

print("DMFM 模型結構圖已生成：DMFM_Model_Structure.pdf")
print("請用 PDF 閱讀器開啟，圖非常詳細漂亮！")