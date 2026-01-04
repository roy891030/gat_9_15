# showModelGatRegressor.py
# 生成 GATRegressor 簡潔摘要（論文最愛這種！）

from torchinfo import summary
from model_dmfm_wei2022 import GATRegressor

# 建立模型
model = GATRegressor(
    in_dim=56,
    hid=64,
    heads=2,
    dropout=0.1,
    tanh_cap=1.0
)

# 生成摘要（batch_size=1, N=771 股票）
summary(model, input_data=[torch.randn(771, 56), torch.randint(0, 771, (2, 27923))],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3,  # 顯示層級
        verbose=2)

print("\n這就是論文中最常見的模型摘要表！直接截圖放論文超專業")