# summary_dmfm.py
from torchinfo import summary
from model_dmfm_wei2022 import DMFM_Wei2022 as DMFM
import torch
model = DMFM(num_features=56, hidden_dim=64, heads=2, dropout=0.1)

summary(model, 
        input_data=[torch.randn(771, 56), 
                    torch.randint(0, 771, (2, 27923)), 
                    torch.randint(0, 771, (2, 594441))],
        col_names=["input_size", "output_size", "num_params"],
        depth=4,
        verbose=2)