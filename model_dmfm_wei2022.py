# -*- coding: utf-8 -*-
"""
Deep Multi-Factor Model (DMFM) - Wei et al. (2022) 完整實作

論文架構對齊說明：
1. Stock Context Encoder: BatchNorm + MLP
2. Industry Neutralization: GAT(Industry Graph) + Subtraction
3. Universe Neutralization: GAT(Universe Graph) + Subtraction
4. Hierarchical Feature Concatenation: [C || C_I || C_U]
5. Deep Factor Learning: MLP Decoder
6. Factor Attention Module: Interpretability

關鍵差異：
- GAT 使用 concat=False（平均多頭輸出）
- 產業中性化：C_I = C - H_I
- 全市場中性化：C_U = C_I - H_U（注意輸入是 C_I）
- 三種特徵拼接後輸出深度因子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DMFM_Wei2022(nn.Module):
    """
    Deep Multi-Factor Model（完全對齊 Wei et al. 2022 論文）

    架構流程：
    原始特徵 F^t [N, F]
        ↓
    [BatchNorm + MLP Encoder]
        ↓
    編碼特徵 C^t [N, hidden_dim]  ← 第一種特徵
        ↓
    [GAT on Industry Graph]
        ↓
    產業影響 H^t_I [N, hidden_dim]
        ↓
    C^t - H^t_I = C̄^t_I [N, hidden_dim]  ← 第二種特徵 (產業中性化)
        ↓
    [GAT on Universe Graph]
        ↓
    全市場影響 H^t_U [N, hidden_dim]
        ↓
    C̄^t_I - H^t_U = C̄^t_U [N, hidden_dim]  ← 第三種特徵 (全市場中性化)
        ↓
    [Concatenate: C^t || C̄^t_I || C̄^t_U]
        ↓
    拼接特徵 [N, 3*hidden_dim]
        ↓
    [MLP Decoder]
        ↓
    深度因子 f^t [N, 1]
        ↓
    [Factor Attention Module] ← 解釋模組
        ↓
    注意力權重 ā^t [F]  (顯示哪些原始特徵重要)
    """

    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 heads: int = 2,
                 dropout: float = 0.0,
                 use_factor_attention: bool = True):
        """
        參數：
            num_features: 原始特徵數量 F
            hidden_dim: 隱藏層維度
            heads: GAT 注意力頭數
            dropout: Dropout 比例
            use_factor_attention: 是否使用因子注意力模組
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_factor_attention = use_factor_attention

        # ==================== 1. Stock Context Encoder ====================
        # BatchNorm (相當於截面 z-score normalization)
        self.batch_norm = nn.BatchNorm1d(num_features)

        # MLP Encoder: F -> hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # ==================== 2. Industry Influence Learning ====================
        # GAT for Industry Graph
        # ⭐ 關鍵：concat=False，平均多頭輸出，維度保持為 hidden_dim
        self.gat_industry = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,  # ← 論文要求：平均多頭，不拼接
            dropout=dropout,
            add_self_loops=True
        )

        # ==================== 3. Universe Influence Learning ====================
        # GAT for Universe Graph
        # ⭐ 關鍵：輸入是產業中性化後的 C_I，維度仍為 hidden_dim
        self.gat_universe = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,  # ← 論文要求：平均多頭
            dropout=dropout,
            add_self_loops=True
        )

        # ==================== 4. Deep Factor Learning ====================
        # 輸入是三種特徵的拼接：[C || C_I || C_U]，維度 = 3 * hidden_dim
        self.factor_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # ==================== 5. Factor Attention Module ====================
        # 用於解釋深度因子來自哪些原始特徵
        if use_factor_attention:
            self.factor_attention = nn.Linear(num_features, num_features)

    def forward(self, x, industry_edge_index, universe_edge_index):
        """
        前向傳播（完全對齊論文 Section 3.2）

        參數：
            x: [N, F] 原始特徵矩陣
            industry_edge_index: [2, E_ind] 產業圖的邊索引
            universe_edge_index: [2, E_uni] 全市場圖的邊索引

        回傳：
            deep_factor: [N, 1] 深度因子
            attn_weights: [N, F] 注意力權重（若啟用）
            contexts: dict 包含三種特徵 (用於分析)
        """
        N, num_features = x.shape

        # ==================== Step 1: Stock Context Encoding ====================
        # BatchNorm（等價於截面 z-score）
        x_norm = self.batch_norm(x)  # [N, num_features]

        # MLP Encoding
        C = self.encoder(x_norm)  # [N, hidden_dim]

        # ==================== Step 2: Industry Influence & Neutralization ====================
        # 學習產業影響（論文公式 3）
        H_I = self.gat_industry(C, industry_edge_index)  # [N, hidden_dim]
        H_I = F.elu(H_I)

        # 產業中性化（論文公式 4）：移除產業影響
        C_I = C - H_I  # [N, hidden_dim]

        # ==================== Step 3: Universe Influence & Neutralization ====================
        # ⭐ 關鍵：Universe GAT 作用在產業中性化後的特徵 C_I！
        H_U = self.gat_universe(C_I, universe_edge_index)  # [N, hidden_dim]
        H_U = F.elu(H_U)

        # 全市場中性化（論文公式 6）：移除全市場影響
        C_U = C_I - H_U  # [N, hidden_dim]

        # ==================== Step 4: Hierarchical Feature Concatenation ====================
        # 拼接三種特徵（論文公式 7）
        hierarchical_features = torch.cat([C, C_I, C_U], dim=-1)  # [N, 3*hidden_dim]

        # ==================== Step 5: Learn Deep Factor ====================
        deep_factor = self.factor_decoder(hierarchical_features)  # [N, 1]

        # ==================== Step 6: Factor Attention (Interpretation) ====================
        if self.use_factor_attention:
            # 學習原始特徵的注意力權重（論文公式 9-12）
            U = F.leaky_relu(self.factor_attention(x), negative_slope=0.2)  # [N, F]
            attn_weights = F.softmax(U, dim=-1)  # [N, F] 每個股票對每個特徵的注意力
        else:
            attn_weights = None

        # ==================== Collect Contexts for Analysis ====================
        contexts = {
            'C': C,       # 原始編碼特徵
            'C_I': C_I,   # 產業中性化特徵
            'C_U': C_U,   # 全市場中性化特徵
            'H_I': H_I,   # 產業影響
            'H_U': H_U    # 全市場影響
        }

        return deep_factor, attn_weights, contexts

    def interpret_factor(self, x, attn_weights):
        """
        解釋深度因子：f̂^t = F^T · ā^t

        參數：
            x: [N, F] 原始特徵
            attn_weights: [N, F] 注意力權重

        回傳：
            f_hat: [N, 1] 注意力估計的因子
        """
        if attn_weights is None:
            return None

        # 用注意力權重加權原始特徵（論文公式 12）
        f_hat = (x * attn_weights).sum(dim=-1, keepdim=True)  # [N, 1]
        return f_hat

    def get_attention_importance(self, x, attn_weights):
        """
        計算每個特徵的平均重要性

        參數：
            x: [N, F] 原始特徵
            attn_weights: [N, F] 注意力權重

        回傳：
            importance: [F] 每個特徵的平均注意力權重
        """
        if attn_weights is None:
            return None

        # 平均所有股票的注意力權重 → 得到特徵重要性
        importance = attn_weights.mean(dim=0)  # [F]
        return importance


class DMFM_Lite(nn.Module):
    """
    DMFM 的輕量版（用於記憶體受限的環境）

    差異：
    - 全市場圖使用 K-近鄰代替全連接
    - 減少隱藏層維度
    - 簡化 MLP 結構
    """

    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 32,  # 減小隱藏層
                 heads: int = 2,
                 dropout: float = 0.0):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Encoder (簡化版)
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.encoder = nn.Linear(num_features, hidden_dim)

        # GAT (簡化版)
        self.gat_industry = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gat_universe = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Decoder (簡化版)
        self.factor_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.factor_attention = nn.Linear(num_features, num_features)

    def forward(self, x, industry_edge_index, universe_edge_index):
        """輕量版前向傳播"""
        x_norm = self.batch_norm(x)
        C = F.relu(self.encoder(x_norm))

        H_I = F.elu(self.gat_industry(C, industry_edge_index))
        C_I = C - H_I

        H_U = F.elu(self.gat_universe(C_I, universe_edge_index))
        C_U = C_I - H_U

        hierarchical_features = torch.cat([C, C_I, C_U], dim=-1)
        deep_factor = self.factor_decoder(hierarchical_features)

        U = F.leaky_relu(self.factor_attention(x), negative_slope=0.2)
        attn_weights = F.softmax(U, dim=-1)

        contexts = {'C': C, 'C_I': C_I, 'C_U': C_U, 'H_I': H_I, 'H_U': H_U}

        return deep_factor, attn_weights, contexts


if __name__ == "__main__":
    """簡單測試"""
    print("=" * 60)
    print("DMFM Wei et al. (2022) 模型測試")
    print("=" * 60)

    # 模擬資料
    N, F = 100, 56  # 100 檔股票，56 個特徵
    x = torch.randn(N, F)

    # 模擬產業圖（20 檔股票為一個產業）
    industry_edges = []
    for i in range(0, N, 20):
        group = list(range(i, min(i+20, N)))
        for a in group:
            for b in group:
                industry_edges.append([a, b])
    industry_edge_index = torch.tensor(industry_edges, dtype=torch.long).t()

    # 模擬全市場圖（全連接，記憶體測試用較小圖）
    universe_edges = [[i, j] for i in range(N) for j in range(N)]
    universe_edge_index = torch.tensor(universe_edges, dtype=torch.long).t()

    # 建立模型
    model = DMFM_Wei2022(num_features=F, hidden_dim=64, heads=2)

    # 前向傳播
    deep_factor, attn_weights, contexts = model(x, industry_edge_index, universe_edge_index)

    print(f"\n輸入特徵: {x.shape}")
    print(f"產業圖邊數: {industry_edge_index.shape[1]}")
    print(f"全市場圖邊數: {universe_edge_index.shape[1]}")
    print(f"\n輸出:")
    print(f"  深度因子: {deep_factor.shape}")
    print(f"  注意力權重: {attn_weights.shape if attn_weights is not None else 'None'}")
    print(f"\n階層式特徵:")
    print(f"  C (原始編碼): {contexts['C'].shape}")
    print(f"  C_I (產業中性): {contexts['C_I'].shape}")
    print(f"  C_U (全市場中性): {contexts['C_U'].shape}")
    print(f"  H_I (產業影響): {contexts['H_I'].shape}")
    print(f"  H_U (全市場影響): {contexts['H_U'].shape}")

    # 計算特徵重要性
    importance = model.get_attention_importance(x, attn_weights)
    if importance is not None:
        print(f"\n特徵重要性: {importance.shape}")
        print(f"  Top 5 特徵索引: {importance.argsort(descending=True)[:5].tolist()}")
        print(f"  Top 5 權重: {importance.sort(descending=True)[0][:5].tolist()}")

    # 解釋因子
    f_hat = model.interpret_factor(x, attn_weights)
    if f_hat is not None:
        print(f"\n注意力估計因子: {f_hat.shape}")
        print(f"  深度因子 vs 注意力因子誤差: {torch.norm(deep_factor - f_hat, p=2).item():.4f}")

    # 參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型參數:")
    print(f"  總參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)
