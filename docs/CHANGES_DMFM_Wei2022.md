# DMFM Wei et al. (2022) 實作 - 變更清單

## 📋 變更摘要

本次更新完全對齊 Wei et al. (2022) 論文架構，新增了以下檔案和修改：

---

## 🆕 新增檔案

### 1. `model_dmfm_wei2022.py`
**完整的 DMFM 模型實作**

**核心功能：**
- Stock Context Encoder (BatchNorm + MLP)
- Industry Neutralization (產業中性化)
- Universe Neutralization (全市場中性化)
- Hierarchical Feature Concatenation (階層式特徵拼接)
- Deep Factor Learning (深度因子學習)
- Factor Attention Module (因子注意力模組)

**關鍵特性：**
- GAT 使用 `concat=False` (平均多頭輸出)
- 全市場 GAT 輸入是產業中性化後的 `C_I`
- 三種特徵拼接：`[C || C_I || C_U]`

---

### 2. `train_dmfm_wei2022.py`
**訓練腳本，使用論文的損失函數**

**損失函數（論文公式 13）：**
```
L = λ_attn · d + λ_IC · (1 - IC) - b
```

其中：
- `d`: Attention estimate loss = ||f - f_hat||²
- `b`: Factor return (cross-sectional regression)
- `IC`: Information Coefficient

**訓練指標：**
- Train/Test IC
- Train/Test ICIR
- Cumulative Factor Return
- Attention Distance

---

### 3. `visualize_factor_attention.py`
**Factor Attention 視覺化工具**

**生成的圖表：**
1. `factor_attention_top_features.png` - Top K 特徵重要性
2. `factor_attention_all_features.png` - 所有特徵排序
3. `factor_attention_timeseries.png` - Top 5 特徵時間序列
4. `factor_attention_heatmap.png` - Top 20 特徵熱力圖
5. `factor_attention_pie.png` - 特徵重要性分布餅圖
6. `factor_attention_summary.txt` - 統計摘要

**分析內容：**
- 哪些原始特徵對深度因子最重要
- 特徵權重的時間序列變化
- Top 10/20 特徵的總權重佔比

---

### 4. `analyze_contexts.py`
**階層式特徵分析工具**

**生成的圖表：**
1. `context_distributions.png` - 三種特徵分布比較
2. `variance_reduction.png` - 變異數降低效果
3. `variance_reduction_percentage.png` - 變異數降低百分比
4. `influence_magnitude.png` - 影響力大小分布
5. `context_pca_projection.png` - 2D PCA 投影
6. `influence_comparison.png` - 產業 vs 全市場影響力
7. `context_analysis_summary.txt` - 統計摘要

**分析內容：**
- 產業中性化效果（C → C_I）
- 全市場中性化效果（C_I → C_U）
- 產業影響 (H_I) vs 全市場影響 (H_U)
- 變異數降低百分比

---

### 5. `run_dmfm_wei2022.sh`
**完整執行流程腳本**

**執行步驟：**
1. 建立 Artifacts（新的預處理方式）
2. 訓練 DMFM 模型
3. 視覺化 Factor Attention
4. 分析階層式特徵
5. 評估模型指標
6. 投資組合回測

**使用方式：**
```bash
chmod +x run_dmfm_wei2022.sh
bash run_dmfm_wei2022.sh
```

---

### 6. `README_DMFM_Wei2022.md`
**完整的說明文件**

**包含內容：**
- 與原始代碼的差異說明
- 快速開始指南
- 分步執行教學
- 關鍵概念解釋（產業中性化、全市場中性化、Factor Attention）
- 預期結果
- 故障排除

---

## 🔧 修改檔案

### 1. `build_artifacts.py`

#### 修改 1: 移除截面標準化（Line 349-351）

**修改前：**
```python
Z = xsec_zscore(A)  # 每日截面 z-score
Ft[:,:,k] = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
```

**修改後：**
```python
# ⭐ 修改：不做截面標準化，保留原始特徵值
# DMFM 模型會使用 BatchNorm 做標準化
Ft[:,:,k] = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
```

**原因：** 論文使用 BatchNorm 在模型內部做標準化

---

#### 修改 2: 移除標籤去均值（Line 314-315）

**修改前：**
```python
# Label：未來 k 日報酬 → 每日截面去均值
df["yt"] = df.groupby(cm["date"])["fwd_ret_k"].transform(lambda s: s - np.nanmean(s.values))
```

**修改後：**
```python
# Label：未來 k 日報酬（保留原始值，不做截面去均值）
# ⭐ 修改：不做截面去均值，因為 DMFM 模型會在訓練時處理
df["yt"] = df["fwd_ret_k"]
```

**原因：** 保留原始報酬率，讓模型學習絕對收益

---

## 📊 架構對比

### 原始 DMFM (train_gat_fixed.py:176-279)

```python
class DMFM(nn.Module):
    def __init__(self, in_dim, hid=64, heads=2, ...):
        self.encoder = nn.Sequential(...)
        self.gat_industry = GATConv(hid, hid, heads=heads)  # concat=True (預設)
        self.gat_universe = GATConv(hid*heads, hid, heads=heads)
        self.factor_head = nn.Sequential(...)

    def forward(self, x_raw, edge_industry, edge_universe):
        C_t = self.encoder(x_raw)  # [N, hid]
        H_I = self.gat_industry(C_t, edge_industry)  # [N, hid*heads]
        C_t_expanded = C_t.repeat(1, self.gat_industry.heads)  # 需要擴展
        C_bar_I = C_t_expanded - H_I
        H_U = self.gat_universe(C_bar_I, edge_universe)
        C_bar_U = C_bar_I - H_U
        hierarchical_features = torch.cat([C_t_expanded, C_bar_I, C_bar_U], dim=-1)
        deep_factor = self.factor_head(hierarchical_features).squeeze(-1)
        # ... (Factor Attention)
```

**問題：**
- GAT 使用 `concat=True`，導致維度變化複雜
- 需要手動擴展 `C_t` 的維度
- 階層式中性化不夠清晰

---

### 新版 DMFM_Wei2022 (model_dmfm_wei2022.py)

```python
class DMFM_Wei2022(nn.Module):
    def __init__(self, num_features, hidden_dim=64, heads=2, ...):
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.encoder = nn.Sequential(...)
        self.gat_industry = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.gat_universe = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.factor_decoder = nn.Sequential(...)
        self.factor_attention = nn.Linear(num_features, num_features)

    def forward(self, x, industry_edge_index, universe_edge_index):
        # Step 1: BatchNorm + Encoding
        x_norm = self.batch_norm(x)  # [N, F]
        C = self.encoder(x_norm)  # [N, hidden_dim]

        # Step 2: Industry Neutralization
        H_I = self.gat_industry(C, industry_edge_index)  # [N, hidden_dim]
        H_I = F.elu(H_I)
        C_I = C - H_I  # 產業中性化

        # Step 3: Universe Neutralization
        H_U = self.gat_universe(C_I, universe_edge_index)  # [N, hidden_dim]
        H_U = F.elu(H_U)
        C_U = C_I - H_U  # 全市場中性化

        # Step 4: Hierarchical Feature Concatenation
        hierarchical_features = torch.cat([C, C_I, C_U], dim=-1)  # [N, 3*hidden_dim]

        # Step 5: Deep Factor
        deep_factor = self.factor_decoder(hierarchical_features)  # [N, 1]

        # Step 6: Factor Attention
        U = F.leaky_relu(self.factor_attention(x), negative_slope=0.2)
        attn_weights = F.softmax(U, dim=-1)

        contexts = {'C': C, 'C_I': C_I, 'C_U': C_U, 'H_I': H_I, 'H_U': H_U}
        return deep_factor, attn_weights, contexts
```

**優點：**
- GAT 使用 `concat=False`，維度統一為 `hidden_dim`
- 不需要手動擴展維度
- 階層式中性化清晰明瞭：`C → C_I → C_U`
- 完全對齊論文架構

---

## 🔑 關鍵改進

### 1. 產業中性化（Industry Neutralization）

**公式：**
```
H_I = GAT(C, Industry_Graph)  # 學習產業內影響
C_I = C - H_I                  # 移除產業影響
```

**意義：** 移除「產業效應」，保留「個股超額表現」

---

### 2. 全市場中性化（Universe Neutralization）

**公式：**
```
H_U = GAT(C_I, Universe_Graph)  # ⭐ 輸入是 C_I，不是 C！
C_U = C_I - H_U                  # 移除全市場影響
```

**意義：** 移除「全市場共同影響」，保留「純個股效應」

---

### 3. 階層式特徵拼接（Hierarchical Feature Concatenation）

**公式：**
```
[C || C_I || C_U]
```

**意義：**
- **C**：包含所有信息（原始編碼特徵）
- **C_I**：移除產業效應
- **C_U**：移除產業 + 市場效應

讓模型同時學習不同層次的信息。

---

### 4. Factor Attention 模組

**公式：**
```
U = LeakyReLU(W · F)  # 學習注意力邏輯
A = Softmax(U)        # 歸一化為權重
f_hat = F^T · A       # 注意力估計的因子
```

**損失：**
```
d = ||f - f_hat||²  # 最小化深度因子與注意力估計的差異
```

**意義：** 提高模型可解釋性，了解哪些原始特徵最重要。

---

## 📈 使用流程

### 快速開始

```bash
# 一鍵執行完整流程
bash run_dmfm_wei2022.sh
```

### 分步執行

```bash
# Step 1: 建立 Artifacts
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv unique_2019q3to2025q3.csv \
  --artifact_dir gat_artifacts_wei2022 \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5

# Step 2: 訓練模型
python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_wei2022 \
  --epochs 200 \
  --lr 1e-3 \
  --device cuda

# Step 3: 視覺化 Factor Attention
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts_wei2022 \
  --output_dir plots_attention_wei2022

# Step 4: 分析階層式特徵
python analyze_contexts.py \
  --artifact_dir gat_artifacts_wei2022 \
  --output_dir plots_contexts_wei2022
```

---

## 📝 輸出檔案

### 模型檔案
- `gat_artifacts_wei2022/dmfm_wei2022_best.pt` - 最佳模型
- `gat_artifacts_wei2022/train_log_wei2022.txt` - 訓練日誌

### 視覺化圖表
- `plots_attention_wei2022/` - Factor Attention 分析
- `plots_contexts_wei2022/` - 階層式特徵分析

### 評估結果
- `results_dmfm_wei2022_metrics.txt` - 模型指標
- `results_dmfm_wei2022_portfolio.txt` - 投資組合回測

---

## ✅ 驗證清單

- [x] 移除 build_artifacts.py 的截面標準化
- [x] 移除標籤去均值
- [x] 實作完整的 DMFM_Wei2022 模型
- [x] GAT 使用 concat=False
- [x] 產業中性化：C_I = C - H_I
- [x] 全市場中性化：C_U = C_I - H_U（輸入是 C_I）
- [x] 階層式特徵拼接：[C || C_I || C_U]
- [x] Factor Attention 模組
- [x] 論文損失函數：d - b + IC_penalty
- [x] Factor Attention 視覺化
- [x] 階層式特徵分析
- [x] 完整執行流程腳本
- [x] 完整說明文件

---

## 🎯 測試建議

1. **先使用小資料集測試**
   ```bash
   python build_artifacts.py --end_date 2020-12-31 --artifact_dir gat_artifacts_test
   python train_dmfm_wei2022.py --artifact_dir gat_artifacts_test --epochs 10
   ```

2. **檢查視覺化輸出**
   ```bash
   python visualize_factor_attention.py --artifact_dir gat_artifacts_test
   ls plots_attention_wei2022/
   ```

3. **驗證階層式特徵**
   ```bash
   python analyze_contexts.py --artifact_dir gat_artifacts_test
   cat plots_contexts_wei2022/context_analysis_summary.txt
   ```

---

## 📚 參考文獻

Wei, L., Li, B., & Chen, Y. (2022). Deep Multi-Factor Model for Stock Prediction.

---

## 作者

**Lo Yi (羅頤)**
National Yang Ming Chiao Tung University
E-mail: roy60404@gmail.com
