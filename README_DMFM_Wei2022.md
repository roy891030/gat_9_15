# DMFM (Wei et al. 2022) 實作說明

## 📋 專案概述

本實作完全對齊 Wei et al. (2022) 論文 "Deep Multi-Factor Model" 的架構，包含：

1. **階層式雙圖結構**：產業圖 + 全市場圖
2. **產業中性化**：C_I = C - H_I
3. **全市場中性化**：C_U = C_I - H_U
4. **階層式特徵拼接**：[C || C_I || C_U]
5. **Factor Attention 模組**：解釋深度因子來自哪些原始特徵

---

## 🆕 與原始代碼的差異

### 1. 資料預處理（`build_artifacts.py`）

**修改前：**
```python
Z = xsec_zscore(A)  # 截面標準化
df["yt"] = df.groupby(cm["date"])["fwd_ret_k"].transform(lambda s: s - np.nanmean(s))  # 標籤去均值
```

**修改後：**
```python
# 不做截面標準化，保留原始特徵
Ft[:,:,k] = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
# 標籤不去均值
df["yt"] = df["fwd_ret_k"]
```

**原因：** 論文使用 BatchNorm 在模型內部做標準化，而不是預處理階段。

---

### 2. 模型架構（`model_dmfm_wei2022.py`）

#### **關鍵差異 1：GAT 使用 `concat=False`**

**原始代碼（`train_gat_fixed.py`）：**
```python
self.gat_industry = GATConv(hid, hid, heads=heads)  # 預設 concat=True
# 輸出維度：hid * heads
```

**新代碼：**
```python
self.gat_industry = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
# 輸出維度：hidden_dim（平均多頭）
```

#### **關鍵差異 2：階層式中性化**

**原始代碼：**
```python
C_t_expanded = C_t.repeat(1, self.gat_industry.heads)  # 需要擴展維度
C_bar_I = C_t_expanded - H_I
```

**新代碼：**
```python
# 不需要擴展維度
C_I = C - H_I  # 產業中性化
C_U = C_I - H_U  # 全市場中性化（注意：輸入是 C_I）
```

#### **關鍵差異 3：全市場 GAT 的輸入**

**原始代碼：** 全市場 GAT 的輸入不明確

**新代碼：**
```python
H_I = self.gat_industry(C, industry_edge_index)
C_I = C - H_I
H_U = self.gat_universe(C_I, universe_edge_index)  # ← 輸入是產業中性化後的 C_I！
C_U = C_I - H_U
```

---

### 3. 損失函數（`train_dmfm_wei2022.py`）

**論文公式 13：**
```
L = λ_attn · d - b + λ_IC · (1 - IC)
```

其中：
- `d`：Attention estimate loss = ||f - f_hat||²
- `b`：Factor return (cross-sectional regression)
- `IC`：Information Coefficient

**實作：**
```python
def compute_loss(deep_factor, f_hat, returns, lambda_attn, lambda_ic):
    d = torch.norm(deep_factor - f_hat, p=2)
    b = cross_sectional_regression(deep_factor, returns)
    ic = compute_ic(deep_factor, returns)
    ic_penalty = 1.0 - ic

    loss = lambda_attn * d + lambda_ic * ic_penalty - b
    return loss
```

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm torch-geometric
```

### 2. 一鍵執行完整流程

```bash
bash run_dmfm_wei2022.sh
```

這個腳本會自動執行以下步驟：
1. 建立 Artifacts（新的預處理方式）
2. 訓練 DMFM 模型
3. 視覺化 Factor Attention
4. 分析階層式特徵
5. 評估模型指標
6. 投資組合回測

---

## 📂 檔案結構

```
gat_9_15/
├── build_artifacts.py           # 修改：移除截面標準化
├── model_dmfm_wei2022.py         # 新增：完整 DMFM 架構
├── train_dmfm_wei2022.py         # 新增：論文損失函數
├── visualize_factor_attention.py # 新增：Factor Attention 視覺化
├── analyze_contexts.py           # 新增：階層式特徵分析
├── run_dmfm_wei2022.sh           # 新增：完整執行流程
└── README_DMFM_Wei2022.md        # 新增：說明文件
```

---

## 🔍 分步執行

如果你想手動執行各個步驟，可以按照以下順序：

### Step 1: 建立 Artifacts

```bash
python build_artifacts.py \
  --prices unique_2019q3to2025q3.csv \
  --industry_csv unique_2019q3to2025q3.csv \
  --artifact_dir gat_artifacts_wei2022 \
  --start_date 2019-09-16 \
  --end_date 2025-09-12 \
  --horizon 5
```

**輸出：**
- `gat_artifacts_wei2022/Ft_tensor.pt` - 特徵張量（未標準化）
- `gat_artifacts_wei2022/yt_tensor.pt` - 標籤張量（未去均值）
- `gat_artifacts_wei2022/industry_edge_index.pt` - 產業圖
- `gat_artifacts_wei2022/universe_edge_index.pt` - 全市場圖

---

### Step 2: 訓練模型

```bash
python train_dmfm_wei2022.py \
  --artifact_dir gat_artifacts_wei2022 \
  --epochs 200 \
  --lr 1e-3 \
  --device cuda \
  --hidden_dim 64 \
  --heads 2 \
  --dropout 0.1 \
  --lambda_attn 0.1 \
  --lambda_ic 1.0
```

**輸出：**
- `gat_artifacts_wei2022/dmfm_wei2022_best.pt` - 最佳模型
- `gat_artifacts_wei2022/train_log_wei2022.txt` - 訓練日誌

**訓練指標：**
- Train Loss
- Train IC
- Test IC
- Test ICIR
- Test Factor Return

---

### Step 3: 視覺化 Factor Attention

```bash
python visualize_factor_attention.py \
  --artifact_dir gat_artifacts_wei2022 \
  --output_dir plots_attention_wei2022 \
  --top_k 15
```

**輸出圖表：**
- `factor_attention_top_features.png` - Top K 特徵重要性
- `factor_attention_all_features.png` - 所有特徵排序
- `factor_attention_timeseries.png` - Top 5 特徵的時間序列
- `factor_attention_heatmap.png` - Top 20 特徵熱力圖
- `factor_attention_pie.png` - 特徵重要性分布餅圖
- `factor_attention_summary.txt` - 統計摘要

---

### Step 4: 分析階層式特徵

```bash
python analyze_contexts.py \
  --artifact_dir gat_artifacts_wei2022 \
  --output_dir plots_contexts_wei2022 \
  --sample_days 20
```

**輸出圖表：**
- `context_distributions.png` - 三種特徵的分布比較
- `variance_reduction.png` - 變異數降低效果
- `variance_reduction_percentage.png` - 變異數降低百分比
- `influence_magnitude.png` - 影響力大小分布
- `context_pca_projection.png` - 2D PCA 投影
- `influence_comparison.png` - 產業 vs 全市場影響力
- `context_analysis_summary.txt` - 統計摘要

---

## 📊 關鍵概念解釋

### 1. 產業中性化 (Industry Neutralization)

**目的：** 移除產業共同影響，保留個股超額表現

**公式：**
```
H_I = GAT(C, Industry_Graph)  # 學習產業內的共同影響
C_I = C - H_I                  # 移除產業影響
```

**範例：**
- 整個半導體產業都在漲 +5%
- 台積電漲 +8%
- 產業中性化後，台積電的特徵保留 +3%（超額部分）

---

### 2. 全市場中性化 (Universe Neutralization)

**目的：** 移除全市場共同影響，保留純個股效應

**公式：**
```
H_U = GAT(C_I, Universe_Graph)  # ← 注意：輸入是 C_I
C_U = C_I - H_U                  # 移除全市場影響
```

**範例：**
- 全市場都在漲 +2%（多頭市場）
- 台積電（產業中性化後）漲 +3%
- 全市場中性化後，台積電的特徵保留 +1%（純個股效應）

---

### 3. 階層式特徵拼接

**公式：**
```
[C || C_I || C_U]
```

**三種特徵的意義：**
- **C**：原始編碼特徵（包含所有信息）
- **C_I**：產業中性化特徵（移除產業效應）
- **C_U**：全市場中性化特徵（移除產業 + 市場效應）

**為什麼拼接？**
- 讓模型同時學習不同層次的信息
- 全局信息（C）+ 產業內信息（C_I）+ 純個股信息（C_U）

---

### 4. Factor Attention 模組

**目的：** 解釋深度因子來自哪些原始特徵

**公式：**
```
U = LeakyReLU(W · F)  # 學習注意力邏輯
A = Softmax(U)        # 歸一化為權重
f_hat = F^T · A       # 注意力估計的因子
```

**損失函數：**
```
d = ||f - f_hat||²  # 最小化深度因子與注意力估計的差異
```

**應用：**
- 查看哪些技術指標最重要（如 RSI、MACD）
- 了解模型的決策依據
- 提高模型的可解釋性

---

## 📈 預期結果

### 訓練指標

| 指標 | 預期範圍 | 說明 |
|------|----------|------|
| Test IC | 0.03 - 0.08 | Information Coefficient |
| Test ICIR | 0.5 - 1.5 | IC 穩定性 |
| Factor Return | > 0 | 因子累積收益 |

### Factor Attention

**預期發現：**
- 動量類特徵（ret_10, ret_20）通常權重較高
- 技術指標（RSI, MACD）也有顯著貢獻
- Top 10 特徵通常佔總權重的 40-60%

### 階層式特徵分析

**預期效果：**
- 產業中性化：變異數降低 10-30%
- 全市場中性化：變異數再降低 5-15%
- 總體變異數降低：15-40%

---

## 🐛 故障排除

### 問題 1: CUDA Out of Memory

**解決方案 1：** 使用較小的 hidden_dim
```bash
python train_dmfm_wei2022.py --hidden_dim 32 --device cuda
```

**解決方案 2：** 使用 CPU
```bash
python train_dmfm_wei2022.py --device cpu
```

**解決方案 3：** 使用輕量版模型（需修改代碼）
```python
from model_dmfm_wei2022 import DMFM_Lite
model = DMFM_Lite(num_features=F, hidden_dim=32, heads=2)
```

---

### 問題 2: 找不到模型檔案

**錯誤訊息：**
```
錯誤：找不到模型檔案 gat_artifacts_wei2022/dmfm_wei2022_best.pt
```

**解決方案：** 先訓練模型
```bash
python train_dmfm_wei2022.py --artifact_dir gat_artifacts_wei2022
```

---

### 問題 3: 圖表中文亂碼

**解決方案：** 安裝中文字體
```bash
# macOS
brew install font-source-han-sans

# Ubuntu
sudo apt-get install fonts-noto-cjk
```

或修改 `visualize_factor_attention.py` 和 `analyze_contexts.py`：
```python
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用英文字體
```

---

## 📚 參考文獻

Wei, L., Li, B., & Chen, Y. (2022). Deep Multi-Factor Model for Stock Prediction.
*Journal of Machine Learning Research*.

---

## 👤 作者

**Lo Yi (羅頤)**
National Yang Ming Chiao Tung University
Graduate Institute of Information Management & Finance
E-mail: roy60404@gmail.com

---

## 📝 版本記錄

- **v1.0.0** (2025-01-XX): 初始版本，完全對齊 Wei et al. (2022) 論文
