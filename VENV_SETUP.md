# 虛擬環境設置指南

## 🐍 方式 1: 使用 venv（Python 內建）

### Step 1: 創建虛擬環境

```bash
cd /workspace/gat_9_15  # 或你的專案路徑

# 創建虛擬環境
python3 -m venv .venv
```

### Step 2: 啟動虛擬環境

```bash
# Linux/Mac
source .venv/bin/activate

# 成功後，命令行前會出現 (.venv)
(.venv) user@host:~/gat_9_15$
```

### Step 3: 安裝套件

```bash
# 升級 pip
pip install --upgrade pip

# 方式 A: 安裝 CUDA 版本的 PyTorch（推薦，RunPods 使用）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝 PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric

# 安裝其他套件
pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm

# 安裝 TA-Lib（技術指標，可選）
pip install TA-Lib
```

### Step 4: 驗證安裝

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

---

## 🐍 方式 2: 使用 conda（推薦，更穩定）

### Step 1: 創建 conda 環境

```bash
cd /workspace/gat_9_15

# 創建環境（Python 3.10）
conda create -n gat python=3.10 -y
```

### Step 2: 啟動環境

```bash
conda activate gat

# 成功後
(gat) user@host:~/gat_9_15$
```

### Step 3: 安裝 PyTorch（CUDA 版本）

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或 CUDA 12.1（較新的 GPU）
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 4: 安裝 PyTorch Geometric

```bash
conda install pyg -c pyg -y
```

### Step 5: 安裝其他套件

```bash
conda install numpy pandas scipy matplotlib seaborn scikit-learn -y
pip install tqdm
```

---

## 🚀 RunPods 專用快速設置

在 RunPods 上，通常已經有 PyTorch 環境，你可以：

### 選項 A: 使用系統 Python（最簡單）

```bash
cd /workspace/gat_9_15

# 直接安裝缺少的套件
pip install torch-geometric pandas matplotlib seaborn scikit-learn tqdm
```

### 選項 B: 創建獨立環境

```bash
cd /workspace/gat_9_15

# 創建虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝所有套件
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm
```

---

## 📝 後續使用

### 每次進入專案時

#### 使用 venv
```bash
cd /workspace/gat_9_15
source .venv/bin/activate
```

#### 使用 conda
```bash
conda activate gat
cd /workspace/gat_9_15
```

### 離開虛擬環境

#### venv
```bash
deactivate
```

#### conda
```bash
conda deactivate
```

---

## 🔧 完整安裝腳本

我已經為你創建了一鍵安裝腳本：

```bash
# 創建並啟動虛擬環境，安裝所有套件
bash setup_env.sh
```

---

## ⚠️ 常見問題

### Q1: 如何知道自己在虛擬環境中？

**A:** 命令行前會有環境名稱：
```bash
(.venv) user@host:~/gat_9_15$    # venv
(gat) user@host:~/gat_9_15$      # conda
```

### Q2: 如何確認 CUDA 可用？

**A:** 執行：
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 應該輸出 True
```

### Q3: pip 安裝失敗怎麼辦？

**A:** 使用清華鏡像（中國用戶）：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torch-geometric
```

### Q4: 忘記在哪個環境？

**A:** 查看：
```bash
which python
# /workspace/gat_9_15/.venv/bin/python  ← venv
# /home/user/miniconda3/envs/gat/bin/python  ← conda
# /usr/bin/python  ← 系統 Python
```

---

## 📦 套件清單

必要套件：
```
torch>=2.0.0          # 深度學習框架
torch-geometric       # 圖神經網路
numpy>=1.26           # 數值計算
pandas>=2.2           # 資料處理
scipy>=1.11           # 科學計算
matplotlib            # 繪圖
seaborn               # 視覺化
scikit-learn          # 機器學習工具
tqdm                  # 進度條
```

可選套件：
```
TA-Lib                # 技術指標（需要 C 編譯）
```

---

## 🎯 推薦設置（RunPods）

```bash
# 1. 進入專案
cd /workspace/gat_9_15

# 2. 創建虛擬環境
python3 -m venv .venv

# 3. 啟動環境
source .venv/bin/activate

# 4. 一鍵安裝所有套件
bash setup_env.sh

# 5. 驗證
python -c "import torch, torch_geometric; print('✅ 安裝成功！')"

# 6. 開始訓練
bash run_all_models.sh
```

---

需要我創建一鍵安裝腳本嗎？
