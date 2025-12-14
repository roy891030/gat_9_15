#!/bin/bash

# ============================================================
# 一鍵設置虛擬環境並安裝所有套件
# ============================================================

set -e  # 遇到錯誤立即停止

echo "============================================================"
echo "設置 GAT 專案虛擬環境"
echo "============================================================"

# 檢查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# ============================================================
# Step 1: 創建虛擬環境（如果不存在）
# ============================================================
if [ ! -d ".venv" ]; then
    echo ""
    echo "====== Step 1: 創建虛擬環境 ======"
    python3 -m venv .venv
    echo "✅ 虛擬環境已創建：.venv"
else
    echo ""
    echo "====== 虛擬環境已存在，跳過創建 ======"
fi

# ============================================================
# Step 2: 啟動虛擬環境
# ============================================================
echo ""
echo "====== Step 2: 啟動虛擬環境 ======"
source .venv/bin/activate
echo "✅ 虛擬環境已啟動"

# ============================================================
# Step 3: 升級 pip
# ============================================================
echo ""
echo "====== Step 3: 升級 pip ======"
pip install --upgrade pip
echo "✅ pip 已升級"

# ============================================================
# Step 4: 檢測 CUDA 版本
# ============================================================
echo ""
echo "====== Step 4: 檢測 CUDA 版本 ======"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "檢測到 CUDA 版本: $CUDA_VERSION"

    # 根據 CUDA 版本選擇 PyTorch
    if [[ "$CUDA_VERSION" == "12"* ]]; then
        TORCH_INDEX="cu121"
        echo "使用 PyTorch CUDA 12.1"
    else
        TORCH_INDEX="cu118"
        echo "使用 PyTorch CUDA 11.8"
    fi
else
    echo "⚠️  未檢測到 CUDA，將安裝 CPU 版本的 PyTorch"
    TORCH_INDEX="cpu"
fi

# ============================================================
# Step 5: 安裝 PyTorch
# ============================================================
echo ""
echo "====== Step 5: 安裝 PyTorch ======"

if [ "$TORCH_INDEX" == "cpu" ]; then
    pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_INDEX
fi

echo "✅ PyTorch 已安裝"

# 驗證 PyTorch
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

# ============================================================
# Step 6: 安裝 PyTorch Geometric
# ============================================================
echo ""
echo "====== Step 6: 安裝 PyTorch Geometric ======"

# 獲取 PyTorch 版本
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d'+' -f1)
echo "PyTorch 版本: $TORCH_VERSION"

# 安裝 PyG 依賴
if [ "$TORCH_INDEX" != "cpu" ]; then
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_INDEX}.html
fi

# 安裝 PyG
pip install torch-geometric

echo "✅ PyTorch Geometric 已安裝"

# 驗證 PyG
python -c "import torch_geometric; print(f'PyG 版本: {torch_geometric.__version__}')"

# ============================================================
# Step 7: 安裝其他套件
# ============================================================
echo ""
echo "====== Step 7: 安裝其他套件 ======"

pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm

echo "✅ 其他套件已安裝"

# ============================================================
# Step 8: 嘗試安裝 TA-Lib（可選）
# ============================================================
echo ""
echo "====== Step 8: 安裝 TA-Lib（可選）======"

if command -v gcc &> /dev/null; then
    echo "嘗試安裝 TA-Lib..."
    pip install TA-Lib || echo "⚠️  TA-Lib 安裝失敗，已跳過（不影響主要功能）"
else
    echo "⚠️  未檢測到 gcc，跳過 TA-Lib 安裝（不影響主要功能）"
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "環境設置完成！"
echo "============================================================"

echo ""
echo "📦 已安裝的套件："
pip list | grep -E "torch|numpy|pandas|matplotlib|seaborn|scikit"

echo ""
echo "============================================================"
echo "使用方式："
echo "============================================================"
echo ""
echo "1. 每次進入專案時，啟動虛擬環境："
echo "   source .venv/bin/activate"
echo ""
echo "2. 開始訓練："
echo "   bash run_all_models.sh"
echo ""
echo "3. 離開虛擬環境："
echo "   deactivate"
echo ""
echo "============================================================"

# 生成啟動腳本
cat > activate_env.sh <<'EOF'
#!/bin/bash
# 快速啟動虛擬環境
source .venv/bin/activate
echo "✅ 虛擬環境已啟動"
echo "當前 Python: $(which python)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x activate_env.sh

echo "💡 提示：下次可以使用以下命令快速啟動環境："
echo "   source activate_env.sh"
echo ""
echo "✅ 設置完成！"
