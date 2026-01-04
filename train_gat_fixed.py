# -*- coding: utf-8 -*-
# train_gat_fixed.py
"""
source .venv/bin/activate

執行方式：
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_out_plus \
  --epochs 3 --lr 1e-3 --device auto \
  --industry_csv unique_2019q3to2025q3.csv

50 epoch 訓練，採相關型損失避免常數解
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_out_plus \
  --epochs 50 --lr 1e-3 --device mps \
  --loss corr_mse_ind --alpha_mse 0.03 --lambda_var 0.1 \
  --tanh_cap 1.0 \
  --industry_csv unique_2019q3to2025q3.csv

用能鼓勵截面相關的損失：
試 corr_mse 或（更推薦）分產業的 corr_mse_ind：
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_out_plus \
  --epochs 50 --lr 1e-3 --device mps \
  --loss corr_mse_ind --alpha_mse 0.03 --lambda_var 0.1 \
  --industry_csv unique_2019q3to2025q3.csv

使用完整 DMFM 架構（對齊論文）：
python train_gat_fixed.py \
  --artifact_dir gat_artifacts_out_plus \
  --epochs 50 --lr 1e-3 --device auto \
  --loss dmfm --alpha_mse 0.03 --lambda_var 0.1 --lambda_attn 0.05 \
  --industry_csv unique_2019q3to2025q3.csv
"""


import os, argparse, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv
from model_dmfm_wei2022 import DMFM_Wei2022 as DMFM, GATRegressor

# -------- Device Selection --------
def pick_device(device_str: str) -> torch.device:
    """
    自動選擇可用的計算裝置
    
    參數：
        device_str: 'cpu', 'cuda', 'mps', 或 'auto'
    
    回傳：
        torch.device: 實際可用的裝置
    """
    s = (device_str or "").lower()
    if s in ("cpu", "cuda", "mps"):
        # 若點名某裝置但不可用，就退回 cpu
        if s == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA 不可用，改用 CPU")
            return torch.device("cpu")
        if s == "mps":
            # MPS 只有在 mac + 金屬後端可用
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print("[warn] MPS 不可用，改用 CPU")
                return torch.device("cpu")
        return torch.device(s)
    # s 不是明確字串（例如 auto）→ 自動偵測
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------- IO --------
def load_artifacts(artifact_dir):
    """
    載入所有 artifacts（包含新增的 universe_edge_index）
    
    回傳：
        Ft_tensor: [T, N, F] 特徵張量
        yt_tensor: [T, N] 標籤張量
        industry_edge_index: [2, E_ind] 產業圖邊索引
        universe_edge_index: [2, E_univ] 全市場圖邊索引
        meta: metadata 字典
    """
    Ft_tensor = torch.load(os.path.join(artifact_dir, "Ft_tensor.pt"))
    yt_tensor = torch.load(os.path.join(artifact_dir, "yt_tensor.pt"))
    industry_edge_index = torch.load(os.path.join(artifact_dir, "industry_edge_index.pt"))
    
    # 載入全市場圖（新增）
    universe_path = os.path.join(artifact_dir, "universe_edge_index.pt")
    if os.path.exists(universe_path):
        universe_edge_index = torch.load(universe_path)
    else:
        # 向後相容：若舊版沒有此檔案，則建立全連接圖
        print("[warn] universe_edge_index.pt 不存在，自動建立全連接圖")
        N = Ft_tensor.shape[1]
        adj = np.ones((N, N), dtype=np.uint8)
        rows, cols = np.where(adj == 1)
        universe_edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    
    with open(os.path.join(artifact_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    
    return Ft_tensor, yt_tensor, industry_edge_index, universe_edge_index, meta


def time_split_indices(dates, train_ratio=0.8):
    """時序切分訓練/測試集索引"""
    T = len(dates); split = int(T*train_ratio)
    return list(range(split)), list(range(split, T))


def load_industry_map(industry_csv, stocks):
    """
    載入股票產業對應關係
    
    回傳：
        dict: {stock_code: industry_name}
    """
    if industry_csv is None or not os.path.exists(industry_csv):
        return {s: "UNK" for s in stocks}

    # 讀檔時就把代碼強制成字串，避免 1101 → 1101.0
    df = pd.read_csv(industry_csv, dtype={"證券代碼": str, "證券代碼_純代碼": str})

    # 股票代碼欄
    sid_col = None
    for c in ["證券代碼_純代碼","證券代碼","sid","StockID","stock_id"]:
        if c in df.columns:
            sid_col = c; break

    # 產業欄（加上 TEJ）
    ind_col = None
    for c in ["TEJ產業_名稱","TEJ產業_代碼","TSE產業_名稱","industry_TSE","Industry","industry","產業"]:
        if c in df.columns:
            ind_col = c; break

    if sid_col is None or ind_col is None:
        return {s: "UNK" for s in stocks}

    df = df[[sid_col, ind_col]].dropna().drop_duplicates()
    df[sid_col] = df[sid_col].astype(str)
    m = dict(zip(df[sid_col], df[ind_col]))
    return {s: m.get(s, "UNK") for s in stocks}


# -------- Loss Functions --------
def corr_loss(pred, target):
    """
    相關性損失（最大化 IC）
    
    原理：1 - Pearson correlation
    """
    px = pred - pred.mean()
    ty = target - target.mean()
    denom = (px.std() * ty.std()) + 1e-8
    c = (px * ty).mean() / denom
    return 1.0 - c  # maximize corr == minimize (1 - corr)


# -------- Evaluation --------
@torch.no_grad()
def evaluate_mse(model, Ft, yt, edge_industry, edge_universe, indices, device="cpu"):
    """
    評估測試集 MSE
    
    參數：
        model: 模型
        edge_universe: 全市場圖（若模型不使用則忽略）
    """
    model.eval()
    s, c = 0.0, 0
    for t in indices:
        x = Ft[t].to(device).float()
        y = yt[t].to(device)
        mask = torch.isfinite(y)
        if mask.sum()==0: continue
        x = torch.nan_to_num(x, nan=0.0)

        p = model(x, edge_industry.to(device))

        s += F.mse_loss(p[mask], y[mask]).item()
        c += 1
    return s / max(c,1)


# -------- Training --------
def train(args):
    """主訓練函數"""
    device = args.device
    
    # 載入資料（包含兩種圖）
    Ft, yt, edge_industry, edge_universe, meta = load_artifacts(args.artifact_dir)
    Ft = Ft.to(device).float()
    yt = yt.to(device)
    edge_industry = edge_industry.to(device)
    edge_universe = edge_universe.to(device)
    
    T, N, Fdim = Ft.shape
    train_idx, test_idx = time_split_indices(meta["dates"], train_ratio=0.8)
    
    print("=" * 60)
    print(f"資料載入完成：T={T}, N={N}, F={Fdim}")
    print(f"產業圖邊數: {edge_industry.shape[1]:,}")
    print(f"全市場圖邊數: {edge_universe.shape[1]:,}")
    print(f"訓練集: {len(train_idx)} 天 | 測試集: {len(test_idx)} 天")
    print("=" * 60)

    # 載入產業分組（用於 corr_mse_ind 損失）
    stocks = meta.get("stocks", [str(i) for i in range(N)])
    ind_map = load_industry_map(args.industry_csv, stocks)
    from collections import defaultdict
    ind2idx = defaultdict(list)
    for i, s in enumerate(stocks):
        ind2idx[ind_map.get(s, "UNK")].append(i)
    groups = {g: torch.tensor(ix, dtype=torch.long, device=device) 
              for g,ix in ind2idx.items() if len(ix)>=3}
    
    print(f"產業分組: {len(groups)} 個產業")

    # 建立模型
    model = GATRegressor(
        in_dim=Fdim,
        hid=args.hid,
        heads=args.heads,
        dropout=args.dropout,
        tanh_cap=args.tanh_cap
    ).to(device)
    print("模型架構: GATRegressor (實驗性訓練)")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可訓練參數: {total_params:,}")
    print("=" * 60)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score, best_state, wait = None, None, 0

    def step_loss(pred, y, mask, factor_estimate=None):
        """
        計算單步損失
        
        根據不同的損失類型，返回 (loss, score)
        - loss: 用於反向傳播
        - score: 用於模型選擇（越大越好）
        """
        if args.loss == "mse":
            loss = F.mse_loss(pred[mask], y[mask])
            score = -loss.item()
            
        elif args.loss == "huber":
            loss = F.huber_loss(pred[mask], y[mask], delta=args.huber_delta)
            score = -loss.item()
            
        elif args.loss == "corr_mse":
            # 全市場相關性損失
            c = corr_loss(pred[mask], y[mask])
            mse_anchor = F.mse_loss(pred[mask], y[mask])
            var_pen = torch.var(pred[mask])
            loss = c + args.alpha_mse*mse_anchor - args.lambda_var*var_pen
            score = 1.0 - c.item()  # 用 IC 作為 score
            
        elif args.loss == "corr_mse_ind":
            # 分產業相關性損失
            clist = []
            for _, idxs in groups.items():
                use = mask[idxs]
                if use.sum() >= 3:
                    clist.append(corr_loss(pred[idxs][use], y[idxs][use]))
            cind = torch.stack(clist).mean() if len(clist)>0 else corr_loss(pred[mask], y[mask])
            mse_anchor = F.mse_loss(pred[mask], y[mask])
            var_pen = torch.var(pred[mask])
            loss = cind + args.alpha_mse*mse_anchor - args.lambda_var*var_pen
            score = 1.0 - cind.item()
        else:
            # 預設：MSE
            loss = F.mse_loss(pred[mask], y[mask])
            score = -loss.item()
        
        return loss, score

    # 訓練循環
    for ep in range(1, args.epochs+1):
        model.train()
        total_score, steps = 0.0, 0
        
        for t in train_idx:
            x = Ft[t]; y = yt[t]
            mask = torch.isfinite(y)
            if mask.sum()==0: continue
            x = torch.nan_to_num(x, nan=0.0)

            # 前向傳播
            pred = model(x, edge_industry)

            # 計算損失
            loss, score = step_loss(pred, y, mask, None)
            
            # 反向傳播
            opt.zero_grad()
            loss.backward()
            if args.clip_grad and args.clip_grad>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            
            total_score += score
            steps += 1

        # 驗證集評估
        with torch.no_grad():
            model.eval()
            val_scores, vsteps = 0.0, 0
            for t in test_idx:
                x = Ft[t]; y = yt[t]
                mask = torch.isfinite(y)
                if mask.sum()==0: continue
                x = torch.nan_to_num(x, nan=0.0)

                p = model(x, edge_industry)

                _, s = step_loss(p, y, mask, None)
                val_scores += s
                vsteps += 1
            val_score = val_scores / max(vsteps,1)

        # 測試集 MSE（額外指標）
        test_mse = evaluate_mse(model, Ft, yt, edge_industry, edge_universe, 
                                test_idx, device=device)
        
        # 輸出訓練狀態
        is_best = (best_score is None or val_score > best_score)
        print(f"Epoch {ep:02d} | "
              f"train {args.loss} {total_score/max(steps,1):.6f} | "
              f"test {args.loss} {val_score:.6f} | "
              f"test MSE {test_mse:.6f}"
              + (" <-- best" if is_best else ""))

        # Early Stopping
        if best_score is None or val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {ep} (best {args.loss}={best_score:.6f})")
                break

    # 載入最佳權重並儲存
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    
    out = os.path.join(args.artifact_dir, "gat_regressor.pt")
    torch.save(model.state_dict(), out)
    
    print("=" * 60)
    print(f"訓練完成！最佳 {args.loss} score: {best_score:.6f}")
    print(f"模型權重已儲存至: {out}")
    print("=" * 60)


# -------- CLI --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="訓練 GAT 或 DMFM 模型進行股票報酬預測")
    
    # 基本參數
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus",
                    help="Artifacts 資料夾路徑")
    ap.add_argument("--device", type=str, default="cpu",
                    help="計算裝置: cpu, cuda, mps, 或 auto")
    ap.add_argument("--industry_csv", type=str, default="merged_stock_industry.csv",
                    help="產業對照表 CSV 檔案")
    
    # 訓練參數
    ap.add_argument("--epochs", type=int, default=30,
                    help="訓練週期數")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="學習率")
    ap.add_argument("--patience", type=int, default=6,
                    help="Early stopping 的耐心值")
    ap.add_argument("--clip_grad", type=float, default=0.5,
                    help="梯度裁剪閾值")
    
    # 模型架構
    ap.add_argument("--hid", type=int, default=64,
                    help="隱藏層維度")
    ap.add_argument("--heads", type=int, default=2,
                    help="GAT 注意力頭數")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout 比例")
    ap.add_argument("--tanh_cap", type=float, default=0.2,
                    help="輸出 tanh 限制範圍（None 表示不限制）")
    
    # 損失函數
    ap.add_argument("--loss", type=str, default="corr_mse_ind",
                    choices=["mse", "huber", "corr_mse", "corr_mse_ind"],
                    help="損失函數類型")
    ap.add_argument("--huber_delta", type=float, default=0.02,
                    help="Huber loss 的 delta 參數")
    ap.add_argument("--alpha_mse", type=float, default=0.1,
                    help="MSE 錨定項權重")
    ap.add_argument("--lambda_var", type=float, default=0.2,
                    help="變異數懲罰/鼓勵項權重")
    ap.add_argument("--lambda_attn", type=float, default=0.05,
                    help="DMFM 注意力估計誤差權重")
    
    args = ap.parse_args()

    # 解析裝置
    args.device = pick_device(args.device)
    print("=" * 60)
    print("Deep Multi-Factor Model (DMFM) 訓練腳本")
    print("=" * 60)
    print(f"使用裝置: {args.device}")
    print(f"損失函數: {args.loss}")
    print(f"學習率: {args.lr}")
    print(f"訓練週期: {args.epochs}")
    print("=" * 60)
    
    train(args)