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
"""


import os, argparse, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv
# 在 import 區之後加這段
def pick_device(device_str: str) -> torch.device:
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
    Ft_tensor = torch.load(os.path.join(artifact_dir, "Ft_tensor.pt"))
    yt_tensor = torch.load(os.path.join(artifact_dir, "yt_tensor.pt"))
    edge_index = torch.load(os.path.join(artifact_dir, "industry_edge_index.pt"))
    with open(os.path.join(artifact_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return Ft_tensor, yt_tensor, edge_index, meta

def time_split_indices(dates, train_ratio=0.8):
    T = len(dates); split = int(T*train_ratio)
    return list(range(split)), list(range(split, T))

def load_industry_map(industry_csv, stocks):
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


# -------- Model --------
class GATRegressor(nn.Module):
    def __init__(self, in_dim, hid=64, heads=2, dropout=0.1, tanh_cap=None):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hid*heads, hid, heads=1, dropout=dropout)
        self.lin = nn.Linear(hid, 1)
        self.tanh_cap = tanh_cap
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index); x = F.elu(x)
        x = self.gat2(x, edge_index); x = F.elu(x)
        out = self.lin(x).squeeze(-1)
        if self.tanh_cap is not None:
            out = self.tanh_cap * torch.tanh(out)
        return out

# -------- Loss utils --------
def corr_loss(pred, target):
    px = pred - pred.mean()
    ty = target - target.mean()
    denom = (px.std() * ty.std()) + 1e-8
    c = (px * ty).mean() / denom
    return 1.0 - c  # maximize corr == minimize (1 - corr)

# -------- Eval --------
@torch.no_grad()
def evaluate_mse(model, Ft, yt, edge_index, indices, device="cpu"):
    model.eval()
    s, c = 0.0, 0
    for t in indices:
        x = Ft[t].to(device).float()
        y = yt[t].to(device)
        mask = torch.isfinite(y)
        if mask.sum()==0: continue
        x = torch.nan_to_num(x, nan=0.0)
        p = model(x, edge_index.to(device))
        s += F.mse_loss(p[mask], y[mask]).item()
        c += 1
    return s / max(c,1)

# -------- Train --------
def train(args):
    device = args.device
    Ft, yt, edge_index, meta = load_artifacts(args.artifact_dir)
    Ft = Ft.to(device).float(); yt = yt.to(device); edge_index = edge_index.to(device)
    T, N, Fdim = Ft.shape
    train_idx, test_idx = time_split_indices(meta["dates"], train_ratio=0.8)

    # industry groups (for corr_mse_ind)
    stocks = meta.get("stocks", [str(i) for i in range(N)])
    ind_map = load_industry_map(args.industry_csv, stocks)
    from collections import defaultdict
    ind2idx = defaultdict(list)
    for i, s in enumerate(stocks):
        ind2idx[ind_map.get(s, "UNK")].append(i)
    groups = {g: torch.tensor(ix, dtype=torch.long, device=device) for g,ix in ind2idx.items() if len(ix)>=3}

    model = GATRegressor(in_dim=Fdim, hid=args.hid, heads=args.heads, dropout=args.dropout, tanh_cap=args.tanh_cap).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score, best_state, wait = None, None, 0

    def step_loss(pred, y, mask):
        if args.loss == "mse":
            loss = F.mse_loss(pred[mask], y[mask]); score = -loss.item()
        elif args.loss == "huber":
            loss = F.huber_loss(pred[mask], y[mask], delta=args.huber_delta); score = -loss.item()
        elif args.loss == "corr_mse":
            c = corr_loss(pred[mask], y[mask])
            mse_anchor = F.mse_loss(pred[mask], y[mask])
            var_pen = torch.var(pred[mask])
            loss = c + args.alpha_mse*mse_anchor + args.lambda_var*var_pen
            score = 1.0 - loss.item()
        elif args.loss == "corr_mse_ind":
            clist=[]
            for _, idxs in groups.items():
                use = mask[idxs]
                if use.sum() >= 3:
                    clist.append(corr_loss(pred[idxs][use], y[idxs][use]))
            cind = torch.stack(clist).mean() if len(clist)>0 else corr_loss(pred[mask], y[mask])
            mse_anchor = F.mse_loss(pred[mask], y[mask])
            var_pen = torch.var(pred[mask])
            loss = cind + args.alpha_mse*mse_anchor + args.lambda_var*var_pen
            score = 1.0 - loss.item()
        else:
            loss = F.mse_loss(pred[mask], y[mask]); score = -loss.item()
        return loss, score

    for ep in range(1, args.epochs+1):
        model.train()
        total_score, steps = 0.0, 0
        for t in train_idx:
            x = Ft[t]; y = yt[t]
            mask = torch.isfinite(y)
            if mask.sum()==0: continue
            x = torch.nan_to_num(x, nan=0.0)
            pred = model(x, edge_index)
            loss, score = step_loss(pred, y, mask)
            opt.zero_grad(); loss.backward()
            if args.clip_grad and args.clip_grad>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            total_score += score; steps += 1

        # val score
        with torch.no_grad():
            model.eval()
            val_scores, vsteps = 0.0, 0
            for t in test_idx:
                x = Ft[t]; y = yt[t]
                mask = torch.isfinite(y)
                if mask.sum()==0: continue
                x = torch.nan_to_num(x, nan=0.0)
                p = model(x, edge_index)
                _, s = step_loss(p, y, mask)
                val_scores += s; vsteps += 1
            val_score = val_scores / max(vsteps,1)

        test_mse = evaluate_mse(model, Ft, yt, edge_index, test_idx, device=device)
        print(f"Epoch {ep:02d} | train {args.loss} {total_score/max(steps,1):.6f} | test {args.loss} {val_score:.6f} | test MSE {test_mse:.6f}"
              + (" <-- best" if (best_score is None or val_score > best_score) else ""))

        if best_score is None or val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {ep} (best {args.loss}={best_score:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    out = os.path.join(args.artifact_dir, "gat_regressor.pt")
    torch.save(model.state_dict(), out)
    print("Saved best weights to:", out)

# -------- CLI --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", type=str, default="./gat_artifacts_out_plus")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", type=str, default="huber", choices=["mse","huber","corr_mse","corr_mse_ind"])
    ap.add_argument("--huber_delta", type=float, default=0.02)
    ap.add_argument("--alpha_mse", type=float, default=0.03)
    ap.add_argument("--lambda_var", type=float, default=0.1)
    ap.add_argument("--tanh_cap", type=float, default=0.2)
    ap.add_argument("--clip_grad", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--industry_csv", type=str, default="merged_stock_industry.csv")
    args = ap.parse_args()

    # 新增這行：把 auto/大小寫/不可用裝置解析成實際可用的 torch.device
    args.device = pick_device(args.device)
    print("Using device:", args.device)
    train(args)
