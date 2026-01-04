# -*- coding: utf-8 -*-
# train_baselines.py
"""
Train baseline models (Linear Regression, LSTM, XGBoost) for comparison with DMFM.

Example usages:

# Linear regression (Ridge) baseline
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model linear \
  --train_ratio 0.8

# LSTM baseline with 10-day lookback
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model lstm \
  --lookback 10 \
  --epochs 30 \
  --batch_size 256 \
  --device cuda

# XGBoost baseline
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model xgboost \
  --max_depth 6 \
  --n_estimators 300
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train_gat_fixed import load_artifacts, time_split_indices


def pick_device(device_str: str) -> torch.device:
    s = (device_str or "").lower()
    if s in ("cpu", "cuda", "mps"):
        if s == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA 不可用，改用 CPU")
            return torch.device("cpu")
        if s == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print("[warn] MPS 不可用，改用 CPU")
                return torch.device("cpu")
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < 1e-8 or sb < 1e-8:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def flatten_by_day(
    Ft: torch.Tensor, yt: torch.Tensor, indices: Iterable[int], dates: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    day_labels: List[str] = []
    for t in indices:
        x = Ft[t].float()
        y = yt[t]
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            continue
        x = torch.nan_to_num(x, nan=0.0)
        xs.append(x[mask].cpu().numpy())
        ys.append(y[mask].cpu().numpy())
        day_labels.extend([dates[t]] * int(mask.sum()))
    if not xs:
        return np.empty((0, Ft.shape[-1]), dtype=np.float32), np.empty(0), []
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), day_labels


@dataclass
class SequenceSample:
    seq: torch.Tensor
    target: torch.Tensor
    day: str


def build_lstm_samples(
    Ft: torch.Tensor, yt: torch.Tensor, indices: Iterable[int], dates: List[str], lookback: int
) -> List[SequenceSample]:
    samples: List[SequenceSample] = []
    for t in indices:
        if t < lookback - 1:
            continue
        x_seq = Ft[t - lookback + 1 : t + 1]  # [L, N, F]
        y_t = yt[t]
        mask = torch.isfinite(y_t)
        if mask.sum() == 0:
            continue
        x_seq = torch.nan_to_num(x_seq, nan=0.0)
        for stock_idx in range(x_seq.shape[1]):
            if not mask[stock_idx]:
                continue
            seq = x_seq[:, stock_idx, :]
            target = y_t[stock_idx]
            samples.append(
                SequenceSample(seq=seq.float(), target=target.float(), day=dates[t])
            )
    return samples


def compute_metrics(preds: np.ndarray, truths: np.ndarray, days: List[str]):
    out = {
        "MSE": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "IC": np.nan,
        "ICIR": np.nan,
        "DirAcc": np.nan,
        "DailyIC": np.nan,
    }
    if preds.size == 0:
        return out
    mse = mean_squared_error(truths, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(truths, preds)
    ic = safe_corr(preds, truths)
    daily_ic = []
    unique_days = list(dict.fromkeys(days))
    for d in unique_days:
        sel = np.array(days) == d
        if sel.sum() < 3:
            continue
        c = safe_corr(preds[sel], truths[sel])
        if c == c:
            daily_ic.append(c)
    icir = float(np.nanmean(daily_ic) / (np.nanstd(daily_ic) + 1e-8)) if daily_ic else np.nan
    dir_acc = float((np.sign(preds) == np.sign(truths)).mean())
    out.update({
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": float(mae),
        "IC": ic,
        "ICIR": icir,
        "DirAcc": dir_acc,
        "DailyIC": float(np.nanmean(daily_ic)) if daily_ic else np.nan,
    })
    return out


def train_linear(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xtr, y_train)
    preds_train = model.predict(Xtr)
    preds_test = model.predict(Xte)
    return model, scaler, preds_train, preds_test


def train_xgboost(X_train, y_train, X_test, args):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.seed,
        tree_method="hist",
        device="cuda" if args.device.type == "cuda" else "cpu",
    )
    model.fit(Xtr, y_train)
    preds_train = model.predict(Xtr)
    preds_test = model.predict(Xte)
    return model, scaler, preds_train, preds_test


class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


def train_lstm(samples: List[SequenceSample], args):
    device = args.device
    model = LSTMRegressor(input_dim=samples[0].seq.shape[-1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    def collate(batch):
        seqs = torch.stack([b.seq for b in batch]).to(device)
        targets = torch.stack([b.target for b in batch]).to(device)
        days = [b.day for b in batch]
        return seqs, targets, days

    loader = torch.utils.data.DataLoader(
        samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=False
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for seqs, targets, _ in loader:
            optimizer.zero_grad()
            preds = model(seqs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        avg_loss = total_loss / max(len(samples), 1)
        if (epoch + 1) % max(args.epochs // 5, 1) == 0:
            print(f"[epoch {epoch+1}/{args.epochs}] train MSE: {avg_loss:.6f}")
    return model


@torch.no_grad()
def predict_lstm(model: LSTMRegressor, samples: List[SequenceSample], device: torch.device):
    model.eval()
    preds, truths, days = [], [], []
    for b in samples:
        seq = b.seq.unsqueeze(0).to(device)
        p = model(seq).squeeze(0).item()
        preds.append(p)
        truths.append(float(b.target))
        days.append(b.day)
    return np.array(preds), np.array(truths), days


def save_metrics(out_path: str, metrics: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True, help="資料目錄（需含 Ft_tensor.pt 等）")
    ap.add_argument("--model", choices=["linear", "lstm", "xgboost"], required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8, help="時間切分比例（訓練集）")
    ap.add_argument("--device", default="cpu", help="lstm/xgboost 訓練裝置")
    ap.add_argument("--seed", type=int, default=42)

    # LSTM hyperparameters
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)

    # XGBoost hyperparameters
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)

    args = ap.parse_args()
    args.device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Ft, yt, _, _, meta = load_artifacts(args.artifact_dir)
    dates = meta.get("dates", list(range(Ft.shape[0])))

    train_idx, test_idx = time_split_indices(dates, train_ratio=args.train_ratio)
    print(f"使用 {len(train_idx)} 天做訓練，{len(test_idx)} 天做測試")

    if args.model in {"linear", "xgboost"}:
        Xtr, Ytr, day_tr = flatten_by_day(Ft, yt, train_idx, dates)
        Xte, Yte, day_te = flatten_by_day(Ft, yt, test_idx, dates)
        print(f"訓練樣本: {len(Ytr):,}，測試樣本: {len(Yte):,}")
        if args.model == "linear":
            model, scaler, p_tr, p_te = train_linear(Xtr, Ytr, Xte)
            model_path = os.path.join(args.artifact_dir, "baseline_linear.pkl")
            scaler_path = os.path.join(args.artifact_dir, "baseline_linear_scaler.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        else:
            model, scaler, p_tr, p_te = train_xgboost(Xtr, Ytr, Xte, args)
            model_path = os.path.join(args.artifact_dir, "baseline_xgboost.json")
            scaler_path = os.path.join(args.artifact_dir, "baseline_xgboost_scaler.pkl")
            model.save_model(model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        metrics_train = compute_metrics(p_tr, Ytr, day_tr)
        metrics_test = compute_metrics(p_te, Yte, day_te)

    else:
        samples_train = build_lstm_samples(Ft, yt, train_idx, dates, args.lookback)
        samples_test = build_lstm_samples(Ft, yt, test_idx, dates, args.lookback)
        if not samples_train or not samples_test:
            raise ValueError("LSTM 無可用樣本，請確認資料或 lookback 設定")
        print(f"LSTM 訓練樣本: {len(samples_train):,}，測試樣本: {len(samples_test):,}")
        model = train_lstm(samples_train, args)
        p_tr, Ytr, day_tr = predict_lstm(model, samples_train, args.device)
        p_te, Yte, day_te = predict_lstm(model, samples_test, args.device)
        model_path = os.path.join(args.artifact_dir, "baseline_lstm.pt")
        torch.save(model.state_dict(), model_path)
        scaler = None
        metrics_train = compute_metrics(p_tr, Ytr, day_tr)
        metrics_test = compute_metrics(p_te, Yte, day_te)

    print("\n=== 訓練集指標 ===")
    for k, v in metrics_train.items():
        print(f"{k:>8}: {v:.6f}" if isinstance(v, float) else f"{k:>8}: {v}")

    print("\n=== 測試集指標 ===")
    for k, v in metrics_test.items():
        print(f"{k:>8}: {v:.6f}" if isinstance(v, float) else f"{k:>8}: {v}")

    metrics_out = {
        "train": metrics_train,
        "test": metrics_test,
        "model": args.model,
        "train_ratio": args.train_ratio,
        "lookback": args.lookback if args.model == "lstm" else None,
    }
    save_metrics(os.path.join(args.artifact_dir, f"baseline_{args.model}_metrics.json"), metrics_out)

    print(f"\n模型與指標已儲存至 {args.artifact_dir}")
    print(f"模型檔案: {model_path}")
    if args.model != "lstm":
        print(f"Scaler檔案: {scaler_path}")


if __name__ == "__main__":
    main()
