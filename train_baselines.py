# -*- coding: utf-8 -*-
# train_baselines.py
"""
Train baseline models (Linear Regression, LSTM, XGBoost) for comparison with DMFM.

Example usages:

# Linear regression (Ridge) baseline
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model linear \
  --train_ratio 0.8 \
  --val_ratio 0.1

# LSTM baseline with 10-day lookback
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model lstm \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --lookback 10 \
  --epochs 30 \
  --batch_size 256 \
  --device cuda

# XGBoost baseline
python train_baselines.py \
  --artifact_dir gat_artifacts \
  --model xgboost \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --max_depth 6 \
  --n_estimators 300
"""

import argparse
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

from report_utils import save_json
from train_gat_fixed import load_artifacts, time_split_indices_3


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


def compute_naive_zero_metrics(truths: np.ndarray):
    out = {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0}
    if truths.size == 0:
        return out
    mse = float(np.mean(truths ** 2))
    out.update({
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(truths))),
        "n": int(truths.size),
    })
    return out


def predict_tabular(model, scaler, X):
    if X.size == 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(model.predict(scaler.transform(X)), dtype=np.float32)


def train_linear(X_train, y_train):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xtr, y_train)
    return model, scaler


def train_xgboost(X_train, y_train, args):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
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
    return model, scaler


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


def make_lstm_loader(samples: List[SequenceSample], batch_size: int, device: torch.device, shuffle: bool):
    def collate(batch):
        seqs = torch.stack([b.seq for b in batch]).to(device)
        targets = torch.stack([b.target for b in batch]).to(device)
        days = [b.day for b in batch]
        return seqs, targets, days

    return torch.utils.data.DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=False,
    )


def train_lstm(train_samples: List[SequenceSample], val_samples: List[SequenceSample], args):
    device = args.device
    model = LSTMRegressor(
        input_dim=train_samples[0].seq.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    loader = make_lstm_loader(train_samples, args.batch_size, device, shuffle=True)

    best_state = None
    best_val_mse = float("inf")
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for seqs, targets, _ in loader:
            optimizer.zero_grad()
            preds = model(seqs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        avg_loss = total_loss / max(len(train_samples), 1)

        val_msg = ""
        if val_samples:
            val_preds, val_truths, _ = predict_lstm(
                model, val_samples, device, batch_size=max(args.batch_size, 1024)
            )
            val_mse = mean_squared_error(val_truths, val_preds) if val_preds.size else float("inf")
            val_msg = f" | val MSE: {val_mse:.6f}"

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    print(
                        f"[epoch {epoch+1}/{args.epochs}] train MSE: {avg_loss:.6f}{val_msg}"
                    )
                    print(f"Early stop at epoch {epoch+1} (best val MSE={best_val_mse:.6f})")
                    break

        if (epoch + 1) % max(args.epochs // 5, 1) == 0 or epoch + 1 == args.epochs:
            print(f"[epoch {epoch+1}/{args.epochs}] train MSE: {avg_loss:.6f}{val_msg}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_lstm(
    model: LSTMRegressor,
    samples: List[SequenceSample],
    device: torch.device,
    batch_size: int = 1024,
):
    model.eval()
    preds, truths, days = [], [], []
    if not samples:
        return np.empty(0), np.empty(0), []

    loader = make_lstm_loader(samples, batch_size, device, shuffle=False)
    for seqs, targets, batch_days in loader:
        batch_preds = model(seqs).detach().cpu().numpy().flatten()
        preds.extend(batch_preds.tolist())
        truths.extend(targets.detach().cpu().numpy().flatten().tolist())
        days.extend(batch_days)
    return np.array(preds), np.array(truths), days


def save_metrics(out_path: str, metrics: dict):
    save_json(out_path, metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True, help="資料目錄（需含 Ft_tensor.pt 等）")
    ap.add_argument("--model", choices=["linear", "lstm", "xgboost"], required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8, help="時間切分比例（訓練集）")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="在 train 區段中劃給 validation 的比例")
    ap.add_argument("--device", default="cpu", help="lstm/xgboost 訓練裝置")
    ap.add_argument("--seed", type=int, default=42)

    # LSTM hyperparameters
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)

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

    train_idx, val_idx, test_idx = time_split_indices_3(
        dates, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    print(
        f"使用 {len(train_idx)} 天做訓練，{len(val_idx)} 天做驗證，{len(test_idx)} 天做測試"
    )

    model_path = None
    scaler_path = None

    if args.model in {"linear", "xgboost"}:
        Xtr, Ytr, day_tr = flatten_by_day(Ft, yt, train_idx, dates)
        Xval, Yval, day_val = flatten_by_day(Ft, yt, val_idx, dates)
        Xte, Yte, day_te = flatten_by_day(Ft, yt, test_idx, dates)
        print(
            f"訓練樣本: {len(Ytr):,}，驗證樣本: {len(Yval):,}，測試樣本: {len(Yte):,}"
        )
        if args.model == "linear":
            model, scaler = train_linear(Xtr, Ytr)
            model_path = os.path.join(args.artifact_dir, "baseline_linear.pkl")
            scaler_path = os.path.join(args.artifact_dir, "baseline_linear_scaler.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        else:
            model, scaler = train_xgboost(Xtr, Ytr, args)
            model_path = os.path.join(args.artifact_dir, "baseline_xgboost.json")
            scaler_path = os.path.join(args.artifact_dir, "baseline_xgboost_scaler.pkl")
            model.save_model(model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        p_tr = predict_tabular(model, scaler, Xtr)
        p_val = predict_tabular(model, scaler, Xval)
        p_te = predict_tabular(model, scaler, Xte)
        metrics_train = compute_metrics(p_tr, Ytr, day_tr)
        metrics_val = compute_metrics(p_val, Yval, day_val)
        metrics_test = compute_metrics(p_te, Yte, day_te)

    else:
        samples_train = build_lstm_samples(Ft, yt, train_idx, dates, args.lookback)
        samples_val = build_lstm_samples(Ft, yt, val_idx, dates, args.lookback)
        samples_test = build_lstm_samples(Ft, yt, test_idx, dates, args.lookback)
        if not samples_train or not samples_test:
            raise ValueError("LSTM 無可用樣本，請確認資料或 lookback 設定")
        print(
            f"LSTM 訓練樣本: {len(samples_train):,}，驗證樣本: {len(samples_val):,}，測試樣本: {len(samples_test):,}"
        )
        model = train_lstm(samples_train, samples_val, args)
        p_tr, Ytr, day_tr = predict_lstm(model, samples_train, args.device)
        p_val, Yval, day_val = predict_lstm(model, samples_val, args.device)
        p_te, Yte, day_te = predict_lstm(model, samples_test, args.device)
        model_path = os.path.join(args.artifact_dir, "baseline_lstm.pt")
        torch.save(model.state_dict(), model_path)
        scaler = None
        metrics_train = compute_metrics(p_tr, Ytr, day_tr)
        metrics_val = compute_metrics(p_val, Yval, day_val)
        metrics_test = compute_metrics(p_te, Yte, day_te)

    naive_test = compute_naive_zero_metrics(Yte)
    if np.isfinite(naive_test["MSE"]) and np.isfinite(metrics_test["MSE"]):
        impr = 100.0 * (1.0 - metrics_test["MSE"] / naive_test["MSE"])
    else:
        impr = np.nan

    print("\n=== 訓練集指標 ===")
    for k, v in metrics_train.items():
        print(f"{k:>8}: {v:.6f}" if isinstance(v, float) else f"{k:>8}: {v}")

    print("\n=== 驗證集指標 ===")
    for k, v in metrics_val.items():
        print(f"{k:>8}: {v:.6f}" if isinstance(v, float) else f"{k:>8}: {v}")

    print("\n=== 測試集指標 ===")
    for k, v in metrics_test.items():
        print(f"{k:>8}: {v:.6f}" if isinstance(v, float) else f"{k:>8}: {v}")

    metrics_out = {
        "model_type": args.model,
        "artifact_dir": args.artifact_dir,
        "weights": model_path,
        "scaler": scaler_path,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "train_days": len(train_idx),
            "val_days": len(val_idx),
            "test_days": len(test_idx),
        },
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "naive_test": naive_test,
        "mse_improvement_vs_naive_pct": impr,
        "has_industry_labels": False,
        "lookback": args.lookback if args.model == "lstm" else None,
        "model": args.model,
    }
    save_metrics(os.path.join(args.artifact_dir, f"baseline_{args.model}_metrics.json"), metrics_out)

    print(f"\n模型與指標已儲存至 {args.artifact_dir}")
    print(f"模型檔案: {model_path}")
    if args.model != "lstm":
        print(f"Scaler檔案: {scaler_path}")


if __name__ == "__main__":
    main()
