"""run_v61_compare.py — 65/35 切分下基线公平对照（RF / LSTM / LSTM-RF / CNN-LSTM）。

与 v6.1 已训好的 CNN-LSTM 结果并排，输出 results/v61_compare/summary.json。
用法：python run_v61_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, load_config, set_seed
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.models.evaluate import all_metrics, save_run
from src.models.baselines import naive_predict, fit_rf, predict_rf
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch
from src.models.lstm_rf import fit_lstm_rf, predict_lstm_rf


def load_xichang32():
    paths = load_paths()
    df = pd.read_csv(paths["clean_dir"] / "long.csv", encoding="utf-8-sig",
                     dtype={"station_id": str, "item_code": str})
    mask = (df["station_id"] == "SCXC32") & (df["item_name"] == "水氡")
    s = df.loc[mask, ["date", "value"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.dropna(subset=["value"]).sort_values("date").drop_duplicates("date")
    s = s.set_index("date")["value"]
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx).interpolate(method="linear", limit=3)
    return s.index.to_numpy(), s.to_numpy()


def year_r2(dates, y_true, y_pred) -> dict:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred})
    out = {}
    for y, g in df.groupby(df["date"].dt.year):
        out[str(int(y))] = round(all_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())["R2"], 4)
    return out


def main():
    cfg = load_config("exp_v61")
    train_ratio = cfg["train_ratio"]
    window = cfg["window"]
    seed = cfg["seed"]

    set_seed(seed)
    paths = load_paths()
    out_root = paths["results_dir"] / "v61_compare"
    out_root.mkdir(parents=True, exist_ok=True)

    dates, radon = load_xichang32()
    n = len(radon)
    k_train = int(n * train_ratio)
    print(f"切分 train_ratio={train_ratio} | 训练截止 {pd.Timestamp(dates[k_train-1]).date()} | 测试 {n-k_train} 天")

    mu, sd = zscore_fit(radon[:k_train])
    rz = zscore_apply(radon, mu, sd)
    X, y = make_windows(rz, window)
    tgt_idx = np.arange(window, window + len(y))
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_te, y_te = X[te_m], y[te_m]
    te_dates = dates[tgt_idx[te_m]]
    k_val = int(len(X_tr) * cfg.get("val_ratio_of_train", 0.85))

    summary = {"train_ratio": train_ratio, "window": window, "n_train": int(len(X_tr)),
               "n_test": int(len(X_te)), "models": {}}

    print("[1/4] naive")
    pred = naive_predict(X_te)
    m = save_run(out_root / "naive", "test", y_te, pred, te_dates)
    m["year_r2"] = year_r2(te_dates, y_te, pred)
    summary["models"]["naive"] = m

    print("[2/4] RF")
    rf = fit_rf(X_tr, y_tr, seed=seed)
    pred = predict_rf(rf, X_te)
    m = save_run(out_root / "rf", "test", y_te, pred, te_dates)
    m["year_r2"] = year_r2(te_dates, y_te, pred)
    summary["models"]["rf"] = m

    print("[3/4] LSTM-RF")
    lstm, rf_cascade, info = fit_lstm_rf(
        X_tr[..., None], y_tr, X_tr[k_val:, :, None], y_tr[k_val:],
        epochs=cfg.get("epochs", 200), patience=cfg.get("patience", 15), seed=seed)
    pred = predict_lstm_rf(lstm, rf_cascade, X_te[..., None])
    m = save_run(out_root / "lstm_rf", "test", y_te, pred, te_dates,
                 extra={"epochs_run": info.get("epochs_run")})
    m["year_r2"] = year_r2(te_dates, y_te, pred)
    summary["models"]["lstm_rf"] = m

    print("[4/4] CNN-LSTM (读取已训结果)")
    cnn_path = paths["results_dir"] / "v61_cnn_lstm" / "cnn_lstm_tuned" / "predictions_test.csv"
    if cnn_path.exists():
        cdf = pd.read_csv(cnn_path)
        pred = cdf["y_pred"].to_numpy()
        y_true = cdf["y_true"].to_numpy()
        m = all_metrics(y_true, pred)
        m["year_r2"] = year_r2(cdf["date"].to_numpy(), y_true, pred)
        summary["models"]["cnn_lstm_tuned"] = m
    else:
        print("  警告：未找到 v61_cnn_lstm 结果，请先运行 run_phase3_tuned.py")

    out_path = out_root / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'模型':<16} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("=" * 44)
    for name, m in summary["models"].items():
        print(f"{name:<16} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f}")
    print(f"\n按年 R2:")
    for name, m in summary["models"].items():
        print(f"  {name:<16} {m.get('year_r2')}")
    print(f"\n保存：{out_path}")


if __name__ == "__main__":
    main()
