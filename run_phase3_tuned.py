"""run_phase3_tuned.py — 用 Optuna 最优参数重新训练 CNN-LSTM。

v6.1：默认读取 configs/exp_v61.json（65/35 切分），结果写入 results/v61_cnn_lstm/。
用法：python run_phase3_tuned.py [--config exp_v61]
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from src.common import load_paths, set_seed, load_config
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.models.evaluate import save_run, all_metrics
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch

SEED = 42
WINDOW = 60
EPOCHS = 200
TRAIN_RATIO = 0.65


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp_v61")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ratio = cfg.get("train_ratio", TRAIN_RATIO)
    window = cfg.get("window", WINDOW)
    seed = cfg.get("seed", SEED)
    epochs = cfg.get("epochs", EPOCHS)
    patience = cfg.get("patience", 15)

    set_seed(seed)
    paths = load_paths()

    # 加载最优参数：优先 v61 调参结果，否则回退 v6.0 调参结果
    for tune_rel in ("v61_tuning/tuning_result.json", "phase5_tuning/tuning_result.json"):
        tune_path = paths["results_dir"] / tune_rel
        if tune_path.exists():
            break
    with open(tune_path, encoding="utf-8") as f:
        tune = json.load(f)
    best = tune["best_params"]
    print(f"调参来源：{tune_path.name}")
    print(f"最优参数：{json.dumps(best, indent=2)}")
    print(f"验证集 R²：{tune['best_r2']:.4f}")
    print(f"切分：train_ratio={train_ratio}")

    # 解析 conv_channels
    conv_map = {"64_32": (64, 32), "128_64": (128, 64), "64_64_32": (64, 64, 32)}
    conv_channels = conv_map[best["conv_channels"]]

    # 加载数据
    dates, radon = load_xichang32()
    n = len(radon)
    k_train = int(n * train_ratio)
    print(f"西昌32井水氡: {n} 天 | 训练: {k_train} | 测试: {n - k_train}")
    print(f"训练截止: {pd.Timestamp(dates[k_train - 1]).date()} | 测试起始: {pd.Timestamp(dates[k_train]).date()}")

    mu, sd = zscore_fit(radon[:k_train])
    rz = zscore_apply(radon, mu, sd)
    X, y = make_windows(rz, window)
    tgt_idx = np.arange(window, window + len(y))
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_te, y_te = X[te_m], y[te_m]
    te_dates = dates[tgt_idx[te_m]]

    # 验证集
    k_val = int(len(X_tr) * cfg.get("val_ratio_of_train", 0.85))

    # 训练最优模型
    print("\n[CNN-LSTM 最优参数] 训练中...")
    model = CNNLSTM(n_vars=1, window=window,
                    conv_channels=conv_channels,
                    lstm_hidden=best["lstm_hidden"],
                    dropout=best["dropout"])
    info = train_torch(model, X_tr[..., None], y_tr,
                       X_tr[k_val:, :, None], y_tr[k_val:],
                       epochs=epochs, batch_size=best["batch_size"],
                       lr=best["lr"], patience=best.get("patience", patience),
                       seed=seed)

    # 测试集评估
    pred = predict_torch(model, X_te[..., None])
    m = all_metrics(y_te, pred)
    print(f"\n测试集结果：")
    print(f"  RMSE = {m['RMSE']:.4f}")
    print(f"  MAE  = {m['MAE']:.4f}")
    print(f"  MAPE = {m['MAPE']:.2f}%")
    print(f"  R²   = {m['R2']:.4f}")
    print(f"  Epochs = {info['epochs_run']}")

    # 保存（v6.1 结果独立目录，不覆盖 v6.0）
    out_dir = paths["results_dir"] / "v61_cnn_lstm" / "cnn_lstm_tuned"
    save_run(out_dir, "test", y_te, pred, te_dates,
             extra={"epochs_run": info["epochs_run"], "tuned": True,
                    "train_ratio": train_ratio, "config": args.config,
                    "tune_source": str(tune_path.name)})
    print(f"\n结果保存：{out_dir}")

    # 对比原版
    orig_path = paths["results_dir"] / "phase3_cnn_lstm" / "cnn_lstm" / "metrics.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        print(f"\n{'='*50}")
        print(f"{'指标':<10} {'原版':>10} {'调参后':>10} {'提升':>10}")
        print(f"{'='*50}")
        for k in ["RMSE", "MAE", "MAPE", "R2"]:
            old = orig[k]
            new = m[k]
            if k == "R2":
                diff = new - old
                print(f"{k:<10} {old:>10.4f} {new:>10.4f} {diff:>+10.4f}")
            elif k == "MAPE":
                diff = new - old
                print(f"{k:<10} {old:>10.2f} {new:>10.2f} {diff:>+10.2f}")
            else:
                diff = new - old
                print(f"{k:<10} {old:>10.4f} {new:>10.4f} {diff:>+10.4f}")


if __name__ == "__main__":
    main()
