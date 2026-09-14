"""run_tune_cnn_lstm.py — Optuna 自动调参（CNN-LSTM）。

搜索空间：
  - conv_channels: (64,32) / (128,64) / (64,64,32)
  - lstm_hidden: 64 / 128 / 256
  - dropout: 0.1 ~ 0.5
  - lr: 1e-4 ~ 1e-2 (log)
  - batch_size: 32 / 64 / 128
  - patience: 15 / 20 / 25

目标：验证集 R² 最大化
用法：python run_tune_cnn_lstm.py [--n_trials 50]
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import optuna

from src.common import load_paths, set_seed
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch
from src.models.evaluate import all_metrics

SEED = 42
WINDOW = 60
EPOCHS = 200
TRAIN_RATIO = 0.65  # v6.1: configs/exp_v61.json


def load_xichang32() -> tuple[pd.DatetimeIndex, np.ndarray]:
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


def objective(trial: optuna.Trial) -> float:
    set_seed(SEED)
    dates, radon = load_xichang32()
    n = len(radon)
    k_train = int(n * TRAIN_RATIO)

    mu, sd = zscore_fit(radon[:k_train])
    rz = zscore_apply(radon, mu, sd)
    X, y = make_windows(rz, WINDOW)
    tgt_idx = np.arange(WINDOW, WINDOW + len(y))
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    X_tr, y_tr = X[tr_m], y[tr_m]

    # 验证集（训练段尾部 15%）
    k_val = int(len(X_tr) * 0.85)
    X_val, y_val = X_tr[k_val:], y_tr[k_val:]
    X_tr_sub, y_tr_sub = X_tr[:k_val], y_tr[:k_val]

    # 搜索空间
    conv_choice = trial.suggest_categorical("conv_channels", ["64_32", "128_64", "64_64_32"])
    if conv_choice == "64_32":
        conv_channels = (64, 32)
    elif conv_choice == "128_64":
        conv_channels = (128, 64)
    else:
        conv_channels = (64, 64, 32)

    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience = trial.suggest_categorical("patience", [15, 20, 25])

    model = CNNLSTM(n_vars=1, window=WINDOW,
                    conv_channels=conv_channels,
                    lstm_hidden=lstm_hidden,
                    dropout=dropout)

    info = train_torch(model, X_tr_sub[..., None], y_tr_sub,
                       X_val[..., None], y_val,
                       epochs=EPOCHS, batch_size=batch_size,
                       lr=lr, patience=patience, seed=SEED)

    # 在验证集上计算 R²
    pred_val = predict_torch(model, X_val[..., None])
    m = all_metrics(y_val, pred_val)
    r2 = m["R2"]

    # 记录额外信息
    trial.set_user_attr("epochs_run", info["epochs_run"])
    trial.set_user_attr("val_loss", info["best_val_loss"])
    trial.set_user_attr("rmse", m["RMSE"])
    trial.set_user_attr("mae", m["MAE"])

    return r2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths["results_dir"] / "v61_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"切分：train_ratio={TRAIN_RATIO} | 输出：{out_dir}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="cnn_lstm_tune")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # 保存结果
    best = study.best_trial
    result = {
        "best_r2": best.value,
        "best_params": best.params,
        "best_metrics": {
            "rmse": best.user_attrs["rmse"],
            "mae": best.user_attrs["mae"],
            "epochs_run": best.user_attrs["epochs_run"],
            "val_loss": best.user_attrs["val_loss"],
        },
        "n_trials": args.n_trials,
        "all_trials": [
            {
                "trial": t.number,
                "r2": t.value,
                "params": t.params,
                "rmse": t.user_attrs.get("rmse"),
                "mae": t.user_attrs.get("mae"),
                "epochs": t.user_attrs.get("epochs_run"),
            }
            for t in study.trials if t.value is not None
        ],
    }

    out_path = out_dir / "tuning_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"调参完成！最优 R² = {best.value:.4f}")
    print(f"最优参数：{json.dumps(best.params, indent=2)}")
    print(f"结果保存：{out_path}")
    print(f"{'='*60}")

    # 保存最优配置
    config_path = out_dir / "best_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
