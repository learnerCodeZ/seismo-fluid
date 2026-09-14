"""run_v61_optimize.py — v6.1 Phase 2 深度优化消融。

在 65/35 切分上逐步叠加：
  baseline → +特征工程 → +RobustScaler → +Huber → +Cosine → +集成

评估：整体 R²、分年 R²、残差 std / 偏度 / 峰度。
用法：python run_v61_optimize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, load_config, set_seed
from src.features.build import (
    zscore_fit, zscore_apply,
    build_feature_matrix, robust_fit, robust_apply,
)
from src.models.evaluate import all_metrics, save_run
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch


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
    df = pd.DataFrame({"date": pd.to_datetime(dates),
                       "y_true": y_true, "y_pred": y_pred})
    out = {}
    for y, g in df.groupby(df["date"].dt.year):
        out[str(int(y))] = round(
            all_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())["R2"], 4)
    return out


def residual_stats(res: np.ndarray) -> dict:
    from scipy import stats as st
    return {
        "std": float(np.std(res)),
        "skew": float(st.skew(res)),
        "kurtosis": float(st.kurtosis(res)),
        "p01": float(np.percentile(res, 1)),
        "p99": float(np.percentile(res, 99)),
    }


def prepare(dates, radon, cfg, *, use_features: bool, scaler: str):
    """构造训练/测试张量。scaler: 'zscore' | 'robust'。特征通道各自标准化。"""
    train_ratio = cfg["train_ratio"]
    window = cfg["window"]
    n = len(radon)
    k_train = int(n * train_ratio)

    if use_features:
        feat, names = build_feature_matrix(radon, use_roll=True, use_diff=True)
    else:
        feat = radon[:, None]
        names = ["raw"]

    n_feat = feat.shape[1]
    feat_s = np.empty_like(feat, dtype=float)
    for j in range(n_feat):
        col = feat[:, j]
        if scaler == "robust":
            med, scale = robust_fit(col[:k_train])
            feat_s[:, j] = robust_apply(col, med, scale)
        else:
            mu, sd = zscore_fit(col[:k_train])
            feat_s[:, j] = zscore_apply(col, mu, sd)

    # 目标始终用第 0 列（raw）的标准化值，保证 R² 口径可比
    if scaler == "robust":
        t_med, t_scale = robust_fit(radon[:k_train])
        y_all = robust_apply(radon, t_med, t_scale)
        inv = lambda z: z * t_scale + t_med
    else:
        t_mu, t_sd = zscore_fit(radon[:k_train])
        y_all = zscore_apply(radon, t_mu, t_sd)
        inv = lambda z: z * t_sd + t_mu

    # 用特征矩阵滑窗，但目标取 y_all（raw 标准化后）
    # 这里 X 来自 feat_s，y 来自 y_all，需手工对齐
    T = len(radon)
    n_win = T - window
    idx = np.arange(window) + np.arange(n_win)[:, None]
    X = feat_s[idx]                      # [N, window, n_feat]
    y = y_all[idx[:, -1] + 1]            # 预测下一时刻
    tgt_idx = np.arange(window, window + n_win)  # 目标在原序列中的位置
    # make_windows 的 y 是 arr[idx[:,-1]+horizon]，horizon=1 → tgt = window+i+1?
    # idx[:,-1] = window-1+i, +1 → window+i；tgt_idx 应为 window + arange(n_win)
    # 上面 idx 构造：idx[i] = arange(window)+i，最后元素 window-1+i，+1 → window+i。正确。
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    return {
        "X_tr": X[tr_m], "y_tr": y[tr_m],
        "X_te": X[te_m], "y_te": y[te_m],
        "te_dates": dates[tgt_idx[te_m]],
        "n_feat": n_feat, "names": names, "inv": inv,
        "k_train": k_train,
    }


def train_one(prep, cfg, *, loss="mse", scheduler=None, seed=42,
              conv_channels=(64, 32), lstm_hidden=64, dropout=0.4,
              lr=4.5e-4, batch_size=128, patience=15):
    X_tr, y_tr = prep["X_tr"], prep["y_tr"]
    k_val = int(len(X_tr) * cfg.get("val_ratio_of_train", 0.85))
    model = CNNLSTM(n_vars=prep["n_feat"], window=cfg["window"],
                    conv_channels=conv_channels, lstm_hidden=lstm_hidden,
                    dropout=dropout)
    info = train_torch(model, X_tr, y_tr, X_tr[k_val:], y_tr[k_val:],
                       epochs=cfg.get("epochs", 200), batch_size=batch_size,
                       lr=lr, patience=patience, seed=seed,
                       loss=loss, scheduler=scheduler)
    pred = predict_torch(model, prep["X_te"])
    return pred, info, model


def evaluate(name, prep, pred_z, out_root):
    y_te = prep["y_te"]
    m = all_metrics(y_te, pred_z)
    m["year_r2"] = year_r2(prep["te_dates"], y_te, pred_z)
    res = y_te - pred_z
    m["residual"] = residual_stats(res)
    save_run(out_root / name, "test", y_te, pred_z, prep["te_dates"],
             extra={"year_r2": m["year_r2"], "residual": m["residual"]})
    return m


def main():
    cfg = load_config("exp_v61")
    set_seed(cfg["seed"])
    paths = load_paths()
    out_root = paths["results_dir"] / "v61_optimize"
    out_root.mkdir(parents=True, exist_ok=True)

    dates, radon = load_xichang32()
    print(f"数据 {len(radon)} 天 | train_ratio={cfg['train_ratio']}")

    # 最优超参（来自 v6.0 Optuna）
    best = dict(conv_channels=(64, 32), lstm_hidden=64, dropout=0.4,
                lr=4.501e-4, batch_size=128, patience=15)

    experiments = [
        # name, use_features, scaler, loss, scheduler, ensemble_seeds
        ("A_baseline",      False, "zscore", "mse",   None,      [42]),
        ("B_features",      True,  "zscore", "mse",   None,      [42]),
        ("C_robust",        True,  "robust", "mse",   None,      [42]),
        ("D_huber",         True,  "robust", "huber", None,      [42]),
        ("E_cosine",        True,  "robust", "huber", "cosine",  [42]),
        ("F_ensemble",      True,  "robust", "huber", "cosine",  [42, 123, 456, 789, 1024]),
    ]

    summary = {}
    for name, use_feat, scaler, loss, sched, seeds in experiments:
        print(f"\n{'='*60}\n[{name}] features={use_feat} scaler={scaler} "
              f"loss={loss} sched={sched} seeds={seeds}\n{'='*60}")
        prep = prepare(dates, radon, cfg, use_features=use_feat, scaler=scaler)
        print(f"  n_feat={prep['n_feat']} {prep['names']} | "
              f"train={len(prep['X_tr'])} test={len(prep['X_te'])}")

        if len(seeds) == 1:
            pred, info, _ = train_one(prep, cfg, loss=loss, scheduler=sched, seed=seeds[0], **best)
            print(f"  epochs={info['epochs_run']} val_loss={info['best_val_loss']:.5f}")
        else:
            preds = []
            for sd in seeds:
                p, info, _ = train_one(prep, cfg, loss=loss, scheduler=sched, seed=sd, **best)
                preds.append(p)
                print(f"  seed={sd} epochs={info['epochs_run']}")
            pred = np.mean(preds, axis=0)

        m = evaluate(name, prep, pred, out_root)
        print(f"  R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f}")
        print(f"  year_r2={m['year_r2']}")
        print(f"  residual std={m['residual']['std']:.4f} skew={m['residual']['skew']:.3f} "
              f"kurt={m['residual']['kurtosis']:.3f}")
        summary[name] = {
            "use_features": use_feat, "scaler": scaler, "loss": loss,
            "scheduler": sched, "seeds": seeds,
            "n_feat": prep["n_feat"],
            "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
            "year_r2": m["year_r2"], "residual": m["residual"],
        }

    # 汇总表
    out_path = out_root / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "experiments": summary}, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'实验':<16} {'R2':>8} {'RMSE':>8} {'res_std':>8} {'skew':>8} {'kurt':>8}")
    print("=" * 60)
    for name, s in summary.items():
        r = s["residual"]
        print(f"{name:<16} {s['R2']:>8.4f} {s['RMSE']:>8.4f} "
              f"{r['std']:>8.4f} {r['skew']:>8.3f} {r['kurtosis']:>8.3f}")

    print(f"\n按年 R2:")
    for name, s in summary.items():
        print(f"  {name:<16} {s['year_r2']}")

    print(f"\n保存：{out_path}")


if __name__ == "__main__":
    main()
