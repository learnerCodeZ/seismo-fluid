"""受控 CNN-GRU 实验：单次训练入口。

用法: python run_train.py --model lstm|gru --seed 42

数据协议与 v6.2 完全一致（configs/exp_controlled_gru.json；见
docs/CNN-GRU_实验计划书.md）。load_radon / prepare 与旧仓 run_v62.py
逐行等价，仅数据路径指向本工作区冻结快照。每次运行：
- set_seed(seed) → 载入数据 → prepare → 构建模型 → train_torch(seed)，
  与旧 run_v62.main 的种子顺序一致（模型初始化前种子已重置）；
- 输出目录 results/controlled_gru/{model}_s{seed}/，已存在则拒绝覆盖；
- 保存 predictions_test.csv（原尺度）、history.csv（逐epoch验证损失）、
  weights.pt + sha256、metrics.json（含参数量、最佳epoch、耗时）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.common import DATA_DIR, RESULTS_DIR, load_config, set_seed
from src.features.build import build_feature_matrix, robust_apply, robust_fit
from src.models.evaluate import all_metrics
from src.models.models import CNNLSTM, CNNGRU, predict_torch, train_torch


def load_radon(cfg):
    frame = pd.read_csv(DATA_DIR / "clean" / "long.csv", encoding="utf-8-sig",
                        dtype={"station_id": str, "item_code": str})
    station = frame[(frame.station_id == cfg["station_id"])
                    & (frame.item_name == cfg["item_name"])][["date", "value"]].copy()
    station["date"] = pd.to_datetime(station["date"])
    series = station.drop_duplicates("date").set_index("date").sort_index()["value"]
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))


def prepare(series, cfg):
    values = series.to_numpy(dtype=float)
    window = cfg["window"]
    test_start = int(len(values) * cfg["train_test_ratio"])
    fit_end = int(test_start * cfg["fit_ratio_within_training"])
    features, names = build_feature_matrix(values)
    scaled = np.column_stack([
        robust_apply(features[:, col], *robust_fit(features[:fit_end, col]))
        for col in range(features.shape[1])
    ])
    center, scale = robust_fit(values[:fit_end])
    target = robust_apply(values, center, scale)
    targets = np.arange(window, len(values))
    inputs = scaled[targets[:, None] - window + np.arange(window)]
    valid = np.isfinite(inputs).all(axis=(1, 2)) & np.isfinite(target[targets])
    partitions = {}
    for name, mask in {
        "fit": targets < fit_end,
        "val": (targets >= fit_end) & (targets < test_start),
        "test": targets >= test_start,
    }.items():
        selected = mask & valid
        partitions[name] = {
            "X": inputs[selected], "y": target[targets[selected]],
            "date": series.index[targets[selected]],
            "raw": values[targets[selected]],
            "previous": values[targets[selected] - 1],
        }
        if not len(partitions[name]["X"]):
            raise ValueError(f"{name} 无有效样本")
    return partitions, (center, scale), names


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def year_metrics(dates, observed, predicted):
    frame = pd.DataFrame({"date": dates, "observed": observed, "predicted": predicted})
    return {str(year): all_metrics(part.observed.to_numpy(), part.predicted.to_numpy())
            for year, part in frame.groupby(frame.date.dt.year)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config("exp_controlled_gru")
    out = RESULTS_DIR / "controlled_gru" / f"{args.model}_s{args.seed}"
    if out.exists():
        print(f"REFUSING: 运行目录已存在，禁止覆盖 -> {out}")
        return 1

    set_seed(args.seed)              # 与 run_v62 相同顺序：先设种子再准备数据、构建模型
    series = load_radon(cfg)
    partitions, (center, scale), names = prepare(series, cfg)
    fit, val, test = (partitions[n] for n in ("fit", "val", "test"))

    if args.model == "lstm":
        model = CNNLSTM(n_vars=len(names), window=cfg["window"],
                        conv_channels=tuple(cfg["conv_channels"]),
                        lstm_hidden=cfg["rnn_hidden"], dropout=cfg["dropout"])
    else:
        model = CNNGRU(n_vars=len(names), window=cfg["window"],
                       conv_channels=tuple(cfg["conv_channels"]),
                       gru_hidden=cfg["rnn_hidden"], dropout=cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model} s{args.seed}] n_fit={len(fit['X'])} n_val={len(val['X'])} "
          f"n_test={len(test['X'])} params={n_params}")

    t0 = time.perf_counter()
    info = train_torch(model, fit["X"], fit["y"], val["X"], val["y"],
                       epochs=cfg["epochs"], patience=cfg["patience"],
                       batch_size=cfg["batch_size"], lr=cfg["lr"], seed=args.seed,
                       loss=cfg["loss"], scheduler=cfg["scheduler"])
    elapsed = time.perf_counter() - t0

    predicted = predict_torch(model, test["X"]) * scale + center
    predictions = pd.DataFrame({"date": test["date"], "y_true": test["raw"],
                                "y_pred": predicted})
    baseline_valid = np.isfinite(test["previous"])
    key = f"cnn_{args.model}"
    metrics = {
        "model": args.model, "seed": args.seed,
        key: all_metrics(test["raw"], predicted),
        "year_metrics": year_metrics(test["date"], test["raw"], predicted),
        "persistence": all_metrics(test["raw"][baseline_valid],
                                   test["previous"][baseline_valid]),
        "n_fit": len(fit["X"]), "n_val": len(val["X"]), "n_test": len(test["X"]),
        "fit_last": str(fit["date"][-1].date()), "val_first": str(val["date"][0].date()),
        "test_first": str(test["date"][0].date()), "test_last": str(test["date"][-1].date()),
        "n_params": n_params, "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"], "best_val_loss": info["best_val_loss"],
        "train_seconds": round(elapsed, 2),
        "torch": torch.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_version": cfg["version"],
    }

    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"epoch": range(1, len(info["history"]) + 1),
                  "val_loss": info["history"]}).to_csv(out / "history.csv", index=False)
    torch.save(model.state_dict(), out / "weights.pt")
    metrics["weights_sha256"] = sha256_file(out / "weights.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    m = metrics[key]
    print(f"[{args.model} s{args.seed}] MAE={m['MAE']:.6f} RMSE={m['RMSE']:.6f} "
          f"R2={m['R2']:.6f} epochs={info['epochs_run']} best_epoch={info['best_epoch']} "
          f"{elapsed:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
