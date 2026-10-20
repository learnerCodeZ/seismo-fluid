"""models/evaluate.py — 评估指标与结果落盘。

迁移自 dachuang_cnn_lstm/src/models/evaluate.py，逐行一致。

指标与参照文献公式一致：RMSE / MAE / MAPE / R²。
每次实验落盘 results/<exp>/：metrics.json + predictions.csv。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "MAPE": mape(y_true, y_pred), "R2": r2(y_true, y_pred)}


def save_run(exp_dir: Path, name: str, y_true: np.ndarray, y_pred: np.ndarray,
             dates=None, extra: dict | None = None) -> dict:
    """一次实验 = 一个目录；指标 json + 预测 csv。"""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    m = all_metrics(y_true, y_pred)
    if extra:
        m.update(extra)
    (exp_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    import pandas as pd

    df = {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    if dates is not None:
        df["date"] = np.asarray(dates)
    pd.DataFrame(df).to_csv(exp_dir / f"predictions_{name}.csv",
                            index=False, encoding="utf-8-sig")
    return m
