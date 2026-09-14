"""模型 API — v6.1：主推 E_cosine 与同切分基线；多变量结果保留作对照。

所有 v6.1 预测会反变换到原始水氡尺度（Bq/L）再输出，避免
Z-score 与 Robust 两种标准化混在同一张图里造成“整体偏移”错觉。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
LONG_CSV = ROOT / "data" / "clean" / "long.csv"

# v6.1 65/35 主结果；scaler 用于反变换
MODEL_CONFIGS = [
    {"name": "naive", "display": "Naive", "color": "#999999",
     "dir": "v61_compare/naive", "version": "v6.1", "scaler": "zscore"},
    {"name": "rf", "display": "随机森林 (RF)", "color": "#5470c6",
     "dir": "v61_compare/rf", "version": "v6.1", "scaler": "zscore"},
    {"name": "lstm_rf", "display": "LSTM-RF", "color": "#fac858",
     "dir": "v61_compare/lstm_rf", "version": "v6.1", "scaler": "zscore"},
    {"name": "cnn_lstm", "display": "CNN-LSTM (tuned)", "color": "#ee6666",
     "dir": "v61_cnn_lstm/cnn_lstm_tuned", "version": "v6.1", "scaler": "zscore"},
    {"name": "e_cosine", "display": "E_cosine ★", "color": "#7b61ff",
     "dir": "v61_optimize/E_cosine", "version": "v6.1", "scaler": "robust"},
]

TRAIN_RATIO = 0.65
_cache: dict = {}


@lru_cache(maxsize=1)
def _train_scale_params() -> dict:
    """从西昌32井水氡训练段估计 zscore / robust 参数。"""
    df = pd.read_csv(LONG_CSV, encoding="utf-8-sig",
                     dtype={"station_id": str, "item_code": str})
    mask = (df["station_id"] == "SCXC32") & (df["item_name"] == "水氡")
    s = df.loc[mask, ["date", "value"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.dropna(subset=["value"]).sort_values("date").drop_duplicates("date")
    s = s.set_index("date")["value"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).interpolate(method="linear", limit=3)
    radon = s.to_numpy()
    k = int(len(radon) * TRAIN_RATIO)
    train = radon[:k]
    mu, sd = float(np.nanmean(train)), float(np.nanstd(train))
    if sd <= 1e-12:
        sd = 1.0
    med = float(np.nanmedian(train))
    q75, q25 = np.nanpercentile(train, [75, 25])
    iqr = float(q75 - q25)
    if iqr <= 1e-12:
        iqr = float(np.nanstd(train) or 1.0)
    return {"z_mu": mu, "z_sd": sd, "r_med": med, "r_scale": iqr}


def _to_raw(arr, scaler: str) -> np.ndarray:
    p = _train_scale_params()
    a = np.asarray(arr, dtype=float)
    if scaler == "robust":
        return a * p["r_scale"] + p["r_med"]
    return a * p["z_sd"] + p["z_mu"]


def _metrics_raw(y_true_raw, y_pred_raw) -> dict:
    yt = np.asarray(y_true_raw, float)
    yp = np.asarray(y_pred_raw, float)
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(yt), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "MAPE": round(mape, 2), "R2": round(r2, 4)}


def load_model_data(model_name: str):
    if model_name in _cache:
        return _cache[model_name]

    config = next((m for m in MODEL_CONFIGS if m["name"] == model_name), None)
    if not config:
        return None

    base_dir = RESULTS_DIR / config["dir"]
    metrics_path = base_dir / "metrics.json"
    predictions_path = base_dir / "predictions_test.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df = pd.read_csv(predictions_path, encoding="utf-8-sig")
    if "date" not in df.columns:
        df["date"] = range(len(df))

    scaler = config.get("scaler", "zscore")
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    y_true_raw = _to_raw(y_true, scaler)
    y_pred_raw = _to_raw(y_pred, scaler)

    # 图用原始尺度；R² 与尺度无关，用反变换后的值重算保证一致
    raw_metrics = _metrics_raw(y_true_raw, y_pred_raw)

    result = {
        "config": config,
        "metrics": {**metrics, **raw_metrics, "scale": "Bq/L"},
        "dates": df["date"].tolist(),
        "y_true": [round(float(v), 4) if np.isfinite(v) else None for v in y_true_raw],
        "y_pred": [round(float(v), 4) if np.isfinite(v) else None for v in y_pred_raw],
    }
    _cache[model_name] = result
    return result


@router.get("/models")
async def get_models():
    out = []
    for m in MODEL_CONFIGS:
        data = load_model_data(m["name"])
        out.append({
            "name": m["name"],
            "display": m["display"],
            "color": m["color"],
            "version": m.get("version", "v6.1"),
            "has_data": data is not None,
        })
    return out


@router.get("/predictions/all")
async def get_all_predictions():
    result = {"dates": [], "y_true": [], "models": {}, "version": "v6.1",
              "split": "65/35 test from 2022-05-16"}
    for m in MODEL_CONFIGS:
        data = load_model_data(m["name"])
        if not data:
            continue
        if not result["dates"]:
            result["dates"] = data["dates"]
            result["y_true"] = data["y_true"]
        result["models"][m["name"]] = {
            "display": m["display"],
            "color": m["color"],
            "y_pred": data["y_pred"],
        }
    return result


@router.get("/predictions/{model}")
async def get_predictions(model: str, start: str = Query(None), end: str = Query(None)):
    data = load_model_data(model)
    if not data:
        return {"error": f"Model '{model}' not found or no data"}

    y_true, y_pred, dates = data["y_true"], data["y_pred"], data["dates"]
    residual = [
        round(t - p, 4) if t is not None and p is not None else None
        for t, p in zip(y_true, y_pred)
    ]
    return {
        "model": model,
        "display": data["config"]["display"],
        "version": data["config"].get("version", "v6.1"),
        "dates": dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": residual,
    }


@router.get("/metrics")
async def get_metrics():
    metrics = {}
    year_r2 = {}
    for m in MODEL_CONFIGS:
        data = load_model_data(m["name"])
        if not data:
            continue
        raw = data["metrics"]
        mape = raw.get("MAPE", 0)
        # z-score 尺度下 MAPE 会爆炸（近零/负值），超过 200% 视为无效
        mape_disp = round(mape, 2) if abs(mape) < 200 else None
        metrics[m["name"]] = {
            "display": m["display"],
            "color": m["color"],
            "RMSE": round(raw.get("RMSE", 0), 4),
            "MAE": round(raw.get("MAE", 0), 4),
            "MAPE": mape_disp,
            "R2": round(raw.get("R2", 0), 4),
        }
        if "year_r2" in raw:
            year_r2[m["name"]] = raw["year_r2"]
    # 从 v61_compare/summary.json 补全分年 R²
    summary_path = RESULTS_DIR / "v61_compare" / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for name, mm in summary.get("models", {}).items():
                key = {"cnn_lstm_tuned": "cnn_lstm"}.get(name, name)
                if "year_r2" in mm and key not in year_r2:
                    year_r2[key] = mm["year_r2"]
                if key in metrics and abs(metrics[key].get("R2", 0) - mm.get("R2", 0)) > 1e-9 and "R2" in mm:
                    metrics[key]["R2"] = round(mm.get("R2", metrics[key]["R2"]), 4)
        except Exception:
            pass
    # E_cosine 的 year_r2
    e_path = RESULTS_DIR / "v61_optimize" / "summary.json"
    if e_path.exists():
        try:
            opt = json.loads(e_path.read_text(encoding="utf-8"))
            e = opt.get("experiments", {}).get("E_cosine", {})
            if "year_r2" in e:
                year_r2["e_cosine"] = e["year_r2"]
            if "e_cosine" in metrics and "R2" in e:
                metrics["e_cosine"]["R2"] = round(float(e["R2"]), 4)
        except Exception:
            pass
    return {"models": metrics, "year_r2": year_r2, "version": "v6.1"}


@router.get("/year_r2")
async def get_year_r2():
    m = await get_metrics()
    return m.get("year_r2", {})


@router.get("/predictions/window/{window}")
async def get_window_predictions(window: int):
    """v6.0 多窗口实验（对照用）。"""
    window_dir = RESULTS_DIR / f"window_{window}"
    if not window_dir.exists():
        return {"error": f"Window {window} not found", "dates": [], "y_true": [], "models": {}}

    result = {"dates": [], "y_true": [], "models": {}, "note": "v6.0 多窗口实验（80/20）"}
    first = True
    for model_name in ["rf", "cnn_lstm", "lstm_rf"]:
        pred_file = window_dir / model_name / "predictions_test.csv"
        metrics_file = window_dir / model_name / "metrics.json"
        if not pred_file.exists():
            continue
        df = pd.read_csv(pred_file, encoding="utf-8-sig")
        metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        if first:
            result["dates"] = df["date"].tolist()
            result["y_true"] = [round(float(v), 4) if pd.notna(v) else None for v in df["y_true"]]
            first = False
        display = {"cnn_lstm": "CNN-LSTM", "lstm_rf": "LSTM-RF", "rf": "随机森林"}.get(model_name, model_name)
        color = {"cnn_lstm": "#ee6666", "lstm_rf": "#fac858", "rf": "#5470c6"}.get(model_name, "#999")
        result["models"][model_name] = {
            "display": display, "color": color,
            "y_pred": [round(float(v), 4) if pd.notna(v) else None for v in df["y_pred"]],
            "metrics": {k: round(metrics.get(k, 0), 4) for k in ("RMSE", "MAE", "MAPE", "R2")},
        }
    return result


@router.get("/windows")
async def get_windows():
    windows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("window_"):
            try:
                windows.append(int(d.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return {"windows": windows}


@router.get("/variables")
async def get_variables():
    variables = []
    var_configs = [
        {"name": "radon", "display": "水氡", "description": "单变量 - 水氡浓度"},
        {"name": "level", "display": "水位", "description": "单变量 - 动水位"},
        {"name": "radon_level", "display": "水氡+水位", "description": "双变量"},
        {"name": "radon_level_weather", "display": "水氡+水位+气象", "description": "多变量"},
    ]
    for vc in var_configs:
        var_dir = RESULTS_DIR / vc["name"]
        if var_dir.exists() and any(var_dir.iterdir()):
            variables.append(vc)
    return {"variables": variables, "note": "多变量结果来自 v6.0（80/20），v6.1 主结果见 /metrics"}


@router.get("/predictions/{var_name}/{window}")
async def get_var_predictions(var_name: str, window: int, model: str = "cnn_lstm"):
    pred_file = RESULTS_DIR / var_name / f"window_{window}" / model / "predictions_test.csv"
    metrics_file = RESULTS_DIR / var_name / f"window_{window}" / model / "metrics.json"
    if not pred_file.exists():
        return {"error": f"no predictions for var={var_name}, window={window}, model={model}",
                "dates": [], "y_true": [], "y_pred": []}
    df = pd.read_csv(pred_file, encoding="utf-8-sig")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    return {
        "var_name": var_name, "window": window, "model": model,
        "note": "v6.0 多变量实验（80/20）",
        "dates": df["date"].tolist(),
        "y_true": [round(float(v), 4) if pd.notna(v) else None for v in df["y_true"]],
        "y_pred": [round(float(v), 4) if pd.notna(v) else None for v in df["y_pred"]],
        "metrics": {k: round(metrics.get(k, 0), 4) for k in ("RMSE", "MAE", "MAPE", "R2")},
    }


@router.get("/multi_var_comparison")
async def get_multi_var_comparison():
    path = RESULTS_DIR / "multi_var_comparison.json"
    if not path.exists():
        return {"error": "comparison not found"}
    return json.loads(path.read_text(encoding="utf-8"))
