"""异常检测 API — v6.1：读取 results/v61_anomaly/（E_cosine 残差 LOF）。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
V61_ANOM = RESULTS / "v61_anomaly"
V61_OPT = RESULTS / "v61_optimize"
ORIDATA3 = Path("D:/桌面/地质/大创/dataAnalysis/OriData3")

WORK_QUANTILE = 0.98


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_catalog() -> pd.DataFrame:
    path = ORIDATA3 / "earthquake_catalog.csv"
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    else:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_weather() -> pd.DataFrame:
    path = ORIDATA3 / "西昌.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="gbk")
    except Exception:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(
        df["年"].astype(str) + "-"
        + df["月"].astype(str).str.zfill(2) + "-"
        + df["日"].astype(str).str.zfill(2)
    )
    df = df.set_index("date")[["降水量（mm）"]].copy()
    mask_early = df.index.year < 2020
    df.loc[mask_early, "降水量（mm）"] = df.loc[mask_early, "降水量（mm）"] / 10
    df[df >= 9999] = pd.NA
    return df.astype(float)


def subsequent_eqs(end: str, days: int = 90, min_mag: float = 4.0) -> list[dict]:
    cat = load_catalog()
    if cat.empty:
        return []
    end_ts = pd.Timestamp(end)
    win = cat[(cat["time"] >= end_ts) & (cat["time"] <= end_ts + pd.Timedelta(days=days))]
    win = win[win["mag"] >= min_mag].sort_values("time")
    out = []
    for _, r in win.iterrows():
        out.append({
            "date": str(r["time"].date()) if pd.notna(r["time"]) else "",
            "mag": float(r["mag"]),
            "place": str(r.get("place", "")),
        })
    return out


def precip_sum(start: str, end: str) -> float:
    weather = load_weather()
    if weather.empty:
        return 0.0
    rng = pd.date_range(pd.Timestamp(start), pd.Timestamp(end))
    sub = weather.reindex(rng)
    val = sub["降水量（mm）"].sum()
    return float(val) if pd.notna(val) else 0.0


def load_events_full() -> list[dict]:
    df = _read_csv(V61_ANOM / "events.csv")
    if df.empty:
        return []
    events = []
    for _, r in df.iterrows():
        start = str(r.get("start", ""))[:10]
        end = str(r.get("end", ""))[:10]
        eqs = subsequent_eqs(end, days=90, min_mag=4.0)
        m5 = [e for e in eqs if e["mag"] >= 5.0]
        m5_details = "; ".join(f"{e['date']} M{e['mag']} {e['place']}" for e in m5)
        precip = precip_sum(start, end)
        weather_hit = precip > 10
        events.append({
            "id": int(r.get("event_id", 0)),
            "start": start,
            "end": end,
            "n_points": int(r.get("n_points", 0)),
            "peak_score": round(float(r.get("peak_score", 0)), 3),
            "mean_score": round(float(r.get("mean_score", 0)), 3),
            "precip_mm": round(precip, 1),
            "attribution": "气象干扰" if weather_hit else "非气象",
            "eq_m4": len(eqs),
            "eq_m5": len(m5),
            "eq_m5_details": m5_details if m5_details else "无",
            "correspondence": "✓" if len(eqs) > 0 else "✗",
            "subsequent": eqs[:6],
        })
    return events


def load_lof_series() -> dict:
    """从 labels_residual.csv 读 LOF 得分序列。"""
    path = V61_ANOM / "labels_residual.csv"
    if not path.exists():
        # fallback: 用 E_cosine 残差现算 |res|
        pred = V61_OPT / "E_cosine" / "predictions_test.csv"
        if not pred.exists():
            return {"dates": [], "scores": [], "threshold": 0}
        df = pd.read_csv(pred)
        res = (df["y_true"] - df["y_pred"]).abs()
        thr = float(res.quantile(WORK_QUANTILE))
        return {
            "dates": df["date"].tolist(),
            "scores": [round(float(v), 4) for v in res.tolist()],
            "threshold": round(thr, 4),
        }
    df = pd.read_csv(path)
    scores = df["lof_score"].fillna(0).astype(float)
    thr = float(scores[scores > 0].quantile(WORK_QUANTILE)) if (scores > 0).any() else 0.0
    return {
        "dates": df["date"].tolist(),
        "scores": [round(float(v), 4) for v in scores.tolist()],
        "threshold": round(thr, 4),
    }


def load_sensitivity() -> list[dict]:
    df = _read_csv(V61_ANOM / "sensitivity.csv")
    if df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        out.append({
            "quantile": float(r.get("quantile", 0)),
            "threshold": round(float(r.get("threshold", 0)), 4),
            "n_points": int(r.get("n_anomaly_points", r.get("n_points", 0)) or 0),
            "n_anomalies": int(r.get("n_anomaly_points", r.get("n_points", 0)) or 0),
        })
    return out


@router.get("/anomalies/events")
async def get_events():
    return load_events_full()


@router.get("/anomalies/sensitivity")
async def get_sensitivity():
    return load_sensitivity()


@router.get("/anomalies/scores")
async def get_scores(quantile: float = Query(WORK_QUANTILE)):
    return load_lof_series()


@router.get("/anomalies/labels")
async def get_labels():
    path = V61_ANOM / "labels_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/anomalies/summary")
async def get_summary():
    events = load_events_full()
    sens = load_sensitivity()
    return {
        "total_events": len(events),
        "verified_events": sum(1 for e in events if e["correspondence"] == "✓"),
        "total_m4_eq": sum(e["eq_m4"] for e in events),
        "total_m5_eq": sum(e["eq_m5"] for e in events),
        "model": "E_cosine",
        "split": "65/35, test from 2022-05-16",
        "lof_quantile": WORK_QUANTILE,
        "n_anomaly_points": next(
            (s["n_points"] for s in sens if abs(s["quantile"] - WORK_QUANTILE) < 1e-6),
            sum(e["n_points"] for e in events),
        ),
        "events": events,
    }
