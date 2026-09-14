"""震例验证 API — 读取 results/v61_anomaly/ 与 v61_optimize 结果。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
V61_ANOM = RESULTS / "v61_anomaly"
V61_OPT = RESULTS / "v61_optimize"
ORIDATA3 = Path("D:/桌面/地质/大创/dataAnalysis/OriData3")


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


def subsequent_eqs(end: str, days: int = 90, min_mag: float = 4.5) -> list[dict]:
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
            "lat": float(r["lat"]) if pd.notna(r.get("lat")) else None,
            "lon": float(r["lon"]) if pd.notna(r.get("lon")) else None,
        })
    return out


@router.get("/earthquakes/verification")
async def get_verification():
    df = _read_csv(V61_ANOM / "eq_verification.csv")
    if df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        in_test = str(r.get("在测试段", "")).lower() in ("true", "1", "yes")
        rows.append({
            "date": str(r.get("地震日期", "")),
            "place": str(r.get("地点", "")),
            "mag": str(r.get("震级", "")),
            "in_test": in_test,
            "n_events": int(r.get("异常事件数", 0) or 0),
            "days_before": None if pd.isna(r.get("震前天数")) else int(r.get("震前天数")),
            "weather": str(r.get("气象干扰", "")),
            "result": str(r.get("对应结果", "")),
            "detail": str(r.get("详情", "")),
        })
    return rows


@router.get("/earthquakes/events")
async def get_events():
    df = _read_csv(V61_ANOM / "events.csv")
    if df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        end = str(r.get("end", ""))[:10]
        eqs = subsequent_eqs(end, days=90, min_mag=4.5)
        m5 = [e for e in eqs if e["mag"] >= 5.0]
        rows.append({
            "id": int(r.get("event_id", 0)),
            "start": str(r.get("start", ""))[:10],
            "end": end,
            "n_points": int(r.get("n_points", 0)),
            "peak_score": round(float(r.get("peak_score", 0)), 3),
            "mean_score": round(float(r.get("mean_score", 0)), 3),
            "n_m45": len(eqs),
            "n_m5": len(m5),
            "subsequent": eqs[:8],
        })
    return rows


@router.get("/earthquakes/sensitivity")
async def get_sensitivity():
    df = _read_csv(V61_ANOM / "sensitivity.csv")
    if df.empty:
        return []
    return [
        {
            "quantile": float(r.get("quantile", 0)),
            "threshold": round(float(r.get("threshold", 0)), 4),
            "n_points": int(r.get("n_anomaly_points", 0)),
        }
        for _, r in df.iterrows()
    ]


@router.get("/earthquakes/timeline")
async def get_timeline():
    pred_path = V61_OPT / "E_cosine" / "predictions_test.csv"
    if not pred_path.exists():
        return {"dates": [], "residual": [], "events": [], "key_eqs": []}
    df = pd.read_csv(pred_path)
    residual = (df["y_true"] - df["y_pred"]).tolist()
    events = await get_events()
    verif = await get_verification()
    key_eqs = [
        {"date": v["date"], "mag": v["mag"], "place": v["place"],
         "in_test": v["in_test"], "result": v["result"]}
        for v in verif
    ]
    return {
        "dates": df["date"].tolist(),
        "residual": [round(float(v), 4) for v in residual],
        "events": events,
        "key_eqs": key_eqs,
    }


@router.get("/earthquakes/summary")
async def get_summary():
    verif = await get_verification()
    events = await get_events()
    in_test = [v for v in verif if v["in_test"]]
    matched = [v for v in in_test if "✓" in v["result"]]
    return {
        "n_key_eqs": len(verif),
        "n_in_test": len(in_test),
        "n_matched": len(matched),
        "n_events": len(events),
        "events_with_m5": sum(1 for e in events if e["n_m5"] > 0),
        "model": "E_cosine (features+robust+huber+cosine)",
        "split": "65/35, test from 2022-05-16",
        "lof_quantile": 0.98,
    }
