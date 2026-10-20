"""anomaly/events.py — 异常点聚合成事件 + 多参数重合统计（对齐文献"事件组"口径）。

迁移自 dachuang_cnn_lstm/src/anomaly/events.py，逐行一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_events(dates, scores: np.ndarray, is_anomaly: np.ndarray,
                     max_gap_days: int = 3) -> pd.DataFrame:
    """相邻（间隔 ≤ max_gap_days）异常点合并为一个异常事件。

    返回事件表：event_id, start, end, n_points, peak_score, mean_score。
    """
    dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    idx = np.where(np.asarray(is_anomaly))[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["event_id", "start", "end", "n_points",
                                     "peak_score", "mean_score"])
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if (dt.iloc[i] - dt.iloc[cur[-1]]).days <= max_gap_days:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    rows = []
    for eid, g in enumerate(groups, 1):
        rows.append({"event_id": eid, "start": dt.iloc[g[0]], "end": dt.iloc[g[-1]],
                     "n_points": len(g), "peak_score": float(np.max(scores[g])),
                     "mean_score": float(np.mean(scores[g]))})
    return pd.DataFrame(rows)


def coincidence(events_by_series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """多参数互证：各序列事件两两之间的重叠计数（事件区间有交集即记一次）。"""
    names = list(events_by_series)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = 0
            for _, ea in events_by_series[a].iterrows():
                hit = ((events_by_series[b]["start"] <= ea["end"])
                       & (events_by_series[b]["end"] >= ea["start"]))
                n += int(hit.any())
            rows.append({"series_a": a, "series_b": b, "coincident_events_a": n})
    return pd.DataFrame(rows)
