"""本地可靠性复核：v6.1 结果一致性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import load_paths
from src.features.build import robust_fit, zscore_fit


def main() -> None:
    paths = load_paths()

    # 1) E_cosine R2 重算
    pred = pd.read_csv("results/v61_optimize/E_cosine/predictions_test.csv")
    y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    m = json.loads(Path("results/v61_optimize/E_cosine/metrics.json").read_text(encoding="utf-8"))
    print(f"E_cosine stored R2={m['R2']:.6f} recomputed={r2:.6f} match={abs(m['R2'] - r2) < 1e-6}")
    print(f"n_test={len(pred)} {pred['date'].min()} -> {pred['date'].max()}")

    # 2) Robust 反变换 vs 原始序列
    df = pd.read_csv("data/clean/long.csv", encoding="utf-8-sig", dtype={"station_id": str})
    mask = (df["station_id"] == "SCXC32") & (df["item_name"] == "水氡")
    s = df.loc[mask, ["date", "value"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.dropna(subset=["value"]).sort_values("date").drop_duplicates("date")
    s = s.set_index("date")["value"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).interpolate(method="linear", limit=3)
    radon = s.to_numpy()
    k = int(len(radon) * 0.65)
    med, scale = robust_fit(radon[:k])
    raw_from_robust = y * scale + med
    true_raw = s.reindex(pd.to_datetime(pred["date"])).to_numpy()
    err = float(np.nanmean(np.abs(raw_from_robust - true_raw)))
    print(f"robust inverse vs series mean_abs_err={err:.6f} (应≈0)")
    mu, sd = zscore_fit(radon[:k])
    print(f"train zscore mu={mu:.4f} sd={sd:.4f} | robust med={med:.4f} scale={scale:.4f}")

    # 3) compare summary
    sumj = json.loads(Path("results/v61_compare/summary.json").read_text(encoding="utf-8"))
    print("v61_compare models:")
    for name, mm in sumj["models"].items():
        print(f"  {name}: R2={mm['R2']:.4f} n_test={mm.get('n_test')}")

    # 4) 异常/震例
    ev = pd.read_csv("results/v61_anomaly/events.csv")
    print(f"events={len(ev)} total_points={int(ev['n_points'].sum())}")
    ver = pd.read_csv("results/v61_anomaly/eq_verification.csv")
    in_test_col = "在测试段"
    if in_test_col in ver.columns:
        n_in = int((ver[in_test_col].astype(str).str.lower().isin(["true", "1", "yes"])).sum())
        print(f"eq_verification rows={len(ver)} in_test={n_in}")
    lab = pd.read_csv("results/v61_anomaly/labels_residual.csv")
    if "lof_flag" in lab.columns:
        print(f"labels_residual n={len(lab)} lof_points={int(lab['lof_flag'].sum())}")
    sens = pd.read_csv("results/v61_anomaly/sensitivity.csv")
    print("sensitivity:")
    print(sens.to_string(index=False))

    # 5) 与 LOF 事件日期交叉：events start 应都在 labels 的 lof_flag=1 内
    if "lof_flag" in lab.columns and "date" in lab.columns:
        lab["date"] = pd.to_datetime(lab["date"])
        flagged = set(lab.loc[lab["lof_flag"] > 0, "date"].dt.strftime("%Y-%m-%d"))
        ok = True
        for _, row in ev.iterrows():
            st = str(row["start"])[:10]
            if st not in flagged:
                # 事件起点可能是聚合窗口内，允许 start-3 天内有 flag
                near = any(
                    (pd.Timestamp(st) - pd.Timedelta(days=d)).strftime("%Y-%m-%d") in flagged
                    for d in range(0, 4)
                )
                if not near:
                    ok = False
                    print(f"  WARN event start not in lof flags: {st}")
        print(f"events vs labels cross-check ok={ok}")

    print("\nRELIABILITY_CHECK_DONE")


if __name__ == "__main__":
    main()
