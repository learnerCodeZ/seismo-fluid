"""run_v61_label.py — v6.1 Phase 4 数据自动标记（路径 A）。

对水氡原始序列与 CNN-LSTM 残差序列分别做三种统计标记：
  - zscore: |z| > 3
  - iqr:    超出 Q1-1.5*IQR / Q3+1.5*IQR
  - mad:    滑动窗口中位数绝对偏差，|x-med|/(1.4826*MAD) > 3

输出：
  results/v61_anomaly/labels_auto.csv
  results/v61_anomaly/labels_summary.json
  results/v61_anomaly/labels_overlap.png

用法：python run_v61_label.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.anomaly.lof_detect import lof_scores, detect, k_strategy, window_features
from src.anomaly.events import aggregate_events

MAD_SCALE = 1.4826
MAD_WINDOW = 30
Z_THRESH = 3.0
IQR_K = 1.5
MAD_THRESH = 3.0


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


def flag_zscore(x: np.ndarray, thresh: float = Z_THRESH) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = np.nanmean(x), np.nanstd(x)
    sd = sd if sd > 1e-12 else 1.0
    z = (x - mu) / sd
    return np.abs(z) > thresh, z


def flag_iqr(x: np.ndarray, k: float = IQR_K) -> tuple[np.ndarray, np.ndarray]:
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    score = np.maximum((lo - x) / (iqr + 1e-12), (x - hi) / (iqr + 1e-12))
    score = np.clip(score, 0, None)
    return (x < lo) | (x > hi), score


def flag_mad(x: np.ndarray, window: int = MAD_WINDOW,
             thresh: float = MAD_THRESH) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(np.asarray(x, float))
    med = s.rolling(window, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=5).median()
    score = ((s - med).abs() / (MAD_SCALE * mad.replace(0, np.nan))).to_numpy()
    score = np.nan_to_num(score, nan=0.0)
    return score > thresh, score


def main():
    set_seed(42)
    paths = load_paths()
    out_dir = paths["results_dir"] / "v61_anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 原始序列标记 ----
    dates, radon = load_xichang32()
    raw_zf, raw_z = flag_zscore(radon)
    raw_if, raw_i = flag_iqr(radon)
    raw_mf, raw_m = flag_mad(radon)

    # ---- 残差序列标记（E_cosine） ----
    pred = pd.read_csv(out_dir.parent / "v61_optimize" / "E_cosine" / "predictions_test.csv")
    pred["date"] = pd.to_datetime(pred["date"])
    res = (pred["y_true"] - pred["y_pred"]).to_numpy()
    res_dates = pred["date"].to_numpy()
    res_zf, res_z = flag_zscore(res)
    res_if, res_i = flag_iqr(res)
    res_mf, res_m = flag_mad(res, window=15)

    # ---- LOF 标记（对照） ----
    feat = window_features(res, 7)
    valid = ~np.isnan(feat).any(axis=1)
    k = k_strategy(int(valid.sum()), "residual")
    scores = lof_scores(feat[valid], feat[valid], k)
    lof_flags_full = np.zeros(len(res), bool)
    lof_d = res_dates[6:6 + len(feat)][valid]
    flags, thr = detect(scores, 0.98)
    # 映射回残差索引
    res_dt = pd.DatetimeIndex(res_dates)
    lof_flag_series = pd.Series(False, index=res_dt)
    lof_flag_series.loc[pd.DatetimeIndex(lof_d)] = flags
    lof_flags = lof_flag_series.to_numpy()
    # LOF 得分对齐
    lof_score_series = pd.Series(np.nan, index=res_dt)
    lof_score_series.loc[pd.DatetimeIndex(lof_d)] = scores
    lof_sc = lof_score_series.fillna(0).to_numpy()

    # ---- 落盘 labels_auto ----
    raw_df = pd.DataFrame({
        "date": dates, "value": radon,
        "zscore_flag": raw_zf, "zscore_z": raw_z,
        "iqr_flag": raw_if, "iqr_score": raw_i,
        "mad_flag": raw_mf, "mad_score": raw_m,
    })
    raw_df.to_csv(out_dir / "labels_raw.csv", index=False, encoding="utf-8-sig")

    res_df = pd.DataFrame({
        "date": res_dates, "residual": res,
        "zscore_flag": res_zf, "zscore_z": res_z,
        "iqr_flag": res_if, "iqr_score": res_i,
        "mad_flag": res_mf, "mad_score": res_m,
        "lof_flag": lof_flags, "lof_score": lof_sc,
    })
    res_df.to_csv(out_dir / "labels_residual.csv", index=False, encoding="utf-8-sig")

    # 合并主表（以残差测试段为主，附原始值）
    raw_lookup = pd.Series(radon, index=pd.DatetimeIndex(dates))
    merged = res_df.copy()
    merged["radon"] = merged["date"].map(raw_lookup)
    merged.to_csv(out_dir / "labels_auto.csv", index=False, encoding="utf-8-sig")

    # ---- 汇总 ----
    def overlap(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return {"n_a": int(a.sum()), "n_b": int(b.sum()),
                "intersection": inter, "jaccard": round(inter / union, 4) if union else 0.0}

    summary = {
        "raw_series": {
            "n": len(radon),
            "zscore": int(raw_zf.sum()), "iqr": int(raw_if.sum()), "mad": int(raw_mf.sum()),
            "any": int((raw_zf | raw_if | raw_mf).sum()),
            "all_three": int((raw_zf & raw_if & raw_mf).sum()),
        },
        "residual_series": {
            "n": len(res),
            "zscore": int(res_zf.sum()), "iqr": int(res_if.sum()),
            "mad": int(res_mf.sum()), "lof": int(lof_flags.sum()),
            "any": int((res_zf | res_if | res_mf).sum()),
            "all_three_stat": int((res_zf & res_if & res_mf).sum()),
        },
        "residual_overlap_with_lof": {
            "zscore": overlap(res_zf, lof_flags),
            "iqr": overlap(res_if, lof_flags),
            "mad": overlap(res_mf, lof_flags),
            "stat_any": overlap(res_zf | res_if | res_mf, lof_flags),
        },
        "params": {
            "z_thresh": Z_THRESH, "iqr_k": IQR_K,
            "mad_window": MAD_WINDOW, "mad_thresh": MAD_THRESH,
            "lof_quantile": 0.98, "lof_k": int(k),
        },
    }
    with open(out_dir / "labels_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== 原始序列 ===")
    print(f"  n={len(radon)}  zscore={int(raw_zf.sum())}  iqr={int(raw_if.sum())}  "
          f"mad={int(raw_mf.sum())}  任一={int((raw_zf|raw_if|raw_mf).sum())}  "
          f"三者交集={int((raw_zf&raw_if&raw_mf).sum())}")
    print("=== 残差序列（测试段） ===")
    print(f"  n={len(res)}  zscore={int(res_zf.sum())}  iqr={int(res_if.sum())}  "
          f"mad={int(res_mf.sum())}  lof={int(lof_flags.sum())}")
    print("=== 统计法 vs LOF 重合（Jaccard） ===")
    for name, o in summary["residual_overlap_with_lof"].items():
        print(f"  {name:<10} |a|={o['n_a']:<4} |b|={o['n_b']:<4} "
              f"∩={o['intersection']:<4} J={o['jaccard']}")

    # ---- 图 ----
    plot_labels(dates, radon, res_dates, res, res_zf, res_if, res_mf,
                lof_flags, lof_sc, out_dir)
    print(f"\n保存：{out_dir}/labels_auto.csv, labels_raw.csv, labels_residual.csv, "
          f"labels_summary.json, labels_overlap.png")


def plot_labels(dates, radon, res_dates, res, zf, inf, mf, lof_f, lof_s, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    a1, a2, a3 = axes

    # 上：原始 + 统计标记
    a1.plot(pd.DatetimeIndex(dates), radon, lw=0.5, color="#4a90d9", label="水氡")
    for flag, color, name in [(zf, "#d62728", "Z>3"), (inf, "#ff7f0e", "IQR"),
                              (mf, "#2ca02c", "MAD")]:
        idx = np.where(flag)[0]
        if len(idx):
            a1.scatter(pd.DatetimeIndex(dates)[idx], radon[idx], c=color, s=12,
                       zorder=5, label=name)
    a1.set_ylabel("水氡 (Bq/L)")
    a1.set_title("原始序列自动标记（Z-score / IQR / MAD）")
    a1.legend(loc="upper left", fontsize=8, ncol=4)
    a1.spines["top"].set_visible(False)
    a1.spines["right"].set_visible(False)

    # 中：残差 + 统计标记 + LOF
    res_dt = pd.DatetimeIndex(res_dates)
    a2.plot(res_dt, res, lw=0.5, color="#999", label="残差")
    for flag, color, name in [(zf, "#d62728", "Z>3"), (inf, "#ff7f0e", "IQR"),
                              (mf, "#2ca02c", "MAD"), (lof_f, "#9467bd", "LOF")]:
        idx = np.where(flag)[0]
        if len(idx):
            a2.scatter(res_dt[idx], res[idx], c=color, s=14, zorder=5, label=name)
    a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("残差")
    a2.set_title("残差序列自动标记 vs LOF")
    a2.legend(loc="upper left", fontsize=8, ncol=5)
    a2.spines["top"].set_visible(False)
    a2.spines["right"].set_visible(False)
    a2.grid(True, ls="--", lw=0.5, alpha=0.3)

    # 下：LOF 得分
    a3.plot(res_dt, lof_s, lw=0.6, color="#7b61ff", label="LOF 得分")
    thr = float(np.quantile(lof_s[lof_s > 0], 0.98)) if (lof_s > 0).any() else 1.8
    a3.axhline(thr, color="red", ls="--", lw=0.8, alpha=0.7, label="q=0.98")
    a3.set_ylabel("LOF 得分")
    a3.set_xlabel("日期")
    a3.legend(loc="upper left", fontsize=8)
    a3.spines["top"].set_visible(False)
    a3.spines["right"].set_visible(False)
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(out_dir / "labels_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
