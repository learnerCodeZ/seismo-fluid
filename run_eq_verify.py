"""run_eq_verify.py — 震例对应验证（v6.1 Phase 3）。

对 6 个关键地震逐一检查：
1. 震前 90 天窗口内是否有 LOF 异常事件
2. 排除气象干扰后是否仍为真异常
3. 输出震例对应表 + 时间轴标注图

默认读取 results/v61_optimize/E_cosine/predictions_test.csv。
用法：python run_eq_verify.py [--source E_cosine|F_ensemble|cnn_lstm_tuned]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.anomaly.lof_detect import lof_scores, detect, k_strategy, window_features
from src.anomaly.events import aggregate_events

SEED = 42
QUANTILES = [0.98, 0.985, 0.987, 0.99, 0.995]
WORK_QUANTILE = 0.98
PRE_EQ_DAYS = 90

KEY_EARTHQUAKES = [
    {"date": "2018-10-31", "place": "四川凉山州西昌市", "mag": 5.1},
    {"date": "2019-06-17", "place": "四川宜宾市长宁县", "mag": 6.0},
    {"date": "2022-01-02", "place": "云南丽江市宁蒗县", "mag": 5.5},
    {"date": "2022-06-01", "place": "四川省雅安市芦山县", "mag": 6.1},
    {"date": "2022-09-05", "place": "四川省甘孜泸定县", "mag": 6.8},
    {"date": "2023-01-26", "place": "四川甘孜泸定县", "mag": 5.6},
]


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


def load_weather():
    paths = load_paths()
    df = pd.read_csv(paths["oridata3_dir"] / "西昌.csv", encoding="gbk")
    df["date"] = pd.to_datetime(df["年"].astype(str) + "-" +
                                df["月"].astype(str).str.zfill(2) + "-" +
                                df["日"].astype(str).str.zfill(2))
    df = df.set_index("date")[["气温(℃)", "气压(hpa)", "降水量（mm）"]].copy()
    mask_early = df.index.year < 2020
    df.loc[mask_early, "降水量（mm）"] = df.loc[mask_early, "降水量（mm）"] / 10
    df[df >= 9999] = pd.NA
    df = df.astype(float)
    return df


def lof_on_residual(res, dates, quantile=WORK_QUANTILE):
    feat = window_features(res, 7)
    valid = ~np.isnan(feat).any(axis=1)
    k = k_strategy(int(valid.sum()), role="residual")
    scores = lof_scores(feat[valid], feat[valid], k)
    d = dates[6:6 + len(feat)][valid]
    flags, thr = detect(scores, quantile)
    return d, scores, flags, thr, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="E_cosine",
                        choices=["E_cosine", "F_ensemble", "cnn_lstm_tuned"])
    args = parser.parse_args()

    set_seed(SEED)
    paths = load_paths()
    out_dir = paths["results_dir"] / "v61_anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)

    dates, radon = load_xichang32()
    weather = load_weather()
    n = len(radon)
    k_train = int(n * 0.65)
    split_date = pd.Timestamp(dates[k_train])
    print(f"切分：train_ratio=0.65 | 训练截止 {pd.Timestamp(dates[k_train - 1]).date()}"
          f" | 测试起始 {split_date.date()}")

    if args.source == "cnn_lstm_tuned":
        pred_path = paths["results_dir"] / "v61_cnn_lstm" / "cnn_lstm_tuned" / "predictions_test.csv"
    else:
        pred_path = paths["results_dir"] / "v61_optimize" / args.source / "predictions_test.csv"
    if not pred_path.exists():
        print(f"错误：预测结果不存在 {pred_path}")
        print("请先运行 run_v61_optimize.py 或 run_phase3_tuned.py")
        return
    print(f"预测来源：{pred_path}")

    pred_df = pd.read_csv(pred_path)
    pred_dates = pd.to_datetime(pred_df["date"])
    te_residual = (pred_df["y_true"] - pred_df["y_pred"]).to_numpy()
    te_dates_arr = pred_dates.to_numpy()

    print("\n阈值敏感性扫描：")
    sens_rows = []
    for q in QUANTILES:
        _, _, fl, thr, k = lof_on_residual(te_residual, te_dates_arr, q)
        sens_rows.append({"quantile": q, "threshold": thr,
                          "n_anomaly_points": int(fl.sum()), "k": k})
        print(f"  q={q}: thr={thr:.4f}, 异常点={int(fl.sum())}")
    pd.DataFrame(sens_rows).to_csv(out_dir / "sensitivity.csv",
                                   index=False, encoding="utf-8-sig")

    print(f"\nLOF 异常检测（工作阈值 q={WORK_QUANTILE}）...")
    feat_dates, scores, flags, thr, k = lof_on_residual(te_residual, te_dates_arr, WORK_QUANTILE)
    events = aggregate_events(feat_dates, scores, flags)
    print(f"  k={k}, thr={thr:.4f}, 异常点={int(flags.sum())}, 事件={len(events)}")
    if len(events):
        print(events.to_string(index=False))
    events.to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")

    results = []
    print(f"\n{'=' * 70}")
    print("震例对应验证（震前 90 天窗口）")
    print(f"{'=' * 70}")

    for eq in KEY_EARTHQUAKES:
        eq_date = pd.Timestamp(eq["date"])
        in_test = eq_date >= split_date
        window_start = eq_date - pd.Timedelta(days=PRE_EQ_DAYS)

        if not in_test:
            results.append({
                "地震日期": eq["date"], "地点": eq["place"], "震级": f"M{eq['mag']}",
                "在测试段": False, "异常事件数": 0, "震前天数": None,
                "气象干扰": "—", "对应结果": "— 训练段",
                "详情": "地震早于测试段，无法用测试残差检验",
            })
            print(f"  {eq['date']} M{eq['mag']} {eq['place']}: — 训练段（跳过）")
            continue

        matched = events[
            (pd.to_datetime(events["start"]) >= window_start)
            & (pd.to_datetime(events["end"]) <= eq_date)
        ]

        weather_ok = True
        weather_note = ""
        if len(matched) > 0:
            for _, ev in matched.iterrows():
                ev_range = pd.date_range(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]))
                w_sub = weather.reindex(ev_range).dropna(how="all")
                if len(w_sub) > 0:
                    precip = float(w_sub["降水量（mm）"].sum())
                    if precip > 10:
                        weather_ok = False
                        weather_note = f"降水={precip:.1f}mm"

        matched_count = len(matched)
        if matched_count > 0:
            days_before = (eq_date - pd.to_datetime(matched.iloc[0]["start"])).days
            result = "✓ 对应" if weather_ok else "⚠ 待确认"
            detail = (f"{matched_count}个事件, 最近震前{days_before}天, "
                      f"峰值={matched.iloc[0]['peak_score']:.3f}")
        else:
            result = "✗ 未检出"
            detail = "窗口内无异常事件"
            days_before = None

        results.append({
            "地震日期": eq["date"], "地点": eq["place"], "震级": f"M{eq['mag']}",
            "在测试段": True, "异常事件数": matched_count, "震前天数": days_before,
            "气象干扰": weather_note if not weather_ok else "无",
            "对应结果": result, "详情": detail,
        })
        print(f"  {eq['date']} M{eq['mag']} {eq['place']}: {result} ({detail})")

    result_df = pd.DataFrame(results)
    result_path = out_dir / "eq_verification.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n结果保存：{result_path}")

    in_test_rows = [r for r in results if r["在测试段"]]
    matched_count = sum(1 for r in in_test_rows if "✓" in r["对应结果"])
    total = len(in_test_rows)
    rate = f" = {matched_count / total * 100:.1f}%" if total else ""
    print(f"测试段内对应率：{matched_count}/{total}{rate}")

    generate_timeline_plot(dates, radon, pred_dates, te_residual,
                           feat_dates, scores, flags, results, out_dir, split_date)


def generate_timeline_plot(dates, radon, pred_dates, residual,
                           anomaly_dates, scores, anomaly_flags,
                           eq_results, out_dir, split_date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.2, 0.9]})
    a1, a2, a3 = axes

    for ax in axes:
        ax.axvspan(pd.Timestamp(dates[0]), split_date, color="#e8f0fe", alpha=0.5, lw=0)
        ax.axvspan(split_date, pd.Timestamp(dates[-1]), color="#fff4e5", alpha=0.5, lw=0)

    a1.plot(pd.DatetimeIndex(dates), radon, lw=0.6, color="#4a90d9", label="水氡观测值")
    a1.axvline(split_date, color="#666", lw=1, alpha=0.6)
    a1.set_ylabel("水氡 (Bq/L)")
    a1.set_title("西昌32井水氡 · 残差LOF · 震例对应（v6.1 / 65/35 / E_cosine）")
    a1.legend(loc="upper left", fontsize=8)
    a1.spines["top"].set_visible(False)
    a1.spines["right"].set_visible(False)

    pred_dt = pd.DatetimeIndex(pred_dates)
    res_series = pd.Series(residual, index=pred_dt)
    a2.plot(pred_dt, residual, lw=0.5, color="#999", label="预测残差")

    ad = pd.DatetimeIndex(anomaly_dates)
    af = np.asarray(anomaly_flags, bool)
    anom_dates = ad[af]
    anom_vals = res_series.reindex(anom_dates).to_numpy()
    a2.scatter(anom_dates, anom_vals, c="red", s=18, zorder=5, label="LOF 异常点")
    a2.axhline(0, color="k", lw=0.5)
    a2.axvline(split_date, color="#666", lw=1, alpha=0.6)
    a2.set_ylabel("残差")
    a2.legend(loc="upper left", fontsize=8)
    a2.spines["top"].set_visible(False)
    a2.spines["right"].set_visible(False)
    a2.grid(True, ls="--", lw=0.5, alpha=0.3)

    a3.plot(ad, scores, lw=0.6, color="#7b61ff", label="LOF 得分")
    thr = float(np.quantile(scores, WORK_QUANTILE))
    a3.axhline(thr, color="red", lw=0.8, ls="--", alpha=0.7, label=f"阈值 q={WORK_QUANTILE}")
    a3.scatter(ad[af], scores[af], c="red", s=12, zorder=5)
    a3.axvline(split_date, color="#666", lw=1, alpha=0.6)
    a3.set_ylabel("LOF 得分")
    a3.set_xlabel("日期")
    a3.legend(loc="upper left", fontsize=8)
    a3.spines["top"].set_visible(False)
    a3.spines["right"].set_visible(False)

    colors = ["#d46060", "#e8963a", "#fdd663", "#91cc75", "#4a90d9", "#9b59b6"]
    for i, eq in enumerate(eq_results):
        eq_date = pd.Timestamp(eq["地震日期"])
        color = colors[i % len(colors)]
        ls = "--" if eq["在测试段"] else ":"
        for ax in (a1, a2, a3):
            ax.axvline(eq_date, color=color, lw=1.4, ls=ls, alpha=0.85)
        mark = "Y" if "✓" in str(eq["对应结果"]) else (
            "N" if "✗" in str(eq["对应结果"]) else "-")
        a1.annotate(f"{mark} M{eq['震级'][1:]}\n{eq['地震日期'][5:]}",
                    xy=(eq_date, 1.0), xycoords=("data", "axes fraction"),
                    fontsize=7, color=color, ha="center", va="top", fontweight="bold")

    a3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.tight_layout()
    fig_path = out_dir / "eq_timeline.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"时间轴图：{fig_path}")


if __name__ == "__main__":
    main()
