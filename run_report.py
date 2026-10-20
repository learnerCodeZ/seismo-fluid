"""步骤4-6：汇总评价、残差 LOF、图件与 report.md 生成。

读取 10 次训练产物（results/controlled_gru/{model}_s{seed}/），
生成 comparison/、lof/、figures/ 与 report.md，并回写 manifest.json。
评价口径与 run_v62.py 等价函数逐行一致；井位缺失时命中/误报记"不可评估"。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.anomaly.events import aggregate_events  # noqa: E402
from src.anomaly.lof_detect import (  # noqa: E402
    blocked_self_reference_scores, k_strategy, window_features)
from src.common import DATA_DIR, FIGURES_DIR, RESULTS_DIR, load_config  # noqa: E402
from src.models.evaluate import all_metrics, mae, r2, rmse  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = RESULTS_DIR / "controlled_gru"
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["lstm", "gru"]
LABEL = {"lstm": "CNN-LSTM", "gru": "CNN-GRU"}
COLOR = {"lstm": "#7b61ff", "gru": "#f28e2b"}

KEY_EARTHQUAKES = [
    ("2018-10-31", "西昌", "M5.1"),
    ("2019-06-17", "长宁", "M6.0"),
    ("2022-01-02", "宁蒗", "M5.5"),
    ("2022-06-01", "芦山", "M6.1"),
    ("2022-09-05", "泸定", "M6.8"),
    ("2023-01-26", "泸定", "M5.6"),
    ("2026-01-19", "巧家（项目参考震例）", "Ms5.1"),
]


# ---------- 运行读取 ----------
def run_dir(model: str, seed: int) -> Path:
    return BASE / f"{model}_s{seed}"


def load_predictions(model: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(run_dir(model, seed) / "predictions_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_metrics(model: str, seed: int) -> dict:
    return json.loads((run_dir(model, seed) / "metrics.json").read_text(encoding="utf-8"))


# ---------- LOF / 事件 / 震例（与 run_v62.py 逐行等价，路径本地化） ----------
def score_residuals(predictions, cfg):
    residuals = (predictions.y_true - predictions.y_pred).to_numpy(dtype=float)
    features = window_features(residuals, cfg["lof_window"])
    dates = pd.DatetimeIndex(predictions.date)
    window = cfg["lof_window"]
    consecutive = np.asarray((dates[window - 1:] - dates[:-(window - 1)]).days == window - 1)
    valid = np.isfinite(features).all(axis=1) & consecutive
    k = k_strategy(int(valid.sum()), role="residual")
    scores = blocked_self_reference_scores(features[valid], k,
                                           block_size=cfg["lof_block_size"],
                                           embargo=cfg["lof_embargo"])
    return pd.DataFrame({"date": dates[window - 1:][valid], "score": scores}), k


def events_at_quantile(scores, quantile, window):
    threshold = float(scores.score.quantile(quantile))
    flagged = (scores.score > threshold).to_numpy()
    events = aggregate_events(scores.date, scores.score.to_numpy(), flagged)
    events = events.rename(columns={"n_points": "n_flagged_windows"})
    events["alarm_date"] = events["end"]
    events["support_start"] = events["start"] - pd.Timedelta(days=window - 1)
    return events, threshold, int(flagged.sum())


def key_earthquake_table(events, score_dates, cfg):
    first, last = pd.Timestamp(score_dates.iloc[0]), pd.Timestamp(score_dates.iloc[-1])
    rows = []
    for date, place, magnitude in KEY_EARTHQUAKES:
        earthquake = pd.Timestamp(date)
        start = earthquake - pd.Timedelta(days=cfg["pre_eq_days"])
        if earthquake < first:
            status = "训练段或无 LOF 得分"
        elif earthquake > last:
            status = "超出观测段"
        elif start < first:
            status = "震前窗口不完整"
        else:
            status = "窗口完整"
        alarms = events[(events.alarm_date > start) & (events.alarm_date < earthquake)]
        first_alarm = pd.Timestamp(alarms.iloc[0].alarm_date) if len(alarms) else None
        rows.append({
            "date": date, "place": place, "magnitude": magnitude,
            "window_status": status,
            "observable_days": max(0, min((earthquake - first).days, cfg["pre_eq_days"])),
            "n_events": len(alarms) if status == "窗口完整" else None,
            "days_before": (earthquake - first_alarm).days if first_alarm is not None
                           and status == "窗口完整" else None,
            "detection": ("候选异常" if len(alarms) else "未检出")
                         if status == "窗口完整" else "不可评估",
            "attribution": "未排除气象/仪器干扰",
            "spatial_status": "井位坐标未核实，不能判断空间相关性",
        })
    return pd.DataFrame(rows)


def load_catalog(cfg):
    latitude, longitude, radius = (cfg.get(name) for name in
                                   ("station_lat", "station_lon", "earthquake_radius_km"))
    if any(value is None for value in (latitude, longitude, radius)):
        return None
    raise ValueError("井位已配置时须走空间筛选路径；本实验按计划保持井位缺失状态")


def alarm_metrics(events, scores, cfg, targets):
    first, last = pd.Timestamp(scores.date.iloc[0]), pd.Timestamp(scores.date.iloc[-1])
    horizon = pd.Timedelta(days=cfg["pre_eq_days"])
    days = pd.date_range(first + horizon, last, freq="D")
    alarmed = np.zeros(len(days), dtype=bool)
    for event in events.itertuples():
        alarmed |= (days > event.alarm_date) & (days <= event.alarm_date + horizon)
    result = {"alarm_days": int(alarmed.sum()), "evaluated_days": len(days),
              "alarm_fraction": float(alarmed.mean()) if len(days) else None}
    if targets is None:
        result.update({"target_status": "井位坐标/半径缺失，地震命中和误报不可评估",
                       "target_count": None, "miss_fraction": None,
                       "false_alarm_events": None})
        return result
    raise ValueError("井位已配置时须走空间筛选路径；本实验按计划保持井位缺失状态")


def weather_context(events):
    path = DATA_DIR / "context" / "西昌.csv"
    weather = pd.read_csv(path, encoding="gbk")
    weather["date"] = pd.to_datetime(dict(year=weather["年"], month=weather["月"],
                                          day=weather["日"]))
    weather = weather.set_index("date").sort_index()
    rain = pd.to_numeric(weather["降水量（mm）"], errors="coerce")
    rain = rain.mask(rain.isin([9999, 32766]))
    rain.loc[rain.index.year < 2020] /= 10
    temperature = pd.to_numeric(weather["气温(℃)"], errors="coerce").mask(
        lambda values: values.isin([9999, 32766]))
    pressure = pd.to_numeric(weather["气压(hpa)"], errors="coerce").mask(
        lambda values: values.isin([9999, 32766]))
    rows = []
    for event in events.itertuples():
        dates = pd.date_range(event.alarm_date - pd.Timedelta(days=6),
                              event.alarm_date, freq="D")
        period = rain.reindex(dates)
        temp_period = temperature.reindex(dates)
        pressure_period = pressure.reindex(dates)
        rows.append({"event_id": event.event_id, "alarm_date": event.alarm_date,
                     "rain_7d_mm": period.sum(min_count=7) if period.notna().all() else np.nan,
                     "rain_days_observed": int(period.notna().sum()),
                     "temperature_mean_7d": temp_period.mean() if temp_period.notna().all() else np.nan,
                     "pressure_range_7d": (pressure_period.max() - pressure_period.min())
                                          if pressure_period.notna().all() else np.nan,
                     "attribution": "仅有气象上下文；未排除仪器/人为/地质干扰"})
    return pd.DataFrame(rows)


# ---------- 不确定性 ----------
def block_resample_ci(daily_diff: np.ndarray, block: int, n_draws: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(daily_diff)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_draws, n_blocks))
    offsets = np.arange(block)
    means = np.array([daily_diff[(starts[i][:, None] + offsets).ravel()].mean()
                      for i in range(n_draws)])
    return {"point": float(daily_diff.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5))}


# ---------- 图件 ----------
def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.5, alpha=0.45)
    ax.tick_params(labelsize=9)


def fig_ppt(pred: pd.DataFrame, year: int | None, model: str, out: Path) -> dict:
    df = pred if year is None else pred[pred.date.dt.year == year]
    period = (f"{year} 年" if year is not None
              else f"测试全段 {df.date.min():%Y-%m-%d}~{df.date.max():%Y-%m-%d}")
    x = df.date.to_numpy()
    y = df.y_true.to_numpy()
    p = df.y_pred.to_numpy()
    res = y - p
    stats = {"R2": r2(y, p), "RMSE": rmse(y, p), "MAE": mae(y, p), "n": len(df)}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    ax1.plot(x, y, lw=1.1, color="#5b8ff9", label="观测值", zorder=2)
    ax1.plot(x, p, lw=1.3, color="#7b61ff", label=f"{LABEL[model]} 预测", zorder=3)
    ax1.set_ylabel("水氡", fontsize=11)
    ax1.set_title(
        f"西昌32井水氡 · {period}预测 vs 观测（{LABEL[model]}，seed=42）\n"
        f"R² = {stats['R2']:.3f} ｜ RMSE = {stats['RMSE']:.3f} ｜ MAE = {stats['MAE']:.3f}",
        fontsize=12, pad=10)
    ax1.legend(loc="upper right", frameon=False, fontsize=10)
    _style_axis(ax1)

    ax2.axhline(0, color="#333", lw=0.8)
    ax2.fill_between(x, 0, res, where=res >= 0, color="#5b8ff9", alpha=0.45,
                     label="残差 > 0")
    ax2.fill_between(x, 0, res, where=res < 0, color="#e8684a", alpha=0.45,
                     label="残差 < 0")
    ax2.set_ylabel("残差", fontsize=10)
    ax2.set_xlabel("日期", fontsize=10)
    ax2.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    _style_axis(ax2)
    ax2.xaxis.set_major_locator(
        plt.matplotlib.dates.MonthLocator(interval=1 if year is not None else 6))
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.text(0.995, 0.005, "纵轴为原数据单位（水氡单位待核实）", ha="right", va="bottom",
             fontsize=8, color="#777")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def fig_ppt_combined(preds: dict, year: int | None, out: Path) -> dict:
    """PPT 风格双图合一：CNN-GRU（本轮）与 v6.2 CNN-LSTM 上下对照，共用时间轴。

    LSTM 面板用 seed=42 新跑结果（已验证与 v6.2 参照预测逐位一致）。
    year=None 时绘制整个测试段。
    """
    full = preds[("gru", 42)]
    period = (f"{year} 年" if year is not None
              else f"测试全段 {full.date.min():%Y-%m-%d}~{full.date.max():%Y-%m-%d}")
    panels = []
    for model in ("gru", "lstm"):
        df = preds[(model, 42)]
        df = df if year is None else df[df.date.dt.year == year]
        y = df.y_true.to_numpy()
        p = df.y_pred.to_numpy()
        panels.append({"model": model, "x": df.date.to_numpy(), "y": y, "p": p,
                       "res": y - p,
                       "stats": {"R2": r2(y, p), "RMSE": rmse(y, p),
                                 "MAE": mae(y, p), "n": len(df)}})
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 10.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]})
    ax1, ax2, ax3, ax4 = axes
    tag = {"gru": "CNN-GRU（本轮，seed=42）",
           "lstm": "CNN-LSTM（= v6.2，seed=42 逐位复现）"}
    for ax_top, ax_res, panel in ((ax1, ax2, panels[0]), (ax3, ax4, panels[1])):
        st = panel["stats"]
        ax_top.plot(panel["x"], panel["y"], lw=1.1, color="#5b8ff9",
                    label="观测值", zorder=2)
        ax_top.plot(panel["x"], panel["p"], lw=1.3, color="#7b61ff",
                    label=f"{LABEL[panel['model']]} 预测", zorder=3)
        ax_top.set_ylabel("水氡", fontsize=10)
        ax_top.set_title(
            f"{tag[panel['model']]}  R² = {st['R2']:.3f} ｜ RMSE = {st['RMSE']:.3f} "
            f"｜ MAE = {st['MAE']:.3f}", fontsize=11, pad=6)
        ax_top.legend(loc="upper right", frameon=False, fontsize=9)
        _style_axis(ax_top)

        ax_res.axhline(0, color="#333", lw=0.8)
        ax_res.fill_between(panel["x"], 0, panel["res"], where=panel["res"] >= 0,
                            color="#5b8ff9", alpha=0.45, label="残差 > 0")
        ax_res.fill_between(panel["x"], 0, panel["res"], where=panel["res"] < 0,
                            color="#e8684a", alpha=0.45, label="残差 < 0")
        ax_res.set_ylabel("残差", fontsize=9)
        ax_res.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
        _style_axis(ax_res)
    ax4.set_xlabel("日期", fontsize=10)
    ax4.xaxis.set_major_locator(
        plt.matplotlib.dates.MonthLocator(interval=1 if year is not None else 6))
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.subplots_adjust(hspace=0.20, bottom=0.075, top=0.94)
    fig.suptitle(f"西昌32井水氡 · {period}预测 vs 观测：CNN-GRU（本轮）与 v6.2 CNN-LSTM 同图对照",
                 fontsize=12.5, y=0.99)
    fig.text(0.995, 0.002, "纵轴为原数据单位（水氡单位待核实）", ha="right", va="bottom",
             fontsize=8, color="#777")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"gru": panels[0]["stats"], "lstm": panels[1]["stats"]}


def fig_compare(preds: dict, year_lo: int, year_hi: int, out: Path) -> dict:
    dl = preds[("lstm", 42)]
    dg = preds[("gru", 42)]
    mask = (dl.date >= pd.Timestamp(year_lo, 1, 1)) & (dl.date <= pd.Timestamp(year_hi, 12, 31))
    dl, dg = dl[mask].reset_index(drop=True), dg[mask].reset_index(drop=True)
    x = dl.date.to_numpy()
    res_l = dl.y_true.to_numpy() - dl.y_pred.to_numpy()
    res_g = dg.y_true.to_numpy() - dg.y_pred.to_numpy()
    mae_l, mae_g = mae(dl.y_true, dl.y_pred), mae(dg.y_true, dg.y_pred)
    period = f"{year_lo}" if year_lo == year_hi else f"{year_lo}–{year_hi}"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    ax1.plot(x, dl.y_true, lw=1.0, color="#5b8ff9", label="观测值", zorder=2)
    ax1.plot(x, dl.y_pred, lw=1.05, color=COLOR["lstm"], label="CNN-LSTM 预测", zorder=3)
    ax1.plot(x, dg.y_pred, lw=1.05, color=COLOR["gru"], label="CNN-GRU 预测", zorder=4)
    ax1.set_ylabel("水氡", fontsize=11)
    ax1.set_title(
        f"西昌32井水氡 · {period}：CNN-LSTM vs CNN-GRU（seed=42）\n"
        f"MAE：LSTM = {mae_l:.3f} ｜ GRU = {mae_g:.3f} ｜ "
        f"GRU−LSTM = {mae_g - mae_l:+.3f}", fontsize=12, pad=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=10)
    _style_axis(ax1)

    ax2.axhline(0, color="#333", lw=0.8)
    ax2.plot(x, res_l, lw=0.9, color=COLOR["lstm"], label="LSTM 残差")
    ax2.plot(x, res_g, lw=0.9, color=COLOR["gru"], label="GRU 残差")
    ax2.set_ylabel("残差", fontsize=10)
    ax2.set_xlabel("日期", fontsize=10)
    ax2.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    _style_axis(ax2)
    step = 1 if year_lo == year_hi else 2
    ax2.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=step))
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.text(0.995, 0.005, "纵轴为原数据单位（水氡单位待核实）", ha="right", va="bottom",
             fontsize=8, color="#777")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"mae_lstm": mae_l, "mae_gru": mae_g, "n": len(dl)}


def fig_summary(summary: pd.DataFrame, year_df: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.plot(summary.seed, summary.lstm_MAE, "o-", color=COLOR["lstm"], label="CNN-LSTM")
    ax1.plot(summary.seed, summary.gru_MAE, "s-", color=COLOR["gru"], label="CNN-GRU")
    for _, row in summary.iterrows():
        ax1.annotate("", xy=(row.seed, row.gru_MAE), xytext=(row.seed, row.lstm_MAE),
                     arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7, zorder=1))
    ax1.set_xlabel("随机种子")
    ax1.set_ylabel("测试 MAE")
    ax1.set_title("五种子配对：测试 MAE（同种子同数据同洗牌）", fontsize=11)
    ax1.set_xticks(SEEDS)
    ax1.legend(frameon=False, fontsize=9)
    _style_axis(ax1)

    x = np.arange(len(year_df))
    ax2.bar(x - 0.2, year_df.lstm_MAE, 0.4, color=COLOR["lstm"], label="CNN-LSTM")
    ax2.bar(x + 0.2, year_df.gru_MAE, 0.4, color=COLOR["gru"], label="CNN-GRU")
    for i, row in year_df.iterrows():
        better = row.gru_MAE <= row.lstm_MAE
        ax2.annotate(f"{row.gru_vs_lstm_pct:+.1f}%", (i, max(row.lstm_MAE, row.gru_MAE)),
                     ha="center", va="bottom", fontsize=8,
                     color="#1a7f4b" if better else "#b3261e")
    ax2.set_xticks(x, year_df.year)
    ax2.set_xlabel("测试年")
    ax2.set_ylabel("MAE")
    ax2.set_title("逐年五种子平均 MAE（标注 GRU−LSTM %）", fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    _style_axis(ax2)
    fig.tight_layout()
    fig.text(0.995, 0.002, "纵轴为原数据单位（水氡单位待核实）", ha="right", va="bottom",
             fontsize=8, color="#777")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- markdown 工具 ----------
def df_to_md(df: pd.DataFrame, fmt: str = "{:.6f}") -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else fmt.format(v))
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "|" + "---|" * len(out.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in out.iterrows()]
    return "\n".join([header, sep] + rows)


def main() -> int:
    cfg = load_config("exp_controlled_gru")

    # 完整性前置断言
    preds, mets = {}, {}
    for m in MODELS:
        for s in SEEDS:
            preds[(m, s)] = load_predictions(m, s)
            mets[(m, s)] = load_metrics(m, s)
    ref = pd.read_csv(PROJ_REF := DATA_DIR.parent / "reference" / "v62" / "predictions_test.csv")
    ref_dates = list(pd.to_datetime(ref.date))
    ref_y = ref.y_true.to_numpy(float)
    for (m, s), df in preds.items():
        if len(df) != 1451 or list(df.date) != ref_dates \
                or not np.array_equal(df.y_true.to_numpy(float), ref_y) \
                or not np.isfinite(df.y_pred).all():
            print(f"FAIL: 运行 {m}_s{s} 预测与切分不一致，终止出报告")
            return 1

    out_cmp, out_lof, out_fig = BASE / "comparison", BASE / "lof", BASE / "figures"
    for d in (out_cmp, out_lof, out_fig):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 步骤4：汇总与配对差 ----------
    rows = []
    for s in SEEDS:
        ml = mets[("lstm", s)]["cnn_lstm"]
        mg = mets[("gru", s)]["cnn_gru"]
        rows.append({
            "seed": s,
            "lstm_MAE": ml["MAE"], "lstm_RMSE": ml["RMSE"], "lstm_R2": ml["R2"],
            "gru_MAE": mg["MAE"], "gru_RMSE": mg["RMSE"], "gru_R2": mg["R2"],
            "mae_diff_gru_minus_lstm": mg["MAE"] - ml["MAE"],
            "mae_improve_frac": (ml["MAE"] - mg["MAE"]) / ml["MAE"],
            "gru_improved": bool(mg["MAE"] < ml["MAE"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_cmp / "summary_per_seed.csv", index=False, encoding="utf-8-sig")

    mean_lstm, mean_gru = summary.lstm_MAE.mean(), summary.gru_MAE.mean()
    persistence = mets[("lstm", 42)]["persistence"]
    for (m, s), met in mets.items():
        if abs(met["persistence"]["MAE"] - persistence["MAE"]) > 1e-12:
            print(f"FAIL: {m}_s{s} 昨日值基线不一致，终止")
            return 1

    years = sorted({y for met in mets.values() for y in met["year_metrics"]})
    year_rows = []
    for y in years:
        lm = float(np.mean([mets[("lstm", s)]["year_metrics"][y]["MAE"] for s in SEEDS]))
        gm = float(np.mean([mets[("gru", s)]["year_metrics"][y]["MAE"] for s in SEEDS]))
        lr2v = float(np.mean([mets[("lstm", s)]["year_metrics"][y]["R2"] for s in SEEDS]))
        gr2v = float(np.mean([mets[("gru", s)]["year_metrics"][y]["R2"] for s in SEEDS]))
        year_rows.append({"year": y, "lstm_MAE": lm, "gru_MAE": gm,
                          "gru_vs_lstm_pct": (gm - lm) / lm * 100,
                          "lstm_R2": lr2v, "gru_R2": gr2v})
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(out_cmp / "year_metrics.csv", index=False, encoding="utf-8-sig")

    # 块重抽样（对同日配对绝对误差差的五种子均值序列）
    diff = np.mean([np.abs(preds[("gru", s)].y_pred.to_numpy()
                           - preds[("gru", s)].y_true.to_numpy())
                    - np.abs(preds[("lstm", s)].y_pred.to_numpy()
                             - preds[("lstm", s)].y_true.to_numpy())
                    for s in SEEDS], axis=0)
    ci = block_resample_ci(diff, cfg["resample"]["block_days"],
                           cfg["resample"]["n_draws"], cfg["resample"]["seed"])
    per_seed_ci = []
    for s in SEEDS:
        d_s = (np.abs(preds[("gru", s)].y_pred - preds[("gru", s)].y_true)
               - np.abs(preds[("lstm", s)].y_pred - preds[("lstm", s)].y_true)).to_numpy(float)
        ci_s = block_resample_ci(d_s, cfg["resample"]["block_days"],
                                 cfg["resample"]["n_draws"], cfg["resample"]["seed"] + s)
        per_seed_ci.append({"seed": s, **ci_s})
    per_seed_ci.append({"seed": "five_seed_mean", **ci})
    pd.DataFrame(per_seed_ci).to_csv(out_cmp / "resample_ci.csv", index=False,
                                     encoding="utf-8-sig")

    # ---------- 筛选规则判定 ----------
    rule = cfg["selection_rule"]
    improved = int(summary.gru_improved.sum())
    worst_year = float(year_df.gru_vs_lstm_pct.max())
    rule1 = mean_gru <= mean_lstm * (1 - rule["mae_improve_min_frac"])
    rule2 = improved >= rule["min_improved_seeds"]
    rule3 = worst_year <= rule["max_year_regression_frac"] * 100
    if rule1 and rule2 and rule3:
        decision = ("GRU 可作为后续验证候选（仅此表述；不得改称正式主模型，"
                    "仍需冻结模型与阈值后的独立前瞻验证）")
    else:
        decision = "证据不足，保留 CNN-LSTM，并如实报告本轮差异（回顾性比较）"

    # ---------- 步骤5：LOF ----------
    targets = load_catalog(cfg)
    lof_rows = []
    for m in MODELS:
        for s in SEEDS:
            scores, k = score_residuals(preds[(m, s)], cfg)
            scores.to_csv(out_lof / f"scores_{m}_s{s}.csv", index=False,
                          encoding="utf-8-sig")
            events, thr, flagged = events_at_quantile(scores, cfg["display_quantile"],
                                                      cfg["lof_window"])
            events.to_csv(out_lof / f"events_{m}_s{s}.csv", index=False,
                          encoding="utf-8-sig")
            sens = []
            for q in cfg["quantiles"]:
                ev_q, thr_q, cnt_q = events_at_quantile(scores, q, cfg["lof_window"])
                sens.append({"model": m, "seed": s, "k": k, "quantile": q,
                             "threshold": thr_q, "n_flagged_windows": cnt_q,
                             "n_events": len(ev_q),
                             **alarm_metrics(ev_q, scores, cfg, targets)})
            sens_df = pd.DataFrame(sens)
            sens_df.to_csv(out_lof / f"sensitivity_{m}_s{s}.csv", index=False,
                           encoding="utf-8-sig")
            key_table = key_earthquake_table(events, scores.date, cfg)
            key_table.to_csv(out_lof / f"eq_verification_{m}_s{s}.csv", index=False,
                             encoding="utf-8-sig")
            weather_context(events).to_csv(out_lof / f"weather_context_{m}_s{s}.csv",
                                           index=False, encoding="utf-8-sig")
            qiaojia = key_table.loc[key_table.date == "2026-01-19"].iloc[0]
            disp = sens_df.loc[sens_df["quantile"] == cfg["display_quantile"]].iloc[0]
            lof_rows.append({
                "model": m, "seed": s, "k": k, "n_score_dates": len(scores),
                "n_flagged_windows": int(disp.n_flagged_windows),
                "n_events": int(disp.n_events),
                "alarm_fraction": disp.alarm_fraction,
                "qiaojia_detection": qiaojia.detection,
                "qiaojia_n_events": qiaojia.n_events,
            })
    lof_df = pd.DataFrame(lof_rows)
    lof_df.to_csv(out_lof / "lof_summary.csv", index=False, encoding="utf-8-sig")

    # 同种子两模型事件重合
    overlap_rows = []
    for s in SEEDS:
        el = pd.read_csv(out_lof / f"events_lstm_s{s}.csv",
                         parse_dates=["start", "end", "alarm_date"])
        eg = pd.read_csv(out_lof / f"events_gru_s{s}.csv",
                         parse_dates=["start", "end", "alarm_date"])

        def hits(a: pd.DataFrame, b: pd.DataFrame) -> int:
            if not len(a) or not len(b):
                return 0
            return int(sum(((b.start <= ea.end) & (b.end >= ea.start)).any()
                           for ea in a.itertuples()))

        overlap_rows.append({"seed": s, "n_events_lstm": len(el), "n_events_gru": len(eg),
                             "lstm_events_overlapped_by_gru": hits(el, eg),
                             "gru_events_overlapped_by_lstm": hits(eg, el)})
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_cmp / "event_overlap.csv", index=False, encoding="utf-8-sig")

    # ---------- 图件 ----------
    stats_gru = fig_ppt(preds[("gru", 42)], 2023, "gru",
                        out_fig / "PPT_预测vs观测_2023_CNN-GRU.png")
    stats_lstm = fig_ppt(preds[("lstm", 42)], 2023, "lstm",
                         out_fig / "PPT_预测vs观测_2023_CNN-LSTM.png")
    combined = fig_ppt_combined(preds, 2023,
                                out_fig / "PPT_预测vs观测_2023_CNN-GRU_vs_v6.2.png")
    fig_ppt(preds[("gru", 42)], None, "gru",
            out_fig / "PPT_预测vs观测_全段_CNN-GRU.png")
    fig_ppt_combined(preds, None,
                     out_fig / "PPT_预测vs观测_全段_CNN-GRU_vs_v6.2.png")
    cmp23 = fig_compare(preds, 2023, 2023, out_fig / "对照_2023_CNN-LSTM_vs_CNN-GRU.png")
    cmp2526 = fig_compare(preds, 2025, 2026, out_fig / "对照_2025-2026_CNN-LSTM_vs_CNN-GRU.png")
    fig_summary(summary, year_df, out_fig / "汇总_五种子与逐年.png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("PPT_预测vs观测_2023_CNN-GRU.png", "PPT_预测vs观测_2023_CNN-LSTM.png",
                 "PPT_预测vs观测_2023_CNN-GRU_vs_v6.2.png",
                 "PPT_预测vs观测_全段_CNN-GRU.png",
                 "PPT_预测vs观测_全段_CNN-GRU_vs_v6.2.png"):
        shutil.copy2(out_fig / name, FIGURES_DIR / name)

    # ---------- report.md ----------
    params_lstm = mets[("lstm", 42)]["n_params"]
    params_gru = mets[("gru", 42)]["n_params"]
    secs_lstm = float(np.mean([mets[("lstm", s)]["train_seconds"] for s in SEEDS]))
    secs_gru = float(np.mean([mets[("gru", s)]["train_seconds"] for s in SEEDS]))
    mean_ci_txt = (f"{ci['point']:+.5f}（60日块重抽样1000次 95% 描述性区间 "
                   f"[{ci['ci_low']:+.5f}, {ci['ci_high']:+.5f}]）")
    lof_lstm = lof_df[lof_df.model == "lstm"]
    lof_gru = lof_df[lof_df.model == "gru"]
    qj_txt = ("CNN-LSTM："
              f"{ {k: int(v) for k, v in lof_lstm.qiaojia_detection.value_counts().items()} }；"
              "CNN-GRU："
              f"{ {k: int(v) for k, v in lof_gru.qiaojia_detection.value_counts().items()} }")

    report = f"""# CNN-GRU 受控替换实验报告

- 生成时间：{datetime.now().isoformat(timespec="seconds")}
- 工作区：`D:\\MYCODE\\dachuang_CNN_GRU`（旧仓库全程只读）
- 协议：`controlled_gru_v1`（configs/exp_controlled_gru.json；计划书 docs/CNN-GRU_实验计划书.md）
- 环境：Python 3.14.5，torch 2.13.0+cpu，numpy 2.5.1，pandas 3.0.3，scikit-learn 1.9.0，无 CUDA，28 核 CPU
- 单位：水氡单位未核实，本报告全部量值为**原数据单位**

## 1. 验收门槛通过情况

| 门槛 | 结果 |
|---|---|
| 步骤0 数据冻结：7 个文件 SHA-256 | **通过**（5 个基准哈希全 MATCH） |
| 步骤1 最小迁移：冒烟 + 独立运行 | **通过**（两模型输出 [batch] 有限；不依赖旧仓路径） |
| 步骤2 结构断言（run_checks.py 15 项） | **通过** |
| 步骤2 LSTM s42 复现 v6.2 | **通过**：dMAE=dRMSE=dR2=0.0，测试 date/y_true/y_pred 逐项一致 |
| 步骤3 10 次配对训练 | **通过**（10/10 完成，无失败种子；同种子测试日期与 y_true 逐项一致） |
| 协议锁定 | **已执行**：首个 GRU 测试预测前写入 protocol_locked.json |
| 旧仓库修改 | 无（全程只读） |

## 2. 主结果（测试 1451 日：2022-05-16 至 2026-05-05，原数据单位）

### 每种子配对结果

{df_to_md(summary, '{:.6f}')}

（mae_improve_frac = (LSTM−GRU)/LSTM，正值表示 GRU 更优）

### 五种子汇总

| 量 | CNN-LSTM | CNN-GRU |
|---|---|---|
| MAE 均值 | {mean_lstm:.6f} | {mean_gru:.6f} |
| MAE 范围 | [{summary.lstm_MAE.min():.6f}, {summary.lstm_MAE.max():.6f}] | [{summary.gru_MAE.min():.6f}, {summary.gru_MAE.max():.6f}] |
| RMSE 均值 | {summary.lstm_RMSE.mean():.6f} | {summary.gru_RMSE.mean():.6f} |
| R² 均值 | {summary.lstm_R2.mean():.6f} | {summary.gru_R2.mean():.6f} |
| 可训练参数量 | {params_lstm} | {params_gru} |
| 平均训练耗时 (s) | {secs_lstm:.1f} | {secs_gru:.1f} |
| 昨日值基线 MAE | {persistence['MAE']:.6f} | {persistence['MAE']:.6f} |

- 逐种子 GRU−LSTM 的 MAE 差五种子均值：{summary.mae_diff_gru_minus_lstm.mean():+.6f}；改善种子数 {improved}/5。
- 五种子平均逐日 MAE 差（GRU−LSTM）：{mean_ci_txt}。窗口高度重叠、测试段曾参与 v6.1 选型，该区间仅为描述性统计，不构成显著性检验。

## 3. 逐年表现（五种子平均）

{df_to_md(year_df, '{:.4f}')}

注意：v6.2 整体 R²=0.893 掩盖低方差年份问题（2025、2026 年 R² 为负）；上表单列逐年，两模型同口径对比。

## 4. 预先登记筛选规则判定

| 规则 | 阈值 | 实际 | 结果 |
|---|---|---|---|
| 五种子平均 MAE 降低 ≥5% | ≤ {mean_lstm * 0.95:.6f} | {mean_gru:.6f} | {'满足' if rule1 else '不满足'} |
| 改善种子数 ≥4/5 | ≥4 | {improved} | {'满足' if rule2 else '不满足'} |
| 任一测试年不劣于 LSTM 5% | 最差年 ≤ +5% | {worst_year:+.2f}% | {'满足' if rule3 else '不满足'} |

**结论：{decision}**

## 5. 残差 LOF（同口径：7 日窗、k=25、30 窗口留出块、embargo=6、间隔≤3 日聚合）

- 得分日期 {len(scores)} 个（2022-05-22 至 2026-04-29），两模型同种子得分日期完全一致。
- q=0.98 仅为各模型自身分数的历史展示阈值，标记窗数按构造接近，**不能解释为异常能力相等**。
- 井位经纬度缺失 → 地震命中/误报一律"不可评估"，不填零。

每运行汇总：

{df_to_md(lof_df, '{:.4f}')}

- 事件重合（同种子，区间有交集记一次）：

{df_to_md(overlap_df, '{:.0f}')}

- 参考震例巧家 Ms5.1（2026-01-19，震前 90 日窗口完整）检出情况：{qj_txt}
- 完整产物：lof/scores_*.csv、events_*.csv、sensitivity_*.csv（含 7 档分位数与警报覆盖率）、eq_verification_*.csv、weather_context_*.csv（仅气象背景，未排除仪器/人为干扰）。

## 6. 图件

| 图 | 内容 |
|---|---|
| figures/PPT_预测vs观测_2023_CNN-GRU.png | PPT 风格 2023 年 GRU 预测 vs 观测（R²={stats_gru['R2']:.3f}） |
| figures/PPT_预测vs观测_2023_CNN-LSTM.png | 同款 LSTM 图（R²={stats_lstm['R2']:.3f}） |
| figures/PPT_预测vs观测_2023_CNN-GRU_vs_v6.2.png | **GRU 与 v6.2 CNN-LSTM 同图上下对照**（2023，各含残差面板，共用时间轴） |
| figures/PPT_预测vs观测_全段_CNN-GRU_vs_v6.2.png | **同图对照 · 测试全段**（2022-05-16~2026-05-05，1451 日） |
| figures/PPT_预测vs观测_全段_CNN-GRU.png | GRU 测试全段单图 |
| figures/对照_2023_CNN-LSTM_vs_CNN-GRU.png | 2023 年两模型共轴对照（MAE {cmp23['mae_lstm']:.3f} vs {cmp23['mae_gru']:.3f}） |
| figures/对照_2025-2026_CNN-LSTM_vs_CNN-GRU.png | 2025–2026 两模型共轴对照（MAE {cmp2526['mae_lstm']:.3f} vs {cmp2526['mae_gru']:.3f}） |
| figures/汇总_五种子与逐年.png | 五种子配对 MAE + 逐年对比 |

以上均同时存于 results/controlled_gru/figures/（PPT 两张另复制到工作区 figures/）。图不代替误差表。

## 7. 结论边界与限制

1. 测试段（2022-05-16 起）曾参与 v6.1 配方选择，本轮结果为**回顾性探索**，不是独立前瞻验证。
2. 水氡单位未核实（原数据单位）；西昌 32 井水位、井位、仪器维护记录缺失，双流体互证与空间对应**不可评估**。
3. LOF 自参照得分使用完整测试段，不构成可实时运行的预警系统；未来数据应在模型与阈值冻结后逐日评估。
4. 本轮结论仅适用于 v6.2 固定训练协议；不推广到所有 GRU 配置，也不构成对申请书 CNN-LSTM/MATLAB 方案的变更。
5. 巧家等震例"候选异常"仅为残差统计重合，**不构成前兆证据**（未排除气象/仪器/人为干扰，无异常真值）。

## 8. 复算方式

```bash
# 解释器：D:/桌面/地质/大创/dataAnalysis/.venv/Scripts/python.exe
python run_freeze.py                                  # 步骤0 哈希+环境
python run_checks.py                                  # 步骤2 结构断言
python run_train.py --model lstm --seed 42            # 复现锚点（与 v6.2 逐位一致）
python run_train.py --model lstm|gru --seed 42..46    # 10 次配对训练
python run_report.py                                  # 本报告全部表格与图
python run_verify.py                                  # 训后验收
```

## 9. 失败与缺失

- 训练失败种子：无（10/10 成功）。
- 数据/哈希异常：无。
- 缺失待人工确认：水氡单位来源；西昌水位与井位；巧家目标台站双流体时序；申请书表述差异处理（见计划书第 8 节）。
"""
    (BASE / "report.md").write_text(report, encoding="utf-8")

    # ---------- manifest 回写 ----------
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_locked"] = True
    manifest["protocol_lock_file"] = "results/controlled_gru/protocol_locked.json"
    manifest["runs"] = [
        {"model": m, "seed": s, "dir": f"{m}_s{s}",
         "n_params": mets[(m, s)]["n_params"], "epochs_run": mets[(m, s)]["epochs_run"],
         "best_epoch": mets[(m, s)]["best_epoch"],
         "train_seconds": mets[(m, s)]["train_seconds"],
         "weights_sha256": mets[(m, s)]["weights_sha256"]}
        for m in MODELS for s in SEEDS]
    manifest["final"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report": "results/controlled_gru/report.md",
        "decision": decision,
        "figures": sorted(p.name for p in out_fig.glob("*.png")),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"summary: LSTM MAE={mean_lstm:.6f} | GRU MAE={mean_gru:.6f} | "
          f"improved {improved}/5 | worst year {worst_year:+.2f}%")
    print(f"decision: {decision}")
    print(f"report -> {BASE / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
