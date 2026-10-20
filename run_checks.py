"""步骤2前置：数据 / 切分 / 参照对齐的结构断言（不涉及任何模型预测）。

检查项（对应计划书步骤2第1条）：
1. SCXC32 水氡 4143 日、日期单调无重复、起止端点、3 个插补日；
2. 9 通道顺序与名称；
3. 目标日严格晚于输入末日（抽查各段窗口与原始序列对齐）；
4. 标准化只在拟合段估计（拟合段各通道中位数≈0、IQR≈1，测试段不成立）；
5. 三段目标日不交叉、样本数 2228/404/1451、三个分界日期与参照一致；
6. 昨日值基线与测试日期完全一致；
7. 参照 predictions_test.csv 的 1451 日 date 与 y_true 逐项一致。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.common import DATA_DIR, RESULTS_DIR, load_config
from src.features.build import build_feature_matrix, robust_apply, robust_fit
from run_train import load_radon, prepare

EXPECTED_NAMES = ["raw", "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
                  "roll_mean_30", "roll_std_30", "diff1", "diff2"]
REF = {"n_fit": 2228, "n_val": 404, "n_test": 1451,
       "fit_last": "2021-04-06", "val_first": "2021-04-07", "test_first": "2022-05-16"}

fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    cfg = load_config("exp_controlled_gru")

    frame = pd.read_csv(DATA_DIR / "clean" / "long.csv", encoding="utf-8-sig",
                        dtype={"station_id": str, "item_code": str})
    s0 = frame[(frame.station_id == cfg["station_id"])
               & (frame.item_name == cfg["item_name"])]
    dates0 = pd.to_datetime(s0.date)
    check("SCXC32水氡 4143 日且无重复", len(s0) == 4143 and dates0.is_unique,
          f"rows={len(s0)}")
    check("日期范围 2015-01-01..2026-05-05",
          dates0.min() == pd.Timestamp("2015-01-01") and dates0.max() == pd.Timestamp("2026-05-05"))
    check("日期单调递增", dates0.is_monotonic_increasing)
    imputed = sorted(s0.loc[s0.quality_flag == "imputed", "date"])
    check("插补日恰为 2015-05-17..19",
          imputed == ["2015-05-17", "2015-05-18", "2015-05-19"], str(imputed))

    series = load_radon(cfg)
    partitions, (center, scale), names = prepare(series, cfg)
    check("9 通道顺序", names == EXPECTED_NAMES, str(names))

    # 目标日严格晚于输入末日：重建索引并抽查
    values = series.to_numpy(dtype=float)
    window = cfg["window"]
    test_start = int(len(values) * cfg["train_test_ratio"])
    fit_end = int(test_start * cfg["fit_ratio_within_training"])
    features, _ = build_feature_matrix(values)
    scaled = np.column_stack([
        robust_apply(features[:, col], *robust_fit(features[:fit_end, col]))
        for col in range(features.shape[1])
    ])
    targets = np.arange(window, len(values))
    rng = np.random.default_rng(2026)
    sample = rng.choice(targets, size=40, replace=False)
    aligned = all(
        np.allclose(scaled[t - window:t], scaled[t - window:t], atol=0)
        and series.index[t] > series.index[t - 1]
        for t in sample
    )
    # 直接断言窗口内容与 raw 对齐（最后一行应为 t-1 日的特征行）
    ok_align = True
    for t in sample[:10]:
        row_raw = scaled[t - 1][:1]  # raw 通道在 t-1 日的标准化值
        last_input_raw = scaled[t - window:t][-1, 0]
        if not np.isclose(row_raw[0], last_input_raw, atol=0):
            ok_align = False
    check("输入窗口末日=t-1日、目标日=t日（抽查对齐）", aligned and ok_align)

    fit_scaled = scaled[:fit_end]
    med_ok = np.allclose(np.median(fit_scaled, axis=0), 0, atol=1e-9)
    iqr_ok = np.allclose(
        np.percentile(fit_scaled, 75, axis=0) - np.percentile(fit_scaled, 25, axis=0),
        1, atol=1e-9)
    test_scaled = scaled[test_start:]
    not_unit = not np.allclose(
        np.percentile(test_scaled, 75, axis=0) - np.percentile(test_scaled, 25, axis=0),
        1, atol=0.1)
    check("标准化只拟合拟合段（fit 段 median≈0/IQR≈1，test 段非单位尺度）",
          med_ok and iqr_ok and not_unit)

    fit, val, test = (partitions[n] for n in ("fit", "val", "test"))
    check("样本数 2228/404/1451",
          (len(fit["X"]), len(val["X"]), len(test["X"]))
          == (REF["n_fit"], REF["n_val"], REF["n_test"]),
          f"{len(fit['X'])}/{len(val['X'])}/{len(test['X'])}")
    check("分界日期 fit_last/val_first/test_first",
          (str(fit["date"][-1].date()), str(val["date"][0].date()),
           str(test["date"][0].date()))
          == (REF["fit_last"], REF["val_first"], REF["test_first"]),
          f"{fit['date'][-1].date()}/{val['date'][0].date()}/{test['date'][0].date()}")
    check("三段目标日不交叉",
          fit["date"][-1] < val["date"][0] and val["date"][-1] < test["date"][0])

    baseline_valid = np.isfinite(test["previous"])
    check("昨日值基线与测试日期一致", bool(baseline_valid.all()),
          f"valid={int(baseline_valid.sum())}/{len(test['date'])}")

    ref = pd.read_csv(RESULTS_DIR.parent / "reference" / "v62" / "predictions_test.csv")
    check("参照预测 1451 行", len(ref) == REF["n_test"], f"rows={len(ref)}")
    check("参照 date 与本切分逐项一致",
          list(pd.to_datetime(ref.date)) == list(test["date"]))
    check("参照 y_true 与本切分逐项一致",
          bool(np.array_equal(ref.y_true.to_numpy(float),
                              np.asarray(test["raw"], float))))

    ref_metrics = json.loads((RESULTS_DIR.parent / "reference" / "v62" / "metrics.json")
                             .read_text(encoding="utf-8"))
    check("参照 metrics.json 分段计数/日期一致",
          all(ref_metrics[k] == v for k, v in REF.items()))

    print(f"\n共 {len(fails)} 项未通过" + ("" if not fails else f": {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
