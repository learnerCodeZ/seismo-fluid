"""训后验收：产物完整性、口径一致、指标复算、LOF 产物、图件与 manifest 全项检查。

全部通过返回 0；任一失败返回 1 并列出失败项。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, RESULTS_DIR, load_config
from src.models.evaluate import all_metrics
from run_freeze import EXPECTED_SHA, sha256_file

BASE = RESULTS_DIR / "controlled_gru"
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["lstm", "gru"]
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    cfg = load_config("exp_controlled_gru")

    # 1 数据冻结未变
    for rel, expected in EXPECTED_SHA.items():
        check(f"数据未变 {rel}",
              sha256_file(PROJECT_ROOT / rel).upper() == expected.upper())

    # 2 10 次运行产物完整
    runs = {}
    for m in MODELS:
        for s in SEEDS:
            d = BASE / f"{m}_s{s}"
            complete = all((d / f).exists()
                           for f in ("metrics.json", "predictions_test.csv",
                                     "history.csv", "weights.pt"))
            check(f"运行完整 {m}_s{s}", complete)
            runs[(m, s)] = (json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                            if complete else None)
    if any(v is None for v in runs.values()):
        print(f"\n{len(fails)} 项未通过；存在缺失运行，终止后续检查")
        return 1

    # 3 预测对齐：1451 行、日期与 y_true 与参照一致、预测有限
    ref = pd.read_csv(PROJECT_ROOT / "reference" / "v62" / "predictions_test.csv")
    ref_dates = ref.date.tolist()
    ref_y = ref.y_true.to_numpy(float)
    dfs = {(m, s): pd.read_csv(BASE / f"{m}_s{s}" / "predictions_test.csv")
           for m in MODELS for s in SEEDS}
    for (m, s), df in dfs.items():
        ok = (len(df) == 1451 and df.date.tolist() == ref_dates
              and np.array_equal(df.y_true.to_numpy(float), ref_y)
              and bool(np.isfinite(df.y_pred).all()))
        check(f"预测对齐 {m}_s{s}", ok, f"rows={len(df)}")

    # 4 指标可从逐日 CSV 复算（CSV 沿用参照格式，约 7 位有效数字；全精度以 metrics.json
    #    为准。复算容差 1e-6，仅为存储往返损耗，远小于复现门槛 0.005）
    for (m, s), df in dfs.items():
        stored = runs[(m, s)][f"cnn_{m}"]
        recomputed = all_metrics(df.y_true.to_numpy(float), df.y_pred.to_numpy(float))
        ok = all(abs(recomputed[k] - stored[k]) <= 1e-6 for k in ("MAE", "RMSE", "R2"))
        check(f"指标复算 {m}_s{s}", ok,
              f"dMAE={abs(recomputed['MAE'] - stored['MAE']):.2e}")

    # 5 步骤2 复现门槛
    ref_m = json.loads((PROJECT_ROOT / "reference" / "v62" / "metrics.json")
                       .read_text(encoding="utf-8"))["cnn_lstm"]
    new_m = runs[("lstm", 42)]["cnn_lstm"]
    gate = (abs(new_m["MAE"] - ref_m["MAE"]) <= cfg["reference_gate"]["mae_rmse_abs_tol"]
            and abs(new_m["RMSE"] - ref_m["RMSE"]) <= cfg["reference_gate"]["mae_rmse_abs_tol"]
            and abs(new_m["R2"] - ref_m["R2"]) <= cfg["reference_gate"]["r2_abs_tol"])
    check("步骤2 复现门槛（MAE/RMSE≤0.005, R²≤0.01）", gate,
          f"dMAE={new_m['MAE'] - ref_m['MAE']:.2e} dR2={new_m['R2'] - ref_m['R2']:.2e}")

    # 6 昨日值基线各运行一致
    p0 = runs[("lstm", 42)]["persistence"]["MAE"]
    check("昨日值基线 10 次运行一致",
          all(abs(r["persistence"]["MAE"] - p0) < 1e-12 for r in runs.values()))

    # 7 训练曲线与记录一致
    for (m, s) in runs:
        h = pd.read_csv(BASE / f"{m}_s{s}" / "history.csv")
        met = runs[(m, s)]
        best_row = h.loc[h.val_loss.idxmin()]
        ok = (len(h) == met["epochs_run"] and bool(np.isfinite(h.val_loss).all())
              and int(best_row.epoch) == met["best_epoch"]
              and abs(float(best_row.val_loss) - met["best_val_loss"]) <= 1e-9)
        check(f"训练曲线 {m}_s{s}", ok, f"epochs={len(h)} best={int(best_row.epoch)}")

    # 8 权重哈希
    for (m, s) in runs:
        ok = sha256_file(BASE / f"{m}_s{s}" / "weights.pt") == runs[(m, s)]["weights_sha256"]
        check(f"权重哈希 {m}_s{s}", ok)

    # 9 LOF 产物
    lof_dir = BASE / "lof"
    for m in MODELS:
        for s in SEEDS:
            paths = [lof_dir / f"scores_{m}_s{s}.csv", lof_dir / f"events_{m}_s{s}.csv",
                     lof_dir / f"sensitivity_{m}_s{s}.csv",
                     lof_dir / f"eq_verification_{m}_s{s}.csv",
                     lof_dir / f"weather_context_{m}_s{s}.csv"]
            ok = all(p.exists() for p in paths)
            if ok:
                sdf = pd.read_csv(paths[0])
                sn = pd.read_csv(paths[2])
                ok = (len(sdf) == 1445 and sdf.date.tolist() == ref_dates[6:]
                      and (sn.k == 25).all())
            check(f"LOF 产物 {m}_s{s}", ok, "rows=1445, k=25" if ok else "")

    # 10 汇总表可复算
    summ = pd.read_csv(BASE / "comparison" / "summary_per_seed.csv")
    ok = all(abs(summ.loc[summ.seed == s, f"{m}_MAE"].iloc[0]
                 - runs[(m, s)][f"cnn_{m}"]["MAE"]) < 1e-12
             for m in MODELS for s in SEEDS)
    check("汇总表可从 metrics 复算", ok)

    # 11 图件
    figs = ["PPT_预测vs观测_2023_CNN-GRU.png", "PPT_预测vs观测_2023_CNN-LSTM.png",
            "PPT_预测vs观测_2023_CNN-GRU_vs_v6.2.png",
            "PPT_预测vs观测_全段_CNN-GRU.png", "PPT_预测vs观测_全段_CNN-GRU_vs_v6.2.png",
            "对照_2023_CNN-LSTM_vs_CNN-GRU.png", "对照_2025-2026_CNN-LSTM_vs_CNN-GRU.png",
            "汇总_五种子与逐年.png"]
    for f in figs:
        p = BASE / "figures" / f
        check(f"图件 {f}", p.exists() and p.stat().st_size > 30_000,
              f"{p.stat().st_size // 1024}KB" if p.exists() else "缺失")

    # 12 协议锁定与 manifest
    check("协议锁定文件存在", (BASE / "protocol_locked.json").exists())
    mf = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    check("manifest 完整（锁定 + 10 runs + 无阻塞项）",
          mf.get("protocol_locked") is True and len(mf.get("runs", [])) == 10
          and not mf.get("blocking_issues"))
    check("报告存在", (BASE / "report.md").exists())

    print(f"\n共 {len(fails)} 项未通过" + ("" if not fails else f": {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
