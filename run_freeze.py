"""步骤0：盘点并冻结输入 → results/controlled_gru/manifest.json。

复核计划书第 2 节 7 个文件的 SHA-256（5 个有基准值，2 个参照 JSON 记录实测值），
记录环境版本与命令入口。哈希不匹配时写入 blocking_issues，由执行者停下反馈。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime

from src.common import PROJECT_ROOT, RESULTS_DIR, load_config

EXPECTED_SHA = {
    "data/clean/long.csv":
        "ED04D36EB64307CB92BEDF0AB0F1CC86511EFAA679DB4064CA235AAEB5A678BD",
    "data/clean/dataset_info.csv":
        "60A571762B9B1F800A55021EC9F813081C1972C8B73DBBDF391156A5A9B578F9",
    "data/context/西昌.csv":
        "837BC79562A8F848688B20B30EA75F79F7693BE411FD1BBA05619AADCCA05B69",
    "data/context/earthquake_catalog.csv":
        "3C0509FF0318246C0F04E58B73CA71D7D7C4F041200C4C3EE9C53C83AC9F782A",
    "reference/v62/predictions_test.csv":
        "F66997630D09D60E0CDCFF00CF85E0498A3636A802CF54C79576569BD8AD84D4",
}
RECORD_ONLY = ("reference/v62/exp_v62.json", "reference/v62/metrics.json")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = load_config("exp_controlled_gru")
    files, mismatches = {}, []
    for rel, expected in EXPECTED_SHA.items():
        actual = sha256_file(PROJECT_ROOT / rel)
        ok = actual.upper() == expected.upper()
        files[rel] = {"sha256": actual, "expected": expected, "match": ok}
        if not ok:
            mismatches.append(rel)
    for rel in RECORD_ONLY:
        files[rel] = {"sha256": sha256_file(PROJECT_ROOT / rel),
                      "expected": None, "match": None}

    import numpy
    import pandas
    import sklearn
    import torch
    try:
        import matplotlib
        mpl = matplotlib.__version__
    except ImportError:
        mpl = None
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "venv_python": sys.executable,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": mpl,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    out_dir = RESULTS_DIR / "controlled_gru"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(PROJECT_ROOT),
        "protocol_version": cfg["version"],
        "files": files,
        "environment": environment,
        "commands": [
            "python run_freeze.py",
            "python run_checks.py",
            "python run_train.py --model lstm --seed 42   # 步骤2复现锚点",
            "python run_train.py --model lstm|gru --seed 42..46   # 步骤3配对训练",
            "python run_report.py",
            "python run_verify.py",
        ],
        "unit_status": "水氡单位未核实，全部量值标注'原数据单位'",
        "protocol_locked": False,
        "runs": [],
        "blocking_issues": [f"哈希不匹配: {rel}" for rel in mismatches],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"manifest -> {out_dir / 'manifest.json'}")
    for rel, info in files.items():
        status = "MATCH" if info["match"] else ("RECORD" if info["match"] is None else "MISMATCH")
        print(f"  [{status}] {rel}")
    print(f"environment: python={environment['python']} torch={environment['torch']} "
          f"numpy={environment['numpy']} pandas={environment['pandas']} "
          f"sklearn={environment['scikit-learn']} cuda={environment['cuda_available']} "
          f"cpus={environment['cpu_count']}")
    if mismatches:
        print(f"BLOCKING: {len(mismatches)} 个文件哈希不匹配，须停下反馈")
        return 1
    print("步骤0 验收：PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
