# 西昌32井 CNN-GRU 受控对照

本分支保存西昌32井水氡的 CNN-GRU/CNN-LSTM 回顾性对照代码与说明。数值预测与地震相关异常判断是两个不同任务；现有结果**不能证明可以预测地震**。

## 内容

- `run_freeze.py`、`run_checks.py`、`run_train.py`、`run_report.py`、`run_verify.py`：原受控实验入口。
- `src/`、`configs/`：模型与实验配置。
- `data/clean/`、`data/context/`、`reference/v62/`：本地复核所需快照，**不进入 Git**；原始 `OriData*` 也不上传。
- `results/`、`figures/`：生成产物，不进入 Git；`local/`：本地计划与工作资料，明确不推送。

## 复核

先由项目组按权限取得并核验 `data/clean/`、`data/context/` 与 `reference/v62/` 的冻结快照；**仅克隆代码分支不能直接复跑实验**。在满足 `requirements.txt` 的环境中，先运行 `python run_freeze.py` 与 `python run_checks.py`，再按 `docs/CNN-GRU_实验计划书.md` 执行训练和核查。脚本在 `results/controlled_gru/` 写产物，已有结果时不会覆盖。原数据单位尚未核实，报告不应写成 Bq/L。

当前分支是**优化前基线快照**。后续双任务优化在本地继续，经过独立验证前不推送优化成果。
