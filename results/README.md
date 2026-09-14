# results/ — 实验结果

每次实验一个子目录，内含 `metrics.json`（指标）与预测/事件等文件。  
**禁止手改**：全部由 `src/` + `run_*.py` 生成。

> **默认不进入 git**（见根目录 `.gitignore`）。新克隆仓库时本目录可能为空或不存在；Web 的预测/异常/震例页依赖下列 v6.1 目录。

---

## 对外协作最小集（Web + 震例）

交接给工程师时，至少提供：

```text
results/
├── v61_anomaly/                 # LOF 事件、震例验证、labels_*
├── v61_compare/                 # 多模型 metrics + predictions
├── v61_cnn_lstm/                # CNN-LSTM / tuned 预测
├── v61_optimize/                # 深度优化（若存在，如 E_cosine）
├── multi_var_comparison.json    # 多变量汇总
└── window_comparison.json       # 多窗口汇总（可选）
```

其余 `phase*`、`radon/`、`level/` 等为 v6.0 对照，可选。

---

## 主要子目录

| 路径 | 含义 |
|------|------|
| `pipeline/` | 阶段1 数据管道日志 |
| `smoke/` | 冒烟测试产物（合成数据） |
| `phase2_baselines/` | 阶段2 基线复现 |
| `phase3_cnn_lstm/` | 阶段3 CNN-LSTM |
| `phase4_anomaly/` | 阶段4 残差 LOF + 震例对应 |
| `v61_compare/` | v6.1 多模型对照 |
| `v61_anomaly/` | v6.1 异常事件 / 震例表 / 自动标签 |
| `v61_cnn_lstm/` | v6.1 CNN-LSTM 预测 |
| `radon/`、`level/`… | 多变量 × 多窗口实验 |
| `window_*/` | 单变量多窗口 |

---

## 文件格式（约定）

### `metrics.json`

```json
{
  "RMSE": 0.1715,
  "MAE": 0.13,
  "R2": 0.67
}
```

字段名以各脚本实际输出为准；v6.1 预测页会结合 `predictions_*.csv` 做反变换与按年 R²。

### `predictions_test.csv` / `predictions_*.csv`

通常含 `date`、`y_true`、`y_pred`（或等价列）。尺度可能是标准化后的，Web 侧会按配置反变换到原始单位（Bq/L 等）。

### `events.csv`

LOF 异常事件：起止时间、峰值得分、点数等（列以生成脚本为准）。

---

## 如何再生

```bash
# 在已配置 configs/paths.json、且 data/clean 或 OriData 可用的前提下
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_v61_optimize.py
python run_v61_compare.py
python run_eq_verify.py
```

或从项目方提供的 **结果 zip** 解压到本目录。

---

## 纪律

1. 结果只增不改；对比实验用新子目录，勿覆盖旧 `metrics.json` 写论文引用。  
2. 未落盘的数字不写入报告。  
3. 分年指标优先于单一整体 R²（见 `notes/v6.1_总结.md`）。  
