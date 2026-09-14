# dachuang_cnn_lstm

大创项目「基于 CNN-LSTM 的巧家地震震前水氡–水位双流体联合建模」。

> 状态：阶段 0–6 + 多变量 + Web ✅ → **v6.1（65/35 切分 + 深度优化 + 震例验证）✅**

## 新协作者 / 工程师请先读

| 文档 | 用途 |
|------|------|
| **[docs/交接说明.md](docs/交接说明.md)** | **上手：环境、路径、数据包、验收命令、故障排查** |
| [AGENTS.md](AGENTS.md) | 数据格式陷阱、已定技术决策（勿回退） |
| [data/README.md](data/README.md) | `data/clean/` 字段与统计 |
| [web/README.md](web/README.md) | Web 启动与页面/API |

**仓库里有什么、缺什么（重要）**

| 内容 | 在 git？ | 说明 |
|------|----------|------|
| 代码 / 文档 / 笔记 | 是 | `src/`、`run_*.py`、`web/`、`docs/`、`notes/` |
| 清洗长表 `data/clean/` | 是 | 主分析表；**禁止手改** |
| 原始数据 `OriData*` | **否** | 在 `data_root`，需单独打包；只读 |
| 实验结果 `results/` | **否** | 可脚本再生或从结果包拷入；Web 完整页依赖它 |
| 模型权重 `*.pt` 等 | **否** | 需本地重训 |

完整说明见 [docs/交接说明.md](docs/交接说明.md)。

## 目录

```
dachuang_cnn_lstm/
├── README.md                 ← 本文件
├── AGENTS.md                 ← 协作者纪律与数据陷阱
├── requirements.txt          ← 依赖版本
├── smoke_test.py             ← 合成数据冒烟（不依赖原始数据）
├── run_phase2.py …           ← v6.0 阶段脚本
├── run_v61_*.py              ← v6.1 优化 / 对比 / 震例 / 标记
├── configs/
│   ├── paths.example.json    ← 路径模板（新机器复制后改）
│   ├── paths.json            ← 本机路径配置
│   ├── exp_v61.json
│   └── exp_v61_best.json     ← v6.1 最优管线 E_cosine
├── src/                      ← 数据管道 / 特征 / 模型 / LOF / 画图
├── data/clean/               ← 清洗长表（程序生成）
├── results/                  ← 实验产物（默认不进 git）
├── web/                      ← FastAPI + ECharts
├── docs/                     ← 数据/技术方案 + 交接说明
├── notes/                    ← 阶段笔记与 v6.1 总结
├── papers/                   ← 文献 PDF + 精读
└── proposal/                 ← 大创申请书
```

## 快速开始

### 1. 环境

```bash
# Python 3.11+（本项目在 3.14.5 上验证；约定不用 pyarrow/pyyaml）
python -m pip install -r requirements.txt
```

Windows 本机若使用独立 venv，可改用该解释器执行下文命令。

### 2. 配置数据路径（新机器必做）

```bash
copy configs\paths.example.json configs\paths.json
# 编辑 paths.json：data_root、project_root 改成你的绝对路径
# data_root 下应有 OriData、OriData2、OriData3
```

若只跑冒烟测试或只看 `data/clean/`，可暂不配 `data_root`。

### 3. 验收命令

```bash
# 合成数据端到端（推荐先跑）
python smoke_test.py

# 已有 data/clean/ 时：读主表
python -c "import pandas as pd; print(len(pd.read_csv('data/clean/long.csv')))"

# 需要原始数据与 clean 时：重生成长表
python src/data/run_pipeline.py

# Web（完整图表需要 results/ 结果目录）
python web/app.py
# http://127.0.0.1:8002/
```

### 4. 常用实验脚本

```bash
# v6.0：西昌32井水氡基线 → CNN-LSTM → LOF
python run_phase2.py
python run_phase3.py
python run_phase4.py

# v6.1：65/35 + E_cosine + 震例（当前主口径）
python run_v61_optimize.py
python run_v61_compare.py
python run_eq_verify.py
python run_v61_label.py
```

## Web 可视化平台

| 页面 | 路径 | 主要数据源 |
|------|------|------------|
| 数据概览 | `/` | `data/clean/long.csv` |
| 模型预测 | `/prediction` | `results/v61_*`（可切 v6.0 多变量） |
| 异常检测 | `/anomaly` | `results/v61_anomaly/` |
| 震例验证 | `/earthquake` | `results/v61_anomaly/` + 地震目录 |

启动细节、端口、常见问题见 [web/README.md](web/README.md)。

## 数据路径

- 配置：`configs/paths.json`（模板：`configs/paths.example.json`）
- **external**（相对 `data_root`，只读）：
  - `OriData/`：云南台站流体日值
  - `OriData2/`：整理后 xlsx（含西昌32井长序列）
  - `OriData3/`：西昌气象三要素 + 地震目录
- **internal**（相对本仓库根）：`data/clean/`、`results/`、`figures/`

代码禁止硬编码绝对路径；一律经 `load_paths()`。

## 当前进度

| 阶段 | 状态 | 产出 |
|------|------|------|
| 0 环境搭建 | ✅ | requirements + 目录骨架 + 种子复现验证 |
| 1 数据管道 | ✅ | data/clean/long.csv（66,034 行级长表） |
| 2 基线复现 | ✅ | RF < LSTM < LSTM-RF，排序与文献一致 |
| 3 CNN-LSTM | ✅ | CNN-LSTM > LSTM-RF，卷积带来增益 |
| 4 异常检测 | ✅ | 残差 LOF + 干扰排除 + 震例对应 |
| 5 多变量实验 | ✅ | 水氡单变量常最优；盲目加特征可能降 R² |
| 6 Web 可视化 | ✅ | http://127.0.0.1:8002 |
| 7 v6.1 优化+震例 | ✅ | 65/35；E_cosine；LOF 事件与后续地震；关键大震可阴性 |

v6.1 结论摘要见 [notes/v6.1_总结.md](notes/v6.1_总结.md)。

## 三条纪律

1. **原始数据不可变**：`OriData*` 只读  
2. **先跑通再调优**：新模块先在西昌32井单变量上端到端验证  
3. **结果必须落盘**：没有写进 `results/` 的实验等于没做过  

另：时序禁止随机打乱划分；异常检测保持无监督（预测 → 残差 → LOF）。

## 打包给外部协作者（项目方）

```bash
# 原始数据包（通常约几 MB）
powershell -File scripts/package_handoff.ps1 -IncludeData
# 数据 + results 结果包
powershell -File scripts/package_handoff.ps1 -IncludeData -IncludeResults
```

产物落在 `handoff/` 目录（已 gitignore）。发送时附上本 README 与 `docs/交接说明.md`。
