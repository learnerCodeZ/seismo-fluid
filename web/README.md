# Web 可视化平台

> **当前数据口径：v6.1**（主模型 E_cosine ★，65/35 切分，测试段自 2022-05-16）

基于 **FastAPI + ECharts** 的交互式界面，把 `data/` 与 `results/` 里的结果变成可点、可缩放的图表。

---

## 端口与页面地址

**固定端口：`8002`**（`localhost` 与 `127.0.0.1` 等价）

| 页面 | 地址 | 数据源 |
|------|------|--------|
| 数据概览 | **http://127.0.0.1:8002/** | `data/clean/long.csv` |
| 模型预测 | **http://127.0.0.1:8002/prediction** | `results/v61_compare`、`results/v61_optimize/E_cosine`；可切 v6.0 多变量 |
| 异常检测 | **http://127.0.0.1:8002/anomaly** | `results/v61_anomaly/` |
| 震例验证 | **http://127.0.0.1:8002/earthquake** | `results/v61_anomaly/` + 地震目录 |

健康检查：`http://127.0.0.1:8002/health`

---

## 如何启动

### 方式一：一键脚本（推荐）

双击：

```
<项目根>\web\start_web.bat
```

会自动检查依赖、启动服务，并打开震例验证页。

### 方式二：命令行（可移植，推荐在新机器使用）

```bash
# 在项目根目录
python -m pip install -r requirements.txt
python web/app.py
# 或
python -m uvicorn web.app:app --host 127.0.0.1 --port 8002
```

停止服务：在运行窗口按 `Ctrl+C`。

> `start_web.bat` 会优先使用项目 `.venv`，否则用 `PATH` 里的 `python`；本机若长期用外部 venv，可自行改 bat 或直接命令行启动。

---

## 页面说明

### 1. 数据概览 `/`
选择台站/测项，看水氡等时序曲线与缺测统计。

### 2. 模型预测 `/prediction`
- 默认 **v6.1 主结果**（65/35）：观测 + E_cosine ★，可勾选 Naive / RF / LSTM-RF / CNN-LSTM 对比
- 预测已**反变换到原始水氡 Bq/L**，避免 Z-score 与 Robust 尺度混画
- 滚轮缩放后线条自动加粗；工具栏可框选缩放/重置/存图
- 下方有整体指标表 + **按年 R²**（整体 R² 须结合分年阅读）
- 可切换 **v6.0 多变量对照**（80/20）

### 3. 异常检测 `/anomaly`
E_cosine 残差 LOF：得分时序、5 个异常事件、气象归因、后续 M4+/M5+、阈值敏感性。

### 4. 震例验证 `/earthquake`
六个关键地震对应表、LOF 事件→后续地震、残差时间轴。

---

## 常见问题

**Q: 端口 8002 被占用？**  
说明服务已在运行，直接打开上面地址即可；或结束占用进程后重启。

**Q: 页面数字还是旧的（如 R²=0.67）？**  
浏览器缓存。按 `Ctrl+F5` 强制刷新。

**Q: 启动报缺 fastapi / uvicorn？**  
```bash
python -m pip install fastapi uvicorn
# 或
python -m pip install -r requirements.txt
```

**Q: 想改端口？**  
改 `web/start_web.bat` 里的 `set PORT=8002`，以及启动命令中的 `--port`。

---

## 目录结构

```
web/
├── app.py                 # FastAPI 入口
├── start_web.bat          # 一键启动（端口 8002）
├── api/
│   ├── data.py            # 台站/时序
│   ├── models.py          # v6.1 预测与指标（反变换到 Bq/L）
│   ├── anomalies.py       # v6.1 LOF 事件
│   └── earthquakes.py     # 震例验证
└── static/
    ├── index.html         # 数据概览
    ├── prediction.html    # 模型预测
    ├── anomaly.html       # 异常检测
    ├── earthquake.html    # 震例验证
    ├── css/style.css
    └── js/
        ├── data.js
        ├── prediction.js
        ├── anomaly.js
        └── earthquake.js
```

### 主要 API

| 端点 | 说明 |
|------|------|
| `GET /api/stations` | 台站列表 |
| `GET /api/data/{id}` | 时序 |
| `GET /api/metrics` | v6.1 各模型指标 + 按年 R² |
| `GET /api/year_r2` | 按年 R² |
| `GET /api/predictions/all` | 全模型预测（Bq/L） |
| `GET /api/predictions/{var}/{window}` | v6.0 多变量对照 |
| `GET /api/anomalies/summary` | LOF 事件汇总 |
| `GET /api/anomalies/events` | 异常事件列表 |
| `GET /api/anomalies/scores` | LOF 得分序列 |
| `GET /api/earthquakes/verification` | 六地震对应表 |
| `GET /api/earthquakes/events` | 事件→后续地震 |

### 技术栈

- 后端：FastAPI  
- 前端：HTML + JS + ECharts 5  
- 无 Node / 数据库
