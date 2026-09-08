"""
地震流体异常检测可视化平台 - FastAPI 后端
启动命令: python web/app.py 或 uvicorn web.app:app --reload
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.data import router as data_router
from api.models import router as models_router
from api.anomalies import router as anomalies_router

app = FastAPI(
    title="地震流体异常检测可视化平台",
    description="基于 CNN-LSTM 的地震前兆异常检测结果展示",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(data_router, prefix="/api", tags=["数据"])
app.include_router(models_router, prefix="/api", tags=["模型"])
app.include_router(anomalies_router, prefix="/api", tags=["异常检测"])

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def index():
    """数据概览页"""
    return STATIC_DIR / "index.html"


@app.get("/prediction", response_class=FileResponse)
async def prediction():
    """模型预测页"""
    return STATIC_DIR / "prediction.html"


@app.get("/anomaly", response_class=FileResponse)
async def anomaly():
    """异常检测页"""
    return STATIC_DIR / "anomaly.html"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "地震流体异常检测可视化平台"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  地震流体异常检测可视化平台")
    print("  访问地址: http://localhost:8000")
    print("=" * 50)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
