"""
静态文件服务模块 - 提供HTML和前端资源
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StaticServer")

# 创建目录
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# 创建FastAPI应用
app = FastAPI(
    title="LangGraph对话系统",
    description="企业级对话系统前端",
    version="1.0.0"
)

# 设置静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """返回主页"""
    return templates.TemplateResponse("index.html", {"request": request})

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 