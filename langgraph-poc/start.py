"""
启动脚本 - 启动API服务和前端服务
"""

import os
import sys
import logging
import asyncio
import subprocess
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StartScript")

# 确保目录结构
def ensure_directories():
    """确保必要的目录结构存在"""
    logger.info("确保目录结构...")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("agent_states", exist_ok=True)
    os.makedirs("knowledge_base", exist_ok=True)

# 检查环境变量
def check_environment():
    """检查环境变量设置"""
    logger.info("检查环境变量...")
    if not os.path.exists(".env"):
        logger.warning("未找到.env文件，将使用默认配置")

# 确保index.html文件在templates目录中
def ensure_templates():
    """确保模板文件存在"""
    logger.info("确保模板文件...")
    if not os.path.exists("templates/index.html"):
        # 如果不存在，复制一份
        source_path = Path("index.html")
        if source_path.exists():
            import shutil
            shutil.copy2(source_path, "templates/index.html")
            logger.info("已复制index.html到templates目录")
        else:
            logger.error("未找到index.html模板文件")
            sys.exit(1)

# 启动API服务
async def start_api_service():
    """启动API服务"""
    logger.info("启动API服务...")
    cmd = [sys.executable, "-m", "uvicorn", "async_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    logger.info(f"API服务已启动 (PID: {process.pid})")
    return process

# 启动前端服务
async def start_frontend_service():
    """启动前端服务"""
    logger.info("启动前端服务...")
    cmd = [sys.executable, "-m", "uvicorn", "static_server:app", "--host", "0.0.0.0", "--port", "8001"]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    logger.info(f"前端服务已启动 (PID: {process.pid})")
    return process

# 监控日志
async def monitor_logs(process, name):
    """监控进程日志"""
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        logger.info(f"{name}: {line.decode().strip()}")

# 主函数
async def main():
    """主函数"""
    logger.info("启动LangGraph对话系统...")
    
    # 准备工作
    ensure_directories()
    check_environment()
    ensure_templates()
    
    # 启动服务
    api_process = await start_api_service()
    frontend_process = await start_frontend_service()
    
    # 监控日志
    api_log_task = asyncio.create_task(monitor_logs(api_process, "API"))
    frontend_log_task = asyncio.create_task(monitor_logs(frontend_process, "Frontend"))
    
    # 等待所有任务完成
    try:
        await asyncio.gather(api_log_task, frontend_log_task)
    except asyncio.CancelledError:
        logger.info("服务停止中...")
    finally:
        # 停止所有进程
        if api_process:
            api_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        logger.info("所有服务已停止")

# 启动主函数
if __name__ == "__main__":
    try:
        # 检查是否可以使用asyncio.run (Python 3.7+)
        asyncio.run(main())
    except AttributeError:
        # 兼容旧版Python
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close() 