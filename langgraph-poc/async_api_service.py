"""
异步API服务层 - 提供REST API和WebSocket接口
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AsyncAPIService")

# 导入增强对话代理
try:
    from enhanced_agent import process_message, aprocess_message, astream_message, get_session_state
except ImportError as e:
    logger.error(f"导入对话代理失败: {str(e)}")
    logger.error("请确保enhanced_agent.py文件已正确配置")
    raise

# 创建FastAPI应用
app = FastAPI(
    title="LangGraph对话系统",
    description="企业级对话系统API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API密钥安全头
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# 从环境变量获取API密钥
API_KEYS = os.getenv("API_KEYS", "test-key").split(",")

# 请求和响应模型
class MessageRequest(BaseModel):
    """对话请求模型"""
    message: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    """对话响应模型"""
    session_id: str
    response: str
    completed: bool
    metadata: Optional[Dict[str, Any]] = None

class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str
    created_at: str
    last_updated_at: str
    message_count: int
    current_intent: Optional[str] = None

# 验证API密钥
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"新的WebSocket连接: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket连接关闭: {session_id}")
    
    async def send_message(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

manager = ConnectionManager()

# REST API路由
@app.post("/api/chat/{session_id}", response_model=MessageResponse, dependencies=[Depends(verify_api_key)])
async def chat(session_id: str, request: MessageRequest):
    """处理对话请求并返回响应"""
    try:
        # 使用异步处理消息
        response = await aprocess_message(session_id, request.message)
        
        return MessageResponse(
            session_id=session_id,
            response=response.get("response", ""),
            completed=response.get("completed", True),
            metadata=response.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理消息时出错: {str(e)}"
        )

@app.get("/api/sessions/{session_id}", response_model=SessionInfo, dependencies=[Depends(verify_api_key)])
async def get_session(session_id: str):
    """获取会话信息"""
    try:
        state = await get_session_state(session_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到会话: {session_id}"
            )
        
        return SessionInfo(
            session_id=session_id,
            created_at=state.get("created_at", ""),
            last_updated_at=state.get("last_updated_at", ""),
            message_count=len(state.get("messages", [])),
            current_intent=state.get("current_intent", {}).get("name") if state.get("current_intent") else None
        )
    except Exception as e:
        logger.error(f"获取会话信息时出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话信息时出错: {str(e)}"
        )

# WebSocket路由
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """处理WebSocket对话"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            request_data = json.loads(data)
            message = request_data.get("message", "")
            
            # 异步流式处理
            async for chunk in astream_message(session_id, message):
                # 发送每个响应块
                await websocket.send_json({
                    "type": "chunk",
                    "data": chunk
                })
            
            # 发送完成信号
            await websocket.send_json({
                "type": "complete",
                "session_id": session_id
            })
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket处理时出错: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
        manager.disconnect(session_id)

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    logger.info("API服务启动")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API服务关闭")

# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 