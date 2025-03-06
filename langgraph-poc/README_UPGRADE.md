# LangGraph 对话系统改造

本项目是对原有 LangGraph POC 的全面改造，目标是构建一个企业级的对话系统，充分发挥 LangGraph 的潜力。

## 改造亮点

1. **异步架构改造**
   - 将同步代码改造为异步
   - 实现流式响应支持
   - 添加 WebSocket 通信

2. **断点和人机协作**
   - 实现断点设置机制
   - 添加人工干预通道
   - 提供调试控制界面

3. **持久化机制升级**
   - 支持多种持久化选项（内存、SQLite、PostgreSQL）
   - 实现会话管理系统
   - 确保状态一致性

4. **API 服务层**
   - FastAPI RESTful 接口
   - WebSocket 流式通信
   - API 密钥认证机制

5. **前端界面**
   - 美观的聊天界面
   - 实时调试控制
   - 会话信息面板

## 目录结构

```
langgraph-poc/
├── async_api_service.py  - 异步API服务
├── enhanced_agent.py     - 增强对话代理
├── static_server.py      - 静态文件服务
├── start.py              - 启动脚本
├── templates/            - 前端模板
│   └── index.html        - 对话界面
├── static/               - 静态资源
├── agent_states/         - 会话状态存储
└── knowledge_base/       - 知识库文件
```

## 安装要求

请确保安装以下依赖：

```bash
pip install -r requirements.txt
```

## 环境变量配置

在 `.env` 文件中配置以下变量：

```
# OpenAI API 配置
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o

# 持久化配置
PERSISTENCE_TYPE=memory  # memory, sqlite, postgres
SQLITE_DB_PATH=agent_states/sessions.db
# POSTGRES_CONN_STRING=postgresql://user:password@localhost/dbname

# API 配置
API_KEYS=test-key,another-key
```

## 运行方法

### 方法一：使用启动脚本

```bash
python start.py
```

这将同时启动 API 服务（端口 8000）和前端服务（端口 8001）。

### 方法二：单独启动服务

**启动 API 服务：**

```bash
uvicorn async_api_service:app --host 0.0.0.0 --port 8000
```

**启动前端服务：**

```bash
uvicorn static_server:app --host 0.0.0.0 --port 8001
```

## 使用方法

1. 访问前端界面：http://localhost:8001

2. 与对话代理交流，尝试以下功能：
   - 计算：`计算 (3 + 4) * 5`
   - 天气查询：`北京今天天气如何？`
   - 翻译：`把"你好"翻译成英语`
   - 业务报表：`生成上个季度的财务报表`

3. 使用调试控制：
   - 设置断点：点击"设置断点"按钮，输入节点名称（如 `intent_recognizer`）
   - 继续执行：当流程在断点处暂停时，点击"继续执行"按钮

4. 指令式控制：
   - `/breakpoint <node_name> on` - 设置断点
   - `/breakpoint <node_name> off` - 移除断点
   - `/continue` - 继续执行
   - `/retry` - 重试当前步骤
   - `/abort` - 中止当前意图

## API 文档

访问 API 文档：http://localhost:8000/docs

## 下一步计划

1. **多模态支持**
   - 添加图像处理能力
   - 实现文档分析功能
   - 支持可视化输出

2. **可视化与监控系统**
   - 实现对话流程可视化
   - 添加性能监控仪表板
   - 设计日志分析系统

3. **业务场景深化**
   - 实现更多业务报表类型
   - 添加复杂查询解析
   - 实现业务分析推荐 