# LangGraph Minimal POC

This project is a minimal Proof of Concept (POC) for LangGraph, showcasing 80% of LangGraph's core functionalities. This POC uses a third-party OpenAI-compatible API service (api2d.net).

## Features

This POC includes the following examples:

1. **Simple Agent (simple_agent.py)**: A LangGraph agent built from scratch, demonstrating basic state graphs and tool calling.
2. **Prebuilt Agent (simple_agent_prebuilt.py)**: An agent quickly created using LangGraph's prebuilt functions.
3. **Advanced Agent (advanced_agent.py)**: Demonstrates more advanced features, including memory and human intervention.
4. **Multi-Agent System (multi_agent_system.py)**: Demonstrates a workflow where multiple agents collaborate to complete tasks.
5. **Streaming and Breakpoints (streaming_breakpoints.py)**: Demonstrates streaming output and breakpoint functionality.

## Installation and Startup

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Internet connection for API access

### Quickest Method (Recommended)

For Windows users, you can double-click the `start_langgraph.bat` file, which provides an all-in-one menu with the following options:

1. Install dependencies
2. Clean temporary files
3. Start LangGraph server
4. Execute all operations (recommended for first-time users)

This batch file will:
- Install all required dependencies
- Clean up any temporary files that might cause errors
- Configure CORS settings for cross-origin requests
- Start the LangGraph development server using multiple fallback methods

Alternatively, you can run the `langgraph_fix.py` script directly:

```bash
python langgraph_fix.py
```

This script offers the following options:
1. Clean temporary files
2. Terminate LangGraph processes
3. Check and install dependencies
4. Start LangGraph server (with CORS support)
5. Execute all steps (clean + terminate + check + start)
6. Create a quick start batch file

### Manual Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure langgraph-cli is correctly installed:
   ```bash
   pip install langgraph-cli
   ```
4. Set up your API key in the `.env` file (pre-configured for api2d.net)

### Common Issues

If you encounter a "command not found" error when running `langgraph dev`, try these steps:

1. Use our provided `start_langgraph.bat` or `start.py` script, which automatically handles these issues
2. Or, ensure langgraph-cli and langgraph-api are installed:
   ```bash
   pip install -U "langgraph-cli[inmem]"
   ```
   This command installs both langgraph-cli and langgraph-api
3. Close and reopen your terminal/command prompt
4. Ensure Python's Scripts directory is in your PATH environment variable
   - Windows: Typically at `C:\Users\<username>\AppData\Local\Programs\Python\Python3x\Scripts`
   - Linux/Mac: Typically at `~/.local/bin` or in the `bin` directory of your virtual environment

If you encounter a "Required package 'langgraph-api' is not installed" error, run:
```bash
pip install -U "langgraph-cli[inmem]"
```

## Detailed Usage Guide

### Using the Startup Script (Recommended)

1. For Windows users, double-click the `start_langgraph.bat` file, or run `python start.py`
2. Select the example you want to run from the menu
3. Follow the on-screen instructions to interact with the selected example

### Starting the LangGraph Server with CORS Support

For the best experience with the web interface, use our all-in-one batch file:

```bash
# Double-click start_langgraph.bat and select option 3 or 4
```

Or use the Python script directly:

```bash
python langgraph_fix.py --start
```

This script will:
1. Check and install required dependencies
2. Clean up any temporary files that might cause errors
3. Configure CORS settings for cross-origin requests
4. Start the LangGraph development server
5. Provide links to access the Studio UI and API documentation

### Running Python Files Directly

You can run individual Python files directly to test different functionalities:

```bash
python simple_agent.py
python simple_agent_prebuilt.py
python advanced_agent.py
python multi_agent_system.py
python streaming_breakpoints.py
```

When running files directly:
- The output will be displayed in the console
- You can interact with the agent through the command line
- No visualization will be available (use LangGraph Studio for visualization)

### Using LangGraph Studio UI

After starting the server with `langgraph dev` or `python start_with_cors.py`:

1. Open your browser and navigate to http://localhost:3000
2. You'll see a list of available graphs from the `langgraph.json` configuration
3. Click on a graph to open its visualization and interaction panel
4. Use the chat interface to interact with the agent
5. Observe the graph execution in real-time as nodes are activated

#### Studio UI Features:

- **Graph Visualization**: See the state graph structure and watch nodes activate in real-time
- **Chat Interface**: Interact with the agent through a chat-like interface
- **Execution History**: Review previous runs and their execution paths
- **Debugging Tools**: Set breakpoints and inspect state at each step
- **API Documentation**: Access the OpenAPI documentation at http://localhost:8000/docs

### Testing the API Directly

You can test the API endpoints directly using our included HTML test tool:

1. Open `test_api.html` in your browser after starting the server
2. Use the connection test tab to verify the API is accessible
3. View available graphs and test chat functionality
4. Troubleshoot common issues with the provided guidance

## Tool Functionality and Limitations

### Available Tools

The agents in this POC have access to the following tools:

1. **Calculator**: Performs basic arithmetic operations
   - Example: "Calculate 25 * 4 + 10"
   - Supports: addition, subtraction, multiplication, division, and parentheses

2. **Weather Information**: Retrieves weather information for specific cities
   - Example: "What's the weather in Beijing?"
   - Supported cities: Beijing, Shanghai, Guangzhou, Shenzhen
   - Note: This is a simulated tool with hardcoded responses

3. **Translation Tool**: Translates text between languages (in updated version)
   - Example: "Translate 'hello' to Spanish"
   - Supports multiple languages including English, Chinese, Spanish, French, and German

### Tool Limitations

- The weather tool only supports the listed cities with simulated data
- The calculator handles basic arithmetic but not advanced mathematical functions
- Tools are simulated and do not connect to external services

## Core Functionality Showcase

1. **StateGraph**: All examples use StateGraph to define workflows and control flow.
2. **Persistence**: Uses MemorySaver for state persistence.
3. **Human-in-the-loop**: Human intervention functionality is demonstrated in advanced_agent.py.
4. **Tool Calling**: All examples include tool calling functionality.
5. **Streaming**: Streaming output functionality is demonstrated in streaming_breakpoints.py.
6. **Multi-Agent System**: Multi-agent collaboration is demonstrated in multi_agent_system.py.
7. **Memory**: Cross-session memory functionality is demonstrated in advanced_agent.py.
8. **Breakpoints**: Breakpoint functionality is demonstrated in streaming_breakpoints.py.

## Example Descriptions

### Simple Agent (simple_agent.py)
A basic agent that can use tools to answer questions. This example demonstrates:
- Creating a state graph from scratch
- Implementing tool calling functionality
- Basic agent decision-making logic

### Prebuilt Agent (simple_agent_prebuilt.py)
A simplified agent implementation using LangGraph's prebuilt components. This example shows:
- How to quickly create an agent using prebuilt functions
- Minimal code required for a functional agent

### Advanced Agent (advanced_agent.py)
An enhanced agent with memory and human intervention capabilities. This example demonstrates:
- Implementing memory across conversations
- Adding human-in-the-loop functionality
- More complex routing logic

### Multi-Agent System (multi_agent_system.py)
A system where multiple agents collaborate to solve tasks. This example shows:
- How to create multiple specialized agents
- Implementing communication between agents
- Coordinating workflow across multiple agents

### Streaming and Breakpoints (streaming_breakpoints.py)
Demonstrates streaming output and debugging capabilities. This example shows:
- How to implement streaming responses
- Setting and using breakpoints for debugging
- Inspecting state during execution

## Third-Party API Configuration

This POC uses the OpenAI-compatible API service provided by api2d.net. Configuration information is set in the `.env` file:

```
OPENAI_API_KEY=fk222719-4TlnHx5wbaXtUm4CcneT1oLogM3TKGDB
OPENAI_API_BASE=https://oa.api2d.net
OPENAI_MODEL_NAME=o3-mini
```

### Using Your Own API Key

If you want to use your own OpenAI API key or another compatible service:

1. Edit the `.env` file
2. Replace the values with your own credentials:
   - For OpenAI direct access:
     ```
     OPENAI_API_KEY=your_openai_api_key
     OPENAI_API_BASE=https://api.openai.com/v1
     OPENAI_MODEL_NAME=gpt-3.5-turbo
     ```
   - For other compatible services, use their respective base URLs and API keys

### Model Recommendations

- The default `o3-mini` model is sufficient for basic testing
- For better performance, consider using `gpt-3.5-turbo` or `gpt-4` if available
- More capable models will provide better tool use and reasoning capabilities

## Configuration

You can configure graphs and dependencies in the `langgraph.json` file:

```json
{
  "dependencies": ["langchain_core", "langchain_openai", "langgraph"],
  "graphs": {
    "simple_agent": "./simple_agent.py:app",
    "simple_agent_prebuilt": "./simple_agent_prebuilt.py:agent",
    "advanced_agent": "./advanced_agent.py:agent",
    "multi_agent": "./multi_agent_system.py:agent",
    "streaming": "./streaming_breakpoints.py:agent"
  },
  "env_file": "./.env"
}
```

To add your own graph:
1. Create a new Python file with your graph implementation
2. Export your graph as a variable (e.g., `app` or `agent`)
3. Add an entry to the `graphs` section in `langgraph.json`

## Troubleshooting

### Server Won't Start
- Run the `start_langgraph.bat` file and select option 4 to perform a complete setup and start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8000 is already in use by another application
- Try running with administrator/sudo privileges
- Check the console for specific error messages

### CORS Issues
- Use the `start_with_cors.py` script which sets up proper CORS headers
- Try accessing the Studio UI from the same origin as the API
- If using a custom frontend, ensure it's sending the correct CORS headers

### API Connection Failures
- Verify the server is running (check for messages in the console)
- Ensure you're using the correct URL (default: http://localhost:8000)
- Check network connectivity and firewall settings
- Try accessing the API documentation at http://localhost:8000/docs

### Model API Issues
- Verify your API key is correct in the `.env` file
- Check if you've exceeded rate limits or quotas
- Try a different model if specified model is unavailable
- Ensure internet connectivity for API requests

## Environment Variables

Set the following environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your API key
- `OPENAI_API_BASE`: API base URL
- `OPENAI_MODEL_NAME`: Model name to use
- `LANGCHAIN_TRACING_V2`: Whether to enable LangSmith tracing
- `LANGCHAIN_API_KEY`: Your LangSmith API key
- `LANGCHAIN_PROJECT`: LangSmith project name

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## 访问 LangGraph

服务器成功启动后，您可以通过以下方式访问：

### LangGraph Studio UI

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### LangGraph API

```
http://localhost:2024
```
或
```
http://127.0.0.1:2024
```

### API 文档

```
http://localhost:2024/docs
```

## 使用 LangGraph Studio UI

1. 在浏览器中打开 Studio UI：`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

## 使用 API 测试工具

如果您想直接测试 API：

1. 打开 `test_api.html` 文件
2. 在 API URL 输入框中输入：`http://localhost:2024`（默认已设置为 `http://127.0.0.1:2024`）
3. 点击 "测试连接" 按钮

### 服务器无法启动

1. 确保已安装所有依赖：
   ```
   pip install -r requirements.txt
   pip install -U "langgraph-cli[inmem]"
   ```

2. 清理临时文件（使用批处理文件的选项2）

3. 检查端口占用情况：
   ```
   netstat -ano | findstr :2024
   netstat -ano | findstr :3000
   ```

### 无法访问 Studio UI

1. 确认服务器正在运行（命令行窗口应显示服务器日志）
2. 尝试使用 `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
3. 检查浏览器控制台是否有 CORS 错误
4. 尝试使用不同的浏览器

### API 连接问题

1. 确保使用正确的端口：2024
2. 检查服务器是否正在运行
3. 尝试访问 API 文档：`http://localhost:2024/docs`

## Environment Variables

Set the following environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your API key
- `OPENAI_API_BASE`: API base URL
- `OPENAI_MODEL_NAME`: Model name to use
- `LANGCHAIN_TRACING_V2`: Whether to enable LangSmith tracing
- `LANGCHAIN_API_KEY`: Your LangSmith API key
- `LANGCHAIN_PROJECT`: LangSmith project name

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## 项目结构

```
langgraph-poc/
├── .env                      # 环境变量配置
├── api_config.py             # API 配置
├── advanced_agent.py         # 高级代理示例
├── langgraph.json            # LangGraph 配置
├── langgraph_fix.py          # LangGraph 修复工具
├── multi_agent_system.py     # 多代理系统示例
├── open_test_api.bat         # 测试工具启动脚本
├── requirements.txt          # 依赖列表
├── simple_agent.py           # 简单代理示例
├── simple_agent_basic.py     # 简化版代理（不依赖 LangGraph）
├── simple_agent_prebuilt.py  # 预构建代理示例
├── start_langgraph.bat       # 主启动脚本
├── streaming_breakpoints.py  # 流式处理和断点示例
└── test_api.html             # API 测试工具
```

## 快速开始

### 方法 1: 使用批处理文件（推荐）

1. 双击运行 `start_langgraph.bat`
2. 在菜单中选择选项 3 启动服务器
3. 访问 LangGraph Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### 方法 2: 使用 Python 脚本

您也可以直接运行 `langgraph_fix.py` 脚本:

```
python langgraph_fix.py --start
```

### 方法 3: 直接运行示例

如果您只想尝试代理功能而不启动服务器，可以直接运行:

```
python simple_agent_basic.py
``` 