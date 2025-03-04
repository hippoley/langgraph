import os
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.pregel import Pregel
from langgraph.graph.message import add_messages
import json
from api_config import get_llm

# 加载环境变量
load_dotenv()

# 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "广州": "雨天，28°C",
        "深圳": "阴天，27°C"
    }
    return weather_data.get(location, f"没有{location}的天气信息")

@tool
def search_database(query: str) -> str:
    """搜索数据库获取信息"""
    database = {
        "人口": "中国人口约14亿，美国人口约3.3亿",
        "GDP": "2023年中国GDP约17.7万亿美元，美国GDP约26.9万亿美元",
        "面积": "中国面积约960万平方公里，美国面积约963万平方公里"
    }
    
    for key, value in database.items():
        if key in query:
            return value
    return "没有找到相关信息"

# 定义状态类型
class AgentState(TypedDict):
    messages: List[Annotated[HumanMessage | AIMessage | ToolMessage | SystemMessage, "对话历史"]]
    memory: Dict[str, Any]
    human_input: Optional[str]

# 定义节点函数
def agent(state: AgentState) -> Dict:
    """调用语言模型生成回复或工具调用"""
    messages = state["messages"]
    memory = state.get("memory", {})
    
    # 创建系统消息，包含记忆中的信息
    memory_str = ""
    if memory:
        memory_str = "你的记忆中包含以下信息:\n"
        for key, value in memory.items():
            memory_str += f"- {key}: {value}\n"
    
    system_message = SystemMessage(content=f"""你是一个智能助手，可以回答问题并使用工具。
{memory_str}
请根据用户的问题，决定是直接回答还是使用工具。""")
    
    # 确保系统消息在最前面
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages
    
    # 调用模型
    model = get_llm(temperature=0)
    model = model.bind_tools([calculator, get_weather, search_database])
    response = model.invoke(messages)
    
    return {"messages": add_messages(messages, [response])}

def route(state: AgentState) -> Literal["agent", "tools", "human_intervention", END]:
    """决定下一步操作"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果有人类输入请求，进入人类干预节点
    if state.get("human_input") == "requested":
        return "human_intervention"
    
    # 如果最后一条消息是AI消息且包含工具调用，则使用工具
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # 如果最后一条消息是工具消息，则返回给AI处理
    if isinstance(last_message, ToolMessage):
        return "agent"
    
    # 如果最后一条消息是AI消息且没有工具调用，则结束
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return END
    
    # 默认情况，交给AI处理
    return "agent"

def update_memory(state: AgentState) -> Dict:
    """更新长期记忆"""
    messages = state["messages"]
    memory = state.get("memory", {})
    
    # 分析最近的消息，提取可能需要记住的信息
    for message in messages[-3:]:  # 只看最近的3条消息
        if isinstance(message, HumanMessage):
            content = message.content
            # 检测是否有记忆相关的请求
            if "记住" in content or "记录" in content:
                # 简单的记忆提取逻辑
                parts = content.split("记住", 1)
                if len(parts) > 1:
                    info = parts[1].strip()
                    # 尝试提取键值对
                    if ":" in info or "：" in info:
                        key, value = info.replace("：", ":").split(":", 1)
                        memory[key.strip()] = value.strip()
                    else:
                        # 如果没有明确的键，使用时间戳作为键
                        import time
                        memory[f"记忆_{int(time.time())}"] = info
    
    return {"memory": memory}

def human_intervention(state: AgentState) -> Dict:
    """处理人类干预"""
    # 在实际应用中，这里会等待人类输入
    # 在这个POC中，我们模拟人类输入
    human_input = "这是人类干预的输入"
    
    # 创建一个新的人类消息
    human_message = HumanMessage(content=f"[人类干预] {human_input}")
    
    # 更新状态
    return {
        "messages": add_messages(state["messages"], [human_message]),
        "human_input": None  # 重置人类输入请求
    }

def request_human_input(state: AgentState) -> Dict:
    """请求人类干预"""
    return {"human_input": "requested"}

# 创建工具节点
tools_node = ToolNode([calculator, get_weather, search_database])

# 创建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)
workflow.add_node("update_memory", update_memory)
workflow.add_node("human_intervention", human_intervention)
workflow.add_node("request_human_input", request_human_input)

# 设置入口点
workflow.add_edge(START, "agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "human_intervention": "human_intervention",
        "request_human_input": "request_human_input",
        "end": END
    }
)

# 添加从工具到更新记忆的边
workflow.add_edge("tools", "update_memory")
workflow.add_edge("update_memory", "agent")

# 添加人类干预相关的边
workflow.add_edge("human_intervention", "agent")

# 编译图
agent = workflow.compile(checkpointer=MemorySaver())

# 示例使用
if __name__ == "__main__":
    print("高级LangGraph智能体示例")
    print("输入'exit'退出，输入'help'查看人类干预")
    
    # 创建会话ID
    session_id = "advanced-demo-session"
    
    # 初始化状态
    state = {
        "messages": [],
        "memory": {}
    }
    
    while True:
        user_input = input("\n您: ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "help":
            print("\n可用命令:")
            print("- 'exit': 退出程序")
            print("- 'memory': 显示当前记忆")
            print("- 'intervene': 触发人类干预")
            print("- '记住xxx': 将信息存入记忆")
            print("- '记住key:value': 将键值对存入记忆")
            continue
            
        if user_input.lower() == "memory":
            # 获取当前状态
            current_state = agent.get_state(session_id)
            if current_state and "memory" in current_state:
                print("\n当前记忆:")
                for key, value in current_state["memory"].items():
                    print(f"- {key}: {value}")
            else:
                print("\n当前没有记忆")
            continue
            
        if user_input.lower() == "intervene":
            # 更新状态请求人类干预
            agent.update_state(
                session_id,
                lambda state: {"human_input": "requested"}
            )
            print("\n已请求人类干预")
            continue
        
        # 创建人类消息
        human_message = HumanMessage(content=user_input)
        
        # 获取当前状态
        current_state = agent.get_state(session_id)
        if current_state:
            # 更新现有状态
            messages = current_state.get("messages", [])
            memory = current_state.get("memory", {})
            
            # 调用图
            result = agent.invoke(
                {"messages": messages + [human_message], "memory": memory},
                config={"configurable": {"thread_id": session_id}}
            )
        else:
            # 首次调用
            result = agent.invoke(
                {"messages": [human_message], "memory": {}},
                config={"configurable": {"thread_id": session_id}}
            )
        
        # 打印AI回复
        messages = result["messages"]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            latest_ai_message = ai_messages[-1]
            print(f"\nAI: {latest_ai_message.content}") 