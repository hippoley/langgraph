import os
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import asyncio
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

# 定义状态类型
class StreamingState(TypedDict):
    messages: List[Annotated[HumanMessage | AIMessage | SystemMessage, "对话历史"]]
    breakpoint_active: bool
    breakpoint_step: str
    user_feedback: Optional[str]

# 定义节点函数
def agent(state: StreamingState) -> Dict:
    """调用语言模型生成回复或工具调用"""
    messages = state["messages"]
    
    # 创建系统消息
    system_message = SystemMessage(content="""你是一个智能助手，可以回答问题并使用工具。
请根据用户的问题，决定是直接回答还是使用工具。""")
    
    # 确保系统消息在最前面
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages
    
    # 调用模型
    model = get_llm(temperature=0)
    model = model.bind_tools([calculator])
    response = model.invoke(messages)
    
    # 模拟处理时间，在实际应用中可以删除
    import time
    time.sleep(1)
    
    return {"messages": add_messages(messages, [response])}

def tools_node(state: StreamingState) -> Dict:
    """执行工具调用"""
    # 使用预构建的ToolNode
    tool_node = ToolNode([calculator])
    result = tool_node.invoke(state)
    
    # 模拟处理时间，在实际应用中可以删除
    import time
    time.sleep(1)
    
    return result

def check_breakpoint(state: StreamingState) -> Dict:
    """检查是否需要暂停执行"""
    breakpoint_active = state.get("breakpoint_active", False)
    breakpoint_step = state.get("breakpoint_step", "")
    
    # 如果断点激活并且当前步骤匹配，则等待用户反馈
    if breakpoint_active and breakpoint_step == "before_agent":
        # 在实际应用中，这里会等待用户输入
        # 在这个POC中，我们模拟等待
        import time
        print("\n[断点激活] 等待用户反馈...")
        time.sleep(3)
        
        # 模拟用户反馈
        user_feedback = state.get("user_feedback", "继续执行")
        print(f"[用户反馈] {user_feedback}")
    
    return {}

def route(state: StreamingState) -> Literal["agent", "tools", END]:
    """决定下一步操作"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果最后一条消息是AI消息且包含工具调用，则使用工具
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # 如果最后一条消息是AI消息且没有工具调用，则结束
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return END
    
    # 默认情况，交给AI处理
    return "agent"

# 创建图
workflow = StateGraph(StreamingState)

# 添加节点
workflow.add_node("check_breakpoint", check_breakpoint)
workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)

# 设置入口点
workflow.add_edge(START, "check_breakpoint")
workflow.add_edge("check_breakpoint", "agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "end": END
    }
)

# 添加从工具到AI的边
workflow.add_edge("tools", "agent")

# 编译图
agent = workflow.compile(checkpointer=MemorySaver())

# 流式处理函数
async def process_stream(stream: AsyncIterator[Dict[str, Any]]) -> None:
    """处理流式输出"""
    print("\n开始流式输出...")
    
    async for chunk in stream:
        # 获取最新的消息
        if "messages" in chunk:
            messages = chunk["messages"]
            if messages and isinstance(messages[-1], AIMessage):
                # 如果是AI消息，打印内容
                content = messages[-1].content
                if content:
                    print(f"\r当前输出: {content}", end="", flush=True)
        
        # 如果有节点更新，显示当前执行的节点
        if "next" in chunk:
            next_node = chunk.get("next")
            if next_node:
                print(f"\n[执行节点] {next_node}")
    
    print("\n流式输出结束")

# 示例使用
if __name__ == "__main__":
    print("流式输出和断点示例")
    print("输入'exit'退出，输入'breakpoint'设置断点")
    
    # 创建会话ID
    session_id = "streaming-demo-session"
    
    # 断点状态
    breakpoint_active = False
    
    while True:
        user_input = input("\n您: ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "breakpoint":
            breakpoint_active = not breakpoint_active
            print(f"\n断点已{'激活' if breakpoint_active else '关闭'}")
            continue
        
        # 创建人类消息
        human_message = HumanMessage(content=user_input)
        
        # 获取当前状态
        current_state = agent.get_state(session_id)
        if current_state:
            # 更新现有状态
            messages = current_state.get("messages", [])
            
            # 初始化状态
            initial_state = {
                "messages": messages + [human_message],
                "breakpoint_active": breakpoint_active,
                "breakpoint_step": "before_agent",
                "user_feedback": "继续执行"
            }
        else:
            # 首次调用
            initial_state = {
                "messages": [human_message],
                "breakpoint_active": breakpoint_active,
                "breakpoint_step": "before_agent",
                "user_feedback": "继续执行"
            }
        
        # 创建流式调用
        stream = agent.astream(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )
        
        # 处理流式输出
        asyncio.run(process_stream(stream))
        
        # 获取最终结果
        final_state = agent.get_state(session_id)
        if final_state and "messages" in final_state:
            messages = final_state["messages"]
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            if ai_messages:
                latest_ai_message = ai_messages[-1]
                print(f"\n最终回复: {latest_ai_message.content}") 