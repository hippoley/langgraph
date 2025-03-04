import os
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ChatMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import json
from api_config import get_llm

# 加载环境变量
load_dotenv()

# 定义状态类型
class MultiAgentState(TypedDict):
    messages: List[Annotated[HumanMessage | AIMessage | ChatMessage | SystemMessage, "对话历史"]]
    current_agent: str
    task: str
    research_results: str
    code_results: str
    review_results: str
    final_answer: str

# 定义不同角色的智能体
def researcher(state: MultiAgentState) -> Dict:
    """研究员智能体，负责收集和分析信息"""
    messages = state["messages"]
    task = state["task"]
    
    # 创建系统消息
    system_message = SystemMessage(content=f"""你是一个专业的研究员。
你的任务是分析问题并提供详细的研究结果。
当前任务: {task}
请提供详细的分析和研究结果。""")
    
    # 准备消息列表
    agent_messages = [system_message]
    
    # 添加之前的消息
    for message in messages:
        if isinstance(message, HumanMessage):
            agent_messages.append(message)
    
    # 调用模型
    model = get_llm(temperature=0)
    response = model.invoke(agent_messages)
    
    # 创建研究员的消息
    researcher_message = ChatMessage(content=response.content, role="researcher")
    
    # 更新状态
    return {
        "messages": add_messages(messages, [researcher_message]),
        "research_results": response.content,
        "current_agent": "coder"  # 下一个是程序员
    }

def coder(state: MultiAgentState) -> Dict:
    """程序员智能体，负责编写代码"""
    messages = state["messages"]
    task = state["task"]
    research_results = state["research_results"]
    
    # 创建系统消息
    system_message = SystemMessage(content=f"""你是一个专业的程序员。
你的任务是根据研究结果编写代码。
当前任务: {task}
研究结果: {research_results}
请提供解决方案的代码实现。""")
    
    # 准备消息列表
    agent_messages = [system_message]
    
    # 添加之前的消息
    for message in messages:
        if isinstance(message, HumanMessage) or (isinstance(message, ChatMessage) and message.role == "researcher"):
            agent_messages.append(message)
    
    # 调用模型
    model = get_llm(temperature=0)
    response = model.invoke(agent_messages)
    
    # 创建程序员的消息
    coder_message = ChatMessage(content=response.content, role="coder")
    
    # 更新状态
    return {
        "messages": add_messages(messages, [coder_message]),
        "code_results": response.content,
        "current_agent": "reviewer"  # 下一个是审核员
    }

def reviewer(state: MultiAgentState) -> Dict:
    """审核员智能体，负责审核研究和代码"""
    messages = state["messages"]
    task = state["task"]
    research_results = state["research_results"]
    code_results = state["code_results"]
    
    # 创建系统消息
    system_message = SystemMessage(content=f"""你是一个专业的审核员。
你的任务是审核研究结果和代码实现。
当前任务: {task}
研究结果: {research_results}
代码实现: {code_results}
请提供审核意见，指出优点和可能的改进点。""")
    
    # 准备消息列表
    agent_messages = [system_message]
    
    # 添加之前的消息
    for message in messages:
        if isinstance(message, HumanMessage) or isinstance(message, ChatMessage):
            agent_messages.append(message)
    
    # 调用模型
    model = get_llm(temperature=0)
    response = model.invoke(agent_messages)
    
    # 创建审核员的消息
    reviewer_message = ChatMessage(content=response.content, role="reviewer")
    
    # 更新状态
    return {
        "messages": add_messages(messages, [reviewer_message]),
        "review_results": response.content,
        "current_agent": "integrator"  # 下一个是整合者
    }

def integrator(state: MultiAgentState) -> Dict:
    """整合者智能体，负责整合所有结果并提供最终答案"""
    messages = state["messages"]
    task = state["task"]
    research_results = state["research_results"]
    code_results = state["code_results"]
    review_results = state["review_results"]
    
    # 创建系统消息
    system_message = SystemMessage(content=f"""你是一个专业的整合者。
你的任务是整合所有结果并提供最终答案。
当前任务: {task}
研究结果: {research_results}
代码实现: {code_results}
审核意见: {review_results}
请提供一个综合的最终答案，包括研究结果、代码实现和改进建议。""")
    
    # 准备消息列表
    agent_messages = [system_message]
    
    # 添加之前的消息
    for message in messages:
        if isinstance(message, HumanMessage) or isinstance(message, ChatMessage):
            agent_messages.append(message)
    
    # 调用模型
    model = get_llm(temperature=0)
    response = model.invoke(agent_messages)
    
    # 创建整合者的消息
    integrator_message = AIMessage(content=response.content)
    
    # 更新状态
    return {
        "messages": add_messages(messages, [integrator_message]),
        "final_answer": response.content,
        "current_agent": "done"  # 标记为完成
    }

def route(state: MultiAgentState) -> Literal["researcher", "coder", "reviewer", "integrator", END]:
    """路由函数，决定下一步由哪个智能体处理"""
    current_agent = state.get("current_agent", "")
    
    if current_agent == "researcher":
        return "researcher"
    elif current_agent == "coder":
        return "coder"
    elif current_agent == "reviewer":
        return "reviewer"
    elif current_agent == "integrator":
        return "integrator"
    elif current_agent == "done":
        return END
    
    # 默认从研究员开始
    return "researcher"

# 创建图
workflow = StateGraph(MultiAgentState)

# 添加节点
workflow.add_node("researcher", researcher)
workflow.add_node("coder", coder)
workflow.add_node("reviewer", reviewer)
workflow.add_node("integrator", integrator)
workflow.add_node("router", route)  # 添加 route 函数作为名为 "router" 的节点

# 设置入口点
workflow.add_edge(START, "router")  # 使用节点名称 "router" 而不是函数 route

# 添加条件边
workflow.add_conditional_edges(
    "router",  # 使用节点名称 "router" 而不是函数 route
    {
        "researcher": lambda x: "researcher",
        "coder": lambda x: "coder",
        "reviewer": lambda x: "reviewer",
        "integrator": lambda x: "integrator",
        END: lambda x: END
    }
)

# 添加从各个智能体回到路由的边
workflow.add_edge("researcher", "router")  # 使用节点名称 "router" 而不是函数 route
workflow.add_edge("coder", "router")  # 使用节点名称 "router" 而不是函数 route
workflow.add_edge("reviewer", "router")  # 使用节点名称 "router" 而不是函数 route
workflow.add_edge("integrator", "router")  # 使用节点名称 "router" 而不是函数 route

# 编译图
agent = workflow.compile(checkpointer=MemorySaver())

# 示例使用
if __name__ == "__main__":
    print("多智能体系统示例")
    print("输入'exit'退出")
    
    # 创建会话ID
    session_id = "multi-agent-demo-session"
    
    while True:
        user_input = input("\n请输入您的任务: ")
        if user_input.lower() == "exit":
            break
        
        # 创建人类消息
        human_message = HumanMessage(content=user_input)
        
        # 初始化状态
        initial_state = {
            "messages": [human_message],
            "task": user_input,
            "current_agent": "researcher",  # 从研究员开始
            "research_results": "",
            "code_results": "",
            "review_results": "",
            "final_answer": ""
        }
        
        print("\n多智能体系统开始工作...")
        print("研究员正在分析问题...")
        
        # 调用图
        result = agent.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )
        
        # 打印各个智能体的结果
        print("\n=== 研究员的分析 ===")
        print(result["research_results"])
        
        print("\n=== 程序员的代码实现 ===")
        print(result["code_results"])
        
        print("\n=== 审核员的审核意见 ===")
        print(result["review_results"])
        
        print("\n=== 最终整合结果 ===")
        print(result["final_answer"]) 