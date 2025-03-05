#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 可重用组件模块

此模块提供了一系列可重用的 LangGraph 组件，包括：
1. 常用节点类型（思考节点、工具节点、路由节点等）
2. 预定义工具集合（搜索、计算、翻译等）
3. 状态管理器
4. 工作流构建器
5. 错误处理组件
"""

import os
import json
import time
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Callable, TypedDict, Annotated, Literal, Type, cast
from enum import Enum
import re
import requests
import numpy as np
from pydantic import BaseModel, Field

# 尝试导入 LangGraph 相关库
try:
    from langchain.schema import HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_core.messages import BaseMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint import MemorySaver
except ImportError:
    print("警告: 未能导入 LangGraph 相关库，某些功能可能不可用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langgraph_components.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("langgraph_components")

# ============================================================================
# 基础状态类型
# ============================================================================

class BaseState(TypedDict):
    """基础状态类型"""
    messages: List[Annotated[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage], "消息历史"]]

class MemoryState(BaseState):
    """带记忆功能的状态类型"""
    memory: Optional[Dict[str, Any]]

class ToolState(BaseState):
    """带工具功能的状态类型"""
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_results: Optional[List[Dict[str, Any]]]

class HumanInteractionState(BaseState):
    """带人类交互功能的状态类型"""
    human_input: Optional[str]
    requires_human: bool

class FullState(MemoryState, ToolState, HumanInteractionState):
    """完整状态类型，包含所有功能"""
    pass

# ============================================================================
# 工具定义
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全地计算表达式
        # 移除所有非法字符
        clean_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
        result = eval(clean_expr)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def search_web(query: str) -> str:
    """搜索网络信息（模拟）"""
    # 这是一个模拟的搜索功能
    search_results = {
        "天气": "今天天气晴朗，温度25°C，适合户外活动。",
        "新闻": "最新新闻: 科技创新大会将在下周举行，多家科技公司将展示最新产品。",
        "股票": "股市今日上涨2.3%，科技股表现强劲。",
        "体育": "昨日足球比赛结果: 主队3-1客队，精彩进球频频。",
        "电影": "本周热映电影: 《星际探险》、《城市之光》、《未来世界》等。"
    }
    
    # 查找最匹配的结果
    for key, value in search_results.items():
        if key in query:
            return value
    
    return "未找到相关信息，请尝试其他关键词。"

@tool
def translate_text(text: str, target_language: str = "英语") -> str:
    """翻译文本（模拟）"""
    # 这是一个模拟的翻译功能
    translations = {
        "你好": {"英语": "Hello", "法语": "Bonjour", "德语": "Hallo", "日语": "こんにちは"},
        "谢谢": {"英语": "Thank you", "法语": "Merci", "德语": "Danke", "日语": "ありがとう"},
        "再见": {"英语": "Goodbye", "法语": "Au revoir", "德语": "Auf Wiedersehen", "日语": "さようなら"}
    }
    
    # 简单模拟翻译
    for key, langs in translations.items():
        if key in text:
            if target_language in langs:
                return f"翻译结果: {langs[target_language]}"
    
    return f"无法翻译文本到{target_language}，请尝试其他语言或文本。"

@tool
def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def generate_image_description(prompt: str) -> str:
    """生成图像描述（模拟）"""
    # 这是一个模拟的图像生成描述功能
    descriptions = {
        "风景": "一幅美丽的风景画，展示了青山绿水，蓝天白云，阳光照耀下的田野和远处的山脉。",
        "城市": "一幅繁华的城市景观，高楼大厦林立，车水马龙，霓虹灯闪烁，人群熙熙攘攘。",
        "动物": "一只可爱的小动物，毛茸茸的，大眼睛炯炯有神，姿态灵动，背景是自然栖息地。",
        "人物": "一位面带微笑的人物肖像，表情生动，眼神传神，背景简洁，突出人物特点。",
        "抽象": "一幅充满活力的抽象画，色彩丰富，线条流畅，形状多变，给人以强烈的视觉冲击。"
    }
    
    # 查找最匹配的描述
    for key, value in descriptions.items():
        if key in prompt:
            return value
    
    return "根据您的提示，生成了一幅融合多种元素的图像，色彩丰富，构图和谐，细节丰富。"

# ============================================================================
# 可重用节点
# ============================================================================

def thinking_node(state: BaseState) -> Dict:
    """思考节点 - 让AI思考下一步行动"""
    # 获取消息历史
    messages = state["messages"]
    
    # 创建思考提示
    thinking_prompt = SystemMessage(content="""
    请思考当前对话的状态和用户的需求。分析以下几点:
    1. 用户的主要问题或需求是什么?
    2. 我目前掌握了哪些信息?
    3. 我还需要哪些信息?
    4. 下一步应该采取什么行动?
    
    以"思考:"开头，简洁地输出你的分析。
    """)
    
    # 添加思考提示到消息历史
    thinking_messages = messages + [thinking_prompt]
    
    # 调用模型进行思考
    model = ChatOpenAI(temperature=0)
    response = model.invoke(thinking_messages)
    
    # 记录思考结果但不返回给用户
    logger.info(f"思考结果: {response.content}")
    
    # 返回原始状态，不修改消息历史
    return {"messages": messages}

def tool_selection_node(state: ToolState) -> Dict:
    """工具选择节点 - 选择合适的工具"""
    # 获取消息历史
    messages = state["messages"]
    
    # 创建工具选择提示
    tool_selection_prompt = SystemMessage(content="""
    根据当前对话，选择最合适的工具来回答用户的问题。可用的工具有:
    - calculator: 计算数学表达式
    - search_web: 搜索网络信息
    - translate_text: 翻译文本
    - get_current_time: 获取当前时间
    - generate_image_description: 生成图像描述
    
    如果需要使用工具，请以JSON格式指定工具名称和参数，例如:
    {"tool": "calculator", "parameters": {"expression": "2+2"}}
    
    如果不需要使用工具，请回复: {"tool": null}
    """)
    
    # 添加工具选择提示到消息历史
    tool_messages = messages + [tool_selection_prompt]
    
    # 调用模型选择工具
    model = ChatOpenAI(temperature=0)
    response = model.invoke(tool_messages)
    
    try:
        # 解析响应中的工具选择
        tool_selection = json.loads(response.content)
        
        # 更新工具调用
        if tool_selection.get("tool"):
            tool_calls = state.get("tool_calls", [])
            tool_calls.append({
                "tool": tool_selection["tool"],
                "parameters": tool_selection.get("parameters", {})
            })
            
            return {
                "messages": messages,
                "tool_calls": tool_calls
            }
    except:
        logger.error(f"工具选择解析失败: {response.content}")
    
    # 如果没有选择工具或解析失败，返回原始状态
    return {"messages": messages, "tool_calls": state.get("tool_calls", [])}

def tool_execution_node(state: ToolState) -> Dict:
    """工具执行节点 - 执行选定的工具"""
    # 获取消息历史和工具调用
    messages = state["messages"]
    tool_calls = state.get("tool_calls", [])
    tool_results = state.get("tool_results", [])
    
    # 检查是否有未执行的工具调用
    if not tool_calls or len(tool_calls) <= len(tool_results):
        return state
    
    # 获取最新的工具调用
    latest_tool_call = tool_calls[len(tool_results)]
    tool_name = latest_tool_call["tool"]
    parameters = latest_tool_call["parameters"]
    
    # 执行工具
    result = None
    try:
        if tool_name == "calculator":
            result = calculator(parameters.get("expression", ""))
        elif tool_name == "search_web":
            result = search_web(parameters.get("query", ""))
        elif tool_name == "translate_text":
            result = translate_text(
                parameters.get("text", ""),
                parameters.get("target_language", "英语")
            )
        elif tool_name == "get_current_time":
            result = get_current_time()
        elif tool_name == "generate_image_description":
            result = generate_image_description(parameters.get("prompt", ""))
        else:
            result = f"未知工具: {tool_name}"
    except Exception as e:
        result = f"工具执行错误: {str(e)}"
    
    # 更新工具结果
    tool_results.append({
        "tool": tool_name,
        "parameters": parameters,
        "result": result
    })
    
    # 添加工具消息到历史
    tool_message = ToolMessage(
        content=result,
        tool_call_id=f"call_{len(tool_results)}",
        name=tool_name
    )
    updated_messages = messages + [tool_message]
    
    return {
        "messages": updated_messages,
        "tool_calls": tool_calls,
        "tool_results": tool_results
    }

def response_generation_node(state: BaseState) -> Dict:
    """响应生成节点 - 生成最终响应"""
    # 获取消息历史
    messages = state["messages"]
    
    # 创建响应生成提示
    response_prompt = SystemMessage(content="""
    根据之前的对话和任何工具调用结果，生成一个有帮助、信息丰富且友好的响应。
    确保你的回答直接解决用户的问题或需求。
    如果有工具提供的信息，请整合这些信息到你的回答中。
    """)
    
    # 添加响应生成提示到消息历史
    response_messages = messages + [response_prompt]
    
    # 调用模型生成响应
    model = ChatOpenAI(temperature=0.7)
    response = model.invoke(response_messages)
    
    # 添加AI响应到消息历史
    updated_messages = messages + [response]
    
    return {"messages": updated_messages}

def human_intervention_node(state: HumanInteractionState) -> Dict:
    """人类干预节点 - 处理需要人类干预的情况"""
    # 获取消息历史
    messages = state["messages"]
    requires_human = state.get("requires_human", False)
    
    if not requires_human:
        return state
    
    # 创建人类干预提示
    intervention_prompt = SystemMessage(content="""
    当前情况需要人类干预。请等待人类操作员的输入。
    """)
    
    # 添加干预提示到消息历史
    updated_messages = messages + [intervention_prompt]
    
    # 如果有人类输入，添加到消息历史
    human_input = state.get("human_input")
    if human_input:
        human_message = HumanMessage(content=f"[人类干预] {human_input}")
        updated_messages.append(human_message)
        
        # 重置人类干预状态
        return {
            "messages": updated_messages,
            "requires_human": False,
            "human_input": None
        }
    
    # 如果没有人类输入，保持需要人类干预的状态
    return {
        "messages": updated_messages,
        "requires_human": True
    }

def memory_update_node(state: MemoryState) -> Dict:
    """记忆更新节点 - 更新记忆状态"""
    # 获取消息历史和当前记忆
    messages = state["messages"]
    memory = state.get("memory", {})
    
    # 检查最新消息是否包含记忆更新指令
    if messages and isinstance(messages[-1], (HumanMessage, AIMessage)):
        latest_message = messages[-1].content
        
        # 检查是否是记忆指令
        if "记住" in latest_message:
            parts = latest_message.split("记住", 1)
            if len(parts) > 1:
                info = parts[1].strip()
                # 尝试提取键值对
                if ":" in info or "：" in info:
                    key, value = info.replace("：", ":").split(":", 1)
                    memory[key.strip()] = value.strip()
                else:
                    # 如果没有明确的键，使用时间戳作为键
                    memory[f"记忆_{int(time.time())}"] = info
    
    # 返回更新后的状态
    return {
        "messages": messages,
        "memory": memory
    }

def error_handling_node(state: BaseState) -> Dict:
    """错误处理节点 - 处理各种错误情况"""
    # 获取消息历史
    messages = state["messages"]
    
    # 检查是否有错误标记
    has_error = False
    error_message = None
    
    # 检查最新消息是否包含错误信息
    if messages and isinstance(messages[-1], (AIMessage, ToolMessage)):
        latest_message = messages[-1].content
        if "错误" in latest_message or "Error" in latest_message or "error" in latest_message:
            has_error = True
            error_message = latest_message
    
    if has_error:
        # 创建错误处理消息
        error_response = AIMessage(content=f"""
        抱歉，在处理您的请求时遇到了问题。错误信息: {error_message}
        
        请尝试以下解决方案:
        1. 重新表述您的问题
        2. 提供更多详细信息
        3. 尝试不同的请求
        
        如果问题持续存在，请联系支持团队。
        """)
        
        # 添加错误响应到消息历史
        updated_messages = messages + [error_response]
        
        return {"messages": updated_messages}
    
    # 如果没有错误，返回原始状态
    return state

def routing_node(state: Union[ToolState, HumanInteractionState]) -> Literal["thinking", "tool_selection", "tool_execution", "response_generation", "human_intervention", "memory_update", "error_handling", "end"]:
    """路由节点 - 决定下一步执行哪个节点"""
    # 获取消息历史
    messages = state["messages"]
    
    # 检查是否需要人类干预
    if state.get("requires_human", False):
        return "human_intervention"
    
    # 检查是否有未执行的工具调用
    tool_calls = state.get("tool_calls", [])
    tool_results = state.get("tool_results", [])
    if tool_calls and len(tool_calls) > len(tool_results):
        return "tool_execution"
    
    # 检查最新消息是否来自用户
    if messages and isinstance(messages[-1], HumanMessage):
        # 检查是否是记忆指令
        if "记住" in messages[-1].content:
            return "memory_update"
        
        # 默认先进行思考
        return "thinking"
    
    # 检查最新消息是否来自工具
    if messages and isinstance(messages[-1], ToolMessage):
        # 工具执行后生成响应
        return "response_generation"
    
    # 检查最新消息是否来自AI且包含错误
    if messages and isinstance(messages[-1], AIMessage):
        if "错误" in messages[-1].content or "Error" in messages[-1].content or "error" in messages[-1].content:
            return "error_handling"
        
        # 如果是思考结果，进行工具选择
        if messages[-1].content.startswith("思考:"):
            return "tool_selection"
        
        # 如果是普通AI响应，结束流程
        return "end"
    
    # 默认生成响应
    return "response_generation"

# ============================================================================
# 工作流构建器
# ============================================================================

class WorkflowBuilder:
    """工作流构建器 - 帮助构建LangGraph工作流"""
    
    def __init__(self, state_type: Type = FullState):
        """初始化工作流构建器"""
        self.state_type = state_type
        self.graph = StateGraph(state_type)
        self.nodes = {}
    
    def add_node(self, name: str, node_func: Callable):
        """添加节点到工作流"""
        self.nodes[name] = node_func
        self.graph.add_node(name, node_func)
        return self
    
    def add_edge(self, source: str, target: str):
        """添加边到工作流"""
        self.graph.add_edge(source, target)
        return self
    
    def add_conditional_edges(self, source: str, condition_func: Callable, targets: Dict[str, str]):
        """添加条件边到工作流"""
        self.graph.add_conditional_edges(source, condition_func, targets)
        return self
    
    def set_entry_point(self, node_name: str):
        """设置工作流入口点"""
        self.graph.set_entry_point(node_name)
        return self
    
    def build(self):
        """构建并返回工作流"""
        return self.graph.compile()

# ============================================================================
# 示例：构建完整工作流
# ============================================================================

def build_complete_workflow():
    """构建完整的工作流示例"""
    # 创建工作流构建器
    builder = WorkflowBuilder(FullState)
    
    # 添加节点
    builder.add_node("thinking", thinking_node)
    builder.add_node("tool_selection", tool_selection_node)
    builder.add_node("tool_execution", tool_execution_node)
    builder.add_node("response_generation", response_generation_node)
    builder.add_node("human_intervention", human_intervention_node)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("error_handling", error_handling_node)
    
    # 设置入口点
    builder.set_entry_point("routing")
    
    # 添加路由节点和条件边
    builder.add_node("routing", routing_node)
    builder.add_conditional_edges(
        "routing",
        lambda x: x,
        {
            "thinking": "thinking",
            "tool_selection": "tool_selection",
            "tool_execution": "tool_execution",
            "response_generation": "response_generation",
            "human_intervention": "human_intervention",
            "memory_update": "memory_update",
            "error_handling": "error_handling",
            "end": END
        }
    )
    
    # 添加其他边
    builder.add_edge("thinking", "routing")
    builder.add_edge("tool_selection", "routing")
    builder.add_edge("tool_execution", "routing")
    builder.add_edge("response_generation", "routing")
    builder.add_edge("human_intervention", "routing")
    builder.add_edge("memory_update", "routing")
    builder.add_edge("error_handling", "routing")
    
    # 构建工作流
    return builder.build()

# ============================================================================
# 辅助函数
# ============================================================================

def create_initial_state(system_prompt: str = None) -> Dict:
    """创建初始状态"""
    messages = []
    
    # 添加系统提示
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    return {
        "messages": messages,
        "memory": {},
        "tool_calls": [],
        "tool_results": [],
        "requires_human": False,
        "human_input": None
    }

def add_human_message(state: Dict, message: str) -> Dict:
    """添加人类消息到状态"""
    messages = state["messages"]
    messages.append(HumanMessage(content=message))
    
    return {**state, "messages": messages}

def trigger_human_intervention(state: Dict, reason: str = None) -> Dict:
    """触发人类干预"""
    messages = state["messages"]
    
    # 添加干预原因
    if reason:
        intervention_message = AIMessage(content=f"需要人类干预: {reason}")
        messages.append(intervention_message)
    
    return {**state, "messages": messages, "requires_human": True}

def provide_human_input(state: Dict, input_text: str) -> Dict:
    """提供人类输入"""
    return {**state, "human_input": input_text}

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 构建工作流
    workflow = build_complete_workflow()
    
    # 创建初始状态
    state = create_initial_state("你是一个智能助手，可以使用各种工具来帮助用户。")
    
    # 添加人类消息
    state = add_human_message(state, "请计算 123 + 456 是多少?")
    
    # 执行工作流
    for s in workflow.stream(state):
        if "messages" in s and s["messages"]:
            latest_message = s["messages"][-1]
            if isinstance(latest_message, (AIMessage, ToolMessage)):
                print(f"AI: {latest_message.content}")
    
    # 触发人类干预
    state = trigger_human_intervention(state, "需要额外信息")
    
    # 提供人类输入
    state = provide_human_input(state, "用户需要高精度计算")
    
    # 继续执行工作流
    for s in workflow.stream(state):
        if "messages" in s and s["messages"]:
            latest_message = s["messages"][-1]
            if isinstance(latest_message, (AIMessage, ToolMessage)):
                print(f"AI: {latest_message.content}") 