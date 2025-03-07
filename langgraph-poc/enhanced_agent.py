"""
增强的对话代理 - 整合意图管理、业务指标和RAG功能
"""

import os
import json
import logging
import uuid
import asyncio
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, TypedDict, Literal, Annotated, AsyncIterator
from dataclasses import dataclass, field, asdict
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedAgent")

# 尝试导入依赖
try:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint import MemorySaver, SQLiteSaver, Checkpointer
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

# 导入自定义模块
from intent_manager import IntentManager, Intent, Slot, IntentState
from business_metrics import BusinessMetricRegistry, create_default_metrics
from business_monitoring import (
    BusinessMetricsMonitor, MetricType, AlertLevel, MetricAlert, 
    get_default_monitor, create_default_metrics_monitor
)

try:
    from enhanced_rag import EnhancedRetriever, RetrievalResult, create_sample_knowledge_base
    from api_config import get_llm
    from integration import retrieve_with_memory, format_retrieval_results, update_memory_with_conversation
except ImportError as e:
    logger.warning(f"导入高级功能模块失败: {str(e)}。部分功能可能不可用。")

# 持久化配置
PERSISTENCE_TYPE = os.getenv("PERSISTENCE_TYPE", "memory")  # memory, sqlite, postgres
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "agent_states/sessions.db")
POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING", "")

# 全局变量
intent_manager = None
metrics_registry = None
metrics_monitor = None
SESSION_STATES = {}  # 会话状态缓存

# 获取Checkpointer实例
def get_checkpointer():
    """获取持久化器"""
    if PERSISTENCE_TYPE == "memory":
        logger.info("使用内存持久化")
        return Checkpointer(MemorySaver())
    elif PERSISTENCE_TYPE == "sqlite":
        logger.info(f"使用SQLite持久化: {SQLITE_DB_PATH}")
        # 确保目录存在
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        return Checkpointer(SQLiteSaver(SQLITE_DB_PATH))
    elif PERSISTENCE_TYPE == "postgres" and POSTGRES_CONN_STRING:
        logger.info("使用PostgreSQL持久化")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return Checkpointer(PostgresSaver(POSTGRES_CONN_STRING))
        except ImportError:
            logger.warning("未安装PostgreSQL支持，回退到SQLite")
            return Checkpointer(SQLiteSaver(SQLITE_DB_PATH))
    else:
        logger.warning(f"未知的持久化类型: {PERSISTENCE_TYPE}，回退到内存")
        return Checkpointer(MemorySaver())

# 对话状态定义
class EnhancedAgentState(TypedDict):
    """增强对话代理状态"""
    # 基础信息
    session_id: str              # 会话ID
    messages: List[Dict[str, Any]]  # 消息历史
    created_at: str              # 创建时间
    last_updated_at: str         # 最后更新时间
    
    # 意图和槽位
    current_intent: Optional[Dict[str, Any]]  # 当前意图
    intent_stack: List[Dict[str, Any]]        # 意图栈
    suspended_intents: List[Dict[str, Any]]   # 挂起的意图
    slots: Dict[str, Any]                    # 槽位值
    
    # 知识和检索
    retrieval_results: List[Dict[str, Any]]   # 检索结果
    
    # 流程控制
    status: str                  # 当前状态
    next_step: str               # 下一步
    error: Optional[str]         # 错误信息
    
    # 断点和人机交互
    breakpoints: Dict[str, Union[bool, Dict[str, Any]]]  # 断点设置，支持条件断点
    breakpoint_history: List[Dict[str, Any]]  # 断点触发历史
    breakpoint_callbacks: Dict[str, str]      # 断点回调设置
    human_input_required: bool    # 是否需要人工输入
    human_input: Optional[str]    # 人工输入
    confidence: float             # 置信度
    
    # 业务指标
    business_metrics: Dict[str, Any]  # 业务指标

# 工具定义
class Tools:
    """工具集合"""
    
    @staticmethod
    def calculator(expression: str) -> str:
        """计算数学表达式"""
        try:
            # 安全的数学表达式计算
            allowed_chars = set("0123456789+-*/().^ ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含无效字符，仅允许数字和+-*/().^"
            
            # 替换 ^ 为 **
            expression = expression.replace("^", "**")
            
            # 使用更安全的方式计算
            import ast
            import math
            
            def safe_eval(expr):
                return ast.literal_eval(expr)
            
            # 尝试计算
            result = safe_eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {str(e)}。请提供有效的数学表达式。"
    
    @staticmethod
    def get_weather(location: str) -> str:
        """获取指定位置的天气信息"""
        weather_data = {
            "北京": "晴朗，25°C",
            "上海": "多云，22°C",
            "广州": "雨，28°C",
            "深圳": "阴，27°C",
            "纽约": "晴朗，18°C",
            "伦敦": "雨，15°C",
            "东京": "晴朗，20°C",
            "巴黎": "多云，17°C",
            "悉尼": "晴朗，30°C"
        }
        return weather_data.get(location, f"没有{location}的天气信息")
    
    @staticmethod
    def translate(text: str, target_language: str) -> str:
        """翻译文本"""
        translations = {
            "中文": {
                "hello": "你好", 
                "goodbye": "再见", 
                "thank you": "谢谢", 
                "yes": "是", 
                "no": "否",
                "how are you": "你好吗"
            },
            "英文": {
                "你好": "hello", 
                "再见": "goodbye", 
                "谢谢": "thank you", 
                "是": "yes", 
                "否": "no",
                "你好吗": "how are you"
            }
        }
        
        if target_language in translations and text.lower() in translations[target_language]:
            return translations[target_language][text.lower()]
        
        return f"无法翻译'{text}'到{target_language}"

# 初始化全局组件
def initialize_components(load_existing: bool = False):
    """初始化组件
    
    Args:
        load_existing: 是否加载已存在的数据
        
    Returns:
        初始化的意图管理器
    """
    global intent_manager, metrics_registry, metrics_monitor
    
    # 初始化意图管理器
    if intent_manager is None:
        intent_manager = IntentManager()
        if load_existing and os.path.exists("data/intents.json"):
            intent_manager.load_from_file("data/intents.json")
        else:
            register_predefined_intents(intent_manager)
    
    # 初始化业务指标注册表
    if metrics_registry is None:
        metrics_registry = create_default_metrics()
    
    # 初始化业务指标监测器
    if metrics_monitor is None:
        metrics_monitor = create_default_metrics_monitor()
        
        # 配置告警处理器
        metrics_monitor.add_alert_handler(handle_metric_alert)
        
        # 启动后台监控
        try:
            metrics_monitor.start_background_monitoring()
        except Exception as e:
            logger.error(f"启动业务指标监控失败: {str(e)}")
    
    return intent_manager

def handle_metric_alert(alert: MetricAlert, value: Any) -> None:
    """处理指标告警
    
    Args:
        alert: 触发的告警
        value: 触发值
    """
    # 记录告警日志
    log_level = logging.WARNING
    if alert.level == AlertLevel.ERROR or alert.level == AlertLevel.CRITICAL:
        log_level = logging.ERROR
    
    logger.log(log_level, f"业务指标告警 [{alert.level.name}]: {alert.metric_name} = {value}, 条件: {alert.condition}")
    
    # 对于严重告警，可以发送通知或执行其他操作
    if alert.level == AlertLevel.CRITICAL:
        # TODO: 实现通知机制
        pass

# 初始化状态
def initialize_state() -> EnhancedAgentState:
    """初始化代理状态
    
    Returns:
        初始状态
    """
    # 确保组件已初始化
    initialize_components()
    
    # 记录会话创建指标
    if metrics_monitor:
        # 获取当前会话数
        current_sessions = metrics_monitor.get_metric_value("session_count") or 0
        # 记录新值
        metrics_monitor.record_metric("session_count", current_sessions + 1)
    
    return {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "last_updated_at": datetime.now().isoformat(),
        "current_intent": None,
        "intent_stack": [],
        "suspended_intents": [],
        "slots": {},
        "retrieval_results": [],
        "status": "initialized",
        "next_step": "user_input_processor",
        "error": None,
        "breakpoints": {},
        "breakpoint_history": [],
        "breakpoint_callbacks": {},
        "human_input_required": False,
        "human_input": None,
        "confidence": 1.0,
        "business_metrics": {
            "intent_counts": {},
            "error_counts": 0,
            "response_times": [],
            "slot_filling_attempts": 0,
            "successful_intents": 0,
            "failed_intents": 0,
            "human_interventions": 0,
            "knowledge_retrieval_counts": 0
        }
    }

# 注册预定义意图
def register_predefined_intents(intent_manager: IntentManager):
    """注册预定义意图"""
    # 财务报表查询意图
    financial_report_intent = Intent(
        name="query_financial_report",
        description="查询财务报表",
        keywords=["财务报表", "财报", "财务数据", "财务状况", "收入", "支出", "利润"],
        examples=[
            "我想查看上个季度的财务报表",
            "请提供公司的财务数据",
            "显示2023年第一季度的财报"
        ],
        handler="handle_financial_report",
        business_metric="financial_report",
        slots=[
            Slot(
                name="time_period",
                description="时间段",
                prompt="您想查询哪个时间段的财务报表？",
                examples=["2023年第一季度", "上个月", "去年"]
            ),
            Slot(
                name="report_type",
                description="报表类型",
                prompt="您需要哪种类型的财务报表？",
                examples=["资产负债表", "利润表", "现金流量表"]
            )
        ]
    )
    
    # 项目报表查询意图
    project_report_intent = Intent(
        name="query_project_report",
        description="查询项目报表",
        keywords=["项目报表", "项目进度", "项目状态", "项目资源", "项目预算"],
        examples=[
            "查看项目A的进度报表",
            "显示所有项目的资源使用情况",
            "我想了解项目B的预算执行情况"
        ],
        handler="handle_project_report",
        business_metric="project_report",
        slots=[
            Slot(
                name="project_name",
                description="项目名称",
                prompt="您想查询哪个项目的报表？",
                examples=["项目A", "项目B", "所有项目"]
            ),
            Slot(
                name="report_aspect",
                description="报表方面",
                prompt="您关注项目的哪个方面？",
                examples=["进度", "资源", "预算", "风险"]
            )
        ]
    )
    
    # 销售报表查询意图
    sales_report_intent = Intent(
        name="query_sales_report",
        description="查询销售报表",
        keywords=["销售报表", "销售数据", "销售情况", "销售业绩", "客户数量", "产品销量"],
        examples=[
            "查看本月的销售报表",
            "显示各区域的销售情况",
            "我想了解产品A的销售情况"
        ],
        handler="handle_sales_report",
        business_metric="sales_report",
        slots=[
            Slot(
                name="time_period",
                description="时间段",
                prompt="您想查询哪个时间段的销售报表？",
                examples=["本月", "上个季度", "今年"]
            ),
            Slot(
                name="product",
                description="产品",
                required=False,
                prompt="您想了解哪个产品的销售情况？",
                examples=["产品A", "产品B", "所有产品"]
            )
        ]
    )
    
    # 人力资源报表查询意图
    hr_report_intent = Intent(
        name="query_hr_report",
        description="查询人力资源报表",
        keywords=["人力资源报表", "人员配置", "招聘情况", "离职情况", "培训情况", "HR报表"],
        examples=[
            "查看部门的人员配置情况",
            "显示本季度的招聘情况",
            "我想了解员工培训的情况"
        ],
        handler="handle_hr_report",
        business_metric="hr_report",
        slots=[
            Slot(
                name="report_aspect",
                description="报表方面",
                prompt="您关注人力资源的哪个方面？",
                examples=["人员配置", "招聘", "离职", "培训"]
            ),
            Slot(
                name="department",
                description="部门",
                required=False,
                prompt="您想了解哪个部门的情况？",
                examples=["技术部", "销售部", "所有部门"]
            )
        ]
    )
    
    # 闲聊意图
    chitchat_intent = Intent(
        name="chitchat",
        description="闲聊",
        keywords=["你好", "您好", "嗨", "早上好", "下午好", "晚上好", "再见", "谢谢"],
        examples=[
            "你好，今天天气怎么样？",
            "你能做什么？",
            "讲个笑话"
        ],
        handler="handle_chitchat"
    )
    
    # 天气查询意图
    weather_intent = Intent(
        name="query_weather",
        description="查询天气",
        keywords=["天气", "气温", "下雨", "晴天", "温度"],
        examples=[
            "北京今天天气怎么样？",
            "上海会下雨吗？",
            "纽约的气温是多少？"
        ],
        handler="handle_weather",
        slots=[
            Slot(
                name="location",
                description="位置",
                prompt="您想查询哪个地方的天气？",
                examples=["北京", "上海", "纽约"]
            )
        ]
    )
    
    # 翻译意图
    translate_intent = Intent(
        name="translate",
        description="翻译文本",
        keywords=["翻译", "转换", "中文", "英文", "翻成"],
        examples=[
            "把'hello'翻译成中文",
            "将'谢谢'翻译成英文",
            "如何用英文说'你好'？"
        ],
        handler="handle_translate",
        slots=[
            Slot(
                name="text",
                description="文本",
                prompt="您想翻译什么文本？"
            ),
            Slot(
                name="target_language",
                description="目标语言",
                prompt="您想翻译成哪种语言？",
                examples=["中文", "英文"]
            )
        ]
    )
    
    # 计算意图
    calculator_intent = Intent(
        name="calculate",
        description="计算数学表达式",
        keywords=["计算", "算一下", "等于多少", "计算器"],
        examples=[
            "计算2+2",
            "3*4等于多少？",
            "计算(10+5)/3"
        ],
        handler="handle_calculator",
        slots=[
            Slot(
                name="expression",
                description="表达式",
                prompt="您想计算什么表达式？",
                examples=["2+2", "3*4", "(10+5)/3"]
            )
        ]
    )
    
    # 帮助意图
    help_intent = Intent(
        name="help",
        description="获取帮助",
        keywords=["帮助", "使用方法", "功能", "能做什么", "说明"],
        examples=[
            "你能做什么？",
            "请提供帮助信息",
            "有哪些功能？"
        ],
        handler="handle_help"
    )
    
    # 注册意图
    intent_manager.register_intent(financial_report_intent)
    intent_manager.register_intent(project_report_intent)
    intent_manager.register_intent(sales_report_intent)
    intent_manager.register_intent(hr_report_intent)
    intent_manager.register_intent(chitchat_intent)
    intent_manager.register_intent(weather_intent)
    intent_manager.register_intent(translate_intent)
    intent_manager.register_intent(calculator_intent)
    intent_manager.register_intent(help_intent)
    
    logger.info("已注册预定义意图")

# 创建全局组件
intent_manager, metric_registry, retriever = initialize_components(load_existing=False)
register_predefined_intents(intent_manager)

# 注册一些常用工具
tools = {
    "calculator": Tools.calculator,
    "get_weather": Tools.get_weather,
    "translate": Tools.translate
}

# ============== 节点函数 ==============

async def user_input_processor(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理用户输入，准备状态
    
    处理用户输入消息，更新状态，准备执行后续节点
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 记录开始时间
    start_time = time.time()
    
    # 复制状态
    new_state = state.copy()
    
    # 更新时间戳
    new_state["last_updated_at"] = datetime.now().isoformat()
    
    # 提取最后一条用户消息
    user_message = None
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_message = msg
            break
    
    if not user_message:
        # 没有找到用户消息，返回错误
        new_state["error"] = "未找到用户消息"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "missing_user_message",
            "component": "user_input_processor",
            "session_id": new_state.get("session_id", "unknown")
        })
        
        return new_state
    
    try:
        # 获取会话ID
        session_id = new_state.get("session_id", "default-session")
        
        # 记录消息指标
        record_business_metric("message_count", 1, {
            "session_id": session_id,
            "is_command": user_message.get("content", "").startswith("/"),
            "message_length": len(user_message.get("content", ""))
        })
        
        # 更新上下文
        try:
            new_state = await update_memory_with_conversation(session_id, new_state)
        except Exception as e:
            logger.warning(f"更新上下文记忆时出错: {str(e)}，继续处理")
            # 记录但不中断流程
            record_business_metric("error_count", 1, {
                "error_type": "memory_update_error",
                "error_message": str(e),
                "session_id": session_id
            })
        
        # 检查置信度（用于断点系统）
        message_content = user_message.get("content", "")
        
        # 计算更全面的置信度指标
        confidence = 0.8  # 默认中等置信度
        
        # 基于消息长度调整置信度
        if len(message_content) < 5:
            confidence -= 0.3  # 极短消息，可能缺乏上下文
        elif len(message_content) < 10:
            confidence -= 0.2  # 较短消息
        elif len(message_content) > 100:
            confidence += 0.1  # 详细消息，可能含有更多信息
        
        # 基于上下文连续性调整置信度
        if len(new_state["messages"]) <= 1:
            confidence -= 0.1  # 首次交互，缺乏上下文
        
        # 基于当前意图调整置信度
        if new_state.get("current_intent"):
            # 已有意图上下文，提高置信度
            confidence += 0.1
            
            # 如果有部分填充的槽位，进一步提高置信度
            if new_state.get("slots") and len(new_state["slots"]) > 0:
                confidence += 0.1
        
        # 确保置信度在合理范围内
        confidence = max(0.1, min(1.0, confidence))
        
        # 更新状态中的置信度
        new_state["confidence"] = confidence
        
        # 记录置信度指标
        record_business_metric("confidence_level", confidence, {
            "session_id": session_id,
            "message_length": len(message_content),
            "has_context": len(new_state["messages"]) > 1,
            "has_intent": bool(new_state.get("current_intent"))
        })
        
        # 设置下一步处理
        if new_state.get("human_input"):
            # 如果有人工输入，优先处理
            new_state["next_step"] = "process_human_input"
        elif new_state.get("error"):
            # 如果有错误，标记状态
            new_state["status"] = "error"
            new_state["next_step"] = "fallback_handler"
        else:
            # 默认流程
            new_state["status"] = "input_processed"
            new_state["next_step"] = "check_return_to_previous"
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 记录处理时间
        record_business_metric("response_time", processing_time, {
            "component": "user_input_processor",
            "session_id": session_id
        })
        
        return new_state
        
    except Exception as e:
        logger.error(f"处理用户输入时出错: {str(e)}")
        new_state["error"] = f"处理用户输入失败: {str(e)}"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "exception",
            "component": "user_input_processor",
            "error_message": str(e),
            "session_id": new_state.get("session_id", "unknown")
        })
        
        return new_state

async def intent_recognizer(state: EnhancedAgentState) -> EnhancedAgentState:
    """意图识别节点
    
    识别用户当前的意图
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 记录开始时间（用于计算耗时）
    start_time = time.time()
    
    new_state = state.copy()
    
    # 如果已经有了当前意图，跳过
    if new_state.get("current_intent") and new_state.get("status") != "need_intent":
        return new_state
    
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有找到用户消息
        new_state["error"] = "无法找到用户消息"
        new_state["status"] = "error"
        
        # 记录错误指标
        if metrics_monitor:
            metrics_monitor.record_metric("error_count", 1, {
                "error_type": "missing_user_message",
                "component": "intent_recognizer"
            })
            
        return new_state
    
    try:
        # 获取意图管理器
        intent_manager = initialize_components()
        
        # 初始化意图识别结果
        intent_result = None
        
        # 首先检查用户是否明确指定了意图
        explicit_intent = check_explicit_intent(user_message)
        if explicit_intent:
            intent_name = explicit_intent
            # 寻找匹配的预定义意图
            for intent in intent_manager.intents:
                if intent.name.lower() == intent_name.lower():
                    intent_result = {
                        "name": intent.name,
                        "confidence": 0.95,  # 明确指定的意图给高置信度
                        "slots": {},
                        "parameters": intent.parameters
                    }
                    
                    # 记录显式意图指标
                    if metrics_monitor:
                        metrics_monitor.record_metric(f"intent_{intent_name}_count", 1, {
                            "method": "explicit",
                            "confidence": 0.95
                        })
                    
                    break
        
        # 如果没有明确指定，使用意图识别
        if not intent_result:
            # 使用异步LLM进行意图识别
            llm = get_async_llm(temperature=0)
            
            # 准备系统提示
            system_prompt = """
你是一个专业的意图识别AI。你的任务是分析用户输入并识别出最可能的意图。
可用的意图有：

{intent_descriptions}

请从以上意图中选择一个最匹配用户输入的，并提取相关参数。回复格式必须是JSON：

{{
    "intent": "意图名称",
    "confidence": 置信度（0-1之间的浮点数）,
    "slots": {{
        "参数名1": "参数值1",
        "参数名2": "参数值2"
    }}
}}

如果你不确定用户意图，请将"intent"设为"unknown"，并给出较低的置信度。
"""
            
            # 准备意图描述
            intent_descriptions = "\n".join([
                f"- {intent.name}: {intent.description}. 参数: {', '.join([f'{p} ({t})' for p, t in intent.parameters.items()])}"
                for intent in intent_manager.intents
            ])
            
            formatted_prompt = system_prompt.format(intent_descriptions=intent_descriptions)
            
            # 调用LLM进行意图识别
            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 使用异步调用
            response = await llm.ainvoke(messages)
            response_content = response.content
            
            try:
                # 解析JSON响应
                intent_data = json.loads(response_content)
                
                # 寻找匹配的预定义意图
                intent_name = intent_data.get("intent")
                confidence = intent_data.get("confidence", 0.0)
                slots = intent_data.get("slots", {})
                
                if intent_name and intent_name.lower() != "unknown":
                    for intent in intent_manager.intents:
                        if intent.name.lower() == intent_name.lower():
                            intent_result = {
                                "name": intent.name,
                                "confidence": confidence,
                                "slots": slots,
                                "parameters": intent.parameters
                            }
                            
                            # 记录LLM识别的意图指标
                            if metrics_monitor:
                                metrics_monitor.record_metric(f"intent_{intent_name}_count", 1, {
                                    "method": "llm",
                                    "confidence": confidence
                                })
                            
                            break
            except json.JSONDecodeError:
                logger.error(f"无法解析意图识别结果: {response_content}")
                new_state["error"] = "意图识别失败：无法解析结果"
                new_state["status"] = "error"
                
                # 记录解析错误指标
                if metrics_monitor:
                    metrics_monitor.record_metric("error_count", 1, {
                        "error_type": "json_parse_error",
                        "component": "intent_recognizer"
                    })
                
                return new_state
        
        # 设置默认意图
        if not intent_result:
            # 如果没有识别出意图，使用fallback
            intent_result = {
                "name": "fallback",
                "confidence": 0.3,
                "slots": {},
                "parameters": {}
            }
            
            # 记录fallback意图指标
            if metrics_monitor:
                metrics_monitor.record_metric("intent_fallback_count", 1, {
                    "confidence": 0.3
                })
        
        # 更新状态
        new_state["current_intent"] = intent_result
        new_state["confidence"] = intent_result["confidence"]
        
        # 如果当前槽位为空初始化槽位
        if "slots" not in new_state or not new_state["slots"]:
            new_state["slots"] = {}
        
        # 合并已识别的槽位到状态中
        for slot_name, slot_value in intent_result.get("slots", {}).items():
            new_state["slots"][slot_name] = slot_value
        
        # 检查是否有足够的置信度
        if intent_result["confidence"] < 0.5:
            # 置信度不足，需要确认
            new_state["status"] = "need_confirmation"
            new_state["next_step"] = "confirm_intent"
        else:
            # 置信度足够，继续处理
            new_state["status"] = "intent_recognized"
            new_state["next_step"] = "slot_filling"
        
        # 更新业务指标
        # 记录意图识别成功率
        intent_name = intent_result["name"]
        confidence = intent_result["confidence"]
        
        if "business_metrics" not in new_state:
            new_state["business_metrics"] = {}
            
        if "intent_counts" not in new_state["business_metrics"]:
            new_state["business_metrics"]["intent_counts"] = {}
            
        # 增加意图计数
        if intent_name in new_state["business_metrics"]["intent_counts"]:
            new_state["business_metrics"]["intent_counts"][intent_name] += 1
        else:
            new_state["business_metrics"]["intent_counts"][intent_name] = 1
        
        # 计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 记录响应时间指标
        if metrics_monitor:
            metrics_monitor.record_metric("response_time", processing_time, {
                "component": "intent_recognizer",
                "intent": intent_name,
                "confidence": confidence
            })
            
            # 记录意图成功率指标
            if confidence >= 0.5:
                success_rate = 100.0  # 100%成功
            else:
                success_rate = confidence * 100.0  # 按置信度比例
                
            metrics_monitor.record_metric("intent_success_rate", success_rate)
        
        return new_state
        
    except Exception as e:
        logger.error(f"意图识别过程中出错: {str(e)}")
        new_state["error"] = f"意图识别失败: {str(e)}"
        new_state["status"] = "error"
        
        # 记录错误指标
        if metrics_monitor:
            metrics_monitor.record_metric("error_count", 1, {
                "error_type": "exception",
                "component": "intent_recognizer",
                "error_message": str(e)
            })
            
        return new_state

async def check_return_to_previous(state: EnhancedAgentState) -> EnhancedAgentState:
    """检查是否返回到之前的对话
    
    检查用户是否想要返回到之前的主题，例如"我们回到之前的话题"
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    new_state = state.copy()
    
    # 获取挂起的意图列表
    suspended_intents = new_state.get("suspended_intents", [])
    
    # 如果没有挂起的意图，无需检查
    if not suspended_intents:
        new_state["next_step"] = "intent_recognizer"
        return new_state
    
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有找到用户消息
        new_state["error"] = "无法找到用户消息"
        new_state["status"] = "error"
        return new_state
    
    try:
        # 检查用户是否想要返回之前的话题
        # 使用异步LLM
        llm = get_async_llm(temperature=0)
        
        # 准备系统提示
        system_prompt = f"""
分析用户消息是否表示想要返回到之前未完成的话题。

你有以下未完成的话题：
{json.dumps([intent.get('name', 'unknown') for intent in suspended_intents], ensure_ascii=False, indent=2)}

判断用户当前的消息是否表示想要返回到之前的某个话题。
如果是，返回JSON格式的话题索引，例如 {{"return_to": 0}} 表示返回到第一个话题。
如果不是，返回 {{"return_to": null}}
"""
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        try:
            # 解析结果
            result = json.loads(response.content)
            return_index = result.get("return_to")
            
            if return_index is not None and isinstance(return_index, int) and 0 <= return_index < len(suspended_intents):
                # 用户想要返回之前的话题
                resumed_intent = suspended_intents.pop(return_index)
                
                # 恢复为当前意图
                new_state["current_intent"] = resumed_intent
                
                # 更新状态
                new_state["status"] = "intent_resumed"
                new_state["next_step"] = "slot_filler"
                
                # 记录日志
                logger.info(f"恢复之前的意图: {resumed_intent['name']}")
                
                # 添加系统消息
                new_state["messages"].append({
                    "role": "system",
                    "content": f"返回之前的话题: {resumed_intent['name']}"
                })
                
                # 更新挂起的意图列表
                new_state["suspended_intents"] = suspended_intents
                
                return new_state
                
        except json.JSONDecodeError as e:
            logger.error(f"解析返回检查结果时出错: {str(e)}, 响应内容: {response.content}")
            # 继续处理，不中断流程
        
        # 默认继续意图识别
        new_state["next_step"] = "intent_recognizer"
        return new_state
        
    except Exception as e:
        logger.error(f"检查返回之前话题时出错: {str(e)}")
        new_state["error"] = f"检查返回之前话题失败: {str(e)}"
        new_state["status"] = "error"
        return new_state

async def slot_filler(state: EnhancedAgentState) -> EnhancedAgentState:
    """填充槽位节点
    
    从用户消息中提取所需的槽位值
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 记录开始时间
    start_time = time.time()
    
    # 使用图流式处理消息
    async for event in compiled_graph.astream(state):
        # 提取当前状态
        current_state = event.get("state", {})
        
        # 检查是否正在等待人工输入
        if current_state.get("human_input_required", False):
            # 生成调试信息
            debug_info = generate_debug_info(current_state)
            
            # 返回等待人工输入的状态
            yield {
                "session_id": session_id,
                "message": "等待人工输入",
                "type": "waiting_for_human",
                "debug_info": debug_info,
                "waiting_for_human": True,
                "state": current_state
            }
            
            # 更新会话状态
            SESSION_STATES[session_id] = current_state
            
            # 保存会话状态
            await save_session_state(session_id, current_state)
            return
        
        # 获取最后一条助手消息
        last_assistant_message = None
        for msg in reversed(current_state.get("messages", [])):
            if msg.get("role") in ["assistant", "ai"]:
                last_assistant_message = msg.get("content", "")
                break
        
        if last_assistant_message:
            # 生成调试信息
            debug_info = generate_debug_info(current_state)
            
            # 返回最新的消息
            yield {
                "session_id": session_id,
                "message": last_assistant_message,
                "type": "message",
                "debug_info": debug_info,
                "node": event.get("node", "unknown")
            }
        
        # 更新会话状态
        SESSION_STATES[session_id] = current_state
    
    # 记录处理结束时间并计算耗时
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 从最终状态获取最后的消息
    final_state = SESSION_STATES.get(session_id, {})
    
    # 获取最后一条助手消息
    last_assistant_message = None
    for msg in reversed(final_state.get("messages", [])):
        if msg.get("role") in ["assistant", "ai"]:
            last_assistant_message = msg.get("content", "")
            break
    
    if not last_assistant_message:
        last_assistant_message = "处理完成，但没有生成响应。"
    
    # 生成最终的调试信息
    final_debug_info = generate_debug_info(final_state)
    final_debug_info["processing_time"] = f"{processing_time:.2f}秒"
    
    # 返回最终结果
    yield {
        "session_id": session_id,
        "message": last_assistant_message,
        "type": "final",
        "debug_info": final_debug_info,
        "processing_time": processing_time
    }
    
    # 保存会话状态
    await save_session_state(session_id, final_state)

# 同步处理消息（用于向后兼容）
def process_message(session_id: str, message: str) -> Dict[str, Any]:
    """同步处理消息"""
    # 使用asyncio运行异步函数
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(aprocess_message(session_id, message))

# 导出代理
agent = {
    "process_message": process_message,
    "aprocess_message": aprocess_message,
    "compiled_graph": compiled_graph
} 

async def save_session_state(session_id: str, state: Dict[str, Any]) -> None:
    """保存会话状态到持久化存储
    
    Args:
        session_id: 会话ID
        state: 会话状态
    """
    try:
        # 更新内存缓存
        SESSION_STATES[session_id] = state
        
        # 尝试保存到持久化存储
        checkpointer = get_checkpointer()
        await checkpointer.put(session_id, state)
        logger.debug(f"会话状态已保存: {session_id}")
    except Exception as e:
        logger.error(f"保存会话状态失败: {str(e)}")
        # 即使保存失败，我们仍然保留内存中的状态

def check_explicit_intent(user_message: str) -> Optional[str]:
    """检查用户是否明确指定了意图
    
    检查用户消息中是否包含明确的意图指令，如"我想查询天气"
    
    Args:
        user_message: 用户消息
        
    Returns:
        识别出的意图名称，如果没有明确指定则返回None
    """
    # 定义意图关键词映射
    intent_keywords = {
        "计算": "calculate",
        "算一下": "calculate",
        "计算器": "calculate",
        "天气": "query_weather",
        "查询天气": "query_weather",
        "天气预报": "query_weather",
        "翻译": "translate",
        "帮我翻译": "translate",
        "聊天": "chitchat",
        "帮助": "help",
        "使用帮助": "help",
        "财务报表": "query_financial_report",
        "财务报告": "query_financial_report",
        "项目报表": "query_project_report",
        "项目报告": "query_project_report",
        "销售报表": "query_sales_report",
        "销售报告": "query_sales_report",
        "人力资源报表": "query_hr_report",
        "人员报表": "query_hr_report",
        "HR报表": "query_hr_report"
    }
    
    # 清理用户消息
    cleaned_message = user_message.lower().strip()
    
    # 检查是否有明确的意图指示
    # 1. 检查是否有"我想要..."格式
    explicit_patterns = [
        r"我想要(.*)",
        r"我需要(.*)",
        r"我想(.*)",
        r"请帮我(.*)",
        r"请(.*)",
        r"帮我(.*)"
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, cleaned_message)
        if match:
            action = match.group(1).strip()
            # 检查操作是否匹配意图关键词
            for keyword, intent in intent_keywords.items():
                if keyword in action:
                    return intent
    
    # 2. 直接检查关键词
    for keyword, intent in intent_keywords.items():
        if keyword in cleaned_message:
            return intent
    
    # 没有找到明确的意图
    return None

def generate_debug_info(state: EnhancedAgentState) -> Dict[str, Any]:
    """生成丰富的调试信息
    
    为前端界面提供详细的调试数据，用于可视化和交互
    
    Args:
        state: 当前状态
        
    Returns:
        包含调试信息的字典
    """
    debug_info = {
        "status": state.get("status", "unknown"),
        "current_node": state.get("status", "unknown"),
        "next_step": state.get("next_step", ""),
        "current_intent": state.get("current_intent", {}).get("name", "无") if state.get("current_intent") else "无",
        "confidence": state.get("confidence", 0),
        "error": state.get("error", None),
        "human_input_required": state.get("human_input_required", False),
        "breakpoints": {
            # 格式化断点信息，使其更易于前端理解
            node: {
                "active": True if isinstance(config, bool) and config else 
                        (True if isinstance(config, dict) else False),
                "type": "simple" if isinstance(config, bool) else 
                        ("conditional" if isinstance(config, dict) and "condition" in config else "unknown"),
                "condition": config.get("condition", "") if isinstance(config, dict) else None,
                "description": config.get("description", "") if isinstance(config, dict) else None
            }
            for node, config in state.get("breakpoints", {}).items()
        },
        "breakpoint_history": state.get("breakpoint_history", [])[-5:],  # 只返回最近的5条记录
        "breakpoint_callbacks": state.get("breakpoint_callbacks", {}),
        "slots_filled": len(state.get("slots", {})),
        "slot_details": state.get("slots", {}),
        "retrieval_count": len(state.get("retrieval_results", [])),
        "message_count": len(state.get("messages", [])),
        "session_age": calculate_session_age(state.get("created_at", "")),
        "graph_nodes": get_available_nodes(),
        "active_breakpoints_count": sum(
            1 for config in state.get("breakpoints", {}).values() 
            if (isinstance(config, bool) and config) or (isinstance(config, dict) and config.get("condition"))
        )
    }
    
    return debug_info

def calculate_session_age(created_at: str) -> str:
    """计算会话年龄
    
    根据创建时间计算会话已存在的时间
    
    Args:
        created_at: ISO格式的创建时间
        
    Returns:
        格式化的会话年龄字符串
    """
    if not created_at:
        return "未知"
    
    try:
        create_time = datetime.fromisoformat(created_at)
        now = datetime.now()
        delta = now - create_time
        
        if delta.days > 0:
            return f"{delta.days}天 {delta.seconds // 3600}小时"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}小时 {(delta.seconds % 3600) // 60}分钟"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}分钟"
        else:
            return f"{delta.seconds}秒"
    except Exception:
        return "未知"

def get_available_nodes() -> List[str]:
    """获取可用的图节点
    
    返回状态图中所有可用的节点名称，用于断点设置
    
    Returns:
        节点名称列表
    """
    # 硬编码关键节点，也可以通过图的编译信息动态获取
    return [
        "user_input_processor",
        "check_return_to_previous",
        "intent_recognizer",
        "slot_filler",
        "knowledge_retriever",
        "calculator_handler",
        "weather_handler",
        "translate_handler",
        "chitchat_handler",
        "help_handler",
        "fallback_handler",
        "financial_report_handler",
        "project_report_handler",
        "sales_report_handler",
        "hr_report_handler"
    ]

def record_business_metric(metric_name: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """记录业务指标
    
    将指标记录到监测系统中
    
    Args:
        metric_name: 指标名称
        value: 指标值
        metadata: 元数据
    """
    # 使用全局监测器
    global metrics_monitor
    
    if metrics_monitor:
        try:
            metrics_monitor.record_metric(metric_name, value, metadata or {})
            logger.debug(f"已记录业务指标: {metric_name} = {value}")
        except Exception as e:
            logger.warning(f"记录业务指标失败: {str(e)}")


def update_state_metrics(state: EnhancedAgentState, metrics_update: Dict[str, Any]) -> EnhancedAgentState:
    """更新状态中的业务指标
    
    将指标更新到状态对象中
    
    Args:
        state: 当前状态
        metrics_update: 要更新的指标
        
    Returns:
        更新后的状态
    """
    # 复制状态
    new_state = state.copy()
    
    # 确保存在业务指标字段
    if "business_metrics" not in new_state:
        new_state["business_metrics"] = {}
    
    # 更新指标
    for key, value in metrics_update.items():
        if key in new_state["business_metrics"] and isinstance(new_state["business_metrics"][key], dict) and isinstance(value, dict):
            # 如果是嵌套字典，进行深度更新
            new_state["business_metrics"][key].update(value)
        else:
            # 否则直接替换
            new_state["business_metrics"][key] = value
    
    return new_state

async def aprocess_message(session_id: str, message: str) -> Dict[str, Any]:
    """异步处理用户消息
    
    处理用户输入的消息，执行对话代理的完整流程
    
    Args:
        session_id: 会话ID
        message: 用户消息
        
    Returns:
        处理结果，包括会话状态和响应消息
    """
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 处理用户输入
        state = await user_input_processor(await intent_recognizer(await check_return_to_previous(await slot_filler(await process_message(session_id, message)))))
        
        # 生成响应消息
        response_message = state["messages"][-1]["content"] if state["messages"] else "处理完成，但没有生成响应。"
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 更新业务指标
        record_business_metric("response_time", processing_time, {
            "session_id": session_id,
            "message_length": len(message),
            "component": "full_process"
        })
        
        # 记录消息计数
        record_business_metric("message_count", 1, {
            "session_id": session_id,
            "is_command": message.startswith("/")
        })
        
        # 记录意图相关指标
        if state.get("current_intent"):
            intent_name = state["current_intent"].get("name", "unknown")
            record_business_metric(f"intent_{intent_name}_count", 1, {
                "session_id": session_id,
                "confidence": state.get("confidence", 0)
            })
        
        # 检查是否有错误
        if state.get("error"):
            record_business_metric("error_count", 1, {
                "error_type": "process_error",
                "error_message": state.get("error"),
                "session_id": session_id
            })
        
        # 生成调试信息
        debug_info = generate_debug_info(state)
        
        # 添加业务指标摘要到调试信息
        if metrics_monitor:
            debug_info["metrics_summary"] = {
                "session_count": metrics_monitor.get_metric_value("session_count"),
                "message_count": metrics_monitor.get_metric_value("message_count"),
                "error_count": metrics_monitor.get_metric_value("error_count"),
                "response_time_avg": metrics_monitor.get_metric_value("response_time", "avg", 3600),
                "intent_success_rate": metrics_monitor.get_metric_value("intent_success_rate")
            }
        
        return {
            "session_id": session_id,
            "message": response_message,
            "type": "final",
            "debug_info": debug_info,
            "processing_time": processing_time
        }
    except Exception as e:
        # 记录异常
        logger.error(f"处理消息时出错: {str(e)}")
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "exception",
            "error_message": str(e),
            "component": "aprocess_message",
            "session_id": session_id
        })
        
        return {
            "session_id": session_id,
            "message": f"处理消息时出错: {str(e)}",
            "type": "error",
            "error": str(e),
            "debug_info": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "error_time": time.time() - start_time
            }
        }

def generate_metrics_dashboard() -> Dict[str, Any]:
    """生成业务指标摘要仪表板
    
    生成包含各种业务指标的摘要报告
    
    Returns:
        包含指标摘要的字典
    """
    # 检查监测器是否可用
    if not metrics_monitor:
        return {"error": "监测器不可用"}
    
    # 获取当前时间
    now = time.time()
    
    # 定义时间窗口
    windows = {
        "last_hour": 3600,
        "last_day": 86400,
        "last_week": 604800
    }
    
    # 生成基础指标
    dashboard = {
        "timestamp": now,
        "formatted_time": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
        "basic_metrics": {},
        "intent_metrics": {},
        "error_metrics": {},
        "performance_metrics": {}
    }
    
    # 基础指标
    dashboard["basic_metrics"] = {
        "session_count": metrics_monitor.get_metric_value("session_count", "last"),
        "message_count": metrics_monitor.get_metric_value("message_count", "sum", windows["last_day"]),
        "messages_per_hour": metrics_monitor.get_metric_value("message_count", "sum", windows["last_hour"])
    }
    
    # 意图指标
    intent_counts = {}
    for intent in ["calculate", "query_weather", "translate", "chitchat", "help",
                  "query_financial_report", "query_project_report", 
                  "query_sales_report", "query_hr_report", "fallback"]:
        count = metrics_monitor.get_metric_value(f"intent_{intent}_count", "sum", windows["last_day"]) or 0
        if count > 0:
            intent_counts[intent] = count
    
    # 计算百分比
    total_intents = sum(intent_counts.values())
    intent_percentages = {}
    if total_intents > 0:
        for intent, count in intent_counts.items():
            intent_percentages[intent] = round((count / total_intents) * 100, 1)
    
    dashboard["intent_metrics"] = {
        "counts": intent_counts,
        "percentages": intent_percentages,
        "success_rate": metrics_monitor.get_metric_value("intent_success_rate", "avg", windows["last_day"]) or 0,
        "total_intents": total_intents
    }
    
    # 错误指标
    error_count = metrics_monitor.get_metric_value("error_count", "sum", windows["last_day"]) or 0
    
    dashboard["error_metrics"] = {
        "error_count": error_count,
        "error_rate": round((error_count / max(1, dashboard["basic_metrics"]["message_count"])) * 100, 2)
    }
    
    # 性能指标
    dashboard["performance_metrics"] = {
        "avg_response_time": metrics_monitor.get_metric_value("response_time", "avg", windows["last_day"]) or 0,
        "max_response_time": metrics_monitor.get_metric_value("response_time", "max", windows["last_day"]) or 0,
        "p95_response_time": "N/A"  # 目前不支持百分位数计算
    }
    
    # 生成文本摘要
    summary_lines = [
        f"业务指标摘要 ({dashboard['formatted_time']})",
        f"总会话数: {dashboard['basic_metrics']['session_count']}",
        f"24小时消息量: {dashboard['basic_metrics']['message_count']}",
        f"最近一小时消息量: {dashboard['basic_metrics']['messages_per_hour']}",
        f"平均响应时间: {dashboard['performance_metrics']['avg_response_time']:.2f}秒",
        f"意图识别成功率: {dashboard['intent_metrics']['success_rate']:.1f}%",
        f"错误率: {dashboard['error_metrics']['error_rate']}%"
    ]
    
    # 添加最常用意图
    if intent_percentages:
        top_intents = sorted(intent_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
        intent_summary = ", ".join([f"{intent}: {pct}%" for intent, pct in top_intents])
        summary_lines.append(f"最常用意图: {intent_summary}")
    
    dashboard["text_summary"] = "\n".join(summary_lines)
    
    return dashboard


async def get_metrics_report(session_id: str) -> Dict[str, Any]:
    """获取业务指标报告
    
    生成并返回业务指标报告
    
    Args:
        session_id: 会话ID
        
    Returns:
        包含指标报告的字典
    """
    # 生成仪表板
    dashboard = generate_metrics_dashboard()
    
    # 创建快照
    snapshot_path = ""
    if metrics_monitor:
        snapshot_path = metrics_monitor.create_snapshot()
    
    # 记录报告生成
    record_business_metric("metrics_report_generated", 1, {
        "session_id": session_id,
        "snapshot": os.path.basename(snapshot_path) if snapshot_path else "none"
    })
    
    return {
        "dashboard": dashboard,
        "snapshot": os.path.basename(snapshot_path) if snapshot_path else None,
        "session_id": session_id,
        "timestamp": time.time()
    }

def process_human_input(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理人工输入
    
    处理人工输入的命令或消息，支持调试命令和断点管理
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 复制状态
    new_state = state.copy()
    
    # 检查是否有人工输入
    if not new_state.get("human_input"):
        return new_state
    
    # 处理人工输入
    human_input = new_state["human_input"]
    new_state["human_input"] = None
    new_state["human_input_required"] = False
    new_state["status"] = "processing_human_input"
    
    # 添加人工消息（除非是自动回调）
    if not any(msg.get("content") == f"自动执行断点回调: {human_input}" for msg in new_state["messages"]):
        new_state["messages"].append({
            "role": "system",
            "content": f"人工干预: {human_input}"
        })
    
    # 记录人工干预指标
    record_business_metric("human_interventions", 1, {
        "session_id": new_state.get("session_id", "unknown"),
        "input": human_input[:50]  # 只记录前50个字符，避免敏感信息
    })
    
    # 解析指令
    if human_input.startswith("/"):
        parts = human_input[1:].split(maxsplit=2)
        command = parts[0].lower() if parts else ""
        
        if command == "continue":
            # 继续执行
            new_state["status"] = new_state.get("next_step", "user_input_processor")
            new_state["messages"].append({
                "role": "system",
                "content": "继续执行流程。"
            })
        elif command == "retry":
            # 重试当前步骤
            new_state["status"] = new_state.get("next_step", "user_input_processor")
            new_state["messages"].append({
                "role": "system",
                "content": "重试当前步骤。"
            })
        elif command == "abort":
            # 中止当前意图
            if new_state["current_intent"]:
                new_state["messages"].append({
                    "role": "system",
                    "content": f"中止当前意图: {new_state['current_intent'].get('name')}。"
                })
                new_state["current_intent"] = None
                new_state["status"] = "intent_recognizer"
        elif command == "breakpoint":
            # 设置断点
            if len(parts) > 1:
                node_name = parts[1]
                action = parts[2] if len(parts) > 2 else "toggle"
                
                if action.lower() == "on":
                    new_state["breakpoints"][node_name] = True
                    new_state["messages"].append({
                        "role": "system",
                        "content": f"断点已设置: {node_name}"
                    })
                elif action.lower() == "off":
                    new_state["breakpoints"][node_name] = False
                    new_state["messages"].append({
                        "role": "system",
                        "content": f"断点已移除: {node_name}"
                    })
                else:  # toggle
                    new_state["breakpoints"][node_name] = not new_state["breakpoints"].get(node_name, False)
                    status = "设置" if new_state["breakpoints"][node_name] else "移除"
                    new_state["messages"].append({
                        "role": "system",
                        "content": f"断点已{status}: {node_name}"
                    })
            else:
                new_state["messages"].append({
                    "role": "system",
                    "content": "设置断点需要指定节点名和操作。格式: /breakpoint <node_name> on|off"
                })
        elif command == "condition":
            # 设置条件断点
            if len(parts) >= 3:
                node_name = parts[1]
                condition_expr = parts[2]
                
                # 创建条件断点配置
                condition_config = {
                    "condition": condition_expr,
                    "description": f"用户设置的条件断点: {condition_expr}",
                    "created_at": datetime.now().isoformat()
                }
                
                new_state["breakpoints"][node_name] = condition_config
                new_state["messages"].append({
                    "role": "system",
                    "content": f"条件断点已设置: {node_name}, 条件: {condition_expr}"
                })
            else:
                new_state["messages"].append({
                    "role": "system",
                    "content": "设置条件断点需要指定节点名和条件表达式。格式: /condition <node_name> <条件表达式>"
                })
        elif command == "callback":
            # 设置断点回调
            if len(parts) >= 3:
                node_name = parts[1]
                callback_command = parts[2]
                
                new_state["breakpoint_callbacks"][node_name] = callback_command
                new_state["messages"].append({
                    "role": "system",
                    "content": f"断点回调已设置: {node_name}, 命令: {callback_command}"
                })
            else:
                new_state["messages"].append({
                    "role": "system",
                    "content": "设置断点回调需要指定节点名和回调命令。格式: /callback <node_name> <回调命令>"
                })
        elif command == "inspect":
            # 查看状态字段详情
            if len(parts) > 1:
                field_name = parts[1]
                
                # 获取字段值
                if field_name == "all":
                    # 显示所有可用字段
                    fields = "\n".join(f"- {key}" for key in new_state.keys())
                    new_state["messages"].append({
                        "role": "system",
                        "content": f"当前状态包含以下字段:\n{fields}"
                    })
                elif field_name in new_state:
                    field_value = new_state[field_name]
                    # 将值转换为可读字符串
                    if isinstance(field_value, dict) or isinstance(field_value, list):
                        value_str = json.dumps(field_value, ensure_ascii=False, indent=2)
                    else:
                        value_str = str(field_value)
                    
                    new_state["messages"].append({
                        "role": "system",
                        "content": f"字段 '{field_name}' 的值:\n```\n{value_str}\n```"
                    })
                else:
                    new_state["messages"].append({
                        "role": "system",
                        "content": f"字段 '{field_name}' 不存在。使用 '/inspect all' 查看所有可用字段。"
                    })
            else:
                new_state["messages"].append({
                    "role": "system",
                    "content": "请指定要查看的字段名。格式: /inspect <字段名> 或 /inspect all"
                })
        elif command == "metrics":
            # 生成业务指标报告
            try:
                # 生成指标仪表板
                dashboard = generate_metrics_dashboard()
                
                # 添加报告到消息
                new_state["messages"].append({
                    "role": "system",
                    "content": f"业务指标报告:\n\n{dashboard['text_summary']}"
                })
                
                # 记录指标查看
                record_business_metric("metrics_viewed", 1, {
                    "session_id": new_state.get("session_id", "unknown")
                })
            except Exception as e:
                new_state["messages"].append({
                    "role": "system",
                    "content": f"生成业务指标报告失败: {str(e)}"
                })
        elif command == "help":
            # 显示帮助信息
            help_message = """可用命令:
- /continue: 继续执行
- /retry: 重试当前步骤
- /abort: 中止当前意图
- /breakpoint <node_name> on|off: 设置或移除断点
- /condition <node_name> <条件表达式>: 设置条件断点
- /callback <node_name> <回调命令>: 设置断点回调
- /inspect <字段名>: 查看状态字段详情
- /inspect all: 查看所有可用字段
- /metrics: 查看业务指标报告
- /help: 显示此帮助信息

条件表达式示例:
- confidence < 0.8
- 'error' in state
- intent == 'query_weather'
- retrieval_count < 3
"""
            new_state["messages"].append({
                "role": "system",
                "content": help_message
            })
        else:
            new_state["messages"].append({
                "role": "system",
                "content": f"未知命令: {command}，输入 /help 查看可用命令。"
            })
            new_state["status"] = "user_input_processor"
    else:
        # 非命令输入，视为普通对话
        # 添加到消息历史
        new_state["messages"].append({
            "role": "human",
            "content": human_input
        })
        # 继续处理
        new_state["status"] = "user_input_processor"
    
    return new_state

async def knowledge_retriever(state: EnhancedAgentState) -> EnhancedAgentState:
    """知识检索节点
    
    基于用户意图和槽位执行知识检索
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态，包含检索结果
    """
    # 记录开始时间
    start_time = time.time()
    
    # 复制状态
    new_state = state.copy()
    
    # 获取当前意图
    current_intent = new_state.get("current_intent")
    if not current_intent:
        # 如果没有当前意图，跳过检索
        logger.info("没有当前意图，跳过知识检索")
        new_state["next_step"] = "router"
        return new_state
    
    # 获取用户槽位
    slots = new_state.get("slots", {})
    
    # 获取用户问题
    user_query = ""
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_query = msg.get("content", "")
            break
    
    if not user_query:
        # 如果没有用户问题，返回错误
        new_state["error"] = "没有找到用户问题，无法执行知识检索"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "missing_user_query",
            "component": "knowledge_retriever",
            "session_id": new_state.get("session_id", "unknown")
        })
        
        return new_state
    
    try:
        # 获取当前意图名称
        intent_name = current_intent.get("name", "unknown")
        
        # 准备检索查询
        # 1. 基于意图和槽位构建查询
        structured_query = f"Intent: {intent_name}\n"
        if slots:
            structured_query += "Slots:\n"
            for key, value in slots.items():
                structured_query += f"- {key}: {value}\n"
        
        # 2. 添加用户原始查询
        structured_query += f"User Query: {user_query}"
        
        logger.info(f"执行知识检索，查询: {structured_query[:100]}...")
        
        # 执行检索 (使用带记忆的增强检索)
        session_id = new_state.get("session_id", "default-session")
        retrieval_results = await retrieve_with_memory(
            query=structured_query,
            session_id=session_id,
            top_k=5  # 获取前5个最相关的结果
        )
        
        # 计算检索质量分数
        retrieval_quality = 0.0
        if retrieval_results:
            # 基于最高相关性分数评估质量
            top_score = retrieval_results[0].score if retrieval_results else 0
            retrieval_quality = min(1.0, top_score * 1.2)  # 归一化到0-1
            
            # 基于结果数量调整质量
            result_count_factor = min(1.0, len(retrieval_results) / 3)  # 至少需要3个结果才算完整
            retrieval_quality = retrieval_quality * 0.8 + result_count_factor * 0.2
        
        # 记录检索指标
        record_business_metric("knowledge_retrieval_count", 1, {
            "session_id": session_id,
            "intent": intent_name,
            "result_count": len(retrieval_results),
            "quality_score": retrieval_quality
        })
        
        # 格式化检索结果
        formatted_results = format_retrieval_results(retrieval_results)
        
        # 更新状态
        new_state["retrieval_results"] = formatted_results
        new_state["status"] = "knowledge_retrieved"
        new_state["next_step"] = "router"
        
        # 更新业务指标
        new_state = update_state_metrics(new_state, {
            "knowledge_retrieval_counts": new_state.get("business_metrics", {}).get("knowledge_retrieval_counts", 0) + 1,
            "last_retrieval_quality": retrieval_quality
        })
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 记录处理时间
        record_business_metric("response_time", processing_time, {
            "component": "knowledge_retriever",
            "session_id": session_id,
            "intent": intent_name,
            "result_count": len(retrieval_results)
        })
        
        return new_state
        
    except Exception as e:
        logger.error(f"知识检索过程中出错: {str(e)}")
        new_state["error"] = f"知识检索失败: {str(e)}"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "retrieval_error",
            "component": "knowledge_retriever",
            "error_message": str(e),
            "session_id": new_state.get("session_id", "unknown"),
            "intent": current_intent.get("name", "unknown")
        })
        
        return new_state

async def astream_message(session_id: str, message: str) -> AsyncIterator[Dict[str, Any]]:
    """异步流式处理用户消息
    
    处理用户消息并以流式方式返回结果
    
    Args:
        session_id: 会话ID
        message: 用户消息
        
    Yields:
        以流式方式返回处理结果
    """
    try:
        # 获取会话状态
        state = SESSION_STATES.get(session_id)
        
        # 如果没有会话状态，初始化
        if not state:
            state = initialize_state()
            state["session_id"] = session_id
            state["created_at"] = datetime.now().isoformat()
            logger.info(f"初始化新会话: {session_id}")
            
            # 返回初始化消息
            yield {
                "session_id": session_id,
                "message": "会话已初始化",
                "type": "status",
                "debug_info": generate_debug_info(state)
            }
        
        # 更新最后活动时间
        state["last_updated_at"] = datetime.now().isoformat()
        
        # 检查是否为调试命令
        is_debug_command = message.startswith("/")
        
        # 如果是调试命令且正在等待人工输入，直接设置为人工输入
        if is_debug_command and state.get("human_input_required", False):
            state["human_input"] = message
            # 使用处理人工输入函数处理
            state = process_human_input(state)
            
            # 生成调试信息
            debug_info = generate_debug_info(state)
            
            # 返回命令处理结果
            yield {
                "session_id": session_id,
                "message": "命令已处理",
                "type": "command_result",
                "debug_info": debug_info,
                "waiting_for_human": state.get("human_input_required", False)
            }
            
            # 保存会话状态
            await save_session_state(session_id, state)
            return
        
        # 添加用户消息到历史
        state["messages"].append({
            "role": "human",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 返回已接收消息的确认
        yield {
            "session_id": session_id,
            "message": "消息已接收",
            "type": "status",
            "debug_info": generate_debug_info(state)
        }
        
        # 记录处理开始时间
        start_time = time.time()
        
        # 使用图流式处理消息
        async for event in compiled_graph.astream(state):
            # 提取当前状态
            current_state = event.get("state", {})
            
            # 检查是否正在等待人工输入
            if current_state.get("human_input_required", False):
                # 生成调试信息
                debug_info = generate_debug_info(current_state)
                
                # 返回等待人工输入的状态
                yield {
                    "session_id": session_id,
                    "message": "等待人工输入",
                    "type": "waiting_for_human",
                    "debug_info": debug_info,
                    "waiting_for_human": True,
                    "state": current_state
                }
                
                # 更新会话状态
                SESSION_STATES[session_id] = current_state
                
                # 保存会话状态
                await save_session_state(session_id, current_state)
                return
            
            # 获取最后一条助手消息
            last_assistant_message = None
            for msg in reversed(current_state.get("messages", [])):
                if msg.get("role") in ["assistant", "ai"]:
                    last_assistant_message = msg.get("content", "")
                    break
            
            if last_assistant_message:
                # 生成调试信息
                debug_info = generate_debug_info(current_state)
                
                # 返回最新的消息
                yield {
                    "session_id": session_id,
                    "message": last_assistant_message,
                    "type": "message",
                    "debug_info": debug_info,
                    "node": event.get("node", "unknown")
                }
            
            # 更新会话状态
            SESSION_STATES[session_id] = current_state
        
        # 记录处理结束时间并计算耗时
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 从最终状态获取最后的消息
        final_state = SESSION_STATES.get(session_id, {})
        
        # 获取最后一条助手消息
        last_assistant_message = None
        for msg in reversed(final_state.get("messages", [])):
            if msg.get("role") in ["assistant", "ai"]:
                last_assistant_message = msg.get("content", "")
                break
        
        if not last_assistant_message:
            last_assistant_message = "处理完成，但没有生成响应。"
        
        # 生成最终的调试信息
        final_debug_info = generate_debug_info(final_state)
        final_debug_info["processing_time"] = f"{processing_time:.2f}秒"
        
        # 返回最终结果
        yield {
            "session_id": session_id,
            "message": last_assistant_message,
            "type": "final",
            "debug_info": final_debug_info,
            "processing_time": processing_time
        }
        
        # 保存会话状态
        await save_session_state(session_id, final_state)
        
    except Exception as e:
        logger.error(f"流式处理消息时出错: {str(e)}", exc_info=True)
        
        # 返回错误信息
        yield {
            "session_id": session_id,
            "message": f"处理消息时出错: {str(e)}",
            "type": "error",
            "error": str(e),
            "debug_info": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "stream_processing_error",
            "component": "astream_message",
            "error_message": str(e),
            "session_id": session_id
        })

async def api_get_metrics_dashboard() -> Dict[str, Any]:
    """获取业务指标仪表板的API接口
    
    生成并返回包含各项业务指标的仪表板数据
    
    Returns:
        包含业务指标的仪表板数据
    """
    try:
        # 生成指标仪表板
        dashboard = generate_metrics_dashboard()
        
        # 创建快照
        snapshot_path = ""
        if metrics_monitor:
            snapshot_path = metrics_monitor.create_snapshot()
            
        # 记录API访问指标
        record_business_metric("api_metrics_access", 1, {
            "timestamp": time.time(),
            "snapshot": os.path.basename(snapshot_path) if snapshot_path else "none"
        })
        
        return {
            "status": "success",
            "data": dashboard,
            "snapshot": os.path.basename(snapshot_path) if snapshot_path else None,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取指标仪表板失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_get_metrics_dashboard",
            "error_message": str(e)
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


async def api_get_session_metrics(session_id: str) -> Dict[str, Any]:
    """获取特定会话的指标数据
    
    返回特定会话的详细指标数据
    
    Args:
        session_id: 会话ID
        
    Returns:
        包含会话指标的字典
    """
    try:
        # 检查会话是否存在
        state = SESSION_STATES.get(session_id)
        if not state:
            return {
                "status": "error",
                "error": f"会话不存在: {session_id}",
                "timestamp": time.time()
            }
        
        # 生成会话指标
        session_data = {
            "session_id": session_id,
            "created_at": state.get("created_at"),
            "last_updated_at": state.get("last_updated_at"),
            "age": calculate_session_age(state.get("created_at", "")),
            "message_count": len(state.get("messages", [])),
            "current_intent": state.get("current_intent", {}).get("name", "无") if state.get("current_intent") else "无",
            "slots_count": len(state.get("slots", {})),
            "error_count": state.get("business_metrics", {}).get("error_counts", 0),
            "human_interventions": state.get("business_metrics", {}).get("human_interventions", 0),
            "breakpoints_triggered": len(state.get("breakpoint_history", [])),
            "business_metrics": state.get("business_metrics", {})
        }
        
        # 如果有指标监控器，获取该会话的聚合指标
        if metrics_monitor:
            # 获取过去1小时内该会话的指标数据
            one_hour_ago = time.time() - 3600
            
            # 过滤出该会话的消息计数
            session_messages = []
            for value in metrics_monitor.get_metric_history("message_count", one_hour_ago):
                if value.metadata.get("session_id") == session_id:
                    session_messages.append(value)
            
            # 过滤出该会话的错误计数
            session_errors = []
            for value in metrics_monitor.get_metric_history("error_count", one_hour_ago):
                if value.metadata.get("session_id") == session_id:
                    session_errors.append(value)
            
            # 添加到会话数据
            session_data["metrics"] = {
                "recent_message_count": len(session_messages),
                "recent_error_count": len(session_errors),
                "recent_activity": bool(session_messages)
            }
        
        # 记录API访问指标
        record_business_metric("api_session_metrics_access", 1, {
            "session_id": session_id,
            "timestamp": time.time()
        })
        
        return {
            "status": "success",
            "data": session_data,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取会话指标失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_get_session_metrics",
            "error_message": str(e),
            "session_id": session_id
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


async def api_get_active_sessions() -> Dict[str, Any]:
    """获取活跃会话列表
    
    返回所有活跃会话的列表及其基本信息
    
    Returns:
        包含活跃会话信息的字典
    """
    try:
        # 获取所有会话
        sessions = []
        for session_id, state in SESSION_STATES.items():
            # 计算最后活动时间
            last_updated = state.get("last_updated_at", "")
            if last_updated:
                try:
                    last_updated_time = datetime.fromisoformat(last_updated)
                    last_active_seconds = (datetime.now() - last_updated_time).total_seconds()
                except:
                    last_active_seconds = float('inf')
            else:
                last_active_seconds = float('inf')
            
            # 只包含24小时内活跃的会话
            if last_active_seconds < 86400:  # 24小时
                sessions.append({
                    "session_id": session_id,
                    "created_at": state.get("created_at", ""),
                    "last_updated_at": last_updated,
                    "last_active_seconds": last_active_seconds,
                    "message_count": len(state.get("messages", [])),
                    "current_intent": state.get("current_intent", {}).get("name", "无") if state.get("current_intent") else "无",
                    "status": state.get("status", "unknown"),
                    "has_error": bool(state.get("error"))
                })
        
        # 按最后活动时间排序
        sessions.sort(key=lambda s: s.get("last_active_seconds", float('inf')))
        
        # 记录API访问指标
        record_business_metric("api_active_sessions_access", 1, {
            "timestamp": time.time(),
            "session_count": len(sessions)
        })
        
        return {
            "status": "success",
            "count": len(sessions),
            "sessions": sessions,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取活跃会话失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_get_active_sessions",
            "error_message": str(e)
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

async def report_base_handler(state: EnhancedAgentState, report_type: str) -> EnhancedAgentState:
    """报表处理基础函数
    
    处理各类报表请求的基础函数，包含共享的逻辑和流程
    
    Args:
        state: 当前状态
        report_type: 报表类型 (financial, project, sales, hr)
        
    Returns:
        更新后的状态
    """
    # 记录开始时间
    start_time = time.time()
    
    # 复制状态
    new_state = state.copy()
    
    # 获取会话ID
    session_id = new_state.get("session_id", "default-session")
    
    # 记录报表请求
    record_business_metric(f"report_{report_type}_requested", 1, {
        "session_id": session_id,
        "timestamp": time.time()
    })
    
    # 获取用户槽位
    slots = new_state.get("slots", {})
    
    # 确定时间范围
    time_range = slots.get("time_range", "last_month")
    time_period_start = slots.get("start_date")
    time_period_end = slots.get("end_date")
    
    # 验证参数
    if not time_range and not (time_period_start and time_period_end):
        # 如果没有指定时间范围和具体日期，使用默认的上个月
        time_range = "last_month"
        
        # 更新状态中的槽位
        if "slots" not in new_state:
            new_state["slots"] = {}
        new_state["slots"]["time_range"] = time_range
        
        # 添加说明消息
        new_state["messages"].append({
            "role": "assistant",
            "content": f"我将为您生成{report_type}报表，使用默认时间范围: 上个月。"
        })
    
    # 准备报表参数
    report_params = {
        "report_type": report_type,
        "time_range": time_range,
        "time_period_start": time_period_start,
        "time_period_end": time_period_end,
        "filters": slots.get("filters", {}),
        "format": slots.get("format", "summary"),
        "detail_level": slots.get("detail_level", "medium"),
        "session_id": session_id,
        "timestamp": time.time()
    }
    
    # 记录该类型报表生成参数（用于分析用户偏好）
    record_business_metric(f"report_{report_type}_parameters", 1, report_params)
    
    try:
        # 执行知识检索来获取相关数据
        # 构建检索查询
        retrieval_query = f"Generate {report_type} report for {time_range}"
        if time_period_start and time_period_end:
            retrieval_query += f" from {time_period_start} to {time_period_end}"
        
        # 记录开始检索
        logger.info(f"执行报表相关知识检索: {retrieval_query}")
        
        # 执行检索
        retrieval_results = await retrieve_with_memory(
            query=retrieval_query,
            session_id=session_id,
            top_k=5  # 获取前5个最相关的结果
        )
        
        # 更新检索结果
        formatted_results = format_retrieval_results(retrieval_results)
        new_state["retrieval_results"] = formatted_results
        
        # 检索质量评估
        retrieval_quality = 0.0
        if retrieval_results:
            top_score = retrieval_results[0].score if retrieval_results else 0
            retrieval_quality = min(1.0, top_score * 1.2)  # 归一化到0-1
            
            # 基于结果数量调整质量
            result_count_factor = min(1.0, len(retrieval_results) / 3)
            retrieval_quality = retrieval_quality * 0.8 + result_count_factor * 0.2
        
        # 记录检索指标
        record_business_metric("knowledge_retrieval_quality", retrieval_quality, {
            "purpose": f"{report_type}_report",
            "session_id": session_id,
            "result_count": len(retrieval_results) if retrieval_results else 0
        })
        
        # 检查检索结果是否足够
        if not retrieval_results or len(retrieval_results) < 2:
            if new_state["confidence"] > 0.7:  # 如果整体置信度高，仍继续生成
                new_state["messages"].append({
                    "role": "assistant",
                    "content": f"我找到的{report_type}报表相关数据有限，但我会尽力为您生成报表。"
                })
            else:
                # 置信度不高，请求更多信息
                new_state["messages"].append({
                    "role": "assistant",
                    "content": f"很抱歉，我没有找到足够的数据来生成{report_type}报表。请提供更多具体信息，例如时间范围或特定的报表需求。"
                })
                
                # 更新状态
                new_state["status"] = "need_more_info"
                new_state["next_step"] = "slot_filler"
                
                # 记录处理时间
                processing_time = time.time() - start_time
                record_business_metric("response_time", processing_time, {
                    "component": "report_base_handler",
                    "report_type": report_type,
                    "outcome": "insufficient_data",
                    "session_id": session_id
                })
                
                return new_state
        
        # 构建报表上下文
        report_context = {
            "report_type": report_type,
            "time_range": time_range,
            "time_period_start": time_period_start,
            "time_period_end": time_period_end,
            "retrieval_results": formatted_results,
            "detail_level": slots.get("detail_level", "medium"),
            "format": slots.get("format", "summary")
        }
        
        # 根据报表类型更新状态
        new_state["report_context"] = report_context
        new_state["status"] = "report_ready"
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 记录处理时间指标
        record_business_metric("response_time", processing_time, {
            "component": "report_base_handler",
            "report_type": report_type,
            "outcome": "success",
            "retrieval_quality": retrieval_quality,
            "session_id": session_id
        })
        
        return new_state
    
    except Exception as e:
        logger.error(f"报表处理过程中出错: {str(e)}")
        new_state["error"] = f"生成{report_type}报表失败: {str(e)}"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "report_generation_error",
            "component": "report_base_handler",
            "report_type": report_type,
            "error_message": str(e),
            "session_id": session_id
        })
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 记录处理时间指标
        record_business_metric("response_time", processing_time, {
            "component": "report_base_handler",
            "report_type": report_type,
            "outcome": "error",
            "session_id": session_id
        })
        
        return new_state

async def financial_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """财务报表处理实现
    
    生成财务报表内容
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 记录开始时间
    start_time = time.time()
    
    # 复制状态
    new_state = state.copy()
    
    # 获取会话ID
    session_id = new_state.get("session_id", "default-session")
    
    # 获取报表上下文
    report_context = new_state.get("report_context", {})
    if not report_context:
        # 如果没有报表上下文，返回错误
        new_state["error"] = "缺少报表上下文，无法生成财务报表"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "missing_report_context",
            "component": "financial_report_handler",
            "session_id": session_id
        })
        
        return new_state
    
    # 获取检索结果
    retrieval_results = new_state.get("retrieval_results", [])
    
    # 获取报表参数
    time_range = report_context.get("time_range", "last_month")
    detail_level = report_context.get("detail_level", "medium")
    report_format = report_context.get("format", "summary")
    
    try:
        # 使用LLM生成报表
        llm = get_async_llm(temperature=0.1)  # 降低随机性，提高一致性
        
        # 构建系统提示
        system_prompt = f"""
你是一位专业的财务分析师，正在生成财务报表。请根据提供的信息生成全面且准确的财务报表。

报表时间范围: {time_range}
详细程度: {detail_level} (low=仅关键指标, medium=标准报表, high=详细分析)
格式: {report_format} (summary=摘要形式, table=表格形式, analysis=带分析的报表)

以下是相关信息:
"""
        
        # 添加检索结果
        if retrieval_results:
            for i, result in enumerate(retrieval_results):
                system_prompt += f"\n来源 {i+1}:\n{result['content']}\n"
        else:
            system_prompt += "\n没有找到特定的财务数据，请生成一份基于通用财务报表结构的示例报表。"
        
        # 添加指导信息
        system_prompt += """
报表应当包含以下部分:
1. 收入概览
2. 支出分析
3. 利润率分析
4. 现金流状况
5. 关键财务指标 (ROI, ROA, 负债比率等)
6. 趋势分析和对比
7. 风险评估
8. 结论和建议

对于财务数据，请确保:
- 数据精确到小数点后两位
- 大额数字使用适当的单位（千、百万、亿等）
- 提供同比和环比变化百分比
- 对于关键变化，提供简要解释

根据详细程度和格式调整内容长度和深度。输出应当结构清晰，便于理解。
"""
        
        # 准备用户消息
        user_message = f"请生成一份{time_range}的财务报表，{detail_level}详细程度，使用{report_format}格式。"
        
        # 准备消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        response = await llm.ainvoke(messages)
        report_content = response.content
        
        # 添加报表到消息
        new_state["messages"].append({
            "role": "assistant",
            "content": f"以下是您请求的财务报表:\n\n{report_content}"
        })
        
        # 更新状态
        new_state["status"] = "completed"
        new_state["next_step"] = "user_input_processor"
        
        # 保存报表内容到状态
        if "generated_reports" not in new_state:
            new_state["generated_reports"] = {}
        
        report_id = f"financial_{int(time.time())}"
        new_state["generated_reports"][report_id] = {
            "type": "financial",
            "content": report_content,
            "parameters": report_context,
            "timestamp": time.time(),
            "quality_score": None  # 将在用户反馈后填充
        }
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 评估报表质量
        # 1. 长度检查 (简单启发式方法)
        length_score = min(1.0, len(report_content) / 1500)
        
        # 2. 结构完整性检查 (检查是否包含所有必要部分)
        required_sections = [
            "收入", "支出", "利润", "现金流", "指标", "趋势", "风险", "结论"
        ]
        
        structure_score = sum(1 for section in required_sections if section in report_content) / len(required_sections)
        
        # 3. 数字密度检查 (财务报表应包含足够的数字)
        import re
        numeric_matches = re.findall(r'\d+\.?\d*%?', report_content)
        numeric_density = min(1.0, len(numeric_matches) / 30)  # 假设至少需要30个数字
        
        # 计算综合质量分数
        quality_score = (length_score * 0.3) + (structure_score * 0.5) + (numeric_density * 0.2)
        
        # 记录质量分数
        record_business_metric("report_quality", quality_score, {
            "report_type": "financial",
            "report_id": report_id,
            "length_score": length_score,
            "structure_score": structure_score,
            "numeric_density": numeric_density,
            "session_id": session_id,
            "processing_time": processing_time
        })
        
        # 更新报表质量分数
        new_state["generated_reports"][report_id]["quality_score"] = quality_score
        
        # 记录处理时间
        record_business_metric("response_time", processing_time, {
            "component": "financial_report_handler",
            "outcome": "success",
            "session_id": session_id,
            "report_quality": quality_score
        })
        
        # 记录报表生成完成
        record_business_metric("report_financial_completed", 1, {
            "session_id": session_id,
            "time_range": time_range,
            "detail_level": detail_level,
            "format": report_format,
            "quality_score": quality_score,
            "processing_time": processing_time
        })
        
        return new_state
    
    except Exception as e:
        logger.error(f"生成财务报表时出错: {str(e)}")
        new_state["error"] = f"生成财务报表失败: {str(e)}"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "financial_report_generation_error",
            "component": "financial_report_handler",
            "error_message": str(e),
            "session_id": session_id
        })
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 记录处理时间
        record_business_metric("response_time", processing_time, {
            "component": "financial_report_handler",
            "outcome": "error",
            "session_id": session_id
        })
        
        return new_state

async def process_report_feedback(state: EnhancedAgentState, feedback: Dict[str, Any]) -> EnhancedAgentState:
    """处理用户对报表的反馈
    
    记录用户反馈意见，并更新相关业务指标
    
    Args:
        state: 当前状态
        feedback: 用户反馈信息，包含report_id, rating, comments等
        
    Returns:
        更新后的状态
    """
    # 复制状态
    new_state = state.copy()
    
    # 获取会话ID
    session_id = new_state.get("session_id", "default-session")
    
    # 提取反馈数据
    report_id = feedback.get("report_id")
    rating = feedback.get("rating")  # 1-5分评价
    comments = feedback.get("comments", "")
    
    # 参数验证
    if not report_id or report_id not in new_state.get("generated_reports", {}):
        # 如果报表ID无效，添加错误消息
        new_state["messages"].append({
            "role": "system",
            "content": "无法处理您的反馈，找不到对应的报表。"
        })
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "invalid_report_id",
            "component": "process_report_feedback",
            "session_id": session_id
        })
        
        return new_state
    
    # 获取报表信息
    report_info = new_state["generated_reports"][report_id]
    report_type = report_info.get("type", "unknown")
    
    try:
        # 记录用户反馈
        if "feedback" not in report_info:
            report_info["feedback"] = []
        
        # 添加反馈记录
        feedback_entry = {
            "timestamp": time.time(),
            "rating": rating,
            "comments": comments
        }
        
        report_info["feedback"].append(feedback_entry)
        
        # 记录反馈指标
        if rating is not None:
            record_business_metric("report_feedback", rating, {
                "report_type": report_type,
                "report_id": report_id,
                "has_comments": bool(comments),
                "session_id": session_id
            })
        
        # 如果评分低于3分，记录问题报表指标
        if rating is not None and rating < 3:
            record_business_metric("report_issue", 1, {
                "report_type": report_type,
                "report_id": report_id,
                "rating": rating,
                "has_comments": bool(comments),
                "session_id": session_id
            })
        
        # 添加感谢消息
        new_state["messages"].append({
            "role": "assistant",
            "content": "感谢您的反馈！我们将不断改进报表生成质量。"
        })
        
        # 记录反馈处理
        record_business_metric("report_feedback_processed", 1, {
            "report_type": report_type,
            "session_id": session_id
        })
        
        # 如果评分较低且有评论，尝试提取具体问题并提供改进方向
        if rating is not None and rating < 4 and comments:
            await analyze_feedback_for_improvement(report_type, rating, comments, session_id)
        
        return new_state
    
    except Exception as e:
        logger.error(f"处理报表反馈时出错: {str(e)}")
        
        # 添加错误消息
        new_state["messages"].append({
            "role": "system",
            "content": "处理您的反馈时出现问题，但我们已记录您的意见。"
        })
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "feedback_processing_error",
            "component": "process_report_feedback",
            "error_message": str(e),
            "session_id": session_id
        })
        
        return new_state


async def analyze_feedback_for_improvement(report_type: str, rating: int, comments: str, session_id: str) -> None:
    """分析用户反馈以提供改进建议
    
    从用户评论中提取具体问题，并建议改进方向
    
    Args:
        report_type: 报表类型
        rating: 评分(1-5)
        comments: 用户评论
        session_id: 会话ID
    """
    try:
        # 使用LLM分析反馈
        llm = get_async_llm(temperature=0)
        
        # 准备系统提示
        system_prompt = f"""
分析以下关于{report_type}报表的用户反馈（评分{rating}/5分），并提取具体问题和改进方向。
返回JSON格式:
{{
    "issues": [具体问题列表],
    "improvements": [改进建议列表],
    "priority": "high|medium|low"  // 基于严重程度的优先级
}}
"""
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": comments}
        ]
        
        # 调用LLM
        response = await llm.ainvoke(messages)
        
        try:
            # 解析结果
            result = json.loads(response.content)
            
            # 记录分析结果
            record_business_metric("feedback_analysis", 1, {
                "report_type": report_type,
                "issues_count": len(result.get("issues", [])),
                "priority": result.get("priority", "medium"),
                "session_id": session_id
            })
            
            # 针对每个问题记录详细指标
            for issue in result.get("issues", []):
                # 记录具体问题
                record_business_metric("report_specific_issue", 1, {
                    "report_type": report_type,
                    "issue": issue[:100],  # 限制长度
                    "rating": rating,
                    "priority": result.get("priority", "medium"),
                    "session_id": session_id
                })
        
        except json.JSONDecodeError:
            logger.warning(f"无法解析反馈分析结果: {response.content}")
    
    except Exception as e:
        logger.error(f"分析报表反馈时出错: {str(e)}")
        # 出错时记录但不中断流程

async def api_get_generated_reports(session_id: Optional[str] = None) -> Dict[str, Any]:
    """获取已生成报表列表
    
    返回系统中已生成的报表列表，可按会话ID筛选
    
    Args:
        session_id: 可选的会话ID筛选条件
        
    Returns:
        报表列表数据
    """
    try:
        # 获取所有会话状态
        reports_data = []
        
        for sid, state in SESSION_STATES.items():
            # 如果指定了会话ID且不匹配，则跳过
            if session_id and sid != session_id:
                continue
            
            # 获取该会话中的报表
            generated_reports = state.get("generated_reports", {})
            
            for report_id, report_info in generated_reports.items():
                # 提取报表基本信息
                report_data = {
                    "report_id": report_id,
                    "session_id": sid,
                    "type": report_info.get("type", "unknown"),
                    "timestamp": report_info.get("timestamp"),
                    "created_at": datetime.fromtimestamp(report_info.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S") if report_info.get("timestamp") else "未知",
                    "quality_score": report_info.get("quality_score"),
                    "parameters": report_info.get("parameters", {}),
                    "has_feedback": bool(report_info.get("feedback"))
                }
                
                reports_data.append(report_data)
        
        # 按时间戳排序（最新的在前）
        reports_data.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # 记录API访问
        record_business_metric("api_reports_list_access", 1, {
            "session_id": session_id or "all",
            "report_count": len(reports_data)
        })
        
        return {
            "status": "success",
            "count": len(reports_data),
            "reports": reports_data,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取报表列表失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_get_generated_reports",
            "error_message": str(e),
            "session_id": session_id or "all"
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


async def api_get_report_details(report_id: str, session_id: str) -> Dict[str, Any]:
    """获取报表详情
    
    返回特定报表的详细信息，包括内容和反馈
    
    Args:
        report_id: 报表ID
        session_id: 会话ID
        
    Returns:
        报表详情数据
    """
    try:
        # 获取会话状态
        state = SESSION_STATES.get(session_id)
        
        if not state:
            return {
                "status": "error",
                "error": f"会话不存在: {session_id}",
                "timestamp": time.time()
            }
        
        # 获取报表信息
        generated_reports = state.get("generated_reports", {})
        
        if report_id not in generated_reports:
            return {
                "status": "error",
                "error": f"报表不存在: {report_id}",
                "timestamp": time.time()
            }
        
        # 获取报表详情
        report_info = generated_reports[report_id]
        
        # 构建详细响应
        report_details = {
            "report_id": report_id,
            "session_id": session_id,
            "type": report_info.get("type", "unknown"),
            "timestamp": report_info.get("timestamp"),
            "created_at": datetime.fromtimestamp(report_info.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S") if report_info.get("timestamp") else "未知",
            "quality_score": report_info.get("quality_score"),
            "parameters": report_info.get("parameters", {}),
            "content": report_info.get("content", ""),
            "feedback": report_info.get("feedback", [])
        }
        
        # 记录API访问
        record_business_metric("api_report_details_access", 1, {
            "session_id": session_id,
            "report_id": report_id,
            "report_type": report_info.get("type", "unknown")
        })
        
        return {
            "status": "success",
            "report": report_details,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取报表详情失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_get_report_details",
            "error_message": str(e),
            "report_id": report_id,
            "session_id": session_id
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


async def api_submit_report_feedback(report_id: str, session_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
    """提交报表反馈
    
    处理用户对报表的反馈评价
    
    Args:
        report_id: 报表ID
        session_id: 会话ID
        feedback: 反馈信息，包含rating和comments
        
    Returns:
        处理结果
    """
    try:
        # 获取会话状态
        state = SESSION_STATES.get(session_id)
        
        if not state:
            return {
                "status": "error",
                "error": f"会话不存在: {session_id}",
                "timestamp": time.time()
            }
        
        # 准备反馈数据
        feedback_data = {
            "report_id": report_id,
            "rating": feedback.get("rating"),
            "comments": feedback.get("comments", "")
        }
        
        # 处理反馈
        updated_state = await process_report_feedback(state, feedback_data)
        
        # 更新会话状态
        SESSION_STATES[session_id] = updated_state
        
        # 保存会话状态
        await save_session_state(session_id, updated_state)
        
        return {
            "status": "success",
            "message": "反馈已处理",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"提交报表反馈失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_submit_report_feedback",
            "error_message": str(e),
            "report_id": report_id,
            "session_id": session_id
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


async def api_generate_report(report_type: str, parameters: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """生成新报表
    
    通过API接口生成新的报表
    
    Args:
        report_type: 报表类型 (financial, project, sales, hr)
        parameters: 报表参数
        session_id: 可选的会话ID，如未提供将创建新会话
        
    Returns:
        报表生成结果
    """
    try:
        # 验证报表类型
        valid_report_types = ["financial", "project", "sales", "hr"]
        if report_type not in valid_report_types:
            return {
                "status": "error",
                "error": f"不支持的报表类型: {report_type}。支持的类型: {', '.join(valid_report_types)}",
                "timestamp": time.time()
            }
        
        # 处理会话ID
        if not session_id:
            session_id = f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # 获取或创建会话状态
        state = SESSION_STATES.get(session_id)
        if not state:
            state = initialize_state()
            state["session_id"] = session_id
            state["created_at"] = datetime.now().isoformat()
            SESSION_STATES[session_id] = state
        
        # 更新状态
        state["last_updated_at"] = datetime.now().isoformat()
        
        # 准备槽位
        if "slots" not in state:
            state["slots"] = {}
        
        # 添加报表参数到槽位
        state["slots"].update(parameters)
        
        # 构建虚拟用户请求
        user_message = f"请生成一份{report_type}报表"
        if "time_range" in parameters:
            user_message += f"，时间范围为{parameters['time_range']}"
        if "detail_level" in parameters:
            user_message += f"，详细程度为{parameters['detail_level']}"
        
        # 添加用户消息
        state["messages"].append({
            "role": "human",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
            "source": "api"
        })
        
        # 设置意图
        state["current_intent"] = {
            "name": f"query_{report_type}_report",
            "confidence": 1.0,
            "slots": parameters,
            "parameters": {}  # 可以根据需要填充
        }
        
        # 执行报表处理
        # 1. 基础处理
        state = await report_base_handler(state, report_type)
        
        # 2. 类型特定处理
        handler_map = {
            "financial": financial_report_handler_impl,
            "project": project_report_handler_impl,
            "sales": sales_report_handler_impl,
            "hr": hr_report_handler_impl
        }
        
        if state.get("status") == "report_ready":
            handler = handler_map.get(report_type)
            if handler:
                state = await handler(state)
        
        # 更新会话状态
        SESSION_STATES[session_id] = state
        
        # 保存会话状态
        await save_session_state(session_id, state)
        
        # 查找生成的报表ID
        report_id = None
        for rid, rinfo in state.get("generated_reports", {}).items():
            if rinfo.get("type") == report_type and int(time.time()) - int(rinfo.get("timestamp", 0)) < 60:
                report_id = rid
                break
        
        # 返回结果
        if report_id:
            return {
                "status": "success",
                "message": "报表已生成",
                "session_id": session_id,
                "report_id": report_id,
                "timestamp": time.time()
            }
        elif state.get("error"):
            return {
                "status": "error",
                "error": state.get("error"),
                "session_id": session_id,
                "timestamp": time.time()
            }
        else:
            return {
                "status": "unknown",
                "message": "处理完成但未找到生成的报表",
                "session_id": session_id,
                "timestamp": time.time()
            }
    except Exception as e:
        logger.error(f"通过API生成报表失败: {str(e)}", exc_info=True)
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "api_error",
            "component": "api_generate_report",
            "error_message": str(e),
            "report_type": report_type,
            "session_id": session_id or "new"
        })
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

async def financial_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """财务报表处理入口
    
    处理财务报表请求的主入口函数
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 记录业务指标
    session_id = state.get("session_id", "default-session")
    record_business_metric("report_handler_called", 1, {
        "report_type": "financial",
        "session_id": session_id
    })
    
    try:
        # 基础报表处理
        report_state = await report_base_handler(state, "financial")
        
        # 如果基础处理出错或需要更多信息，直接返回
        if report_state.get("error") or report_state.get("status") == "need_more_info":
            return report_state
        
        # 调用实现函数
        return await financial_report_handler_impl(report_state)
    except Exception as e:
        # 记录错误
        logger.error(f"财务报表处理器出错: {str(e)}")
        
        # 复制状态并添加错误信息
        new_state = state.copy()
        new_state["error"] = f"财务报表处理失败: {str(e)}"
        new_state["status"] = "error"
        
        # 记录错误指标
        record_business_metric("error_count", 1, {
            "error_type": "report_handler_error",
            "component": "financial_report_handler",
            "error_message": str(e),
            "session_id": session_id
        })
        
        return new_state