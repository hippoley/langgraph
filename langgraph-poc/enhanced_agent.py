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
    """处理用户输入，准备状态"""
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
        return new_state
    
    try:
        # 更新记忆系统
        session_id = new_state.get("session_id", "default-session")
        new_state = await update_memory_with_conversation(session_id, new_state)
        
        # 检查置信度（用于断点系统）
        # 计算一个简单的置信度指标：如果消息长度小于10个字符，降低置信度
        message_content = user_message.get("content", "")
        if len(message_content) < 10:
            new_state["confidence"] = 0.6
        else:
            new_state["confidence"] = 0.9
        
        # 设置状态为已处理
        new_state["status"] = "user_input_processed"
        new_state["next_step"] = "intent_recognizer"
        
    except Exception as e:
        # 记录错误
        logger.error(f"处理用户输入时出错: {str(e)}")
        new_state["error"] = f"处理用户输入时出错: {str(e)}"
        new_state["status"] = "error"
    
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