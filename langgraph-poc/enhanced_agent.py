"""
增强的对话代理 - 整合意图管理、业务指标和RAG功能
"""

import os
import json
import logging
import uuid
import asyncio
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
    from intent_manager import IntentManager, Intent, Slot, IntentState
    from business_metrics import BusinessMetricRegistry, create_default_metrics
    from enhanced_rag import EnhancedRetriever, RetrievalResult, create_sample_knowledge_base
    from api_config import get_llm
    from integration import retrieve_with_memory, format_retrieval_results, update_memory_with_conversation
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

# 持久化配置
PERSISTENCE_TYPE = os.getenv("PERSISTENCE_TYPE", "memory")  # memory, sqlite, postgres
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "agent_states/sessions.db")
POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING", "")

# 全局会话管理
SESSION_STATES = {}

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
    breakpoints: Dict[str, bool]  # 断点设置
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
    """初始化全局组件"""
    # 初始化意图管理器
    llm = get_llm()
    intent_manager = IntentManager(llm=llm)
    
    # 初始化业务指标
    if load_existing and os.path.exists("./knowledge_base/metrics.json"):
        try:
            metric_registry = BusinessMetricRegistry.load_from_file("./knowledge_base/metrics.json")
            logger.info("从文件加载业务指标")
        except Exception as e:
            logger.error(f"加载业务指标失败: {str(e)}")
            metric_registry = create_default_metrics()
    else:
        metric_registry = create_default_metrics()
        # 保存业务指标
        os.makedirs("./knowledge_base", exist_ok=True)
        metric_registry.save_to_file("./knowledge_base/metrics.json")
    
    # 初始化检索器
    if load_existing and os.path.exists("./knowledge_base/retriever_state.json"):
        try:
            retriever = EnhancedRetriever.load_state(
                "./knowledge_base/retriever_state.json",
                llm=llm
            )
            logger.info("从文件加载检索器状态")
        except Exception as e:
            logger.error(f"加载检索器状态失败: {str(e)}")
            retriever = EnhancedRetriever(llm=llm)
            create_sample_knowledge_base(retriever)
    else:
        retriever = EnhancedRetriever(llm=llm)
        create_sample_knowledge_base(retriever)
    
    return intent_manager, metric_registry, retriever

# 初始化状态
def initialize_state() -> EnhancedAgentState:
    """初始化对话状态"""
    now = datetime.now().isoformat()
    return EnhancedAgentState(
        session_id=str(uuid.uuid4()),
        messages=[],
        created_at=now,
        last_updated_at=now,
        current_intent=None,
        intent_stack=[],
        suspended_intents=[],
        slots={},
        retrieval_results=[],
        status="initialized",
        next_step="",
        error=None,
        breakpoints={},
        human_input_required=False,
        human_input=None,
        confidence=0.0,
        business_metrics={}
    )

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
                            break
            except json.JSONDecodeError:
                logger.error(f"无法解析意图识别结果: {response_content}")
                new_state["error"] = "意图识别失败：无法解析结果"
                new_state["status"] = "error"
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
        
        return new_state
        
    except Exception as e:
        logger.error(f"意图识别过程中出错: {str(e)}")
        new_state["error"] = f"意图识别失败: {str(e)}"
        new_state["status"] = "error"
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
    new_state = state.copy()
    
    # 获取当前意图
    current_intent = new_state.get("current_intent")
    if not current_intent:
        # 如果没有当前意图，返回状态
        new_state["error"] = "没有当前意图，无法填充槽位"
        new_state["status"] = "error"
        return new_state
    
    # 获取意图的参数
    parameters = current_intent.get("parameters", {})
    if not parameters:
        # 如果没有参数，直接继续
        new_state["status"] = "slots_filled"
        new_state["next_step"] = "router"
        return new_state
    
    # 获取已填充的槽位
    slots = new_state.get("slots", {})
    if "slots" not in new_state:
        new_state["slots"] = slots
    
    # 查找缺失的必要槽位
    missing_slots = {}
    for param_name, param_type in parameters.items():
        if param_name not in slots or slots[param_name] is None or slots[param_name] == "":
            missing_slots[param_name] = param_type
    
    # 如果没有缺失的槽位，继续
    if not missing_slots:
        new_state["status"] = "slots_filled"
        new_state["next_step"] = "router"
        return new_state
    
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有找到用户消息，需要询问用户
        missing_slot_names = list(missing_slots.keys())
        first_missing = missing_slot_names[0]
        
        # 添加系统消息，要求提供信息
        new_state["messages"].append({
            "role": "assistant",
            "content": f"为了处理您的'{current_intent['name']}'请求，我需要了解{first_missing}。请提供这个信息。"
        })
        
        # 更新状态
        new_state["status"] = "awaiting_slot"
        new_state["next_step"] = "slot_filler"
        return new_state
    
    try:
        # 使用异步LLM提取槽位
        llm = get_async_llm(temperature=0)
        
        # 准备槽位提取提示
        system_prompt = f"""
你是一个专业的参数提取AI。从用户的消息中提取以下参数的值：

{json.dumps(missing_slots, ensure_ascii=False, indent=2)}

回复必须是一个JSON对象，包含提取到的参数值。例如：
{{
    "参数名1": "参数值1",
    "参数名2": "参数值2"
}}

如果无法从用户消息中提取某个参数，将其值设为null。
"""
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        response = await llm.ainvoke(messages)
        
        try:
            # 解析结果
            extracted_slots = json.loads(response.content)
            
            # 更新槽位
            extracted_count = 0
            for slot_name, slot_value in extracted_slots.items():
                if slot_name in missing_slots and slot_value is not None and slot_value != "":
                    new_state["slots"][slot_name] = slot_value
                    extracted_count += 1
            
            # 检查是否还有缺失的槽位
            still_missing = {}
            for param_name in missing_slots:
                if param_name not in new_state["slots"] or new_state["slots"][param_name] is None or new_state["slots"][param_name] == "":
                    still_missing[param_name] = missing_slots[param_name]
            
            if still_missing and extracted_count == 0:
                # 如果没有提取到任何槽位，直接询问第一个缺失的槽位
                missing_slot_names = list(still_missing.keys())
                first_missing = missing_slot_names[0]
                
                # 添加系统消息，要求提供信息
                new_state["messages"].append({
                    "role": "assistant",
                    "content": f"为了处理您的'{current_intent['name']}'请求，我需要了解{first_missing}。请提供这个信息。"
                })
                
                # 更新状态
                new_state["status"] = "awaiting_slot"
                new_state["next_step"] = "slot_filler"
            elif still_missing:
                # 如果提取到了一些槽位但仍有缺失，继续尝试提取
                new_state["status"] = "slots_partially_filled"
                new_state["next_step"] = "slot_filler"
            else:
                # 所有槽位已填充
                new_state["status"] = "slots_filled"
                new_state["next_step"] = "router"
            
            return new_state
            
        except json.JSONDecodeError as e:
            logger.error(f"解析槽位提取结果时出错: {str(e)}, 响应内容: {response.content}")
            # 如果解析失败，请求用户直接提供第一个缺失的槽位
            missing_slot_names = list(missing_slots.keys())
            first_missing = missing_slot_names[0]
            
            # 添加系统消息，要求提供信息
            new_state["messages"].append({
                "role": "assistant",
                "content": f"为了处理您的'{current_intent['name']}'请求，我需要了解{first_missing}。请提供这个信息。"
            })
            
            # 更新状态
            new_state["status"] = "awaiting_slot"
            new_state["next_step"] = "slot_filler"
            return new_state
    
    except Exception as e:
        logger.error(f"槽位填充过程中出错: {str(e)}")
        new_state["error"] = f"槽位填充失败: {str(e)}"
        new_state["status"] = "error"
        return new_state

async def knowledge_retriever(state: EnhancedAgentState) -> EnhancedAgentState:
    """知识检索，根据当前意图和用户查询检索相关信息"""
    # 复制状态
    new_state = state.copy()
    
    # 获取当前意图和槽位
    current_intent = new_state.get("current_intent")
    if not current_intent:
        # 如果没有当前意图，跳过检索
        logger.info("无当前意图，跳过知识检索")
        return new_state
    
    intent_name = current_intent.get("name", "")
    
    # 对于某些意图，可能不需要检索知识
    skip_intents = ["calculate", "translate", "help", "chitchat"]
    if intent_name in skip_intents:
        logger.info(f"意图 {intent_name} 无需知识检索")
        return new_state
    
    # 构建查询
    slots = new_state.get("slots", {})
    messages = new_state.get("messages", [])
    
    # 获取最后一条用户消息
    last_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "human":
            last_user_message = msg.get("content", "")
            break
    
    if not last_user_message:
        logger.warning("未找到用户消息，无法构建查询")
        return new_state
    
    # 使用集成检索模块进行检索
    try:
        # 基于意图定制检索参数
        params = {
            "top_k": 5,
            "include_memories": True,
            "memory_weight": 0.3,
            "knowledge_weight": 0.7
        }
        
        # 对特定意图进行参数调整
        if intent_name.startswith("query_"):
            # 对于查询类意图，增加知识库权重
            params["knowledge_weight"] = 0.8
            params["memory_weight"] = 0.2
        elif "report" in intent_name:
            # 对于报表类意图，增加记忆权重
            params["knowledge_weight"] = 0.6
            params["memory_weight"] = 0.4
            params["top_k"] = 7  # 增加结果数量
        
        # 构建查询
        query = last_user_message
        
        # 如果有槽位信息，可以增强查询
        if slots:
            slot_info = ", ".join([f"{k}: {v}" for k, v in slots.items()])
            query = f"{query} ({slot_info})"
        
        # 异步执行检索
        session_id = new_state.get("session_id", "default-session")
        results = await retrieve_with_memory(
            session_id=session_id,
            query=query,
            state=new_state,
            **params
        )
        
        # 更新状态中的检索结果
        new_state["retrieval_results"] = results.get("retrieval_results", [])
        
        # 生成上下文文本
        context_text = format_retrieval_results(results, include_metadata=False)
        if context_text:
            # 添加系统消息，提供检索到的信息
            new_state["messages"].append({
                "role": "system",
                "content": f"已检索到以下相关信息:\n\n{context_text}"
            })
            
            logger.info(f"检索到 {len(results.get('retrieval_results', []))} 条相关信息")
        else:
            logger.info("未检索到相关信息")
        
        # 设置状态
        new_state["status"] = "knowledge_retrieved"
        
    except Exception as e:
        logger.error(f"知识检索出错: {str(e)}")
        # 添加错误信息
        new_state["error"] = f"知识检索出错: {str(e)}"
        # 继续流程
        new_state["status"] = "knowledge_retrieved"
    
    return new_state

async def calculator_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理计算意图"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    expression = slots.get("expression", "")
    
    if not expression:
        # 如果没有表达式
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我需要一个表达式来进行计算。请提供一个数学表达式，例如 '2 + 2'。"
        })
        new_state["status"] = "missing_expression"
        return new_state
    
    try:
        # 使用工具进行计算
        result = Tools.calculator(expression)
        
        # 构建响应
        response = f"计算结果：{expression} = {result}"
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": response
        })
        
        # 更新状态
        new_state["status"] = "calculation_completed"
        
        # 记录日志
        logger.info(f"计算完成: {expression} = {result}")
    
    except Exception as e:
        # 处理错误
        error_message = f"计算时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请提供一个有效的数学表达式。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "calculation_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def weather_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理天气查询意图"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    location = slots.get("location", "")
    
    if not location:
        # 如果没有位置
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我需要知道您想查询哪个地区的天气。请提供一个位置，例如'北京'。"
        })
        new_state["status"] = "missing_location"
        return new_state
    
    try:
        # 使用工具查询天气
        weather_info = Tools.get_weather(location)
        
        # 构建响应
        response = f"以下是{location}的天气信息：\n\n{weather_info}"
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": response
        })
        
        # 更新状态
        new_state["status"] = "weather_queried"
        
        # 记录日志
        logger.info(f"查询天气完成: {location}")
    
    except Exception as e:
        # 处理错误
        error_message = f"查询天气时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请提供一个有效的位置。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "weather_query_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def translate_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理翻译意图"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    text = slots.get("text", "")
    target_language = slots.get("target_language", "")
    
    if not text:
        # 如果没有文本
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我需要知道您想翻译的文本。请提供需要翻译的内容。"
        })
        new_state["status"] = "missing_text"
        return new_state
    
    if not target_language:
        # 如果没有目标语言
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我需要知道您想翻译成哪种语言。请指定目标语言，例如'英语'、'日语'等。"
        })
        new_state["status"] = "missing_target_language"
        return new_state
    
    try:
        # 使用工具进行翻译
        translated_text = Tools.translate(text, target_language)
        
        # 构建响应
        response = f"翻译结果：\n\n原文：{text}\n\n译文：{translated_text}"
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": response
        })
        
        # 更新状态
        new_state["status"] = "translation_completed"
        
        # 记录日志
        logger.info(f"翻译完成: '{text}' -> '{translated_text}' (目标语言: {target_language})")
    
    except Exception as e:
        # 处理错误
        error_message = f"翻译时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请检查您的文本和目标语言。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "translation_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def chitchat_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理闲聊意图"""
    # 复制状态
    new_state = state.copy()
    
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有找到用户消息
        new_state["messages"].append({
            "role": "ai",
            "content": "您好！有什么可以帮助您的吗？"
        })
        new_state["status"] = "chitchat_responded"
        return new_state
    
    try:
        # 使用LLM生成闲聊回复
        llm = get_llm(temperature=0.7)  # 闲聊使用较高的温度
        
        # 收集上下文历史
        history = []
        for msg in new_state["messages"][-5:]:  # 仅使用最近5条消息
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "human":
                history.append({"role": "user", "content": content})
            elif role == "ai":
                history.append({"role": "assistant", "content": content})
        
        # 添加系统提示
        system_prompt = {
            "role": "system", 
            "content": "你是一个友好的对话助手。请保持礼貌和有帮助的态度，提供自然、友好的回复。"
        }
        
        # 构建消息
        messages = [system_prompt] + history
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        # 获取回复
        reply = response.content
        
        # 添加回复到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": reply
        })
        
        # 更新状态
        new_state["status"] = "chitchat_responded"
        
        # 记录日志
        logger.info(f"生成闲聊回复: '{reply[:50]}...'")
    
    except Exception as e:
        # 处理错误
        error_message = f"生成闲聊回复时出错: {str(e)}"
        
        # 添加一个通用回复
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我现在无法正确理解您的意思。请问您能换种方式表达吗？"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "chitchat_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def help_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理帮助意图"""
    # 复制状态
    new_state = state.copy()
    
    # 准备帮助信息
    help_text = """
    我可以帮助您完成以下任务:
    
    1. 计算: 我可以计算数学表达式，例如 "计算 (3 + 4) * 5"
    
    2. 天气查询: 我可以查询城市的天气信息，例如 "北京今天天气怎么样？"
    
    3. 翻译: 我可以将文本翻译为不同语言，例如 "把'你好'翻译成英语"
    
    4. 业务报表:
       - 财务报表: "生成上个季度的财务报表"
       - 项目报表: "查看项目XYZ的进度报表"
       - 销售报表: "显示本月销售报表"
       - 人力资源报表: "生成部门人员配置报表"
    
    如果您有其他问题，请直接询问，我将尽力帮助您！
    """
    
    # 添加帮助信息到消息历史
    new_state["messages"].append({
        "role": "ai",
        "content": help_text
    })
    
    # 更新状态
    new_state["status"] = "help_provided"
    
    # 记录日志
    logger.info("已提供帮助信息")
    
    return new_state

async def fallback_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理未识别意图的回退处理"""
    # 复制状态
    new_state = state.copy()
    
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(new_state["messages"]):
        if msg.get("role") == "human":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有找到用户消息
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我不太理解您的意思。请问您能换个方式表达吗？或者您可以尝试以下功能：计算、天气查询、翻译或查询报表。"
        })
        new_state["status"] = "fallback_responded"
        return new_state
    
    try:
        # 使用LLM生成通用回复
        llm = get_llm(temperature=0.3)
        
        # 准备系统提示
        system_prompt = """
        你是一个助手，用户的请求不在你的主要功能范围内。请礼貌地回应，并引导用户使用你的核心功能。
        
        你的核心功能包括：
        1. 计算数学表达式
        2. 查询天气信息
        3. 文本翻译
        4. 生成业务报表（财务、项目、销售、人力资源）
        
        提供友好、有帮助的回复，并简要介绍你的核心功能。
        """
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        # 添加回复到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": response.content
        })
        
        # 更新状态
        new_state["status"] = "fallback_responded"
        
        # 记录日志
        logger.info(f"生成回退回复: '{response.content[:50]}...'")
    
    except Exception as e:
        # 处理错误
        error_message = f"生成回退回复时出错: {str(e)}"
        
        # 添加一个通用回复
        new_state["messages"].append({
            "role": "ai",
            "content": "抱歉，我现在无法正确处理您的请求。请尝试使用我的核心功能：计算、天气查询、翻译或查询报表。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "fallback_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

def router(state: EnhancedAgentState) -> str:
    """路由节点
    
    根据当前状态和意图，决定下一步应该执行哪个节点
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点的名称
    """
    # 检查是否有错误
    if state.get("error"):
        logger.error(f"流程中存在错误: {state.get('error')}")
        return "fallback_handler"
    
    # 获取当前意图
    current_intent = state.get("current_intent")
    if not current_intent:
        logger.warning("没有识别到意图，使用 fallback 处理")
        return "fallback_handler"
    
    # 获取意图名称
    intent_name = current_intent.get("name", "")
    
    # 记录路由信息
    logger.info(f"路由到意图处理器: {intent_name}")
    
    # 根据意图名称路由到相应的处理器
    intent_handlers = {
        "calculate": "calculator_handler",
        "query_weather": "weather_handler",
        "translate": "translate_handler",
        "chitchat": "chitchat_handler",
        "help": "help_handler",
        "query_financial_report": "financial_report_handler",
        "query_project_report": "project_report_handler",
        "query_sales_report": "sales_report_handler",
        "query_hr_report": "hr_report_handler",
        "fallback": "fallback_handler"
    }
    
    # 获取处理器名称，如果没有匹配则使用 fallback
    handler = intent_handlers.get(intent_name, "fallback_handler")
    
    # 如果状态中指定了 next_step，并且是有效的处理器，使用它
    if state.get("next_step") in intent_handlers.values():
        handler = state.get("next_step")
    
    return handler

async def update_state(state: EnhancedAgentState) -> EnhancedAgentState:
    """更新状态，处理完成后的清理工作"""
    # 复制状态
    new_state = state.copy()
    
    # 更新时间戳
    new_state["last_updated_at"] = datetime.now().isoformat()
    
    # 检查是否有错误
    if new_state.get("error"):
        # 如果有错误，标记为已处理
        new_state["status"] = "error_handled"
    else:
        # 如果没有错误，标记为已完成
        new_state["status"] = "completed"
    
    # 检查是否需要清理当前意图
    current_intent = new_state.get("current_intent")
    if current_intent:
        intent_name = current_intent.get("name", "")
        
        # 对于一次性意图，处理完成后清除
        one_time_intents = ["calculate", "query_weather", "translate", "help"]
        if intent_name in one_time_intents:
            # 保存到意图历史
            if "intent_history" not in new_state:
                new_state["intent_history"] = []
            
            # 添加完成时间
            current_intent["completed_at"] = datetime.now().isoformat()
            
            # 添加到历史
            new_state["intent_history"].append(current_intent)
            
            # 清除当前意图
            new_state["current_intent"] = None
            
            # 记录日志
            logger.info(f"意图已完成并清除: {intent_name}")
    
    # 设置下一步
    new_state["next_step"] = "user_input_processor"
    
    return new_state

# ============== 业务报表处理程序 ==============

async def report_base_handler(state: EnhancedAgentState, report_type: str) -> EnhancedAgentState:
    """报表处理基础函数"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    
    # 检查必要的槽位
    required_slots = {
        "financial_report": ["time_period", "report_type"],
        "project_report": ["project_id", "time_period"],
        "sales_report": ["time_period", "product_category"],
        "hr_report": ["department", "report_type"]
    }
    
    # 获取当前报表类型的必要槽位
    required = required_slots.get(report_type, [])
    missing_slots = []
    
    for slot in required:
        if not slots.get(slot):
            missing_slots.append(slot)
    
    if missing_slots:
        # 如果有缺失的槽位
        slot_prompts = {
            "time_period": "请指定您需要的时间段（例如：上个月、本季度、2023年）",
            "report_type": "请指定报表类型（例如：收入报表、支出报表、利润报表）",
            "project_id": "请提供项目ID或项目名称",
            "product_category": "请指定产品类别",
            "department": "请指定部门名称"
        }
        
        # 构建提示消息
        prompts = [slot_prompts.get(slot, f"请提供{slot}") for slot in missing_slots]
        prompt_text = "\n".join([f"- {p}" for p in prompts])
        
        # 添加提示消息
        new_state["messages"].append({
            "role": "ai",
            "content": f"为了生成{report_type.replace('_', ' ')}，我需要以下信息：\n{prompt_text}"
        })
        
        # 更新状态
        new_state["status"] = "slot_filling"
        return new_state
    
    # 获取检索结果
    retrieval_results = new_state.get("retrieval_results", [])
    
    # 如果没有检索结果，尝试进行检索
    if not retrieval_results:
        # 构建查询
        query_parts = []
        for slot, value in slots.items():
            if value:
                query_parts.append(f"{slot}: {value}")
        
        query = f"{report_type} {' '.join(query_parts)}"
        
        # 更新状态，请求检索
        new_state["status"] = "need_retrieval"
        new_state["next_step"] = "knowledge_retriever"
        
        # 记录日志
        logger.info(f"需要检索知识: {query}")
        
        return new_state
    
    # 根据报表类型调用相应的实现
    if report_type == "financial_report":
        return await financial_report_handler_impl(new_state)
    elif report_type == "project_report":
        return await project_report_handler_impl(new_state)
    elif report_type == "sales_report":
        return await sales_report_handler_impl(new_state)
    elif report_type == "hr_report":
        return await hr_report_handler_impl(new_state)
    else:
        # 未知报表类型
        new_state["error"] = f"未知的报表类型: {report_type}"
        new_state["status"] = "error"
        return new_state

async def financial_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理财务报表意图"""
    return await report_base_handler(state, "financial_report")

async def project_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理项目报表意图"""
    return await report_base_handler(state, "project_report")

async def sales_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理销售报表意图"""
    return await report_base_handler(state, "sales_report")

async def hr_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理人力资源报表意图"""
    return await report_base_handler(state, "hr_report")

async def financial_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """财务报表处理实现"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    time_period = slots.get("time_period", "")
    report_type = slots.get("report_type", "")
    
    # 获取检索结果
    retrieval_results = new_state.get("retrieval_results", [])
    
    try:
        # 构建业务数据
        business_data = {
            "time_period": time_period,
            "report_type": report_type,
            "financial_data": {
                "revenue": {
                    "value": 1250000,
                    "change": 0.15,
                    "breakdown": {
                        "product_sales": 850000,
                        "services": 350000,
                        "other": 50000
                    }
                },
                "expenses": {
                    "value": 750000,
                    "change": 0.08,
                    "breakdown": {
                        "operations": 400000,
                        "marketing": 150000,
                        "r_and_d": 120000,
                        "admin": 80000
                    }
                },
                "profit": {
                    "value": 500000,
                    "change": 0.25,
                    "margin": 0.4
                },
                "cash_flow": {
                    "operating": 450000,
                    "investing": -200000,
                    "financing": -100000,
                    "net_change": 150000
                }
            }
        }
        
        # 使用LLM生成报表分析
        llm = get_llm(temperature=0.2)
        
        # 准备系统提示
        system_prompt = f"""
        你是一位专业的财务分析师。请根据提供的财务数据生成一份专业的财务报表分析。
        
        时间段: {time_period}
        报表类型: {report_type}
        
        财务数据:
        {json.dumps(business_data["financial_data"], indent=2)}
        
        请提供以下内容:
        1. 简短的总体财务状况概述
        2. 关键指标分析（收入、支出、利润等）
        3. 与上期相比的变化趋势
        4. 值得注意的财务亮点或问题
        5. 简短的建议或展望
        
        使用专业但易于理解的语言，保持分析简洁明了。
        """
        
        # 准备检索结果上下文
        context = ""
        if retrieval_results:
            context_items = []
            for item in retrieval_results[:3]:  # 使用前3条检索结果
                content = item.get("content", "")
                source = item.get("source", "未知")
                context_items.append(f"- {content} (来源: {source})")
            
            if context_items:
                context = "相关背景信息:\n" + "\n".join(context_items)
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": context})
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        # 构建报表响应
        report_content = response.content
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"以下是{time_period}的{report_type}分析：\n\n{report_content}"
        })
        
        # 更新状态
        new_state["status"] = "report_generated"
        
        # 记录日志
        logger.info(f"生成财务报表: {time_period} {report_type}")
    
    except Exception as e:
        # 处理错误
        error_message = f"生成财务报表时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请稍后再试。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "report_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def project_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """项目报表处理实现"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    project_id = slots.get("project_id", "")
    time_period = slots.get("time_period", "")
    
    # 获取检索结果
    retrieval_results = new_state.get("retrieval_results", [])
    
    try:
        # 构建业务数据
        business_data = {
            "project_id": project_id,
            "time_period": time_period,
            "project_data": {
                "name": f"Project {project_id}",
                "status": "In Progress",
                "completion": 0.65,
                "start_date": "2023-01-15",
                "end_date": "2023-12-31",
                "budget": {
                    "total": 500000,
                    "spent": 325000,
                    "remaining": 175000,
                    "burn_rate": 35000
                },
                "milestones": [
                    {"name": "Planning", "status": "Completed", "completion": 1.0},
                    {"name": "Design", "status": "Completed", "completion": 1.0},
                    {"name": "Development", "status": "In Progress", "completion": 0.7},
                    {"name": "Testing", "status": "In Progress", "completion": 0.3},
                    {"name": "Deployment", "status": "Not Started", "completion": 0.0}
                ],
                "resources": {
                    "team_members": 12,
                    "allocation": {
                        "developers": 7,
                        "designers": 2,
                        "managers": 1,
                        "qa": 2
                    }
                },
                "risks": [
                    {"name": "Schedule Delay", "severity": "Medium", "mitigation": "Adding resources"},
                    {"name": "Budget Overrun", "severity": "Low", "mitigation": "Cost monitoring"}
                ]
            }
        }
        
        # 使用LLM生成报表分析
        llm = get_llm(temperature=0.2)
        
        # 准备系统提示
        system_prompt = f"""
        你是一位专业的项目管理分析师。请根据提供的项目数据生成一份专业的项目报表分析。
        
        项目ID: {project_id}
        时间段: {time_period}
        
        项目数据:
        {json.dumps(business_data["project_data"], indent=2)}
        
        请提供以下内容:
        1. 项目概述和当前状态
        2. 进度分析（计划vs实际）
        3. 预算使用情况
        4. 里程碑完成情况
        5. 资源分配情况
        6. 风险评估
        7. 建议和下一步行动
        
        使用专业但易于理解的语言，保持分析简洁明了。
        """
        
        # 准备检索结果上下文
        context = ""
        if retrieval_results:
            context_items = []
            for item in retrieval_results[:3]:  # 使用前3条检索结果
                content = item.get("content", "")
                source = item.get("source", "未知")
                context_items.append(f"- {content} (来源: {source})")
            
            if context_items:
                context = "相关背景信息:\n" + "\n".join(context_items)
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": context})
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        # 构建报表响应
        report_content = response.content
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"以下是项目 {project_id} 在 {time_period} 的报表分析：\n\n{report_content}"
        })
        
        # 更新状态
        new_state["status"] = "report_generated"
        
        # 记录日志
        logger.info(f"生成项目报表: 项目 {project_id}, 时间段 {time_period}")
    
    except Exception as e:
        # 处理错误
        error_message = f"生成项目报表时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请稍后再试。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "report_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def sales_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """销售报表处理实现"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    region = slots.get("region", "")
    time_period = slots.get("time_period", "")
    product_line = slots.get("product_line", "")
    
    # 获取检索结果
    retrieval_results = new_state.get("retrieval_results", [])
    
    try:
        # 构建业务数据
        business_data = {
            "region": region,
            "time_period": time_period,
            "product_line": product_line,
            "sales_data": {
                "total_revenue": 2450000,
                "year_over_year_growth": 0.15,
                "quarter_over_quarter_growth": 0.05,
                "units_sold": 12500,
                "average_deal_size": 196000,
                "conversion_rate": 0.28,
                "top_products": [
                    {"name": "Product A", "revenue": 980000, "units": 4900},
                    {"name": "Product B", "revenue": 735000, "units": 3750},
                    {"name": "Product C", "revenue": 490000, "units": 2450},
                    {"name": "Product D", "revenue": 245000, "units": 1400}
                ],
                "sales_channels": {
                    "direct": 0.45,
                    "partners": 0.30,
                    "online": 0.25
                },
                "customer_segments": {
                    "enterprise": 0.55,
                    "mid_market": 0.30,
                    "small_business": 0.15
                },
                "pipeline": {
                    "value": 3675000,
                    "qualified_leads": 85,
                    "average_sales_cycle": 45
                }
            }
        }
        
        # 使用LLM生成报表分析
        llm = get_llm(temperature=0.2)
        
        # 准备系统提示
        system_prompt = f"""
        你是一位专业的销售分析师。请根据提供的销售数据生成一份专业的销售报表分析。
        
        区域: {region}
        时间段: {time_period}
        产品线: {product_line}
        
        销售数据:
        {json.dumps(business_data["sales_data"], indent=2)}
        
        请提供以下内容:
        1. 销售业绩概述
        2. 同比和环比增长分析
        3. 产品表现分析
        4. 销售渠道分析
        5. 客户细分分析
        6. 销售漏斗和转化率分析
        7. 关键洞察和建议
        
        使用专业但易于理解的语言，保持分析简洁明了。
        """
        
        # 准备检索结果上下文
        context = ""
        if retrieval_results:
            context_items = []
            for item in retrieval_results[:3]:  # 使用前3条检索结果
                content = item.get("content", "")
                source = item.get("source", "未知")
                context_items.append(f"- {content} (来源: {source})")
            
            if context_items:
                context = "相关背景信息:\n" + "\n".join(context_items)
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": context})
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        # 构建报表响应
        report_content = response.content
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"以下是{region}地区{product_line}在{time_period}的销售报表分析：\n\n{report_content}"
        })
        
        # 更新状态
        new_state["status"] = "report_generated"
        
        # 记录日志
        logger.info(f"生成销售报表: 区域 {region}, 产品线 {product_line}, 时间段 {time_period}")
    
    except Exception as e:
        # 处理错误
        error_message = f"生成销售报表时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请稍后再试。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "report_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

async def hr_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """人力资源报表处理实现"""
    # 复制状态
    new_state = state.copy()
    
    # 获取槽位
    slots = new_state.get("slots", {})
    department = slots.get("department", "")
    time_period = slots.get("time_period", "")
    report_type = slots.get("report_type", "")
    
    # 获取检索结果
    retrieval_results = new_state.get("retrieval_results", [])
    
    try:
        # 构建业务数据
        business_data = {
            "department": department,
            "time_period": time_period,
            "report_type": report_type,
            "hr_data": {
                "headcount": {
                    "total": 125,
                    "new_hires": 15,
                    "terminations": 8,
                    "growth_rate": 0.06
                },
                "demographics": {
                    "gender_ratio": {"male": 0.55, "female": 0.45},
                    "age_distribution": {
                        "20-30": 0.25,
                        "31-40": 0.45,
                        "41-50": 0.20,
                        "51+": 0.10
                    },
                    "tenure": {
                        "0-1 year": 0.20,
                        "1-3 years": 0.35,
                        "3-5 years": 0.25,
                        "5+ years": 0.20
                    }
                },
                "performance": {
                    "average_rating": 3.7,
                    "rating_distribution": {
                        "5": 0.15,
                        "4": 0.45,
                        "3": 0.30,
                        "2": 0.08,
                        "1": 0.02
                    },
                    "promotion_rate": 0.12
                },
                "compensation": {
                    "average_salary": 85000,
                    "salary_increase": 0.04,
                    "bonus_distribution": 0.08
                },
                "engagement": {
                    "satisfaction_score": 7.8,
                    "participation_rate": 0.85,
                    "top_concerns": [
                        "工作生活平衡",
                        "职业发展",
                        "团队协作"
                    ]
                },
                "turnover": {
                    "rate": 0.064,
                    "voluntary_rate": 0.048,
                    "involuntary_rate": 0.016,
                    "top_reasons": [
                        "职业发展机会",
                        "薪酬",
                        "工作环境"
                    ]
                }
            }
        }
        
        # 使用LLM生成报表分析
        llm = get_llm(temperature=0.2)
        
        # 准备系统提示
        system_prompt = f"""
        你是一位专业的人力资源分析师。请根据提供的HR数据生成一份专业的人力资源报表分析。
        
        部门: {department}
        时间段: {time_period}
        报表类型: {report_type}
        
        HR数据:
        {json.dumps(business_data["hr_data"], indent=2)}
        
        请提供以下内容:
        1. 人员概况分析
        2. 人口统计学分析
        3. 绩效分析
        4. 薪酬分析
        5. 员工敬业度分析
        6. 离职率分析
        7. 关键洞察和建议
        
        使用专业但易于理解的语言，保持分析简洁明了。
        """
        
        # 准备检索结果上下文
        context = ""
        if retrieval_results:
            context_items = []
            for item in retrieval_results[:3]:  # 使用前3条检索结果
                content = item.get("content", "")
                source = item.get("source", "未知")
                context_items.append(f"- {content} (来源: {source})")
            
            if context_items:
                context = "相关背景信息:\n" + "\n".join(context_items)
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": context})
        
        # 请求LLM
        response = await llm.ainvoke(messages)
        
        # 构建报表响应
        report_content = response.content
        
        # 添加响应到消息历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"以下是{department}部门在{time_period}的{report_type}人力资源报表分析：\n\n{report_content}"
        })
        
        # 更新状态
        new_state["status"] = "report_generated"
        
        # 记录日志
        logger.info(f"生成HR报表: 部门 {department}, 时间段 {time_period}, 类型 {report_type}")
    
    except Exception as e:
        # 处理错误
        error_message = f"生成HR报表时出错: {str(e)}"
        
        # 添加错误消息到历史
        new_state["messages"].append({
            "role": "ai",
            "content": f"抱歉，{error_message} 请稍后再试。"
        })
        
        # 设置错误状态
        new_state["error"] = error_message
        new_state["status"] = "report_error"
        
        # 记录日志
        logger.error(error_message)
    
    return new_state

# ============== 图形构建 ==============

def build_graph() -> StateGraph:
    """构建状态图
    
    创建并配置代理状态图，定义各节点和边的连接关系
    
    Returns:
        配置好的状态图
    """
    # 创建状态图
    workflow = StateGraph(EnhancedAgentState)
    
    # 添加节点
    workflow.add_node("user_input_processor", user_input_processor)
    workflow.add_node("check_return_to_previous", check_return_to_previous)
    workflow.add_node("intent_recognizer", intent_recognizer)
    workflow.add_node("slot_filler", slot_filler)
    workflow.add_node("knowledge_retriever", knowledge_retriever)
    workflow.add_node("calculator_handler", calculator_handler)
    workflow.add_node("weather_handler", weather_handler)
    workflow.add_node("translate_handler", translate_handler)
    workflow.add_node("chitchat_handler", chitchat_handler)
    workflow.add_node("help_handler", help_handler)
    workflow.add_node("fallback_handler", fallback_handler)
    workflow.add_node("financial_report_handler", financial_report_handler)
    workflow.add_node("project_report_handler", project_report_handler)
    workflow.add_node("sales_report_handler", sales_report_handler) 
    workflow.add_node("hr_report_handler", hr_report_handler)
    workflow.add_node("router", router)
    workflow.add_node("handle_breakpoint", handle_breakpoint)
    workflow.add_node("process_human_input", process_human_input)
    
    # 设置入口节点
    workflow.set_entry_point("user_input_processor")
    
    # 定义条件函数
    def should_check_return(state: EnhancedAgentState) -> bool:
        """检查是否需要检查返回到之前的话题"""
        return len(state.get("suspended_intents", [])) > 0
    
    def should_handle_breakpoint(state: EnhancedAgentState) -> bool:
        """检查是否需要处理断点"""
        return state.get("human_input_required", False)
    
    def needs_human_input(state: EnhancedAgentState) -> bool:
        """检查是否需要等待人工输入"""
        return state.get("status") == "awaiting_human_input"
    
    def needs_slot_filling(state: EnhancedAgentState) -> bool:
        """检查是否需要填充槽位"""
        return state.get("status") in ["slot_filling", "slots_partially_filled", "awaiting_slot"]
    
    def is_awaiting_intent_confirmation(state: EnhancedAgentState) -> bool:
        """检查是否需要确认意图"""
        return state.get("status") == "need_confirmation"
    
    # 添加边
    # 用户输入处理器 -> 检查返回到之前的话题或意图识别
    workflow.add_conditional_edges(
        "user_input_processor",
        should_check_return,
        {
            True: "check_return_to_previous",
            False: "intent_recognizer"
        }
    )
    
    # 检查返回到之前的话题 -> 意图识别或下一步
    workflow.add_conditional_edges(
        "check_return_to_previous",
        lambda state: state.get("next_step") != "intent_recognizer",
        {
            True: "slot_filler",  # 已恢复意图，继续填充槽位
            False: "intent_recognizer"  # 未恢复意图，继续意图识别
        }
    )
    
    # 意图识别 -> 路由或槽位填充
    workflow.add_conditional_edges(
        "intent_recognizer",
        lambda state: state.get("status") == "intent_recognized" and not needs_slot_filling(state),
        {
            True: "router",  # 意图已识别且不需要槽位填充
            False: "slot_filler"  # 需要填充槽位
        }
    )
    
    # 槽位填充 -> 路由或继续填充
    workflow.add_conditional_edges(
        "slot_filler",
        lambda state: state.get("status") == "slots_filled",
        {
            True: "router",  # 所有槽位已填充
            False: "slot_filler"  # 继续填充槽位
        }
    )
    
    # 所有处理节点 -> 结束
    for handler in [
        "calculator_handler", "weather_handler", "translate_handler",
        "chitchat_handler", "help_handler", "fallback_handler",
        "financial_report_handler", "project_report_handler",
        "sales_report_handler", "hr_report_handler"
    ]:
        workflow.add_edge(handler, END)
    
    # 路由到各处理器
    workflow.add_router("router", router)
    
    # 编译图
    return workflow.compile()

# 创建图
compiled_graph = build_graph()

# 创建内存保存器
memory_saver = MemorySaver()

# 获取异步LLM
def get_async_llm(temperature=0):
    """获取异步LLM实例"""
    # 获取环境变量
    api_key = os.getenv("OPENAI_API_KEY", "fk222719-4TlnHx5wbaXtUm4CcneT1oLogM3TKGDB")
    api_base = os.getenv("OPENAI_API_BASE", "https://oa.api2d.net")
    model_name = os.getenv("OPENAI_MODEL_NAME", "o3-mini")
    
    # 创建异步LLM实例
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base,
        streaming=True  # 启用流式处理
    )
    
    return llm

# 断点检查函数
def should_pause(state: EnhancedAgentState) -> bool:
    """检查是否需要暂停执行"""
    # 检查断点设置
    if state["breakpoints"].get(state["status"], False):
        logger.info(f"触发断点: {state['status']}")
        return True
    
    # 检查置信度
    if state["confidence"] < 0.7 and state.get("status") not in ["waiting_for_human", "processing_human_input"]:
        logger.info(f"置信度低于阈值({state['confidence']}), 当前状态: {state['status']}")
        return True
    
    # 检查错误状态
    if state.get("error") and state.get("status") != "error_handled":
        logger.info(f"发现错误, 暂停执行: {state.get('error')}")
        return True
    
    # 特殊条件检查
    current_intent = state.get("current_intent", {})
    if current_intent:
        intent_name = current_intent.get("name", "")
        
        # 对于某些高风险操作，需要人工确认
        high_risk_intents = ["delete_data", "modify_critical_settings", "approve_large_transaction"]
        if intent_name in high_risk_intents:
            logger.info(f"高风险意图需要人工确认: {intent_name}")
            return True
        
        # 对于报表生成，检查是否有足够的检索结果
        if "report" in intent_name and state.get("status") == "knowledge_retrieved":
            retrieval_results = state.get("retrieval_results", [])
            if len(retrieval_results) < 2:  # 至少需要2条结果
                logger.info(f"报表生成检索结果不足({len(retrieval_results)}), 可能需要人工干预")
                return True
    
    return False

# 处理断点函数
def handle_breakpoint(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理断点状态"""
    # 复制状态
    new_state = state.copy()
    
    # 设置等待人工输入标志
    new_state["human_input_required"] = True
    new_state["status"] = "waiting_for_human"
    
    # 添加系统消息
    new_state["messages"].append({
        "role": "system",
        "content": f"执行暂停，等待人工干预。当前状态: {state['status']}"
    })
    
    return new_state

# 处理人工输入函数
def process_human_input(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理人工输入"""
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
    
    # 添加人工消息
    new_state["messages"].append({
        "role": "system",
        "content": f"人工干预: {human_input}"
    })
    
    # 解析指令
    if human_input.startswith("/"):
        parts = human_input[1:].split()
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
        elif command.startswith("breakpoint"):
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
                "content": f"未知命令: {command}"
            })
            new_state["status"] = "user_input_processor"
    else:
        # 作为普通消息处理
        new_state["messages"].append({
            "role": "human",
            "content": human_input
        })
        new_state["status"] = "user_input_processor"
    
    return new_state

# 异步流式处理
async def astream_message(session_id: str, message: str) -> AsyncIterator[Dict[str, Any]]:
    """异步流式处理消息"""
    # 获取会话状态
    session_state = await get_or_create_session_state(session_id)
    
    # 创建图实例
    checkpointer = get_checkpointer()
    graph = build_graph()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    
    # 添加用户消息
    user_message = {"role": "human", "content": message}
    session_state["messages"].append(user_message)
    session_state["last_updated_at"] = datetime.now().isoformat()
    
    # 如果需要人工输入，将消息作为人工输入处理
    if session_state.get("human_input_required"):
        session_state["human_input"] = message
    
    # 异步流式处理
    config = {"configurable": {"thread_id": session_id}}
    async for chunk in compiled_graph.astream(session_state, config):
        if isinstance(chunk, dict) and chunk.get("messages"):
            # 找到最新的消息
            messages = chunk.get("messages", [])
            if messages and len(messages) > 0:
                latest_message = messages[-1]
                
                # 只考虑AI消息
                if isinstance(latest_message, dict) and latest_message.get("role") == "ai":
                    content = latest_message.get("content", "")
                    yield {"content": content, "session_id": session_id}
                    
    # 更新会话状态
    SESSION_STATES[session_id] = session_state
    
    # 返回完成状态
    yield {"content": "[DONE]", "session_id": session_id}

async def aprocess_message(session_id: str, message: str) -> Dict[str, Any]:
    """异步处理用户消息
    
    处理用户输入的消息，执行对话代理的完整流程
    
    Args:
        session_id: 会话ID
        message: 用户消息
        
    Returns:
        处理结果，包括会话状态和响应消息
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
        
        # 更新最后活动时间
        state["last_updated_at"] = datetime.now().isoformat()
        
        # 添加用户消息到历史
        state["messages"].append({
            "role": "human",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 使用已编译的图执行一次完整流程
        state = await compiled_graph.ainvoke(state)
        
        # 获取最后一条助手消息
        last_assistant_message = None
        for msg in reversed(state["messages"]):
            if msg.get("role") in ["assistant", "ai"]:
                last_assistant_message = msg.get("content", "")
                break
        
        if not last_assistant_message:
            last_assistant_message = "抱歉，我无法处理您的请求。"
        
        # 保存会话状态
        await save_session_state(session_id, state)
        
        # 返回处理结果
        return {
            "session_id": session_id,
            "message": last_assistant_message,
            "state": state
        }
        
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}", exc_info=True)
        # 返回错误信息
        return {
            "session_id": session_id,
            "message": f"处理您的消息时出现错误: {str(e)}",
            "error": str(e)
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