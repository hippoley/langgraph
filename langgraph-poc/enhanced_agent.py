"""
增强的对话代理 - 整合意图管理、业务指标和RAG功能
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, TypedDict, Literal, Annotated
from dataclasses import dataclass, field, asdict

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
    from langgraph.checkpoint import MemorySaver
    from intent_manager import IntentManager, Intent, Slot, IntentState
    from business_metrics import BusinessMetricRegistry, create_default_metrics
    from enhanced_rag import EnhancedRetriever, RetrievalResult, create_sample_knowledge_base
    from api_config import get_llm
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

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
    context: Dict[str, Any]                  # 上下文信息
    
    # 业务数据
    business_data: Optional[Dict[str, Any]]   # 业务数据
    
    # 控制标志
    status: str                   # 状态标志
    error: Optional[str]          # 错误信息
    should_continue: bool         # 是否继续
    human_intervention: bool      # 是否需要人工干预

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
        context={},
        business_data=None,
        status="initialized",
        error=None,
        should_continue=True,
        human_intervention=False
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

def user_input_processor(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理用户输入"""
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有用户消息，不做任何处理
        return state
    
    # 更新状态
    state["last_updated_at"] = datetime.now().isoformat()
    state["status"] = "processing_input"
    
    # 记录日志
    logger.info(f"处理用户输入: {user_message[:50]}...")
    
    return state

def intent_recognizer(state: EnhancedAgentState) -> EnhancedAgentState:
    """识别意图"""
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有用户消息，不做任何处理
        return state
    
    # 识别意图
    intent_name, confidence = intent_manager.identify_intent(user_message)
    
    # 如果识别到意图
    if intent_name:
        # 检查当前是否有活跃意图
        if state["current_intent"]:
            current_intent_name = state["current_intent"].get("name")
            # 如果意图不同，可能是中断
            if intent_name != current_intent_name:
                # 检测是否是中断
                if intent_manager.is_interruption(user_message):
                    # 挂起当前意图
                    current_intent = state["current_intent"].copy()
                    # 标记为挂起
                    current_intent["state"] = "suspended"
                    # 添加到挂起栈
                    state["suspended_intents"].append(current_intent)
                    # 清除当前意图
                    state["current_intent"] = None
                    
                    # 记录中断事件
                    state["context"]["last_interruption"] = {
                        "intent": current_intent_name,
                        "time": datetime.now().isoformat()
                    }
                    
                    # 记录日志
                    logger.info(f"用户中断当前意图: {current_intent_name}")
        
        # 如果没有当前意图或已被挂起，设置新意图
        if not state["current_intent"]:
            # 获取意图定义
            intent = intent_manager.intents[intent_name]
            # 创建新的当前意图
            state["current_intent"] = {
                "name": intent_name,
                "description": intent.description,
                "confidence": confidence,
                "state": "active",
                "slots": {slot.name: None for slot in intent.slots},
                "handler": intent.handler,
                "business_metric": intent.business_metric,
                "created_at": datetime.now().isoformat()
            }
            
            # 添加到意图栈
            state["intent_stack"].append(state["current_intent"])
            
            # 记录日志
            logger.info(f"识别到新意图: {intent_name} (置信度: {confidence:.2f})")
    else:
        # 如果没有识别到明确的意图
        if not state["current_intent"]:
            # 设置默认为闲聊意图
            state["current_intent"] = {
                "name": "chitchat",
                "description": "闲聊",
                "confidence": 0.5,
                "state": "active",
                "slots": {},
                "handler": "handle_chitchat",
                "business_metric": None,
                "created_at": datetime.now().isoformat()
            }
            
            # 添加到意图栈
            state["intent_stack"].append(state["current_intent"])
            
            # 记录日志
            logger.info("未识别到明确意图，默认为闲聊")
    
    # 更新状态
    state["status"] = "intent_recognized"
    
    return state

def check_return_to_previous(state: EnhancedAgentState) -> EnhancedAgentState:
    """检查是否返回到之前的意图"""
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有用户消息，不做任何处理
        return state
    
    # 检查是否是返回之前话题的请求
    if intent_manager.is_return_to_previous(user_message) and state["suspended_intents"]:
        # 从挂起栈中恢复最近的意图
        previous_intent = state["suspended_intents"].pop()
        
        # 恢复为当前意图
        state["current_intent"] = previous_intent
        
        # 更新状态
        state["current_intent"]["state"] = "active"
        state["current_intent"]["resumed_at"] = datetime.now().isoformat()
        
        # 记录日志
        logger.info(f"返回到之前的意图: {previous_intent['name']}")
        
        # 添加系统消息
        state["messages"].append({
            "role": "system",
            "content": f"已返回到关于"{previous_intent['description']}"的话题。"
        })
    
    # 更新状态
    state["status"] = "checked_return"
    
    return state

def slot_filler(state: EnhancedAgentState) -> EnhancedAgentState:
    """填充槽位"""
    # 获取当前意图
    current_intent = state.get("current_intent", {})
    intent_name = current_intent.get("name")
    
    if not intent_name or intent_name not in intent_manager.intents:
        return state
    
    # 获取意图定义
    intent = intent_manager.intents[intent_name]
    
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        # 如果没有用户消息，不做任何处理
        return state
    
    # 尝试从用户消息中提取槽位值
    slots_filled = False
    
    # 获取未填充的槽位
    unfilled_slots = [
        slot for slot in intent.slots 
        if slot.required and not current_intent.get("slots", {}).get(slot.name)
    ]
    
    if unfilled_slots:
        # 获取第一个未填充的槽位
        slot = unfilled_slots[0]
        
        # 尝试提取槽位值
        slot_value = intent_manager.extract_slot_value(slot, user_message)
        
        if slot_value:
            # 更新槽位值
            current_intent["slots"][slot.name] = slot_value
            slots_filled = True
            
            # 记录日志
            logger.info(f"已填充槽位 {slot.name}: {slot_value}")
    
    # 检查是否还有未填充的槽位
    unfilled_required_slots = [
        slot for slot in intent.slots 
        if slot.required and not current_intent.get("slots", {}).get(slot.name)
    ]
    
    if unfilled_required_slots:
        # 获取下一个槽位的提示
        next_slot = unfilled_required_slots[0]
        prompt = next_slot.prompt or f"请提供{next_slot.description}"
        
        # 添加助手消息，请求槽位值
        state["messages"].append({
            "role": "assistant",
            "content": prompt
        })
        
        # 更新状态
        state["status"] = "slot_filling"
    else:
        # 所有必填槽位已填充
        state["status"] = "slots_filled"
    
    return state

def knowledge_retriever(state: EnhancedAgentState) -> EnhancedAgentState:
    """检索知识"""
    # 获取当前意图
    current_intent = state.get("current_intent", {})
    intent_name = current_intent.get("name")
    business_metric = current_intent.get("business_metric")
    
    if not intent_name:
        return state
    
    # 获取用户查询
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        return state
    
    # 确定相关的知识源
    knowledge_sources = []
    if business_metric:
        if business_metric == "financial_report":
            knowledge_sources.append("financial_knowledge")
        elif business_metric == "project_report":
            knowledge_sources.append("project_knowledge")
        elif business_metric == "sales_report":
            knowledge_sources.append("sales_knowledge")
        elif business_metric == "hr_report":
            knowledge_sources.append("hr_knowledge")
    
    # 如果没有明确的知识源，尝试通过向量相似度查找
    if not knowledge_sources and business_metric:
        metric_name = metric_registry.route_query_to_metric(user_message)
        if metric_name:
            if metric_name == "financial_report":
                knowledge_sources.append("financial_knowledge")
            elif metric_name == "project_report":
                knowledge_sources.append("project_knowledge")
            elif metric_name == "sales_report":
                knowledge_sources.append("sales_knowledge")
            elif metric_name == "hr_report":
                knowledge_sources.append("hr_knowledge")
    
    # 如果仍然没有知识源，使用所有知识源
    if not knowledge_sources:
        knowledge_sources = [
            "financial_knowledge",
            "project_knowledge",
            "sales_knowledge",
            "hr_knowledge"
        ]
    
    # 获取对话上下文
    context = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
    context_dicts = [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in context
    ]
    
    # 执行检索
    try:
        results = retriever.retrieve(
            query=user_message,
            source_names=knowledge_sources,
            top_k=3,
            context=context_dicts
        )
        
        # 重新排序结果
        results = retriever.rerank_results(
            query=user_message,
            results=results,
            context=context_dicts
        )
        
        # 更新检索结果
        state["retrieval_results"] = [
            {
                "content": result.content,
                "source": result.source,
                "relevance": result.relevance,
                "metadata": result.metadata
            }
            for result in results
        ]
        
        # 记录日志
        logger.info(f"检索到{len(results)}个结果，来自知识源: {', '.join(knowledge_sources)}")
    except Exception as e:
        # 记录错误
        logger.error(f"检索失败: {str(e)}")
        state["error"] = f"检索失败: {str(e)}"
    
    # 更新状态
    state["status"] = "knowledge_retrieved"
    
    return state

def calculator_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理计算意图"""
    # 获取当前意图和槽位
    current_intent = state.get("current_intent", {})
    slots = current_intent.get("slots", {})
    
    # 获取表达式
    expression = slots.get("expression")
    
    if not expression:
        # 如果没有表达式，请求表达式
        state["messages"].append({
            "role": "assistant",
            "content": "请提供要计算的数学表达式。"
        })
        state["status"] = "slot_filling"
        return state
    
    # 计算表达式
    try:
        result = tools["calculator"](expression)
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": f"计算结果: {result}"
        })
        
        # 记录日志
        logger.info(f"计算表达式: {expression} = {result}")
    except Exception as e:
        # 处理错误
        error_message = f"计算失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": error_message
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def weather_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理天气意图"""
    # 获取当前意图和槽位
    current_intent = state.get("current_intent", {})
    slots = current_intent.get("slots", {})
    
    # 获取位置
    location = slots.get("location")
    
    if not location:
        # 如果没有位置，请求位置
        state["messages"].append({
            "role": "assistant",
            "content": "请问您想查询哪个地方的天气？"
        })
        state["status"] = "slot_filling"
        return state
    
    # 获取天气信息
    try:
        weather_info = tools["get_weather"](location)
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": f"{location}的天气: {weather_info}"
        })
        
        # 记录日志
        logger.info(f"查询天气: {location} - {weather_info}")
    except Exception as e:
        # 处理错误
        error_message = f"获取天气信息失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": error_message
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def translate_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理翻译意图"""
    # 获取当前意图和槽位
    current_intent = state.get("current_intent", {})
    slots = current_intent.get("slots", {})
    
    # 获取文本和目标语言
    text = slots.get("text")
    target_language = slots.get("target_language")
    
    if not text:
        # 如果没有文本，请求文本
        state["messages"].append({
            "role": "assistant",
            "content": "请提供要翻译的文本。"
        })
        state["status"] = "slot_filling"
        return state
    
    if not target_language:
        # 如果没有目标语言，请求目标语言
        state["messages"].append({
            "role": "assistant",
            "content": "请问您想翻译成哪种语言？（支持中文、英文）"
        })
        state["status"] = "slot_filling"
        return state
    
    # 执行翻译
    try:
        translated_text = tools["translate"](text, target_language)
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": f"'{text}'的{target_language}翻译是: {translated_text}"
        })
        
        # 记录日志
        logger.info(f"翻译: {text} -> {translated_text} ({target_language})")
    except Exception as e:
        # 处理错误
        error_message = f"翻译失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": error_message
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def chitchat_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理闲聊意图"""
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        return state
    
    # 使用LLM生成回复
    try:
        llm = get_llm()
        
        system_prompt = """
        你是一个友好、礼貌的助手。请针对用户的问候或一般性问题提供简洁、友好的回复。
        保持简短自然，不要过于正式或机械。
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        # 记录日志
        logger.info(f"闲聊响应: {response.content[:50]}...")
    except Exception as e:
        # 处理错误
        error_message = f"生成响应失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": "抱歉，我现在无法回答您的问题。请稍后再试。"
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def help_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理帮助意图"""
    # 生成帮助信息
    help_message = """
    我可以帮助您完成以下任务：

    1. 查询报表：
       - 财务报表：查看收入、支出、利润等财务数据
       - 项目报表：了解项目进度、资源使用情况等
       - 销售报表：查看销售数据、客户数量、产品销量等
       - 人力资源报表：了解人员配置、招聘情况等

    2. 实用工具：
       - 计算器：计算数学表达式，如"计算2+2"
       - 天气查询：查询各地天气，如"北京今天天气怎么样"
       - 翻译：在中英文之间翻译文本

    您可以直接提问，例如"查看上个季度的财务报表"或"北京今天天气怎么样"。
    如果我需要更多信息，会向您提问。
    
    在对话过程中，您可以随时转换话题，之后可以说"回到之前的话题"继续之前的对话。
    """
    
    # 添加响应
    state["messages"].append({
        "role": "assistant",
        "content": help_message
    })
    
    # 记录日志
    logger.info("提供帮助信息")
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def fallback_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理未知意图"""
    # 获取最新的用户消息
    user_message = None
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        return state
    
    # 使用LLM生成回复
    try:
        llm = get_llm()
        
        system_prompt = """
        你是一个友好、礼貌的助手。请针对用户的问题提供一个友好的回复，
        表示你理解他们的请求，但可能需要更多信息。提供一些可能的帮助选项。
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        # 记录日志
        logger.info(f"备用处理程序响应: {response.content[:50]}...")
    except Exception as e:
        # 处理错误
        error_message = f"生成响应失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": "抱歉，我不太理解您的请求。您可以尝试重新表述，或者输入'帮助'查看我能做什么。"
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def router(state: EnhancedAgentState) -> str:
    """路由到下一个节点"""
    # 获取当前意图和状态
    current_intent = state.get("current_intent", {})
    current_status = state.get("status", "")
    
    # 如果没有当前意图，返回到意图识别
    if not current_intent:
        return "intent_recognizer"
    
    # 检查是否有未填充的槽位
    intent_name = current_intent.get("name", "")
    if intent_name in intent_manager.intents:
        intent = intent_manager.intents[intent_name]
        unfilled_required_slots = [
            slot for slot in intent.slots 
            if slot.required and not current_intent.get("slots", {}).get(slot.name)
        ]
        
        if unfilled_required_slots:
            return "slot_filler"
    
    # 根据意图进行路由
    if intent_name == "query_financial_report":
        return "financial_report_handler"
    elif intent_name == "query_project_report":
        return "project_report_handler"
    elif intent_name == "query_sales_report":
        return "sales_report_handler"
    elif intent_name == "query_hr_report":
        return "hr_report_handler"
    elif intent_name == "query_weather":
        return "weather_handler"
    elif intent_name == "translate":
        return "translate_handler"
    elif intent_name == "calculate":
        return "calculator_handler"
    elif intent_name == "help":
        return "help_handler"
    elif intent_name == "chitchat":
        return "chitchat_handler"
    else:
        return "fallback_handler"

def update_state(state: EnhancedAgentState) -> EnhancedAgentState:
    """更新状态，完成当前处理"""
    # 当前意图已经处理完成
    if state["current_intent"]:
        state["current_intent"]["state"] = "completed"
        state["current_intent"]["completed_at"] = datetime.now().isoformat()
    
    # 更新时间戳
    state["last_updated_at"] = datetime.now().isoformat()
    
    # 设置继续标志
    state["should_continue"] = True
    
    # 更新状态
    state["status"] = "updated"
    
    return state

# ============== 业务报表处理程序 ==============

def report_base_handler(state: EnhancedAgentState, report_type: str) -> EnhancedAgentState:
    """报表处理程序的基础实现"""
    # 获取当前意图和槽位
    current_intent = state.get("current_intent", {})
    slots = current_intent.get("slots", {})
    
    # 确保所有必要的槽位都已填充
    if report_type == "financial_report":
        required_slots = ["time_period", "report_type"]
    elif report_type == "project_report":
        required_slots = ["project_name", "report_aspect"]
    elif report_type == "sales_report":
        required_slots = ["time_period"]
    elif report_type == "hr_report":
        required_slots = ["report_aspect"]
    else:
        required_slots = []
    
    # 检查是否所有必要槽位都已填充
    missing_slots = [slot for slot in required_slots if not slots.get(slot)]
    if missing_slots:
        # 缺少必要槽位，返回并请求填充
        slot_name = missing_slots[0]
        state["messages"].append({
            "role": "assistant",
            "content": f"请提供{slot_name}信息。"
        })
        state["status"] = "slot_filling"
        return state
    
    # 先执行知识检索
    state = knowledge_retriever(state)
    
    # 处理不同类型的报表
    if report_type == "financial_report":
        return financial_report_handler_impl(state)
    elif report_type == "project_report":
        return project_report_handler_impl(state)
    elif report_type == "sales_report":
        return sales_report_handler_impl(state)
    elif report_type == "hr_report":
        return hr_report_handler_impl(state)
    else:
        # 未知报表类型
        state["messages"].append({
            "role": "assistant",
            "content": "抱歉，我不支持这种类型的报表查询。"
        })
        state["status"] = "error"
        state["error"] = f"未知报表类型: {report_type}"
        return state

def financial_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理财务报表意图"""
    return report_base_handler(state, "financial_report")

def project_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理项目报表意图"""
    return report_base_handler(state, "project_report")

def sales_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理销售报表意图"""
    return report_base_handler(state, "sales_report")

def hr_report_handler(state: EnhancedAgentState) -> EnhancedAgentState:
    """处理人力资源报表意图"""
    return report_base_handler(state, "hr_report")

def financial_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """财务报表处理程序实现"""
    # 获取槽位值
    slots = state["current_intent"]["slots"]
    time_period = slots.get("time_period", "")
    report_type = slots.get("report_type", "")
    
    # 获取检索结果
    retrieval_results = state.get("retrieval_results", [])
    
    # 生成报表数据（模拟数据）
    business_data = {
        "report_type": "financial",
        "time_period": time_period,
        "report_subtype": report_type,
        "data": {
            "revenue": "1,350,000",
            "expenses": "950,000",
            "profit": "400,000",
            "growth": "+12%",
            "details": {
                "products": "850,000",
                "services": "500,000",
                "operations": "550,000",
                "marketing": "250,000",
                "administrative": "150,000"
            }
        }
    }
    
    # 更新业务数据
    state["business_data"] = business_data
    
    # 使用LLM生成报表响应
    try:
        llm = get_llm()
        
        # 构建提示
        system_prompt = """
        你是一个专业的财务分析助手。请根据提供的财务数据和检索信息，生成一份简洁、专业的财务报表分析。
        分析应包括以下内容：
        1. 简要概述报表主要指标
        2. 与相关时期的比较和变化趋势
        3. 关键财务指标的解读
        4. 结论或建议（如果适用）
        
        保持分析简洁、专业、易于理解。
        """
        
        # 构建检索信息
        retrieval_info = "\n\n".join([
            f"[检索结果 {i+1}, 相关度: {result.get('relevance', 0):.2f}]\n{result.get('content', '')}"
            for i, result in enumerate(retrieval_results)
        ])
        
        # 构建业务数据信息
        business_data_info = json.dumps(business_data, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        请根据以下信息生成一份财务报表分析：
        
        查询信息：
        - 报表类型: {report_type}
        - 时间段: {time_period}
        
        检索信息：
        {retrieval_info}
        
        报表数据：
        {business_data_info}
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        # 记录日志
        logger.info(f"生成财务报表分析，时间段: {time_period}, 类型: {report_type}")
    except Exception as e:
        # 处理错误
        error_message = f"生成财务报表分析失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"抱歉，生成{time_period}的{report_type}报表分析时出现错误。请稍后再试。"
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def project_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """项目报表处理程序实现"""
    # 获取槽位值
    slots = state["current_intent"]["slots"]
    project_name = slots.get("project_name", "")
    report_aspect = slots.get("report_aspect", "")
    
    # 获取检索结果
    retrieval_results = state.get("retrieval_results", [])
    
    # 生成报表数据（模拟数据）
    business_data = {
        "report_type": "project",
        "project_name": project_name,
        "report_aspect": report_aspect,
        "data": {
            "completion": "65%",
            "timeline": {
                "start_date": "2023-01-15",
                "planned_end_date": "2023-09-30",
                "current_phase": "实施",
                "next_milestone": "系统测试",
                "next_milestone_date": "2023-07-15"
            },
            "resources": {
                "team_members": "8",
                "allocated_budget": "200万",
                "budget_used": "120万",
                "remaining_budget": "80万"
            },
            "risks": [
                {"name": "技术挑战", "level": "中", "mitigation": "增加技术专家支持"},
                {"name": "时间压力", "level": "高", "mitigation": "调整范围和优先级"}
            ]
        }
    }
    
    # 更新业务数据
    state["business_data"] = business_data
    
    # 使用LLM生成报表响应
    try:
        llm = get_llm()
        
        # 构建提示
        system_prompt = """
        你是一个专业的项目管理助手。请根据提供的项目数据和检索信息，生成一份简洁、专业的项目报表分析。
        分析应包括以下内容：
        1. 项目当前状态概述
        2. 具体关注方面的详细分析（进度、资源、预算或风险）
        3. 相关建议或下一步行动
        
        保持分析简洁、专业、易于理解，并聚焦在用户关注的方面。
        """
        
        # 构建检索信息
        retrieval_info = "\n\n".join([
            f"[检索结果 {i+1}, 相关度: {result.get('relevance', 0):.2f}]\n{result.get('content', '')}"
            for i, result in enumerate(retrieval_results)
        ])
        
        # 构建业务数据信息
        business_data_info = json.dumps(business_data, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        请根据以下信息生成一份项目报表分析：
        
        查询信息：
        - 项目名称: {project_name}
        - 关注方面: {report_aspect}
        
        检索信息：
        {retrieval_info}
        
        报表数据：
        {business_data_info}
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        # 记录日志
        logger.info(f"生成项目报表分析，项目: {project_name}, 方面: {report_aspect}")
    except Exception as e:
        # 处理错误
        error_message = f"生成项目报表分析失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"抱歉，生成{project_name}的{report_aspect}报表分析时出现错误。请稍后再试。"
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def sales_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """销售报表处理程序实现"""
    # 获取槽位值
    slots = state["current_intent"]["slots"]
    time_period = slots.get("time_period", "")
    product = slots.get("product", "所有产品")
    
    # 获取检索结果
    retrieval_results = state.get("retrieval_results", [])
    
    # 生成报表数据（模拟数据）
    business_data = {
        "report_type": "sales",
        "time_period": time_period,
        "product": product,
        "data": {
            "total_sales": "480万元",
            "growth_rate": "+6.7%",
            "new_customers": "25",
            "products": {
                "产品A": {"sales": "180万元", "growth": "+3%", "units": "900"},
                "产品B": {"sales": "150万元", "growth": "+30%", "units": "600"},
                "产品C": {"sales": "150万元", "growth": "-5%", "units": "300"}
            },
            "regions": {
                "华北": "150万元",
                "华东": "180万元",
                "华南": "100万元",
                "西部": "50万元"
            }
        }
    }
    
    # 更新业务数据
    state["business_data"] = business_data
    
    # 使用LLM生成报表响应
    try:
        llm = get_llm()
        
        # 构建提示
        system_prompt = """
        你是一个专业的销售分析助手。请根据提供的销售数据和检索信息，生成一份简洁、专业的销售报表分析。
        分析应包括以下内容：
        1. 销售总体情况概述
        2. 产品销售表现分析
        3. 区域销售分布
        4. 重要趋势和亮点
        5. 相关建议或下一步行动
        
        如果查询的是特定产品，则重点分析该产品的销售情况。
        保持分析简洁、专业、易于理解。
        """
        
        # 构建检索信息
        retrieval_info = "\n\n".join([
            f"[检索结果 {i+1}, 相关度: {result.get('relevance', 0):.2f}]\n{result.get('content', '')}"
            for i, result in enumerate(retrieval_results)
        ])
        
        # 构建业务数据信息
        business_data_info = json.dumps(business_data, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        请根据以下信息生成一份销售报表分析：
        
        查询信息：
        - 时间段: {time_period}
        - 产品: {product}
        
        检索信息：
        {retrieval_info}
        
        报表数据：
        {business_data_info}
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        # 记录日志
        logger.info(f"生成销售报表分析，时间段: {time_period}, 产品: {product}")
    except Exception as e:
        # 处理错误
        error_message = f"生成销售报表分析失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"抱歉，生成{time_period}的销售报表分析时出现错误。请稍后再试。"
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

def hr_report_handler_impl(state: EnhancedAgentState) -> EnhancedAgentState:
    """人力资源报表处理程序实现"""
    # 获取槽位值
    slots = state["current_intent"]["slots"]
    report_aspect = slots.get("report_aspect", "")
    department = slots.get("department", "所有部门")
    
    # 获取检索结果
    retrieval_results = state.get("retrieval_results", [])
    
    # 生成报表数据（模拟数据）
    business_data = {
        "report_type": "hr",
        "report_aspect": report_aspect,
        "department": department,
        "data": {
            "总员工数": "500人",
            "部门分布": {
                "技术部": "200人",
                "销售部": "150人",
                "行政部": "50人",
                "其他部门": "100人"
            },
            "招聘情况": {
                "本季度新员工": "30人",
                "技术岗位": "15人",
                "销售岗位": "10人",
                "行政岗位": "5人",
                "在招岗位": "25个"
            },
            "离职情况": {
                "本季度离职": "10人",
                "离职率": "2%"
            },
            "培训情况": {
                "培训次数": "15次",
                "培训人次": "300人次",
                "培训满意度": "4.2/5"
            }
        }
    }
    
    # 更新业务数据
    state["business_data"] = business_data
    
    # 使用LLM生成报表响应
    try:
        llm = get_llm()
        
        # 构建提示
        system_prompt = """
        你是一个专业的人力资源分析助手。请根据提供的人力资源数据和检索信息，生成一份简洁、专业的人力资源报表分析。
        分析应包括以下内容：
        1. 所查询方面的总体情况概述
        2. 相关指标的详细分析
        3. 趋势和特点
        4. 相关建议或改进措施
        
        如果查询的是特定部门，则重点分析该部门的情况。
        保持分析简洁、专业、易于理解。
        """
        
        # 构建检索信息
        retrieval_info = "\n\n".join([
            f"[检索结果 {i+1}, 相关度: {result.get('relevance', 0):.2f}]\n{result.get('content', '')}"
            for i, result in enumerate(retrieval_results)
        ])
        
        # 构建业务数据信息
        business_data_info = json.dumps(business_data, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        请根据以下信息生成一份人力资源报表分析：
        
        查询信息：
        - 关注方面: {report_aspect}
        - 部门: {department}
        
        检索信息：
        {retrieval_info}
        
        报表数据：
        {business_data_info}
        """
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # 添加响应
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        # 记录日志
        logger.info(f"生成人力资源报表分析，方面: {report_aspect}, 部门: {department}")
    except Exception as e:
        # 处理错误
        error_message = f"生成人力资源报表分析失败: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"抱歉，生成{report_aspect}的人力资源报表分析时出现错误。请稍后再试。"
        })
        state["error"] = error_message
        logger.error(error_message)
    
    # 更新状态
    state["status"] = "processed"
    
    return state

# ============== 图形构建 ==============

# 构建图结构
def build_graph():
    """构建对话图"""
    # 创建图
    graph = StateGraph(EnhancedAgentState)
    
    # 添加节点
    graph.add_node("user_input_processor", user_input_processor)
    graph.add_node("intent_recognizer", intent_recognizer)
    graph.add_node("check_return_to_previous", check_return_to_previous)
    graph.add_node("slot_filler", slot_filler)
    graph.add_node("knowledge_retriever", knowledge_retriever)
    
    # 添加处理程序节点
    graph.add_node("calculator_handler", calculator_handler)
    graph.add_node("weather_handler", weather_handler)
    graph.add_node("translate_handler", translate_handler)
    graph.add_node("chitchat_handler", chitchat_handler)
    graph.add_node("help_handler", help_handler)
    graph.add_node("fallback_handler", fallback_handler)
    
    # 添加报表处理程序节点
    graph.add_node("financial_report_handler", financial_report_handler)
    graph.add_node("project_report_handler", project_report_handler)
    graph.add_node("sales_report_handler", sales_report_handler)
    graph.add_node("hr_report_handler", hr_report_handler)
    
    # 添加状态更新节点
    graph.add_node("update_state", update_state)
    
    # 添加条件边
    # 起始 -> 用户输入处理
    graph.set_entry_point("user_input_processor")
    
    # 用户输入处理 -> 检查返回
    graph.add_edge("user_input_processor", "check_return_to_previous")
    
    # 检查返回 -> 意图识别
    graph.add_edge("check_return_to_previous", "intent_recognizer")
    
    # 意图识别 -> 路由
    graph.add_conditional_edges(
        "intent_recognizer",
        router,
        {
            "slot_filler": lambda state: state.get("current_intent") and state.get("status") != "slots_filled",
            "calculator_handler": lambda state: state.get("current_intent", {}).get("name") == "calculate" and state.get("status") == "slots_filled",
            "weather_handler": lambda state: state.get("current_intent", {}).get("name") == "query_weather" and state.get("status") == "slots_filled",
            "translate_handler": lambda state: state.get("current_intent", {}).get("name") == "translate" and state.get("status") == "slots_filled",
            "chitchat_handler": lambda state: state.get("current_intent", {}).get("name") == "chitchat",
            "help_handler": lambda state: state.get("current_intent", {}).get("name") == "help",
            "financial_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_financial_report" and state.get("status") == "slots_filled",
            "project_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_project_report" and state.get("status") == "slots_filled",
            "sales_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_sales_report" and state.get("status") == "slots_filled",
            "hr_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_hr_report" and state.get("status") == "slots_filled",
            "fallback_handler": lambda state: True  # 默认情况
        }
    )
    
    # 槽位填充 -> 路由
    graph.add_conditional_edges(
        "slot_filler",
        router,
        {
            "slot_filler": lambda state: state.get("status") == "slot_filling",
            "calculator_handler": lambda state: state.get("current_intent", {}).get("name") == "calculate" and state.get("status") == "slots_filled",
            "weather_handler": lambda state: state.get("current_intent", {}).get("name") == "query_weather" and state.get("status") == "slots_filled",
            "translate_handler": lambda state: state.get("current_intent", {}).get("name") == "translate" and state.get("status") == "slots_filled",
            "financial_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_financial_report" and state.get("status") == "slots_filled",
            "project_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_project_report" and state.get("status") == "slots_filled",
            "sales_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_sales_report" and state.get("status") == "slots_filled",
            "hr_report_handler": lambda state: state.get("current_intent", {}).get("name") == "query_hr_report" and state.get("status") == "slots_filled",
            "intent_recognizer": lambda state: True  # 默认回到意图识别
        }
    )
    
    # 所有处理程序 -> 状态更新
    handlers = [
        "calculator_handler", "weather_handler", "translate_handler", 
        "chitchat_handler", "help_handler", "fallback_handler",
        "financial_report_handler", "project_report_handler", 
        "sales_report_handler", "hr_report_handler"
    ]
    
    for handler in handlers:
        graph.add_edge(handler, "update_state")
    
    # 状态更新 -> 结束
    graph.add_edge("update_state", END)
    
    # 编译图
    return graph.compile()

# 创建图
compiled_graph = build_graph()

# 创建内存保存器
memory_saver = MemorySaver()

# 设置入口函数
def process_message(session_id: str, message: str):
    """处理用户消息"""
    try:
        # 获取或初始化状态
        try:
            state = compiled_graph.get_state(session_id)
            logger.info(f"加载会话: {session_id}")
        except:
            state = initialize_state()
            state["session_id"] = session_id
            logger.info(f"初始化新会话: {session_id}")
        
        # 添加用户消息
        state["messages"].append({
            "role": "user",
            "content": message
        })
        
        # 执行图
        result = compiled_graph.invoke(state, {"session_id": session_id})
        
        # 返回结果
        return result
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        # 返回错误消息
        return {
            "status": "error",
            "error": str(e),
            "messages": [
                {"role": "assistant", "content": "抱歉，处理您的消息时出现错误。请稍后再试。"}
            ]
        }

# 导出代理
agent = {
    "process_message": process_message,
    "compiled_graph": compiled_graph
} 