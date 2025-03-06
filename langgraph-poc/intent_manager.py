"""
意图管理模块 - 实现完整的意图栈管理和上下文切换
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, TypedDict
from enum import Enum
from dataclasses import dataclass, asdict, field
from datetime import datetime
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntentManager")

# 尝试导入LangGraph依赖
try:
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
except ImportError:
    logger.error("请安装必要的依赖: pip install langgraph langchain-openai")
    raise

# 意图状态枚举
class IntentState(Enum):
    """意图状态"""
    NEW = "new"                # 新创建
    ACTIVE = "active"          # 活跃中
    SUSPENDED = "suspended"    # 被挂起
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败

# 槽位定义
@dataclass
class Slot:
    """槽位定义"""
    name: str                # 槽位名称
    description: str         # 槽位描述
    required: bool = True    # 是否必填
    value: Any = None        # 槽位值
    prompt: str = ""         # 提示语
    examples: List[str] = field(default_factory=list)  # 示例值
    validation_func: Optional[Callable[[Any], bool]] = None  # 验证函数
    
    def is_filled(self) -> bool:
        """检查槽位是否已填充"""
        return self.value is not None
    
    def validate(self) -> bool:
        """验证槽位值"""
        if not self.is_filled():
            return False
        
        if self.validation_func:
            return self.validation_func(self.value)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "value": self.value,
            "prompt": self.prompt,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Slot':
        """从字典创建槽位"""
        return cls(
            name=data["name"],
            description=data["description"],
            required=data.get("required", True),
            value=data.get("value"),
            prompt=data.get("prompt", ""),
            examples=data.get("examples", [])
        )

# 意图定义
@dataclass
class Intent:
    """意图定义"""
    name: str                # 意图名称
    description: str         # 意图描述
    slots: List[Slot] = field(default_factory=list)  # 所需槽位
    keywords: List[str] = field(default_factory=list)  # 关键词
    examples: List[str] = field(default_factory=list)  # 示例语句
    handler: Optional[str] = None  # 处理函数名
    parent: Optional[str] = None  # 父意图
    children: List[str] = field(default_factory=list)  # 子意图
    state: IntentState = IntentState.NEW  # 意图状态
    confidence: float = 1.0  # 置信度
    business_metric: Optional[str] = None  # 关联的业务指标
    
    def get_unfilled_slots(self) -> List[Slot]:
        """获取未填充的槽位"""
        return [slot for slot in self.slots if not slot.is_filled()]
    
    def get_next_slot(self) -> Optional[Slot]:
        """获取下一个需要填充的槽位"""
        unfilled = self.get_unfilled_slots()
        return unfilled[0] if unfilled else None
    
    def are_required_slots_filled(self) -> bool:
        """检查是否所有必填槽位都已填充"""
        return all(slot.is_filled() for slot in self.slots if slot.required)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "slots": [slot.to_dict() for slot in self.slots],
            "keywords": self.keywords,
            "examples": self.examples,
            "handler": self.handler,
            "parent": self.parent,
            "children": self.children,
            "state": self.state.value,
            "confidence": self.confidence,
            "business_metric": self.business_metric
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """从字典创建意图"""
        return cls(
            name=data["name"],
            description=data["description"],
            slots=[Slot.from_dict(slot) for slot in data.get("slots", [])],
            keywords=data.get("keywords", []),
            examples=data.get("examples", []),
            handler=data.get("handler"),
            parent=data.get("parent"),
            children=data.get("children", []),
            state=IntentState(data.get("state", "new")),
            confidence=data.get("confidence", 1.0),
            business_metric=data.get("business_metric")
        )

# 意图栈项
@dataclass
class IntentStackItem:
    """意图栈项"""
    intent: Intent           # 意图
    created_at: datetime     # 创建时间
    suspended_at: Optional[datetime] = None  # 挂起时间
    resumed_at: Optional[datetime] = None    # 恢复时间
    completed_at: Optional[datetime] = None  # 完成时间
    message_index: int = 0   # 消息索引
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "intent": self.intent.to_dict(),
            "created_at": self.created_at.isoformat(),
            "suspended_at": self.suspended_at.isoformat() if self.suspended_at else None,
            "resumed_at": self.resumed_at.isoformat() if self.resumed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "message_index": self.message_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentStackItem':
        """从字典创建意图栈项"""
        return cls(
            intent=Intent.from_dict(data["intent"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            suspended_at=datetime.fromisoformat(data["suspended_at"]) if data.get("suspended_at") else None,
            resumed_at=datetime.fromisoformat(data["resumed_at"]) if data.get("resumed_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            message_index=data.get("message_index", 0)
        )

# 意图管理器
class IntentManager:
    """意图管理器 - 管理意图栈和上下文切换"""
    
    def __init__(self, llm=None):
        self.intents: Dict[str, Intent] = {}  # 注册的意图
        self.intent_stack: List[IntentStackItem] = []  # 意图栈
        self.suspended_intents: List[IntentStackItem] = []  # 挂起的意图
        self.llm = llm or ChatOpenAI(temperature=0)  # 用于意图识别的LLM
        self.business_metrics: Dict[str, Dict[str, Any]] = {}  # 业务指标
    
    def register_intent(self, intent: Intent) -> None:
        """注册意图"""
        self.intents[intent.name] = intent
        logger.info(f"已注册意图: {intent.name}")
    
    def register_business_metric(self, name: str, description: str, vector=None) -> None:
        """注册业务指标"""
        self.business_metrics[name] = {
            "name": name,
            "description": description,
            "vector": vector
        }
        logger.info(f"已注册业务指标: {name}")
    
    def identify_intent(self, message: str) -> Tuple[Optional[str], float]:
        """识别消息中的意图"""
        # 1. 关键词匹配
        for name, intent in self.intents.items():
            if any(keyword.lower() in message.lower() for keyword in intent.keywords):
                return name, 1.0
        
        # 2. 使用LLM进行意图识别
        system_prompt = """
        你是一个专业的意图识别助手。请识别用户消息中的意图。
        
        可能的意图包括:
        """
        
        # 添加所有注册的意图
        for name, intent in self.intents.items():
            system_prompt += f"- {name}: {intent.description}\n"
            if intent.examples:
                system_prompt += "  示例: " + ", ".join(f'"{ex}"' for ex in intent.examples) + "\n"
        
        system_prompt += """
        请以JSON格式返回识别结果，包含以下字段:
        - intent: 识别出的意图名称
        - confidence: 置信度(0-1之间的浮点数)
        - explanation: 简短解释
        
        如果无法识别任何意图，请将intent设为"unknown"。
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"识别以下消息的意图: \"{message}\""}
            ])
            
            # 解析JSON响应
            import json
            result = json.loads(response.content)
            return result.get("intent"), result.get("confidence", 0.7)
            
        except Exception as e:
            logger.error(f"意图识别失败: {str(e)}")
            return None, 0.0
    
    def is_interruption(self, message: str) -> bool:
        """检测是否是打断"""
        # 打断关键词
        interruption_keywords = [
            "等一下", "先等等", "换个话题", "问个问题", 
            "打断一下", "插一句", "等等", "先问一下",
            "转换话题", "换一下", "先说个事"
        ]
        
        # 1. 关键词匹配
        if any(keyword in message for keyword in interruption_keywords):
            return True
        
        # 2. 如果当前有活跃意图，检测是否是新意图
        if self.intent_stack:
            current_intent = self.intent_stack[-1].intent
            if current_intent.state == IntentState.ACTIVE:
                intent_name, confidence = self.identify_intent(message)
                if intent_name and intent_name != current_intent.name and confidence > 0.7:
                    return True
        
        return False
    
    def is_return_to_previous(self, message: str) -> bool:
        """检测是否是返回之前话题的请求"""
        return_keywords = [
            "回到刚才", "继续刚才", "之前的话题", "刚才说到", 
            "回到之前", "继续之前", "回到上一个", "继续上一个"
        ]
        
        return any(keyword in message for keyword in return_keywords)
    
    def push_intent(self, intent_name: str, confidence: float = 1.0) -> Optional[Intent]:
        """将意图压入栈顶"""
        if intent_name not in self.intents:
            logger.warning(f"意图不存在: {intent_name}")
            return None
        
        intent = self.intents[intent_name]
        intent.state = IntentState.ACTIVE
        intent.confidence = confidence
        
        # 创建栈项并压入栈顶
        stack_item = IntentStackItem(
            intent=intent,
            created_at=datetime.now()
        )
        
        self.intent_stack.append(stack_item)
        logger.info(f"意图已压栈: {intent_name}")
        
        return intent
    
    def suspend_current_intent(self) -> Optional[Intent]:
        """挂起当前意图"""
        if not self.intent_stack:
            return None
        
        # 获取栈顶意图
        stack_item = self.intent_stack.pop()
        intent = stack_item.intent
        
        # 更新状态
        intent.state = IntentState.SUSPENDED
        stack_item.suspended_at = datetime.now()
        
        # 添加到挂起列表
        self.suspended_intents.append(stack_item)
        
        logger.info(f"意图已挂起: {intent.name}")
        return intent
    
    def resume_intent(self, index: int = -1) -> Optional[Intent]:
        """恢复挂起的意图"""
        if not self.suspended_intents:
            return None
        
        # 获取指定索引的挂起意图
        try:
            stack_item = self.suspended_intents.pop(index)
        except IndexError:
            logger.warning(f"索引无效: {index}")
            return None
        
        intent = stack_item.intent
        
        # 更新状态
        intent.state = IntentState.ACTIVE
        stack_item.resumed_at = datetime.now()
        
        # 压入栈顶
        self.intent_stack.append(stack_item)
        
        logger.info(f"意图已恢复: {intent.name}")
        return intent
    
    def complete_current_intent(self) -> Optional[Intent]:
        """完成当前意图"""
        if not self.intent_stack:
            return None
        
        # 获取栈顶意图
        stack_item = self.intent_stack[-1]
        intent = stack_item.intent
        
        # 更新状态
        intent.state = IntentState.COMPLETED
        stack_item.completed_at = datetime.now()
        
        logger.info(f"意图已完成: {intent.name}")
        return intent
    
    def get_current_intent(self) -> Optional[Intent]:
        """获取当前意图"""
        if not self.intent_stack:
            return None
        
        return self.intent_stack[-1].intent
    
    def extract_slot_value(self, slot: Slot, message: str) -> Any:
        """从消息中提取槽位值"""
        # 使用LLM提取槽位值
        system_prompt = f"""
        你是一个专业的槽位提取助手。请从用户消息中提取"{slot.name}"的值。
        
        槽位描述: {slot.description}
        
        如果找到了值，请直接返回该值。
        如果没有找到值，请返回"NOT_FOUND"。
        """
        
        if slot.examples:
            system_prompt += f"\n示例值: {', '.join(slot.examples)}"
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
            
            value = response.content.strip()
            if value == "NOT_FOUND":
                return None
            
            return value
            
        except Exception as e:
            logger.error(f"槽位提取失败: {str(e)}")
            return None
    
    def fill_slots_from_message(self, message: str) -> Dict[str, Any]:
        """从消息中填充当前意图的槽位"""
        if not self.intent_stack:
            return {}
        
        intent = self.get_current_intent()
        if not intent:
            return {}
        
        filled_slots = {}
        
        # 获取未填充的槽位
        unfilled_slots = intent.get_unfilled_slots()
        for slot in unfilled_slots:
            value = self.extract_slot_value(slot, message)
            if value is not None:
                slot.value = value
                filled_slots[slot.name] = value
        
        return filled_slots
    
    def get_next_prompt(self) -> Optional[str]:
        """获取下一个提示"""
        intent = self.get_current_intent()
        if not intent:
            return None
        
        # 获取下一个需要填充的槽位
        next_slot = intent.get_next_slot()
        if next_slot:
            return next_slot.prompt or f"请提供{next_slot.description}"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "intents": {name: intent.to_dict() for name, intent in self.intents.items()},
            "intent_stack": [item.to_dict() for item in self.intent_stack],
            "suspended_intents": [item.to_dict() for item in self.suspended_intents],
            "business_metrics": self.business_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm=None) -> 'IntentManager':
        """从字典创建意图管理器"""
        manager = cls(llm=llm)
        
        # 恢复意图
        for name, intent_data in data.get("intents", {}).items():
            manager.intents[name] = Intent.from_dict(intent_data)
        
        # 恢复意图栈
        manager.intent_stack = [
            IntentStackItem.from_dict(item) 
            for item in data.get("intent_stack", [])
        ]
        
        # 恢复挂起的意图
        manager.suspended_intents = [
            IntentStackItem.from_dict(item) 
            for item in data.get("suspended_intents", [])
        ]
        
        # 恢复业务指标
        manager.business_metrics = data.get("business_metrics", {})
        
        return manager 