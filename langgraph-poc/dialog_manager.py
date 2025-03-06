import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import re
import random

class DialogState(Enum):
    """对话状态"""
    IDLE = "idle"                 # 空闲状态
    SLOT_FILLING = "slot_filling" # 槽位填充状态
    INTERRUPTED = "interrupted"   # 被打断状态
    COMPLETED = "completed"       # 完成状态

@dataclass
class DialogSlot:
    """对话槽位定义"""
    name: str                # 槽位名称
    value: Optional[Any] = None     # 槽位值
    required: bool = True    # 是否必填
    prompt: str = ""        # 提示语
    validate_func: Optional[Callable[[str], bool]] = None  # 验证函数
    normalize_func: Optional[Callable[[str], Any]] = None  # 标准化函数
    error_message: str = "输入无效，请重新输入"  # 验证失败提示
    examples: List[str] = field(default_factory=list)  # 示例输入
    
    def validate(self, value: str) -> bool:
        """验证输入值"""
        if self.validate_func:
            return self.validate_func(value)
        return True
    
    def normalize(self, value: str) -> Any:
        """标准化输入值"""
        if self.normalize_func:
            return self.normalize_func(value)
        return value

class BusinessMetric:
    """业务指标定义"""
    def __init__(self, name: str, description: str, query_func: Callable):
        self.name = name
        self.description = description
        self.query_func = query_func

@dataclass
class DialogIntent:
    """对话意图定义"""
    name: str                # 意图名称
    slots: List[DialogSlot]  # 所需槽位
    handler: str            # 处理函数名
    keywords: List[str] = field(default_factory=list)  # 意图关键词
    description: str = ""   # 意图描述
    examples: List[str] = field(default_factory=list)  # 示例语句
    business_metric: Optional[BusinessMetric] = None  # 关联的业务指标
    rag_enabled: bool = False  # 是否启用RAG增强

@dataclass
class DialogContext:
    """对话上下文"""
    intent: Optional[DialogIntent] = None    # 当前意图
    state: DialogState = DialogState.IDLE    # 当前状态
    current_slot: Optional[DialogSlot] = None # 当前正在填充的槽位
    interrupted_stack: List[Dict] = field(default_factory=list)  # 中断栈
    filled_slots: Dict[str, Any] = field(default_factory=dict)  # 已填充的槽位
    retry_count: int = 0    # 重试次数
    last_error: str = ""    # 最后一次错误信息

class DialogManager:
    """对话管理器"""
    
    def __init__(self):
        self.intents: Dict[str, DialogIntent] = {}
        self.context = DialogContext()
        self.history: List[Dict] = []
        self.max_retries = 3  # 最大重试次数
        self.business_metrics: Dict[str, BusinessMetric] = {}
        self.rag_context: Optional[str] = None
    
    def register_intent(self, intent: DialogIntent):
        """注册意图"""
        self.intents[intent.name] = intent
    
    def register_business_metric(self, metric: BusinessMetric):
        """注册业务指标"""
        self.business_metrics[metric.name] = metric
    
    def update_rag_context(self, context: str):
        """更新RAG上下文"""
        self.rag_context = context
    
    def detect_intent(self, user_input: str) -> Optional[str]:
        """检测用户输入的意图"""
        # 1. 首先检查完全匹配的关键词
        for intent_name, intent in self.intents.items():
            if any(keyword in user_input for keyword in intent.keywords):
                return intent_name
        
        # 2. 然后检查模糊匹配
        for intent_name, intent in self.intents.items():
            # 使用正则表达式进行模糊匹配
            pattern = '|'.join(f'.*{keyword}.*' for keyword in intent.keywords)
            if pattern and re.match(pattern, user_input, re.IGNORECASE):
                return intent_name
        
        return None
    
    def is_interruption(self, user_input: str) -> bool:
        """检测是否是打断"""
        # 扩展打断词列表
        interruption_keywords = [
            "等一下", "先等等", "换个话题", "问个问题", 
            "打断一下", "插一句", "等等", "先问一下",
            "转换话题", "换一下", "先说个事"
        ]
        
        # 1. 直接关键词匹配
        if any(keyword in user_input for keyword in interruption_keywords):
            return True
        
        # 2. 检测新的意图
        new_intent = self.detect_intent(user_input)
        if new_intent and self.context.state == DialogState.SLOT_FILLING:
            return True
        
        return False
    
    def save_context(self) -> Dict:
        """保存当前上下文"""
        return {
            "intent": self.context.intent.name if self.context.intent else None,
            "state": self.context.state.value,
            "current_slot": asdict(self.context.current_slot) if self.context.current_slot else None,
            "filled_slots": self.context.filled_slots.copy(),
            "retry_count": self.context.retry_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_interruption(self, user_input: str) -> str:
        """处理打断"""
        # 保存当前上下文到中断栈
        if self.context.state == DialogState.SLOT_FILLING:
            saved_context = self.save_context()
            self.context.interrupted_stack.append(saved_context)
            
            # 记录打断事件
            self.history.append({
                "role": "system",
                "content": "对话被打断",
                "context": saved_context,
                "timestamp": datetime.now().isoformat()
            })
        
        # 检测新的意图
        new_intent = self.detect_intent(user_input)
        if new_intent and new_intent in self.intents:
            # 切换到新意图
            self.context.intent = self.intents[new_intent]
            self.context.state = DialogState.SLOT_FILLING
            self.context.current_slot = self.get_next_empty_slot()
            self.context.filled_slots = {}
            self.context.retry_count = 0
            
            # 构建回复
            response = [
                "好的，让我们先处理这个问题。",
                f"您想{self.context.intent.description}。"
            ]
            
            if self.context.current_slot:
                response.append(self.get_slot_prompt(self.context.current_slot))
            
            return "\n".join(response)
        
        return "好的，我们稍后再回到之前的话题。请问您现在想聊什么？"
    
    def restore_context(self) -> str:
        """恢复之前的上下文"""
        if not self.context.interrupted_stack:
            return "没有需要恢复的对话。"
        
        # 恢复最近的上下文
        last_context = self.context.interrupted_stack.pop()
        
        # 恢复意图
        self.context.intent = self.intents[last_context["intent"]] if last_context["intent"] else None
        self.context.state = DialogState(last_context["state"])
        
        # 恢复槽位信息
        if last_context["current_slot"]:
            slot_data = last_context["current_slot"]
            self.context.current_slot = DialogSlot(**slot_data)
        
        # 恢复已填充的槽位
        self.context.filled_slots = last_context["filled_slots"]
        self.context.retry_count = last_context["retry_count"]
        
        # 记录恢复事件
        self.history.append({
            "role": "system",
            "content": "恢复之前的对话",
            "context": last_context,
            "timestamp": datetime.now().isoformat()
        })
        
        # 构建回复
        response = ["让我们回到之前的话题。"]
        
        # 添加已填充的槽位信息
        if self.context.filled_slots:
            response.append("您之前已经提供了以下信息：")
            for slot_name, value in self.context.filled_slots.items():
                response.append(f"- {slot_name}: {value}")
        
        # 添加当前需要填充的槽位提示
        if self.context.current_slot:
            response.append(self.get_slot_prompt(self.context.current_slot))
        
        return "\n".join(response)
    
    def get_slot_prompt(self, slot: DialogSlot) -> str:
        """获取槽位提示语"""
        prompt = [slot.prompt]
        
        # 如果有示例，添加示例
        if slot.examples:
            prompt.append(f"例如：{' / '.join(slot.examples)}")
        
        return " ".join(prompt)
    
    def get_next_empty_slot(self) -> Optional[DialogSlot]:
        """获取下一个未填充的槽位"""
        if not self.context.intent:
            return None
        
        for slot in self.context.intent.slots:
            if slot.name not in self.context.filled_slots and slot.required:
                return slot
        return None
    
    def validate_and_fill_slot(self, slot: DialogSlot, value: str) -> Tuple[bool, str]:
        """验证并填充槽位值"""
        try:
            # 验证输入
            if not slot.validate(value):
                return False, slot.error_message
            
            # 标准化输入
            normalized_value = slot.normalize(value)
            
            # 填充槽位
            self.context.filled_slots[slot.name] = normalized_value
            return True, ""
            
        except Exception as e:
            return False, f"处理输入时出错: {str(e)}"
    
    def process_user_input(self, user_input: str) -> str:
        """处理用户输入"""
        # 记录用户输入
        self.history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
            "rag_context": self.rag_context
        })
        
        try:
            # 检查是否是恢复上下文的请求
            if any(keyword in user_input for keyword in ["继续", "回到", "返回", "之前"]):
                return self.restore_context()
            
            # 检查是否是打断
            if self.is_interruption(user_input):
                return self.handle_interruption(user_input)
            
            # 如果当前没有活动意图，检测新的意图
            if self.context.state == DialogState.IDLE:
                intent_name = self.detect_intent(user_input)
                if intent_name and intent_name in self.intents:
                    self.context.intent = self.intents[intent_name]
                    self.context.state = DialogState.SLOT_FILLING
                    self.context.current_slot = self.get_next_empty_slot()
                    return self.get_slot_prompt(self.context.current_slot)
                
                return "抱歉，我不太理解您的意图。请您换个方式说明，或者可以参考以下示例：\n" + \
                       "\n".join(f"- {intent.description}：{' / '.join(intent.examples)}" 
                                for intent in self.intents.values())
            
            # 如果正在填充槽位
            if self.context.state == DialogState.SLOT_FILLING:
                # 验证并填充当前槽位
                is_valid, error_msg = self.validate_and_fill_slot(self.context.current_slot, user_input)
                
                if not is_valid:
                    self.context.retry_count += 1
                    self.context.last_error = error_msg
                    
                    # 如果超过最大重试次数
                    if self.context.retry_count >= self.max_retries:
                        self.context = DialogContext()  # 重置上下文
                        return f"抱歉，由于多次输入无效，我们需要重新开始。请您重新描述您的需求。"
                    
                    return f"{error_msg}\n{self.get_slot_prompt(self.context.current_slot)}"
                
                # 重置重试计数
                self.context.retry_count = 0
                
                # 获取下一个未填充的槽位
                next_slot = self.get_next_empty_slot()
                if next_slot:
                    self.context.current_slot = next_slot
                    return self.get_slot_prompt(next_slot)
                
                # 所有槽位都已填充，执行意图处理
                self.context.state = DialogState.COMPLETED
                
                # 构建完成响应
                response = [
                    f"好的，让我帮您{self.context.intent.handler}",
                    "您提供的信息是："
                ]
                
                for slot_name, value in self.context.filled_slots.items():
                    response.append(f"- {slot_name}: {value}")
                
                # 重置上下文
                self.context = DialogContext()
                
                # 如果当前意图启用了RAG
                if self.context.intent and self.context.intent.rag_enabled:
                    # 这里可以添加RAG处理逻辑
                    pass
                
                # 如果所有槽位都已填充，检查是否需要查询业务指标
                if self.context.intent and \
                   self.context.intent.business_metric:
                    
                    metric = self.context.intent.business_metric
                    result = self.context.intent.business_metric.query_func(self.context.filled_slots)
                    return f"根据{self.context.intent.business_metric.description}，{result}"
                
                return "\n".join(response)
            
            return "抱歉，我现在有点混乱。让我们重新开始。"
            
        except Exception as e:
            # 记录错误
            self.history.append({
                "role": "system",
                "content": f"错误: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            
            # 重置上下文
            self.context = DialogContext()
            return f"抱歉，处理您的输入时出现错误。让我们重新开始。"

def validate_month(value: str) -> bool:
    """验证月份输入"""
    try:
        month = int(value.replace("月", ""))
        return 1 <= month <= 12
    except:
        return False

def normalize_month(value: str) -> str:
    """标准化月份输入"""
    month = int(value.replace("月", ""))
    return f"{month}月"

def validate_report_type(value: str) -> bool:
    """验证报表类型"""
    valid_types = ["财务", "项目", "销售"]
    return any(t in value for t in valid_types)

# 模拟数据库
class MockDatabase:
    def __init__(self):
        self.financial_data = {
            "财务报表": {
                "技术部": {
                    "1月": {"收入": 100000, "支出": 80000, "利润": 20000},
                    "2月": {"收入": 120000, "支出": 85000, "利润": 35000},
                    "3月": {"收入": 150000, "支出": 100000, "利润": 50000}
                },
                "销售部": {
                    "1月": {"收入": 200000, "支出": 150000, "利润": 50000},
                    "2月": {"收入": 250000, "支出": 180000, "利润": 70000},
                    "3月": {"收入": 300000, "支出": 200000, "利润": 100000}
                }
            },
            "项目报表": {
                "技术部": {
                    "1月": {"进行中": 5, "已完成": 3, "延期": 1},
                    "2月": {"进行中": 4, "已完成": 4, "延期": 0},
                    "3月": {"进行中": 6, "已完成": 2, "延期": 2}
                },
                "销售部": {
                    "1月": {"进行中": 8, "已完成": 5, "延期": 2},
                    "2月": {"进行中": 7, "已完成": 6, "延期": 1},
                    "3月": {"进行中": 9, "已完成": 4, "延期": 3}
                }
            },
            "销售报表": {
                "技术部": {
                    "1月": {"目标": 10, "完成": 8, "转化率": "80%"},
                    "2月": {"目标": 12, "完成": 10, "转化率": "83%"},
                    "3月": {"目标": 15, "完成": 13, "转化率": "87%"}
                },
                "销售部": {
                    "1月": {"目标": 20, "完成": 18, "转化率": "90%"},
                    "2月": {"目标": 25, "完成": 22, "转化率": "88%"},
                    "3月": {"目标": 30, "完成": 28, "转化率": "93%"}
                }
            }
        }
        
        self.weather_data = {
            "北京": {"温度": range(15, 25), "天气": ["晴", "多云", "小雨"]},
            "上海": {"温度": range(18, 28), "天气": ["多云", "小雨", "阴"]},
            "广州": {"温度": range(20, 30), "天气": ["晴", "多云", "雷阵雨"]}
        }
    
    def query_report(self, report_type: str, month: str, department: str = "全部") -> str:
        try:
            if department == "全部":
                result = []
                for dept in ["技术部", "销售部"]:
                    data = self.financial_data[report_type][dept][month]
                    result.append(f"{dept}:")
                    for key, value in data.items():
                        result.append(f"  - {key}: {value}")
                return "\n".join(result)
            else:
                data = self.financial_data[report_type][department][month]
                result = [f"{department}:"]
                for key, value in data.items():
                    result.append(f"  - {key}: {value}")
                return "\n".join(result)
        except KeyError:
            return "抱歉，未找到相关数据。"
    
    def query_weather(self, city: str, date: str = "今天") -> str:
        try:
            city_data = self.weather_data[city]
            temp = random.choice(list(city_data["温度"]))
            weather = random.choice(city_data["天气"])
            
            date_str = ""
            if date == "明天":
                date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            elif date == "后天":
                date_str = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")
            
            return f"{city} {date_str} 天气预报：\n温度：{temp}℃\n天气：{weather}"
        except KeyError:
            return "抱歉，未找到该城市的天气数据。"

# 创建数据库实例
db = MockDatabase()

def query_financial_report(params: Dict[str, Any]) -> str:
    report_type = params.get("报表类型")
    month = params.get("月份")
    department = params.get("部门", "全部")
    return db.query_report(report_type, month, department)

def create_report_query_intent() -> DialogIntent:
    """创建报表查询意图"""
    return DialogIntent(
        name="report_query",
        description="查询报表",
        keywords=["报表", "报告", "财务报表", "项目报表", "销售报表"],
        examples=["我要查看财务报表", "帮我看看项目报表", "查一下销售报表"],
        slots=[
            DialogSlot(
                name="报表类型",
                prompt="您想查询哪种类型的报表？",
                examples=["财务报表", "项目报表", "销售报表"],
                validate_func=validate_report_type,
                error_message="报表类型必须是：财务、项目或销售"
            ),
            DialogSlot(
                name="月份",
                prompt="请问您要查询哪个月份的报表？",
                examples=["3月", "三月"],
                validate_func=validate_month,
                normalize_func=normalize_month,
                error_message="请输入有效的月份，例如：3月"
            ),
            DialogSlot(
                name="部门",
                required=False,
                prompt="请问是哪个部门的报表？(可选)",
                examples=["技术部", "销售部", "财务部"]
            )
        ],
        handler="查询报表",
        business_metric=BusinessMetric(
            name="financial_report",
            description="财务报表指标",
            query_func=query_financial_report
        ),
        rag_enabled=True  # 启用RAG增强
    )

def validate_city(value: str) -> bool:
    """验证城市名称"""
    # 这里应该使用更复杂的城市名称验证
    return len(value) >= 2 and "市" not in value

def normalize_city(value: str) -> str:
    """标准化城市名称"""
    return value.replace("市", "")

def create_weather_query_intent() -> DialogIntent:
    """创建天气查询意图"""
    return DialogIntent(
        name="weather_query",
        description="查询天气",
        keywords=["天气", "气温", "温度", "下雨", "阴晴"],
        examples=["北京天气怎么样", "查查上海的天气", "明天会下雨吗"],
        slots=[
            DialogSlot(
                name="城市",
                prompt="您想查询哪个城市的天气？",
                examples=["北京", "上海", "广州"],
                validate_func=lambda x: x in db.weather_data,
                normalize_func=normalize_city,
                error_message="目前只支持查询：北京、上海、广州的天气"
            ),
            DialogSlot(
                name="日期",
                required=False,
                prompt="请问您要查询哪一天的天气？(默认今天)",
                examples=["今天", "明天", "后天"],
                validate_func=lambda x: x in ["今天", "明天", "后天"],
                error_message="只支持查询：今天、明天、后天的天气"
            )
        ],
        handler="查询天气",
        business_metric=BusinessMetric(
            name="weather_query",
            description="天气查询",
            query_func=lambda params: db.query_weather(params["城市"], params.get("日期", "今天"))
        )
    )

if __name__ == "__main__":
    # 创建对话管理器
    dialog_manager = DialogManager()
    
    # 注册业务指标
    dialog_manager.register_business_metric(BusinessMetric(
        name="financial_report",
        description="财务报表指标",
        query_func=query_financial_report
    ))
    
    # 注册意图
    dialog_manager.register_intent(create_report_query_intent())
    dialog_manager.register_intent(create_weather_query_intent())
    
    print("=== 智能助手已启动 ===")
    print("您可以：")
    print("1. 查询报表 (例如：'我要查看财务报表')")
    print("2. 查询天气 (例如：'北京天气怎么样')")
    print("3. 随时可以打断当前对话 (例如：'等一下，我想问个问题')")
    print("4. 输入'退出'结束对话")
    print("===========================")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            # 检查是否退出
            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("\n=== 对话结束 ===")
                break
            
            # 处理用户输入
            response = dialog_manager.process_user_input(user_input)
            print(f"助手: {response}")
            
            # 如果对话完成，显示当前状态
            if dialog_manager.context.state == DialogState.COMPLETED:
                print("\n当前对话已完成，您可以开始新的对话。")
            
        except KeyboardInterrupt:
            print("\n\n=== 对话被用户中断 ===")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            print("让我们重新开始对话。")
            dialog_manager.context = DialogContext() 