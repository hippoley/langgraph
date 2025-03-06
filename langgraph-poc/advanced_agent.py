# 全局变量声明
session_id = None
memory_state = {}
message_history = []
intervention_records = []
game_state = {
    "active_game": None,  # 当前活跃的游戏
    "guess_number": {
        "active": False,
        "target": 0,
        "attempts": 0,
        "max_attempts": 10,
        "range": (1, 100)
    },
    "adventure": {
        "active": False,
        "current_location": "start",
        "inventory": [],
        "health": 100,
        "visited": set()
    }
}

# 导入所需库
import os
import json
import time
import random
import datetime
import threading
import queue
import uuid
from enum import Enum
from dotenv import load_dotenv
from typing import Literal, TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from api_config import get_llm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from visualization import (
    GraphVisualizer, 
    debug_info, 
    performance_analyzer, 
    interactive_debugger,
    setup_debugging,
    create_debug_report
)

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

@tool
def search_database(query: str) -> str:
    """搜索数据库获取信息"""
    # 模拟数据库
    database = {
        "人工智能": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法和模型。",
        "深度学习": "深度学习是机器学习的一个子集，使用多层神经网络处理复杂的模式和抽象。",
        "自然语言处理": "自然语言处理（NLP）是人工智能的一个分支，专注于使计算机能够理解、解释和生成人类语言。",
        "计算机视觉": "计算机视觉是人工智能的一个领域，专注于使计算机能够从数字图像或视频中获取高级理解。"
    }
    
    # 简单的模糊匹配
    for key, value in database.items():
        if key in query:
            return value
    
    return f"没有找到与'{query}'相关的信息"

# 定义状态类型
class AgentState(TypedDict):
    messages: List[Annotated[HumanMessage | AIMessage | ToolMessage | SystemMessage, "Messages"]]
    memory: Optional[Dict[str, str]]
    human_input: Optional[str]

# 初始化工具和模型
tools = [calculator, get_weather, search_database]
model = get_llm(temperature=0)

# 定义代理函数
def agent(state: AgentState) -> Dict:
    """处理消息并生成代理响应"""
    messages = state["messages"]
    memory = state.get("memory", {}) or {}
    
    # 检查消息列表是否为空
    if not messages:
        # 如果消息列表为空，创建一个初始系统消息
        system_message = SystemMessage(content="你是一个高级AI助手，可以使用工具并记住用户提供的信息。")
        messages = [system_message]
        return {"messages": messages}
    
    # 获取最近的消息
    last_message = messages[-1]
    
    # 如果最后一条消息是人类消息，且之前没有系统消息，添加系统消息
    if isinstance(last_message, HumanMessage) and not any(isinstance(msg, SystemMessage) for msg in messages):
        system_message = SystemMessage(content="你是一个高级AI助手，可以使用工具并记住用户提供的信息。")
        messages = [system_message] + messages
    
    # 检查是否有"记住"关键词，如果有，直接生成回复
    if isinstance(last_message, HumanMessage) and "记住" in last_message.content:
        response = AIMessage(content="我已经记住了这个信息。")
        messages.append(response)
        return {"messages": messages}
    
    # 生成AI响应
    try:
        # 为了减少API调用的token数量，只保留最近的5条消息
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        response = model.invoke(recent_messages)
    except Exception as e:
        print(f"模型调用失败: {str(e)}")
        response = AIMessage(content="抱歉，我无法处理您的请求。")
    
    messages.append(response)
    return {"messages": messages}

# 自定义工具处理函数
def tools_handler(state: AgentState) -> Dict:
    """处理工具调用"""
    # 简化版本，直接返回状态
    return state

# 定义路由函数
def route(state: AgentState) -> Literal["tools", "human_intervention", "request_human_input", "agent", "end"]:
    """确定下一步操作"""
    messages = state["messages"]
    human_input = state.get("human_input")
    
    # 如果请求了人类输入，进行人类干预
    if human_input == "requested":
        return "human_intervention"
    
    # 获取最近的消息
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    
    # 检查是否有"记住"关键词，如果有，更新记忆
    if isinstance(last_message, HumanMessage) and "记住" in last_message.content:
        return "update_memory"
    
    # 如果最后一条消息是AI消息，结束
    if isinstance(last_message, AIMessage):
        return "end"
    
    # 默认情况，交给AI处理
    return "agent"

def update_memory(state: AgentState) -> Dict:
    """更新长期记忆"""
    messages = state["messages"]
    memory = state.get("memory", {}) or {}
    
    # 分析最近的消息，提取可能需要记住的信息
    for message in messages[-3:]:  # 只看最近的3条消息
        if isinstance(message, HumanMessage):
            content = message.content
            # 检测是否有记忆相关的请求
            if "记住" in content or "记录" in content:
                print(f"检测到记忆请求: {content}")
                # 简单的记忆提取逻辑
                parts = content.split("记住", 1)
                if len(parts) > 1:
                    info = parts[1].strip()
                    # 尝试提取键值对
                    if ":" in info or "：" in info:
                        key, value = info.replace("：", ":").split(":", 1)
                        memory[key.strip()] = value.strip()
                        print(f"已记住键值对: {key.strip()} = {value.strip()}")
                    else:
                        # 如果没有明确的键，使用时间戳作为键
                        memory[f"记忆_{int(time.time())}"] = info
                        print(f"已记住信息: {info}")
    
    # 返回更新后的状态
    return {"messages": messages, "memory": memory}

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

# 创建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent)
workflow.add_node("tools", tools_handler)
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
        "update_memory": "update_memory",
        "end": END
    }
)

# 添加从工具到代理的边
workflow.add_edge("tools", "agent")

# 添加从更新记忆到代理的边
workflow.add_edge("update_memory", "agent")

# 添加人类干预相关的边
workflow.add_edge("human_intervention", "agent")

# 编译图
agent = workflow.compile(checkpointer=MemorySaver())

# 检测是否需要人类干预的函数
def needs_intervention(message: str) -> bool:
    """检测消息是否需要人类干预"""
    # 定义需要干预的关键词和短语
    intervention_triggers = [
        "投诉", "不满意", "要求退款", "要求赔偿", "非常生气", "非常失望",
        "要求经理", "要求上级", "要求人工", "紧急情况", "紧急请求",
        "严重问题", "系统错误", "无法理解", "完全不对", "完全错误",
        "特殊情况", "例外", "政策例外", "特殊请求"
    ]
    
    # 检查是否包含触发词
    for trigger in intervention_triggers:
        if trigger in message:
            return True
    
    # 检查消息长度，过长的消息可能需要干预
    if len(message) > 200:
        # 长消息有50%的概率触发干预
        if random.random() < 0.5:
            return True
    
    return False

# 游戏世界定义 - 文字冒险
adventure_world = {
    "start": {
        "description": "你站在一个分叉路口。向北是茂密的森林，向东是一座古老的城堡，向南是宁静的湖泊。",
        "exits": {"north": "forest", "east": "castle", "south": "lake"},
        "items": ["地图", "火柴"]
    },
    "forest": {
        "description": "你进入了一片茂密的森林。树木高耸，阳光几乎无法穿透树冠。你听到远处有动物的声音。",
        "exits": {"south": "start", "east": "cabin"},
        "items": ["木棍", "浆果"],
        "danger": {"type": "wolf", "probability": 0.3, "damage": 20}
    },
    "castle": {
        "description": "这座古老的城堡看起来已经废弃多年。城墙上爬满了藤蔓，大门半开着。",
        "exits": {"west": "start", "north": "tower", "inside": "hall"},
        "items": ["生锈的钥匙"]
    },
    "lake": {
        "description": "清澈的湖水反射着天空的蓝色。湖边有一条小船，似乎可以使用。",
        "exits": {"north": "start", "boat": "island"},
        "items": ["鱼竿", "水壶"]
    },
    "cabin": {
        "description": "一座隐藏在森林中的小木屋。看起来有人在这里生活过，但现在似乎已经废弃。",
        "exits": {"west": "forest"},
        "items": ["绳子", "罐头食品"]
    },
    "tower": {
        "description": "城堡的高塔提供了俯瞰整个地区的视野。你可以看到远处的森林、湖泊和山脉。",
        "exits": {"south": "castle"},
        "items": ["望远镜"]
    },
    "hall": {
        "description": "城堡的大厅宏伟而空旷。墙上挂着褪色的挂毯和生锈的武器。",
        "exits": {"outside": "castle", "upstairs": "chamber"},
        "items": ["火把", "古老的书籍"]
    },
    "island": {
        "description": "湖中央的小岛上有一座神秘的石头建筑。入口处刻着古老的符文。",
        "exits": {"boat": "lake"},
        "items": ["宝箱", "神秘符文"]
    },
    "chamber": {
        "description": "这是城堡的一个私人房间，可能属于城堡的主人。房间里有一张大床和一个书桌。",
        "exits": {"downstairs": "hall"},
        "items": ["日记", "金币"]
    }
}

# 游戏命令处理函数
def handle_game_command(command: str) -> str:
    """处理游戏相关命令"""
    global game_state
    
    # 检查是否是游戏启动命令
    if command.lower() in ["玩游戏", "开始游戏", "游戏", "play game", "game"]:
        return """
可选游戏:
1. 猜数字 (输入'猜数字'或'guess number'开始)
2. 文字冒险 (输入'冒险'或'adventure'开始)
输入'退出游戏'或'exit game'可以随时退出游戏
"""
    
    # 检查是否是游戏选择命令
    if command.lower() in ["猜数字", "guess number"] and not any(game_state[game]["active"] for game in ["guess_number", "adventure"]):
        # 初始化猜数字游戏
        game_state["active_game"] = "guess_number"
        game_state["guess_number"]["active"] = True
        game_state["guess_number"]["target"] = random.randint(
            game_state["guess_number"]["range"][0], 
            game_state["guess_number"]["range"][1]
        )
        game_state["guess_number"]["attempts"] = 0
        
        return f"""
猜数字游戏已开始!
我已经想好了一个{game_state["guess_number"]["range"][0]}到{game_state["guess_number"]["range"][1]}之间的数字。
你有{game_state["guess_number"]["max_attempts"]}次机会猜出这个数字。
请输入你的猜测:
"""
    
    if command.lower() in ["冒险", "adventure"] and not any(game_state[game]["active"] for game in ["guess_number", "adventure"]):
        # 初始化文字冒险游戏
        game_state["active_game"] = "adventure"
        game_state["adventure"]["active"] = True
        game_state["adventure"]["current_location"] = "start"
        game_state["adventure"]["inventory"] = []
        game_state["adventure"]["health"] = 100
        game_state["adventure"]["visited"] = {"start"}
        
        location = game_state["adventure"]["current_location"]
        description = adventure_world[location]["description"]
        items = adventure_world[location]["items"]
        exits = adventure_world[location]["exits"]
        
        return f"""
文字冒险游戏已开始!
{description}
你可以看到: {', '.join(items)}
可用出口: {', '.join(exits.keys())}

命令: 
- 移动: 去 <方向> (例如: 去 north)
- 查看: 看 <物品> (例如: 看 地图)
- 拾取: 拿 <物品> (例如: 拿 地图)
- 查看背包: 背包
- 查看状态: 状态
- 帮助: 游戏帮助
"""
    
    # 检查是否是退出游戏命令
    if command.lower() in ["退出游戏", "exit game"]:
        if game_state["active_game"]:
            active_game = game_state["active_game"]
            game_state[active_game]["active"] = False
            game_state["active_game"] = None
            return "已退出游戏。"
        else:
            return "当前没有正在进行的游戏。"
    
    # 处理猜数字游戏的猜测
    if game_state["active_game"] == "guess_number" and game_state["guess_number"]["active"]:
        try:
            guess = int(command.strip())
            target = game_state["guess_number"]["target"]
            game_state["guess_number"]["attempts"] += 1
            attempts = game_state["guess_number"]["attempts"]
            max_attempts = game_state["guess_number"]["max_attempts"]
            
            if guess == target:
                game_state["guess_number"]["active"] = False
                game_state["active_game"] = None
                return f"恭喜你猜对了! 答案就是{target}。你用了{attempts}次尝试。"
            elif attempts >= max_attempts:
                game_state["guess_number"]["active"] = False
                game_state["active_game"] = None
                return f"游戏结束! 你已用完{max_attempts}次尝试。正确答案是{target}。"
            elif guess < target:
                return f"太小了! 还剩{max_attempts - attempts}次尝试。"
            else:
                return f"太大了! 还剩{max_attempts - attempts}次尝试。"
        except ValueError:
            return "请输入一个有效的数字。"
    
    # 处理文字冒险游戏命令
    if game_state["active_game"] == "adventure" and game_state["adventure"]["active"]:
        if command.lower() == "游戏帮助":
            return """
文字冒险游戏命令:
- 移动: 去 <方向> (例如: 去 north)
- 查看: 看 <物品> (例如: 看 地图)
- 拾取: 拿 <物品> (例如: 拿 地图)
- 查看背包: 背包
- 查看状态: 状态
- 帮助: 游戏帮助
"""
        
        if command.lower() == "背包":
            if not game_state["adventure"]["inventory"]:
                return "你的背包是空的。"
            else:
                return f"背包内容: {', '.join(game_state['adventure']['inventory'])}"
        
        if command.lower() == "状态":
            return f"生命值: {game_state['adventure']['health']}\n位置: {game_state['adventure']['current_location']}\n背包物品: {len(game_state['adventure']['inventory'])}"
        
        if command.lower().startswith("去 "):
            direction = command[3:].strip().lower()
            current = game_state["adventure"]["current_location"]
            
            if direction in adventure_world[current]["exits"]:
                new_location = adventure_world[current]["exits"][direction]
                game_state["adventure"]["current_location"] = new_location
                
                # 标记为已访问
                game_state["adventure"]["visited"].add(new_location)
                
                # 检查是否有危险
                danger_message = ""
                if "danger" in adventure_world[new_location]:
                    danger = adventure_world[new_location]["danger"]
                    if random.random() < danger["probability"]:
                        game_state["adventure"]["health"] -= danger["damage"]
                        danger_message = f"\n你遭遇了{danger['type']}! 损失了{danger['damage']}点生命值。当前生命值: {game_state['adventure']['health']}"
                        
                        # 检查是否游戏结束
                        if game_state["adventure"]["health"] <= 0:
                            game_state["adventure"]["active"] = False
                            game_state["active_game"] = None
                            return f"{danger_message}\n你的生命值降至0。游戏结束!"
                
                location = new_location
                description = adventure_world[location]["description"]
                items = adventure_world[location]["items"]
                exits = adventure_world[location]["exits"]
                
                return f"""
{description}{danger_message}
你可以看到: {', '.join(items) if items else '这里没有任何物品'}
可用出口: {', '.join(exits.keys())}
"""
            else:
                return f"你不能往{direction}方向走。可用出口: {', '.join(adventure_world[current]['exits'].keys())}"
        
        if command.lower().startswith("拿 "):
            item = command[3:].strip()
            current = game_state["adventure"]["current_location"]
            
            if item in adventure_world[current]["items"]:
                game_state["adventure"]["inventory"].append(item)
                adventure_world[current]["items"].remove(item)
                return f"你拿起了{item}。"
            else:
                return f"这里没有{item}。"
        
        if command.lower().startswith("看 "):
            item = command[3:].strip()
            current = game_state["adventure"]["current_location"]
            
            if item in adventure_world[current]["items"]:
                return f"你仔细查看了{item}。这是一个普通的{item}。"
            elif item in game_state["adventure"]["inventory"]:
                return f"你从背包中取出{item}查看。这是一个普通的{item}。"
            else:
                return f"这里没有{item}，你的背包里也没有。"
        
        return "我不明白你的命令。输入'游戏帮助'查看可用命令。"
    
    return None  # 不是游戏命令

# 添加持久化和恢复功能
import json
import os
import datetime

# 持久化目录
PERSISTENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_states")

# 确保目录存在
if not os.path.exists(PERSISTENCE_DIR):
    os.makedirs(PERSISTENCE_DIR)

def save_agent_state(session_id: str, memory_state: dict, message_history: list, game_state: dict = None) -> str:
    """保存智能体状态到文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{session_id}_{timestamp}.json"
    filepath = os.path.join(PERSISTENCE_DIR, filename)
    
    # 转换消息历史为可序列化格式
    serializable_history = []
    for msg in message_history:
        msg_type = type(msg).__name__
        serializable_history.append({
            "type": msg_type,
            "content": msg.content
        })
    
    # 准备要保存的数据
    data = {
        "session_id": session_id,
        "timestamp": timestamp,
        "memory_state": memory_state,
        "message_history": serializable_history
    }
    
    # 如果有游戏状态，也保存
    if game_state:
        # 处理不可序列化的集合类型
        if "adventure" in game_state and "visited" in game_state["adventure"]:
            game_state["adventure"]["visited"] = list(game_state["adventure"]["visited"])
        data["game_state"] = game_state
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath

def load_agent_state(filepath: str) -> tuple:
    """从文件加载智能体状态"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 恢复会话ID
    session_id = data["session_id"]
    
    # 恢复记忆状态
    memory_state = data["memory_state"]
    
    # 恢复消息历史
    message_history = []
    for msg in data["message_history"]:
        if msg["type"] == "SystemMessage":
            message_history.append(SystemMessage(content=msg["content"]))
        elif msg["type"] == "HumanMessage":
            message_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "AIMessage":
            message_history.append(AIMessage(content=msg["content"]))
        elif msg["type"] == "ToolMessage":
            message_history.append(ToolMessage(content=msg["content"]))
    
    # 恢复游戏状态（如果有）
    game_state_restored = None
    if "game_state" in data:
        game_state_restored = data["game_state"]
        # 恢复不可序列化的集合类型
        if "adventure" in game_state_restored and "visited" in game_state_restored["adventure"]:
            game_state_restored["adventure"]["visited"] = set(game_state_restored["adventure"]["visited"])
    
    return session_id, memory_state, message_history, game_state_restored

def list_saved_states() -> list:
    """列出所有保存的状态"""
    if not os.path.exists(PERSISTENCE_DIR):
        return []
    
    files = os.listdir(PERSISTENCE_DIR)
    states = []
    
    for file in files:
        if file.endswith('.json'):
            filepath = os.path.join(PERSISTENCE_DIR, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                states.append({
                    "filename": file,
                    "session_id": data.get("session_id", "未知"),
                    "timestamp": data.get("timestamp", "未知"),
                    "filepath": filepath
                })
            except Exception as e:
                print(f"读取文件 {file} 时出错: {str(e)}")
    
    # 按时间戳排序，最新的在前
    states.sort(key=lambda x: x["timestamp"], reverse=True)
    return states

# 知识库目录
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

# 确保目录存在
if not os.path.exists(KNOWLEDGE_DIR):
    os.makedirs(KNOWLEDGE_DIR)

# 向量存储路径
VECTOR_STORE_PATH = os.path.join(KNOWLEDGE_DIR, "faiss_index")

# 全局向量存储对象
vector_store = None

def initialize_knowledge_base():
    """初始化知识库"""
    global vector_store
    
    # 检查是否已有向量存储
    if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
        try:
            print("正在加载现有知识库...")
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
            print(f"成功加载知识库，包含 {vector_store.index.ntotal} 个文档")
            return True
        except Exception as e:
            print(f"加载知识库失败: {str(e)}")
            return False
    else:
        print("知识库不存在，请先添加文档")
        return False

def add_document_to_knowledge_base(file_path):
    """向知识库添加文档"""
    global vector_store
    
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # 创建或更新向量存储
        embeddings = OpenAIEmbeddings()
        
        if vector_store is None:
            vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store.save_local(VECTOR_STORE_PATH)
            return f"成功创建知识库并添加文档: {file_path}"
        else:
            vector_store.add_texts(chunks)
            vector_store.save_local(VECTOR_STORE_PATH)
            return f"成功向知识库添加文档: {file_path}"
    
    except Exception as e:
        return f"添加文档失败: {str(e)}"

def query_knowledge_base(query, k=3):
    """查询知识库"""
    global vector_store
    
    if vector_store is None:
        if not initialize_knowledge_base():
            return "知识库未初始化，请先添加文档"
    
    try:
        # 执行相似性搜索
        docs = vector_store.similarity_search(query, k=k)
        
        # 格式化结果
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"结果 {i}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"查询知识库失败: {str(e)}"

# 添加知识库工具
@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库以回答问题"""
    return query_knowledge_base(query)

# 添加并行处理和条件分支功能
import threading
import queue
import uuid
from enum import Enum

# 任务优先级
class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

# 任务状态
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 任务类
class Task:
    def __init__(self, task_id, name, func, args=None, kwargs=None, priority=TaskPriority.MEDIUM, depends_on=None):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.priority = priority
        self.depends_on = depends_on or []  # 依赖的任务ID列表
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None

# 任务管理器
class TaskManager:
    def __init__(self, max_workers=3):
        self.tasks = {}  # 任务字典，键为任务ID
        self.task_queue = queue.PriorityQueue()  # 优先级队列
        self.max_workers = max_workers
        self.workers = []
        self.results = {}  # 任务结果字典
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = False
    
    def add_task(self, name, func, args=None, kwargs=None, priority=TaskPriority.MEDIUM, depends_on=None, task_id=None):
        """添加任务到队列"""
        with self.lock:
            task = Task(task_id, name, func, args, kwargs, priority, depends_on)
            self.tasks[task.task_id] = task
            
            # 检查是否有依赖
            if not depends_on:
                # 如果没有依赖，直接加入队列
                self.task_queue.put((-task.priority.value, task.task_id))
            
            return task.task_id
    
    def start(self):
        """启动任务管理器"""
        if self.running:
            return
        
        self.running = True
        
        # 创建工作线程
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """停止任务管理器"""
        self.running = False
        
        # 通知所有等待的线程
        with self.condition:
            self.condition.notify_all()
        
        # 等待所有工作线程结束
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 尝试从队列获取任务
                try:
                    _, task_id = self.task_queue.get(block=True, timeout=1.0)
                except queue.Empty:
                    continue
                
                with self.lock:
                    task = self.tasks.get(task_id)
                    if not task:
                        self.task_queue.task_done()
                        continue
                    
                    # 检查依赖是否满足
                    dependencies_met = True
                    for dep_id in task.depends_on:
                        dep_task = self.tasks.get(dep_id)
                        if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                            dependencies_met = False
                            break
                    
                    if not dependencies_met:
                        # 依赖未满足，放回队列
                        self.task_queue.put((-task.priority.value, task_id))
                        self.task_queue.task_done()
                        continue
                    
                    # 更新任务状态
                    task.status = TaskStatus.RUNNING
                    task.start_time = datetime.datetime.now()
                
                # 执行任务
                try:
                    result = task.func(*task.args, **task.kwargs)
                    
                    with self.lock:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        task.end_time = datetime.datetime.now()
                        self.results[task_id] = result
                        
                        # 检查是否有依赖于此任务的其他任务
                        for other_task in self.tasks.values():
                            if task_id in other_task.depends_on and other_task.status == TaskStatus.PENDING:
                                # 检查该任务的所有依赖是否都已完成
                                all_deps_completed = True
                                for dep_id in other_task.depends_on:
                                    dep_task = self.tasks.get(dep_id)
                                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                                        all_deps_completed = False
                                        break
                                
                                if all_deps_completed:
                                    # 所有依赖都已完成，将任务加入队列
                                    self.task_queue.put((-other_task.priority.value, other_task.task_id))
                        
                        # 通知等待的线程
                        self.condition.notify_all()
                
                except Exception as e:
                    with self.lock:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.end_time = datetime.datetime.now()
                        
                        # 通知等待的线程
                        self.condition.notify_all()
                
                finally:
                    self.task_queue.task_done()
            
            except Exception as e:
                print(f"工作线程发生错误: {str(e)}")
    
    def wait_for_task(self, task_id, timeout=None):
        """等待任务完成"""
        with self.condition:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task.result
            
            # 等待任务完成
            self.condition.wait(timeout=timeout)
            
            return task.result if task.status == TaskStatus.COMPLETED else None
    
    def wait_all(self, timeout=None):
        """等待所有任务完成"""
        deadline = None
        if timeout is not None:
            deadline = datetime.datetime.now() + datetime.timedelta(seconds=timeout)
        
        with self.condition:
            while any(task.status in [TaskStatus.PENDING, TaskStatus.RUNNING] for task in self.tasks.values()):
                if deadline and datetime.datetime.now() >= deadline:
                    break
                
                remaining_time = None
                if deadline:
                    remaining_time = (deadline - datetime.datetime.now()).total_seconds()
                    if remaining_time <= 0:
                        break
                
                self.condition.wait(timeout=remaining_time)
        
        return {task_id: task.result for task_id, task in self.tasks.items() if task.status == TaskStatus.COMPLETED}
    
    def get_task_status(self, task_id):
        """获取任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            return {
                "task_id": task.task_id,
                "name": task.name,
                "status": task.status.value,
                "start_time": task.start_time.strftime("%Y-%m-%d %H:%M:%S") if task.start_time else None,
                "end_time": task.end_time.strftime("%Y-%m-%d %H:%M:%S") if task.end_time else None,
                "error": task.error
            }
    
    def get_all_tasks(self):
        """获取所有任务状态"""
        with self.lock:
            return {
                task_id: {
                    "task_id": task.task_id,
                    "name": task.name,
                    "status": task.status.value,
                    "priority": task.priority.name,
                    "depends_on": task.depends_on,
                    "start_time": task.start_time.strftime("%Y-%m-%d %H:%M:%S") if task.start_time else None,
                    "end_time": task.end_time.strftime("%Y-%m-%d %H:%M:%S") if task.end_time else None,
                    "error": task.error
                }
                for task_id, task in self.tasks.items()
            }
    
    def cancel_task(self, task_id):
        """取消任务"""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.PENDING]:
                task.status = TaskStatus.CANCELLED
                return True
            
            return False

# 创建全局任务管理器
task_manager = TaskManager()

# 示例任务函数
def example_task(name, delay=1):
    """示例任务，模拟耗时操作"""
    import time
    time.sleep(delay)
    return f"任务 {name} 完成"

# 添加任务管理命令
def handle_task_command(command: str) -> str:
    """处理任务相关命令"""
    global task_manager
    
    # 启动任务管理器
    if not task_manager.running:
        task_manager.start()
    
    # 添加任务
    if command.lower().startswith("task add "):
        parts = command[9:].strip().split(" ", 1)
        if len(parts) < 1:
            return "请指定任务名称"
        
        task_name = parts[0]
        delay = 2  # 默认延迟2秒
        
        if len(parts) > 1:
            try:
                delay = int(parts[1])
            except ValueError:
                pass
        
        task_id = task_manager.add_task(task_name, example_task, args=[task_name, delay])
        return f"已添加任务: {task_name} (ID: {task_id})"
    
    # 添加依赖任务
    if command.lower().startswith("task depend "):
        parts = command[12:].strip().split(" ")
        if len(parts) < 3:
            return "格式: task depend <任务名称> <依赖任务1> [<依赖任务2> ...]"
        
        task_name = parts[0]
        depends_on = parts[1:]
        
        # 检查依赖任务是否存在
        with task_manager.lock:
            for dep_id in depends_on:
                if dep_id not in task_manager.tasks:
                    return f"依赖任务不存在: {dep_id}"
        
        task_id = task_manager.add_task(task_name, example_task, args=[task_name, 1], depends_on=depends_on)
        return f"已添加依赖任务: {task_name} (ID: {task_id}), 依赖于: {', '.join(depends_on)}"
    
    # 列出所有任务
    if command.lower() == "task list":
        tasks = task_manager.get_all_tasks()
        if not tasks:
            return "当前没有任务"
        
        result = ["当前任务列表:"]
        for task_id, task in tasks.items():
            status_emoji = "⏳" if task["status"] == "running" else "✅" if task["status"] == "completed" else "❌" if task["status"] == "failed" else "⏹️" if task["status"] == "cancelled" else "⏱️"
            result.append(f"{status_emoji} {task['name']} (ID: {task_id})")
            result.append(f"   状态: {task['status']}, 优先级: {task['priority']}")
            if task["depends_on"]:
                result.append(f"   依赖: {', '.join(task['depends_on'])}")
            if task["start_time"]:
                result.append(f"   开始时间: {task['start_time']}")
            if task["end_time"]:
                result.append(f"   结束时间: {task['end_time']}")
            if task["error"]:
                result.append(f"   错误: {task['error']}")
        
        return "\n".join(result)
    
    # 等待任务完成
    if command.lower().startswith("task wait "):
        task_id = command[10:].strip()
        
        with task_manager.lock:
            if task_id not in task_manager.tasks:
                return f"任务不存在: {task_id}"
        
        result = task_manager.wait_for_task(task_id, timeout=10)
        if result is not None:
            return f"任务 {task_id} 已完成，结果: {result}"
        else:
            return f"等待任务 {task_id} 超时或任务失败"
    
    # 取消任务
    if command.lower().startswith("task cancel "):
        task_id = command[12:].strip()
        
        with task_manager.lock:
            if task_id not in task_manager.tasks:
                return f"任务不存在: {task_id}"
        
        if task_manager.cancel_task(task_id):
            return f"已取消任务: {task_id}"
        else:
            return f"无法取消任务: {task_id}，可能已经开始执行"
    
    # 等待所有任务
    if command.lower() == "task wait all":
        results = task_manager.wait_all(timeout=30)
        if results:
            result_lines = [f"所有任务已完成，结果:"]
            for task_id, result in results.items():
                task_name = task_manager.tasks[task_id].name
                result_lines.append(f"- {task_name}: {result}")
            return "\n".join(result_lines)
        else:
            return "等待所有任务超时或没有已完成的任务"
    
    return "未知的任务命令。可用命令: task add, task depend, task list, task wait, task cancel, task wait all"

# 全局变量
session_id = None
memory_state = {}
message_history = []
game_state = None

def load_session_state(filepath):
    """加载会话状态"""
    global game_state, session_id, memory_state, message_history
    try:
        loaded_session_id, loaded_memory, loaded_history, loaded_game = load_agent_state(filepath)
        # 更新当前状态
        session_id = loaded_session_id
        memory_state = loaded_memory
        message_history = loaded_history
        if loaded_game:
            game_state = loaded_game
        print(f"\n成功加载会话状态: {session_id}")
    except Exception as e:
        print(f"\n加载会话状态失败: {str(e)}")
        raise e

def setup_workflow():
    """设置并返回工作流程图实例"""
    # 创建可视化工具
    visualizer = GraphVisualizer()
    
    # 添加工作流节点
    visualizer.add_node("start", label="开始")
    visualizer.add_node("process_input", label="处理输入")
    visualizer.add_node("handle_commands", label="处理命令")
    visualizer.add_node("generate_response", label="生成响应")
    visualizer.add_node("update_state", label="更新状态")
    
    # 添加节点之间的连接
    visualizer.add_edge("start", "process_input")
    visualizer.add_edge("process_input", "handle_commands")
    visualizer.add_edge("handle_commands", "generate_response")
    visualizer.add_edge("generate_response", "update_state")
    visualizer.add_edge("update_state", "process_input")
    
    return visualizer

def process_command(user_input: str) -> str:
    """处理用户命令"""
    # 检查是否是游戏命令
    game_response = handle_game_command(user_input)
    if game_response:
        return game_response
        
    # 检查是否是任务命令
    if user_input.lower().startswith("task "):
        return handle_task_command(user_input)
        
    # 检查是否是知识库命令
    if user_input.lower().startswith("kb "):
        if user_input.lower().startswith("kb add "):
            file_path = user_input[7:].strip()
            return add_document_to_knowledge_base(file_path)
        elif user_input.lower().startswith("kb query "):
            query = user_input[9:].strip()
            return query_knowledge_base(query)
            
    # 添加调试和可视化命令
    if user_input.lower() == "debug status":
        return f"""
调试状态:
- 会话ID: {session_id}
- 消息历史: {len(message_history)} 条消息
- 内存状态: {len(memory_state)} 个记录
- 游戏状态: {'活跃' if game_state['active_game'] else '未激活'}
"""

    if user_input.lower() == "debug report":
        report_path = create_debug_report()
        return f"调试报告已生成: {report_path}"
        
    if user_input.lower() == "show workflow":
        workflow_visualizer = setup_workflow()
        workflow_visualizer.draw(save_path="workflow_graph.png")
        return "工作流程图已保存为 workflow_graph.png"
        
    if user_input.lower() == "performance":
        metrics = performance_analyzer.get_metrics()
        summary = performance_analyzer.get_summary()
        return f"""
性能指标摘要:
- 总操作数: {summary.get('total_operations', 0)}
- 总执行时间: {summary.get('total_duration', 0):.2f} 秒
- 平均执行时间: {summary.get('avg_duration', 0):.2f} 秒
- 最长操作: {summary.get('max_duration', 0):.2f} 秒
- 最短操作: {summary.get('min_duration', 0):.2f} 秒
"""
        
    if user_input.lower() == "help":
        return """
可用命令:
基础命令:
- 'exit': 退出程序
- 'help': 显示此帮助信息

游戏命令:
- 'game': 显示可用游戏
- '玩游戏': 同上
- '退出游戏': 退出当前游戏

任务管理:
- 'task add <任务名> [延迟秒数]': 添加任务
- 'task depend <任务名> <依赖任务1> [依赖任务2...]': 添加依赖任务
- 'task list': 列出所有任务
- 'task wait <任务ID>': 等待指定任务完成
- 'task cancel <任务ID>': 取消任务
- 'task wait all': 等待所有任务完成

知识库操作:
- 'kb add <文件路径>': 向知识库添加文档
- 'kb query <查询>': 查询知识库

调试和可视化:
- 'debug status': 显示当前调试状态
- 'debug report': 生成调试报告
- 'show workflow': 显示工作流程图
- 'performance': 显示性能指标

会话管理:
- 'save': 保存当前会话状态
- 'load': 加载已保存的会话状态
- 'list': 列出所有保存的会话状态
"""

    # 处理其他命令...
    if user_input.lower() == "save":
        try:
            filepath = save_agent_state(session_id, memory_state, message_history, game_state)
            return f"会话状态已保存到: {filepath}"
        except Exception as e:
            return f"保存会话状态失败: {str(e)}"
            
    if user_input.lower() == "load":
        states = list_saved_states()
        if not states:
            return "没有找到已保存的会话状态"
            
        result = ["请选择要加载的会话状态 (输入编号):"]
        for i, state in enumerate(states, 1):
            result.append(f"{i}. 会话ID: {state['session_id']}, 时间: {state['timestamp']}")
        return "\n".join(result)
        
    if user_input.lower() == "list":
        states = list_saved_states()
        if not states:
            return "没有找到已保存的会话状态"
            
        result = ["已保存的会话状态:"]
        for i, state in enumerate(states, 1):
            result.append(f"{i}. 会话ID: {state['session_id']}, 时间: {state['timestamp']}, 文件: {state['filename']}")
        return "\n".join(result)
        
    # 如果不是特殊命令，交给AI处理
    try:
        # 使用较短的消息历史减少token使用
        recent_messages = message_history[-5:] if len(message_history) > 5 else message_history
        response = model.invoke(recent_messages + [HumanMessage(content=user_input)])
        return response.content
    except Exception as e:
        return f"处理消息时出错: {str(e)}"

def main():
    global session_id, memory_state, message_history, game_state
    
    # 设置工作流程可视化
    workflow_visualizer = setup_workflow()
    
    # 设置性能分析
    performance_analyzer.start_operation("session_initialization")
    
    try:
        # 初始化会话状态
        session_id = str(uuid.uuid4())
        memory_state = {"context": [], "summary": ""}
        message_history = []
        game_state = {
            "active_game": None,
            "guess_number": {"target": None, "attempts": []},
            "adventure": {"current_scene": None, "inventory": []}
        }
        
        debug_info.log("Session initialized", level="INFO", session_id=session_id)
        performance_analyzer.end_operation("session_initialization")
        
        # 主循环
        while True:
            try:
                # 记录性能指标
                performance_analyzer.start_operation("input_processing")
                
                # 获取用户输入
                user_input = input("\nUser: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    debug_info.log("User requested exit", level="INFO")
                    break
                
                # 处理用户输入
                debug_info.log("Processing user input", level="DEBUG", input=user_input)
                
                # 更新交互式调试器状态
                interactive_debugger.update_state({
                    "session_id": session_id,
                    "memory_state": memory_state,
                    "message_history": message_history,
                    "game_state": game_state
                })
                
                performance_analyzer.end_operation("input_processing")
                
                # 处理命令
                performance_analyzer.start_operation("command_processing")
                response = process_command(user_input)
                performance_analyzer.end_operation("command_processing")
                
                # 打印响应
                print("\nAssistant:", response)
                
                # 更新消息历史
                message_history.append({"role": "user", "content": user_input})
                message_history.append({"role": "assistant", "content": response})
                
                # 定期生成调试报告
                if len(message_history) % 10 == 0:
                    report_path = create_debug_report()
                    debug_info.log(f"Debug report generated", level="INFO", path=report_path)
                
            except Exception as e:
                debug_info.log(f"Error in main loop: {str(e)}", level="ERROR")
                print(f"\nError: {str(e)}")
                continue
    
    except Exception as e:
        debug_info.log(f"Fatal error: {str(e)}", level="ERROR")
        print(f"\nFatal error: {str(e)}")
    
    finally:
        # 保存最终的调试报告
        final_report_path = create_debug_report()
        print(f"\nFinal debug report generated: {final_report_path}")
        
        # 显示工作流程图
        workflow_visualizer.draw(save_path="workflow_graph.png")
        print("\nWorkflow graph saved as workflow_graph.png")

if __name__ == "__main__":
    main()