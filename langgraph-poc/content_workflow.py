"""
内容生成工作流 - 基于LangGraph实现图表中的内容处理流程
支持外部工具调用和实际应用场景
"""

import os
import json
import importlib
import logging
import requests
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set, Literal, Callable, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContentWorkflow")

# 尝试导入LangGraph依赖
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    logger.error("请安装必要的依赖: pip install langgraph")
    raise

# ======== 工具调用框架 ========

class ToolType(Enum):
    """工具类型"""
    LLM = "llm"                # 大语言模型
    VECTOR_DB = "vector_db"    # 向量数据库
    FILE_SYSTEM = "file_system"  # 文件系统
    API = "api"                # 外部API
    IMAGE_GEN = "image_gen"    # 图像生成
    VIDEO_GEN = "video_gen"    # 视频生成
    DOCUMENT = "document"      # 文档处理
    CUSTOM = "custom"          # 自定义工具

@dataclass
class ToolConfig:
    """工具配置"""
    name: str                  # 工具名称
    type: ToolType             # 工具类型
    config: Dict[str, Any] = field(default_factory=dict)  # 配置参数
    description: str = ""      # 工具描述
    enabled: bool = True       # 是否启用

@dataclass
class ToolResult:
    """工具调用结果"""
    success: bool = True       # 是否成功
    data: Any = None           # 返回数据
    error: str = ""            # 错误信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.configs: Dict[str, ToolConfig] = {}
    
    def register_tool(self, name: str, tool_func: Callable, config: ToolConfig):
        """注册工具"""
        if name in self.tools:
            logger.warning(f"工具 '{name}' 已存在，将被覆盖")
        
        self.tools[name] = tool_func
        self.configs[name] = config
        logger.info(f"工具 '{name}' 已注册")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具"""
        if name not in self.tools:
            logger.warning(f"工具 '{name}' 不存在")
            return None
        
        if not self.configs[name].enabled:
            logger.warning(f"工具 '{name}' 已禁用")
            return None
        
        return self.tools[name]
    
    def call_tool(self, name: str, **kwargs) -> ToolResult:
        """调用工具"""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(success=False, error=f"工具 '{name}' 不存在或已禁用")
        
        try:
            result = tool(**kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.error(f"调用工具 '{name}' 失败: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    def load_config_from_file(self, config_path: str):
        """从文件加载配置"""
        if not config_path:
            logger.info("未提供配置文件路径，跳过工具配置加载")
            return
            
        if not os.path.exists(config_path):
            logger.warning(f"配置文件 '{config_path}' 不存在，跳过工具配置加载")
            return
        
        try:
            # 读取并解析配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    logger.error(f"不支持的配置文件格式: {config_path}")
                    return
            
            # 处理工具配置
            tools_config = config.get('tools', [])
            if not tools_config:
                logger.warning(f"配置文件中没有找到工具配置: {config_path}")
                return
                
            for tool_config in tools_config:
                name = tool_config.get('name')
                if not name:
                    logger.warning("发现无名称的工具配置，跳过")
                    continue
                
                try:
                    tool_type = ToolType(tool_config.get('type', 'custom'))
                except ValueError:
                    logger.warning(f"无效的工具类型: {tool_config.get('type')}，使用'custom'")
                    tool_type = ToolType.CUSTOM
                
                self.configs[name] = ToolConfig(
                    name=name,
                    type=tool_type,
                    config=tool_config.get('config', {}),
                    description=tool_config.get('description', ''),
                    enabled=tool_config.get('enabled', True)
                )
                
                logger.info(f"从配置文件加载工具配置: '{name}'")
        
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")

# 创建全局工具注册表
tools = ToolRegistry()

# ======== 示例工具实现 ========

def llm_generate_text(prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> str:
    """调用LLM生成文本"""
    logger.info(f"调用LLM生成文本: {prompt[:50]}...")
    
    # 从配置获取API密钥
    config = tools.configs.get('llm_generate_text', ToolConfig(name="", type=ToolType.LLM)).config
    api_key = config.get('api_key', os.environ.get('OPENAI_API_KEY', ''))
    
    if not api_key:
        # 模拟返回 - 针对不同prompt返回不同结构的模拟数据
        if "提取意图" in prompt:
            # 返回意图提取结果
            if "报告" in prompt or "文档" in prompt:
                return json.dumps({
                    "intent": "生成报告",
                    "theme": "一般报告",
                    "pages": 5,
                    "description": prompt.split("用户输入:")[-1].strip()
                }, ensure_ascii=False)
            elif "问候" in prompt or "你好" in prompt:
                return json.dumps({
                    "intent": "闲聊",
                    "theme": "问候",
                    "pages": 0,
                    "description": "用户只是在打招呼"
                }, ensure_ascii=False)
            else:
                # 默认解析为闲聊
                return json.dumps({
                    "intent": "闲聊",
                    "theme": "一般对话",
                    "pages": 0,
                    "description": prompt.split("用户输入:")[-1].strip()
                }, ensure_ascii=False)
        elif "内容结构" in prompt:
            # 返回内容结构
            return json.dumps({
                "structure_type": "标准报告结构",
                "theme_template": "通用报告模板",
                "outline": [
                    "1. 引言",
                    "2. 背景",
                    "3. 主要内容",
                    "4. 结论",
                    "5. 建议"
                ]
            }, ensure_ascii=False)
        elif "大纲" in prompt:
            # 返回大纲
            return """
            1. 引言
               1.1 报告目的
               1.2 范围界定
               1.3 方法概述
            2. 背景
               2.1 历史情境
               2.2 当前状况
               2.3 相关研究
            3. 主要内容
               3.1 数据分析
               3.2 关键发现
               3.3 趋势观察
            4. 结论
               4.1 关键总结
               4.2 价值评估
            5. 建议
               5.1 近期行动建议
               5.2 长期策略
               5.3 资源需求
            """
        elif "生成" in prompt and "章节" in prompt:
            # 返回章节内容
            section_name = ""
            for line in prompt.split("\n"):
                if line.strip() and "章节" in line:
                    parts = line.split('"')
                    if len(parts) >= 3:
                        section_name = parts[1]
                        break
            
            if not section_name:
                section_name = "未命名章节"
                
            return f"""
            # {section_name}
            
            本章节将详细介绍{section_name}的相关内容。这是一个示例文本，在实际应用中，这里会包含更详细、专业的内容。
            
            ## 主要内容
            
            {section_name}是整个报告中的重要组成部分，它涉及到多个方面的信息和数据分析。通过对相关数据的深入研究，我们得出了以下几点重要发现：
            
            1. 第一个重要发现是关于...
            2. 第二个重要发现表明...
            3. 数据分析还显示...
            
            ## 详细分析
            
            基于上述发现，我们进行了更深入的分析。结果表明这些趋势预计将在未来几个月内持续，并可能对相关领域产生重大影响。
            
            ## 小结
            
            总结来说，{section_name}的分析结果对整体报告具有重要意义。这些发现不仅帮助我们理解当前状况，还为未来的决策提供了依据。
            """
        elif "图表" in prompt:
            # 返回图表数据
            return json.dumps({
                "数据类型": "收入分析",
                "数据": {
                    "Q1": 10000,
                    "Q2": 12000,
                    "Q3": 15000,
                    "Q4": 18000
                },
                "图表类型": "bar"
            }, ensure_ascii=False)
        else:
            # 其他类型的生成，提供默认响应
            return f"这是对'{prompt[:30]}...'的模拟回复。在实际应用中，这里会调用LLM API获取真实响应。"
    
    # 实际实现可以调用OpenAI的API
    # import openai
    # openai.api_key = api_key
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}],
    #     **kwargs
    # )
    # return response.choices[0].message.content
    
    # 模拟返回
    return f"这是'{prompt[:30]}...'的生成结果"

def vector_db_query(query: str, collection: str = "default", top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
    """查询向量数据库"""
    logger.info(f"查询向量数据库: {query[:50]}...")
    
    # 从配置获取API密钥和URL
    config = tools.configs.get('vector_db_query', ToolConfig(name="", type=ToolType.VECTOR_DB)).config
    api_key = config.get('api_key', '')
    url = config.get('url', '')
    
    # 模拟不同集合的查询结果
    if collection == "templates":
        if "财务报告" in query:
            return [
                {
                    "text": """
                    财务报告模板:
                    {
                        "structure_type": "财务报表结构",
                        "theme_template": "季度财务报告模板",
                        "outline": [
                            "1. 执行摘要",
                            "2. 财务概况",
                            "3. 收入分析",
                            "4. 支出分析",
                            "5. 利润率",
                            "6. 未来展望"
                        ]
                    }
                    """,
                    "score": 0.95
                },
                {
                    "text": """
                    财务数据分析:
                    {
                        "structure_type": "财务分析结构",
                        "theme_template": "财务数据分析模板",
                        "outline": [
                            "1. 数据概览",
                            "2. 关键指标",
                            "3. 趋势分析",
                            "4. 风险评估",
                            "5. 建议"
                        ]
                    }
                    """,
                    "score": 0.85
                }
            ]
        elif "项目报告" in query:
            return [
                {
                    "text": """
                    项目报告模板:
                    {
                        "structure_type": "项目报表结构",
                        "theme_template": "项目进度报告模板",
                        "outline": [
                            "1. 项目概述",
                            "2. 里程碑完成情况",
                            "3. 资源使用",
                            "4. 风险与问题",
                            "5. 下一步计划"
                        ]
                    }
                    """,
                    "score": 0.95
                },
                {
                    "text": """
                    项目管理报告:
                    {
                        "structure_type": "项目管理结构",
                        "theme_template": "项目管理报告模板",
                        "outline": [
                            "1. 项目状态摘要",
                            "2. 团队表现",
                            "3. 成本控制",
                            "4. 质量指标",
                            "5. 问题与解决方案"
                        ]
                    }
                    """,
                    "score": 0.88
                }
            ]
        else:
            # 默认报告模板
            return [
                {
                    "text": """
                    通用报告模板:
                    {
                        "structure_type": "标准报告结构",
                        "theme_template": "通用报告模板",
                        "outline": [
                            "1. 引言",
                            "2. 背景",
                            "3. 主要内容",
                            "4. 结论",
                            "5. 建议"
                        ]
                    }
                    """,
                    "score": 0.75
                },
                {
                    "text": """
                    简报模板:
                    {
                        "structure_type": "简报结构",
                        "theme_template": "简报模板",
                        "outline": [
                            "1. 概述",
                            "2. 要点",
                            "3. 数据支持",
                            "4. 总结"
                        ]
                    }
                    """,
                    "score": 0.70
                }
            ]
    else:
        # 其他集合的默认响应
        return [
            {"text": f"向量数据库结果 1 for {query[:20]}", "score": 0.95},
            {"text": f"向量数据库结果 2 for {query[:20]}", "score": 0.85},
        ]

def generate_image(prompt: str, size: str = "1024x1024", **kwargs) -> str:
    """生成图像"""
    logger.info(f"生成图像: {prompt[:50]}...")
    
    # 从配置获取API密钥
    config = tools.configs.get('generate_image', ToolConfig(name="", type=ToolType.IMAGE_GEN)).config
    api_key = config.get('api_key', os.environ.get('OPENAI_API_KEY', ''))
    
    if not api_key:
        # 模拟返回
        image_path = f"./outputs/images/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        return image_path
    
    # 实际实现可以调用OpenAI的DALL-E API
    # import openai
    # openai.api_key = api_key
    # response = openai.Image.create(
    #     prompt=prompt,
    #     n=1,
    #     size=size,
    #     **kwargs
    # )
    # return response['data'][0]['url']
    
    # 模拟返回
    image_path = f"./outputs/images/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    return image_path

def generate_chart(data: Dict[str, Any], chart_type: str = "bar", **kwargs) -> str:
    """生成图表"""
    logger.info(f"生成图表: {chart_type} - {str(data)[:50]}...")
    
    # 模拟返回
    chart_path = f"./outputs/charts/{chart_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    
    # 实际实现可以使用matplotlib等库生成图表
    # import matplotlib.pyplot as plt
    # if chart_type == "bar":
    #     plt.bar(data.keys(), data.values())
    # elif chart_type == "line":
    #     plt.plot(list(data.values()))
    # plt.savefig(chart_path)
    # plt.close()
    
    return chart_path

def create_document(content: Dict[str, Any], template: str = "default", output_format: str = "docx", **kwargs) -> str:
    """创建文档"""
    logger.info(f"创建文档: {template} / {output_format}")
    
    # 模拟返回
    file_name = f"{content.get('title', 'document')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{output_format}"
    file_path = f"./outputs/{file_name}"
    
    # 实际实现可以使用python-docx等库创建文档
    # if output_format == "docx":
    #     from docx import Document
    #     doc = Document()
    #     doc.add_heading(content.get('title', 'Document'), 0)
    #     for section in content.get('sections', []):
    #         doc.add_heading(section.get('title', ''), level=1)
    #         doc.add_paragraph(section.get('content', ''))
    #     doc.save(file_path)
    
    return file_path

# 注册默认工具
tools.register_tool(
    "llm_generate_text", 
    llm_generate_text, 
    ToolConfig(name="llm_generate_text", type=ToolType.LLM, description="使用LLM生成文本")
)

tools.register_tool(
    "vector_db_query", 
    vector_db_query, 
    ToolConfig(name="vector_db_query", type=ToolType.VECTOR_DB, description="查询向量数据库")
)

tools.register_tool(
    "generate_image", 
    generate_image, 
    ToolConfig(name="generate_image", type=ToolType.IMAGE_GEN, description="生成图像")
)

tools.register_tool(
    "generate_chart", 
    generate_chart, 
    ToolConfig(name="generate_chart", type=ToolType.IMAGE_GEN, description="生成图表")
)

tools.register_tool(
    "create_document", 
    create_document, 
    ToolConfig(name="create_document", type=ToolType.DOCUMENT, description="创建文档")
)

# ======== 定义状态模型 ========

@dataclass
class DialogInfo:
    """对话信息"""
    intent: str = ""  # 意图
    theme: str = ""   # 主题
    pages: int = 0    # 页数
    description: str = ""  # 描述
    
@dataclass
class ContentStructure:
    """内容结构"""
    structure_type: str = ""  # 结构类型
    theme_template: str = ""  # 主题模板
    outline: List[str] = field(default_factory=list)  # 内容大纲
    
@dataclass
class MediaContent:
    """媒体内容"""
    text_content: Dict[str, str] = field(default_factory=dict)  # 文本内容
    image_content: Dict[str, str] = field(default_factory=dict)  # 图片内容
    video_content: Dict[str, str] = field(default_factory=dict)  # 视频内容
    
@dataclass
class DocumentInfo:
    """文档信息"""
    format: str = ""  # 文档格式
    file_path: str = ""  # 文件路径
    preview_url: str = ""  # 预览URL
    
@dataclass
class WorkflowState:
    """工作流状态"""
    # 用户输入和对话处理
    user_input: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # 提取的对话信息
    dialog_info: DialogInfo = field(default_factory=DialogInfo)
    
    # 内容结构
    content_structure: ContentStructure = field(default_factory=ContentStructure)
    
    # 页面内容
    pages: List[Dict[str, Any]] = field(default_factory=list)
    
    # 媒体内容
    media_content: MediaContent = field(default_factory=MediaContent)
    
    # 文档输出
    document_info: DocumentInfo = field(default_factory=DocumentInfo)
    
    # 处理状态
    current_step: str = "初始化"
    errors: List[str] = field(default_factory=list)
    
    # 工具调用记录
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self):
        """转换为字典"""
        return asdict(self)
    
    def add_tool_call(self, tool_name: str, inputs: Dict[str, Any], result: ToolResult):
        """添加工具调用记录"""
        self.tool_calls.append({
            "tool": tool_name,
            "inputs": inputs,
            "success": result.success,
            "error": result.error,
            "timestamp": datetime.now().isoformat()
        })
        
        # 同时添加到消息历史
        if result.success:
            self.messages.append({
                "role": "tool",
                "tool": tool_name,
                "content": f"工具 '{tool_name}' 调用成功"
            })
        else:
            self.messages.append({
                "role": "tool",
                "tool": tool_name,
                "content": f"工具 '{tool_name}' 调用失败: {result.error}"
            })

# ======== 前端处理节点 ========

def process_dialog(state: WorkflowState) -> WorkflowState:
    """处理对话，提取意图、主题、页数和描述"""
    logger.info(f"[进行中] 处理对话: '{state.user_input}'")
    
    # 如果用户输入为空或只是简单问候，提供默认响应
    if not state.user_input or state.user_input.strip() in ["你好", "hello", "hi", "嗨"]:
        state.dialog_info.intent = "闲聊"
        state.dialog_info.theme = "问候"
        state.dialog_info.description = "用户只是在打招呼"
        
        # 更新处理状态
        state.current_step = "对话处理完成"
        state.messages.append({
            "role": "system", 
            "content": f"这是一个问候。如果您想生成内容，请提供更多详细信息，例如'我需要一份财务报告'。"
        })
        
        return state
    
    # 调用LLM提取意图和主题
    prompt = f"""
    从以下用户输入中提取意图、主题、页数和描述。
    格式要求:
    {{
        "intent": "生成报告", // 意图，如生成报告、查询数据等
        "theme": "财务报告", // 主题，如财务报告、项目报告等
        "pages": 5, // 页数，整数
        "description": "用户的描述" // 原始描述
    }}
    
    用户输入: {state.user_input}
    """
    
    result = tools.call_tool("llm_generate_text", prompt=prompt, model="gpt-3.5-turbo")
    state.add_tool_call("llm_generate_text", {"prompt": prompt}, result)
    
    if result.success:
        try:
            # 尝试解析JSON
            llm_response = result.data
            # 在实际环境中，可能需要处理LLM返回的非标准JSON
            if isinstance(llm_response, str):
                # 提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    llm_response = json_match.group(0)
                
                try:
                    extracted_data = json.loads(llm_response)
                    state.dialog_info.intent = extracted_data.get("intent", "")
                    state.dialog_info.theme = extracted_data.get("theme", "")
                    state.dialog_info.pages = extracted_data.get("pages", 5)
                    state.dialog_info.description = extracted_data.get("description", state.user_input)
                except json.JSONDecodeError:
                    raise ValueError(f"无法解析LLM返回的JSON: {llm_response}")
            else:
                # 处理LLM直接返回结构化数据的情况
                state.dialog_info.intent = llm_response.get("intent", "")
                state.dialog_info.theme = llm_response.get("theme", "")
                state.dialog_info.pages = llm_response.get("pages", 5)
                state.dialog_info.description = llm_response.get("description", state.user_input)
        except Exception as e:
            logger.error(f"解析LLM响应失败: {str(e)}")
            # 回退到规则方法
            if "报告" in state.user_input or "文档" in state.user_input:
                state.dialog_info.intent = "生成报告"
                
                # 提取主题
                if "财务" in state.user_input:
                    state.dialog_info.theme = "财务报告"
                elif "项目" in state.user_input:
                    state.dialog_info.theme = "项目报告"
                else:
                    state.dialog_info.theme = "一般报告"
                    
                # 提取页数
                if "详细" in state.user_input:
                    state.dialog_info.pages = 10
                else:
                    state.dialog_info.pages = 5
                    
                # 提取描述
                state.dialog_info.description = state.user_input
            else:
                # 对于不明确的输入，默认设置为闲聊
                state.dialog_info.intent = "闲聊"
                state.dialog_info.theme = "一般对话"
                state.dialog_info.description = state.user_input
    else:
        # 工具调用失败，回退到规则方法
        if "报告" in state.user_input or "文档" in state.user_input:
            state.dialog_info.intent = "生成报告"
            
            # 提取主题
            if "财务" in state.user_input:
                state.dialog_info.theme = "财务报告"
            elif "项目" in state.user_input:
                state.dialog_info.theme = "项目报告"
            else:
                state.dialog_info.theme = "一般报告"
                
            # 提取页数
            if "详细" in state.user_input:
                state.dialog_info.pages = 10
            else:
                state.dialog_info.pages = 5
                
            # 提取描述
            state.dialog_info.description = state.user_input
        else:
            # 对于不明确的输入，默认设置为闲聊
            state.dialog_info.intent = "闲聊"
            state.dialog_info.theme = "一般对话"
            state.dialog_info.description = state.user_input
    
    # 对于非内容生成意图，提供相应的响应并结束工作流
    if state.dialog_info.intent == "闲聊":
        # 提供一个友好的响应
        response = "您好！我是内容生成助手。如果您需要生成报告或文档，请提供更具体的描述，例如'生成一份财务报告'或'我需要一份项目进度报告'。"
        
        # 更新处理状态
        state.current_step = "对话处理完成"
        state.messages.append({
            "role": "system", 
            "content": response
        })
        
        # 为便于测试，我们可以继续正常流程，生成一个最小的样本文档
        if not state.dialog_info.theme:
            state.dialog_info.theme = "一般报告"
    
    # 更新处理状态
    state.current_step = "对话处理完成"
    state.messages.append({
        "role": "system", 
        "content": f"已识别意图: {state.dialog_info.intent}, 主题: {state.dialog_info.theme}"
    })
    
    return state

def create_content_structure(state: WorkflowState) -> WorkflowState:
    """创建内容结构，包括目标结构和主题模版"""
    logger.info(f"[进行中] 创建内容结构: 主题 '{state.dialog_info.theme}'")
    
    # 从向量数据库获取相关模板
    query = f"获取{state.dialog_info.theme}的内容结构和模板"
    result = tools.call_tool("vector_db_query", query=query, collection="templates", top_k=3)
    state.add_tool_call("vector_db_query", {"query": query}, result)
    
    # 设置默认结构
    default_structure = {
        "财务报告": {
            "structure_type": "财务报表结构",
            "theme_template": "季度财务报告模板",
            "outline": [
                "1. 执行摘要",
                "2. 财务概况",
                "3. 收入分析",
                "4. 支出分析",
                "5. 利润率",
                "6. 未来展望"
            ]
        },
        "项目报告": {
            "structure_type": "项目报表结构",
            "theme_template": "项目进度报告模板",
            "outline": [
                "1. 项目概述",
                "2. 里程碑完成情况",
                "3. 资源使用",
                "4. 风险与问题",
                "5. 下一步计划"
            ]
        },
        "一般报告": {
            "structure_type": "标准报告结构",
            "theme_template": "通用报告模板",
            "outline": [
                "1. 引言",
                "2. 背景",
                "3. 主要内容",
                "4. 结论",
                "5. 建议"
            ]
        }
    }
    
    # 如果向量数据库查询成功，尝试使用查询结果
    if result.success and result.data:
        try:
            # 使用LLM从查询结果中选择最佳结构
            context = "\n".join([item.get("text", "") for item in result.data])
            prompt = f"""
            基于用户请求和以下模板上下文，创建一个包含结构类型、主题模板和内容大纲的内容结构。
            用户请求: {state.user_input}
            主题: {state.dialog_info.theme}
            
            模板上下文:
            {context}
            
            请以JSON格式返回，格式为:
            {{
                "structure_type": "结构类型",
                "theme_template": "主题模板名称",
                "outline": ["1. 第一章", "2. 第二章", ...]
            }}
            """
            
            llm_result = tools.call_tool("llm_generate_text", prompt=prompt)
            state.add_tool_call("llm_generate_text", {"prompt": prompt}, llm_result)
            
            if llm_result.success:
                try:
                    # 解析LLM响应
                    content_structure = llm_result.data
                    if isinstance(content_structure, str):
                        # 提取JSON
                        import re
                        json_match = re.search(r'\{.*\}', content_structure, re.DOTALL)
                        if json_match:
                            content_structure_json = json_match.group(0)
                            try:
                                content_structure = json.loads(content_structure_json)
                                state.content_structure.structure_type = content_structure.get("structure_type", "")
                                state.content_structure.theme_template = content_structure.get("theme_template", "")
                                state.content_structure.outline = content_structure.get("outline", [])
                            except json.JSONDecodeError:
                                raise ValueError(f"无法解析LLM返回的JSON结构: {content_structure_json}")
                    else:
                        state.content_structure.structure_type = content_structure.get("structure_type", "")
                        state.content_structure.theme_template = content_structure.get("theme_template", "")
                        state.content_structure.outline = content_structure.get("outline", [])
                except Exception as e:
                    logger.error(f"解析LLM结构化内容失败: {str(e)}")
                    # 回退到默认结构
                    theme_key = state.dialog_info.theme if state.dialog_info.theme in default_structure else "一般报告"
                    default = default_structure.get(theme_key)
                    state.content_structure.structure_type = default["structure_type"]
                    state.content_structure.theme_template = default["theme_template"]
                    state.content_structure.outline = default["outline"]
            else:
                # LLM调用失败，使用默认结构
                theme_key = state.dialog_info.theme if state.dialog_info.theme in default_structure else "一般报告"
                default = default_structure.get(theme_key)
                state.content_structure.structure_type = default["structure_type"]
                state.content_structure.theme_template = default["theme_template"]
                state.content_structure.outline = default["outline"]
        except Exception as e:
            logger.error(f"处理内容结构时出错: {str(e)}")
            # 回退到默认结构
            theme_key = state.dialog_info.theme if state.dialog_info.theme in default_structure else "一般报告"
            default = default_structure.get(theme_key)
            state.content_structure.structure_type = default["structure_type"]
            state.content_structure.theme_template = default["theme_template"]
            state.content_structure.outline = default["outline"]
    else:
        # 向量数据库查询失败，使用默认结构
        theme_key = state.dialog_info.theme if state.dialog_info.theme in default_structure else "一般报告"
        default = default_structure.get(theme_key)
        state.content_structure.structure_type = default["structure_type"]
        state.content_structure.theme_template = default["theme_template"]
        state.content_structure.outline = default["outline"]
    
    # 更新处理状态
    state.current_step = "内容结构创建完成"
    state.messages.append({
        "role": "system", 
        "content": f"已创建内容结构，使用模板: {state.content_structure.theme_template}"
    })
    
    return state

# ======== 处理端节点 ========

def generate_outline(state: WorkflowState) -> WorkflowState:
    """生成纲要"""
    logger.info(f"[进行中] 生成纲要: 使用模板 '{state.content_structure.theme_template}'")
    
    # 调用LLM生成详细大纲
    prompt = f"""
    基于以下主题和大纲框架，生成更详细的内容大纲。
    每个章节应该有2-3个子章节。
    
    主题: {state.dialog_info.theme}
    模板: {state.content_structure.theme_template}
    当前大纲:
    {chr(10).join(state.content_structure.outline)}
    
    请生成更详细的大纲，包括子章节，例如:
    1. 执行摘要
       1.1 报告目的
       1.2 主要发现
       1.3 关键建议
    2. ...
    """
    
    result = tools.call_tool("llm_generate_text", prompt=prompt)
    state.add_tool_call("llm_generate_text", {"prompt": prompt}, result)
    
    if result.success:
        try:
            # 解析LLM响应
            detailed_outline = result.data
            
            # 简单处理：按行分割并保留有数字编号的行
            if isinstance(detailed_outline, str):
                lines = detailed_outline.split('\n')
                # 过滤空行并保留有数字编号的行
                import re
                detailed_lines = [line.strip() for line in lines if re.match(r'^\d+(\.\d+)*\.?\s+', line.strip())]
                if detailed_lines:
                    # 更新大纲
                    state.content_structure.outline = detailed_lines
        except Exception as e:
            logger.error(f"解析详细大纲失败: {str(e)}")
            # 保持原大纲不变
    
    # 更新处理状态
    state.current_step = "纲要生成完成"
    state.messages.append({
        "role": "system", 
        "content": f"已生成{len(state.content_structure.outline)}个章节的纲要"
    })
    
    return state

def create_document(state: WorkflowState) -> WorkflowState:
    """生成文档内容"""
    logger.info(f"[进行中] 生成文档: {state.dialog_info.pages}页")
    
    # 检查是否有大纲，如果没有则添加默认大纲
    if not state.content_structure.outline:
        state.content_structure.outline = [
            "1. 引言",
            "2. 主要内容",
            "3. 结论"
        ]
    
    # 添加一个空页面列表，防止后续步骤出错
    if len(state.pages) > 0:
        logger.info(f"已有 {len(state.pages)} 页内容，跳过生成")
    else:
        logger.info(f"开始生成内容，共 {len(state.content_structure.outline)} 个章节")
    
    # 并行生成各章节内容
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 为每个章节创建生成任务
        future_to_section = {}
        
        for i, outline_item in enumerate(state.content_structure.outline):
            # 只处理主章节（不包含小数点的编号）或者如果内容很少，处理所有章节
            if '.' not in outline_item.split(' ')[0] or len(state.content_structure.outline) <= 5:
                try:
                    section_title = outline_item.split('. ', 1)[1] if '. ' in outline_item else outline_item
                except IndexError:
                    section_title = outline_item  # 使用整个字符串作为标题
                
                # 获取该章节的所有子章节
                section_number = outline_item.split('.')[0] if '.' in outline_item else str(i+1)
                sub_sections = [
                    item for item in state.content_structure.outline 
                    if item.startswith(f"{section_number}.") and item != outline_item
                ]
                
                # 构建提示
                prompt = f"""
                为{state.dialog_info.theme}文档生成"{section_title}"章节的内容。
                
                章节结构:
                {outline_item}
                {chr(10).join(sub_sections)}
                
                请生成详细、专业、连贯的内容，长度约500-800字。
                """
                
                # 提交任务
                future = executor.submit(tools.call_tool, "llm_generate_text", prompt=prompt)
                future_to_section[future] = {
                    "section_number": i + 1,
                    "title": section_title,
                    "outline": outline_item,
                    "sub_sections": sub_sections
                }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_section):
            section_info = future_to_section[future]
            try:
                result = future.result()
                if result.success:
                    # 创建页面
                    page = {
                        "page_number": section_info["section_number"],
                        "title": section_info["title"],
                        "content": result.data,
                        "outline": section_info["outline"],
                        "sub_sections": section_info["sub_sections"],
                        "media": []
                    }
                    
                    # 添加到页面列表
                    state.pages.append(page)
                else:
                    # 创建带有错误信息的页面
                    page = {
                        "page_number": section_info["section_number"],
                        "title": section_info["title"],
                        "content": f"无法生成内容: {result.error}\n\n这是{section_info['title']}章节的占位内容。",
                        "outline": section_info["outline"],
                        "sub_sections": section_info["sub_sections"],
                        "media": []
                    }
                    state.pages.append(page)
                    state.errors.append(f"生成章节 '{section_info['title']}' 内容失败: {result.error}")
            except Exception as e:
                logger.error(f"处理章节 '{section_info['title']}' 时出错: {str(e)}")
                # 创建带有错误信息的页面
                page = {
                    "page_number": section_info["section_number"],
                    "title": section_info["title"],
                    "content": f"生成内容时出错: {str(e)}",
                    "outline": section_info["outline"],
                    "sub_sections": section_info["sub_sections"],
                    "media": []
                }
                state.pages.append(page)
                state.errors.append(f"处理章节 '{section_info['title']}' 时出错: {str(e)}")
    
    # 按页码排序
    state.pages.sort(key=lambda x: x["page_number"])
    
    # 更新处理状态
    state.current_step = "文档生成完成"
    state.messages.append({
        "role": "system", 
        "content": f"已生成{len(state.pages)}页文档内容"
    })
    
    return state

def segment_pages(state: WorkflowState) -> WorkflowState:
    """页面拆分"""
    logger.info(f"[进行中] 页面拆分: {len(state.pages)}页")
    
    # 检查是否有超大页面需要拆分
    new_pages = []
    for page in state.pages:
        content = page["content"]
        
        # 计算内容长度
        if isinstance(content, str) and len(content) > 1500:  # 假设1500字符是一页的合理长度
            # 按段落拆分
            paragraphs = content.split("\n\n")
            
            # 拆分成多个页面
            current_content = ""
            current_page_number = page["page_number"]
            sub_page = 1
            
            for para in paragraphs:
                if len(current_content) + len(para) <= 1500:
                    # 添加到当前页
                    if current_content:
                        current_content += "\n\n"
                    current_content += para
                else:
                    # 创建新页
                    new_page = page.copy()
                    new_page["page_number"] = f"{current_page_number}.{sub_page}"
                    new_page["content"] = current_content
                    new_page["title"] = f"{page['title']} (第{sub_page}部分)"
                    new_pages.append(new_page)
                    
                    # 重置当前内容
                    current_content = para
                    sub_page += 1
            
            # 添加最后一页
            if current_content:
                new_page = page.copy()
                new_page["page_number"] = f"{current_page_number}.{sub_page}"
                new_page["content"] = current_content
                new_page["title"] = f"{page['title']} (第{sub_page}部分)"
                new_pages.append(new_page)
        else:
            # 不需要拆分
            new_pages.append(page)
    
    # 更新页面列表
    state.pages = new_pages
    
    # 更新处理状态
    state.current_step = "页面拆分完成"
    state.messages.append({
        "role": "system", 
        "content": f"已完成页面拆分和格式调整"
    })
    
    return state

def create_media_content(state: WorkflowState) -> WorkflowState:
    """创建媒体内容：文字、图片、视频"""
    logger.info(f"[进行中] 创建媒体内容")
    
    # 文字内容已在文档生成阶段创建，这里处理图片和视频
    
    # 为每个页面生成相关图片/图表
    for page in state.pages:
        page_title = page["title"]
        page_content = page["content"]
        
        # 提取关键数据以生成图表
        if state.dialog_info.theme == "财务报告" and ("财务概况" in page_title or "收入分析" in page_title or "支出分析" in page_title):
            # 使用LLM从内容中提取数据
            prompt = f"""
            从以下财务报告内容中提取关键数据，以便生成图表。
            返回JSON格式的数据，例如：
            {{
                "数据类型": "收入",
                "数据": {{
                    "Q1": 10000,
                    "Q2": 12000,
                    "Q3": 15000,
                    "Q4": 18000
                }},
                "图表类型": "bar" // bar, line, pie 中选一个
            }}
            
            内容：
            {page_content}
            """
            
            result = tools.call_tool("llm_generate_text", prompt=prompt)
            state.add_tool_call("llm_generate_text", {"prompt": prompt}, result)
            
            if result.success:
                try:
                    # 解析LLM生成的数据
                    chart_data = result.data
                    if isinstance(chart_data, str):
                        # 提取JSON
                        import re
                        json_match = re.search(r'\{.*\}', chart_data, re.DOTALL)
                        if json_match:
                            chart_data = json.loads(json_match.group(0))
                    
                    # 生成图表
                    chart_type = chart_data.get("图表类型", "bar")
                    data = chart_data.get("数据", {})
                    
                    if data:
                        chart_result = tools.call_tool("generate_chart", data=data, chart_type=chart_type)
                        state.add_tool_call("generate_chart", {"data": data, "chart_type": chart_type}, chart_result)
                        
                        if chart_result.success:
                            # 添加到页面媒体
                            chart_path = chart_result.data
                            page["media"].append({
                                "type": "image",
                                "content": chart_path,
                                "caption": f"{chart_data.get('数据类型', '数据')}图表"
                            })
                            
                            # 添加到媒体内容集合
                            state.media_content.image_content[f"chart_{page['page_number']}"] = chart_path
                except Exception as e:
                    logger.error(f"处理图表数据失败: {str(e)}")
        
        # 为项目报告生成甘特图
        elif state.dialog_info.theme == "项目报告" and ("里程碑" in page_title or "进度" in page_title):
            # 生成甘特图
            # 实际应用中，这里可以调用专门的甘特图生成库
            chart_data = {
                "任务1": {"开始": "2023-01-01", "结束": "2023-02-15"},
                "任务2": {"开始": "2023-02-01", "结束": "2023-03-15"},
                "任务3": {"开始": "2023-03-01", "结束": "2023-04-30"}
            }
            
            chart_result = tools.call_tool("generate_chart", data=chart_data, chart_type="gantt")
            state.add_tool_call("generate_chart", {"data": chart_data, "chart_type": "gantt"}, chart_result)
            
            if chart_result.success:
                chart_path = chart_result.data
                page["media"].append({
                    "type": "image",
                    "content": chart_path,
                    "caption": "项目进度甘特图"
                })
                
                state.media_content.image_content["gantt_chart"] = chart_path
        
        # 生成封面图或其他插图
        if page["page_number"] == 1 or "概述" in page_title:
            prompt = f"为{state.dialog_info.theme}报告创建一个专业的封面图，主题是{page_title}"
            image_result = tools.call_tool("generate_image", prompt=prompt)
            state.add_tool_call("generate_image", {"prompt": prompt}, image_result)
            
            if image_result.success:
                image_path = image_result.data
                page["media"].append({
                    "type": "image",
                    "content": image_path,
                    "caption": f"{state.dialog_info.theme}封面图"
                })
                
                state.media_content.image_content["cover_image"] = image_path
        
        # 添加视频
        if "视频" in state.user_input and page["page_number"] == 1:
            # 模拟视频生成，实际应用中可以调用视频生成API
            video_path = f"./outputs/videos/{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
            
            page["media"].append({
                "type": "video",
                "content": video_path,
                "caption": f"{state.dialog_info.theme}概述视频"
            })
            
            state.media_content.video_content["summary_video"] = video_path
    
    # 更新处理状态
    state.current_step = "媒体内容创建完成"
    state.messages.append({
        "role": "system", 
        "content": f"已创建媒体内容: {len(state.pages)}段文字, " \
                  f"{len(state.media_content.image_content)}张图片, " \
                  f"{len(state.media_content.video_content)}个视频"
    })
    
    return state

def assemble_layout(state: WorkflowState) -> WorkflowState:
    """布局组装"""
    logger.info(f"[进行中] 布局组装")
    
    # 在实际应用中，这里可以调用专门的布局引擎或模板系统
    # 这里我们模拟布局过程
    
    # 为每个页面生成布局信息
    for page in state.pages:
        # 根据媒体类型计算布局
        media_count = len(page["media"])
        
        if media_count == 0:
            # 纯文本布局
            page["layout"] = {
                "type": "text_only",
                "title_position": "top",
                "content_position": "full"
            }
        elif media_count == 1:
            # 单媒体布局
            media_type = page["media"][0]["type"]
            
            if media_type == "image":
                page["layout"] = {
                    "type": "text_with_image",
                    "title_position": "top",
                    "image_position": "right",
                    "content_position": "left"
                }
            elif media_type == "video":
                page["layout"] = {
                    "type": "text_with_video",
                    "title_position": "top",
                    "video_position": "center",
                    "content_position": "bottom"
                }
        else:
            # 多媒体布局
            page["layout"] = {
                "type": "multi_media",
                "title_position": "top",
                "content_position": "left",
                "media_position": "grid"
            }
    
    # 更新处理状态
    state.current_step = "布局组装完成"
    state.messages.append({
        "role": "system", 
        "content": "已完成文档布局组装"
    })
    
    return state

def synthesize_file(state: WorkflowState) -> WorkflowState:
    """文件合成"""
    logger.info(f"[进行中] 文件合成")
    
    # 根据主题选择输出格式
    if state.dialog_info.theme == "财务报告":
        state.document_info.format = "Excel"
        file_name = f"财务报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
    elif state.dialog_info.theme == "项目报告":
        state.document_info.format = "PPT"
        file_name = f"项目报告_{datetime.now().strftime('%Y%m%d')}.pptx"
    else:
        state.document_info.format = "Word"
        file_name = f"报告_{datetime.now().strftime('%Y%m%d')}.docx"
    
    # 准备文档内容
    doc_content = {
        "title": state.dialog_info.theme,
        "sections": []
    }
    
    for page in state.pages:
        section = {
            "title": page["title"],
            "content": page["content"],
            "media": page["media"],
            "layout": page.get("layout", {})
        }
        doc_content["sections"].append(section)
    
    # 调用文档创建工具
    result = tools.call_tool(
        "create_document", 
        content=doc_content, 
        template=state.content_structure.theme_template,
        output_format=state.document_info.format.lower()
    )
    state.add_tool_call("create_document", {"content": "document_content", "template": state.content_structure.theme_template}, result)
    
    if result.success:
        # 更新文档信息
        state.document_info.file_path = result.data
        state.document_info.preview_url = f"http://example.com/preview/{os.path.basename(result.data)}"
    else:
        # 设置默认文件路径
        state.document_info.file_path = f"./outputs/{file_name}"
        state.document_info.preview_url = f"http://example.com/preview/{file_name}"
        state.errors.append(f"文件合成失败: {result.error}")
    
    # 更新处理状态
    state.current_step = "文件合成完成"
    state.messages.append({
        "role": "system", 
        "content": f"已生成{state.document_info.format}格式文件: {state.document_info.file_path}"
    })
    
    return state

# ======== 协作端节点 ========

def integrate_with_systems(state: WorkflowState) -> WorkflowState:
    """与IT系统集成"""
    logger.info(f"[进行中] 系统集成")
    
    # 模拟IT系统集成API
    def mock_it_system_api(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "system_id": f"doc-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "url": f"http://it-system.example.com/documents/{os.path.basename(file_path)}"
        }
    
    # 准备元数据
    metadata = {
        "title": state.dialog_info.theme,
        "author": "内容生成系统",
        "created_at": datetime.now().isoformat(),
        "type": state.document_info.format,
        "pages": len(state.pages),
        "description": state.dialog_info.description
    }
    
    try:
        # 在实际应用中，这里会调用外部API或服务
        # 例如将文档上传到文档管理系统、内容管理系统或其他IT系统
        result = mock_it_system_api(state.document_info.file_path, metadata)
        
        if result.get("success"):
            # 更新预览URL为IT系统URL
            state.document_info.preview_url = result.get("url", state.document_info.preview_url)
            
            # 记录成功
            state.messages.append({
                "role": "system", 
                "content": f"文档已与IT系统集成，系统ID: {result.get('system_id')}"
            })
        else:
            # 记录错误
            error_msg = result.get("error", "未知错误")
            state.errors.append(f"IT系统集成失败: {error_msg}")
            state.messages.append({
                "role": "system", 
                "content": f"文档与IT系统集成失败: {error_msg}"
            })
    except Exception as e:
        # 记录错误
        logger.error(f"IT系统集成出错: {str(e)}")
        state.errors.append(f"IT系统集成出错: {str(e)}")
        state.messages.append({
            "role": "system", 
            "content": f"文档与IT系统集成时发生错误: {str(e)}"
        })
    
    # 更新处理状态
    state.current_step = "系统集成完成"
    state.messages.append({
        "role": "system", 
        "content": f"文档已与IT系统集成，可在 {state.document_info.preview_url} 预览"
    })
    
    return state

def update_vector_db(state: WorkflowState) -> WorkflowState:
    """更新向量数据库"""
    logger.info(f"[进行中] 更新向量数据库")
    
    # 准备向量化的内容
    texts = []
    metadata_list = []
    
    for page in state.pages:
        # 提取文本内容
        if isinstance(page["content"], str):
            # 分段处理长文本
            paragraphs = page["content"].split("\n\n")
            
            for i, para in enumerate(paragraphs):
                if para.strip():
                    texts.append(para)
                    metadata_list.append({
                        "document_id": os.path.basename(state.document_info.file_path),
                        "page": str(page["page_number"]),
                        "title": page["title"],
                        "paragraph": i,
                        "theme": state.dialog_info.theme,
                        "created_at": datetime.now().isoformat()
                    })
    
    # 向量化并存储内容
    try:
        # 在实际应用中，这里会调用向量数据库API
        # 例如将文档内容向量化并存储到Pinecone、Weaviate、Qdrant等
        
        # 模拟向量化过程
        results = []
        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            # 模拟向量嵌入过程
            vector_id = f"vec-{datetime.now().strftime('%Y%m%d%H%M%S')}-{i}"
            
            # 记录结果
            results.append({
                "id": vector_id,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "metadata": metadata
            })
        
        # 记录成功
        num_vectors = len(results)
        state.messages.append({
            "role": "system", 
            "content": f"已向量化并存储{num_vectors}个文本段落"
        })
        
    except Exception as e:
        # 记录错误
        logger.error(f"向量数据库更新出错: {str(e)}")
        state.errors.append(f"向量数据库更新出错: {str(e)}")
        state.messages.append({
            "role": "system", 
            "content": f"更新向量数据库时发生错误: {str(e)}"
        })
    
    # 更新处理状态
    state.current_step = "向量数据库更新完成"
    state.messages.append({
        "role": "system", 
        "content": "文档已索引到向量数据库，可供后续检索"
    })
    
    return state

# ======== 配置管理 ========

def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置"""
    if not config_path:
        # 默认配置文件路径
        config_paths = [
            "./config.yaml",
            "./config.yml",
            "./config.json",
            os.path.join(os.path.dirname(__file__), "config.yaml"),
            os.path.join(os.path.dirname(__file__), "config.json")
        ]
        
        # 查找第一个存在的配置文件
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if not config_path or not os.path.exists(config_path):
        logger.warning("未找到配置文件，使用默认配置")
        return {}
    
    try:
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            else:
                logger.error(f"不支持的配置文件格式: {config_path}")
                return {}
        
        logger.info(f"从 {config_path} 加载配置")
        return config
    
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return {}

def initialize_from_config(config_path: str = None):
    """从配置初始化系统"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建必要的目录
    output_dirs = [
        "./outputs",
        "./outputs/images",
        "./outputs/videos",
        "./outputs/charts"
    ]
    
    for directory in output_dirs:
        # 确保路径是string类型，并创建目录
        directory_path = os.path.normpath(directory)
        os.makedirs(directory_path, exist_ok=True)
    
    # 加载工具配置，只有当config_path不为None且存在时才加载
    if config_path and os.path.exists(config_path):
        tools.load_config_from_file(config_path)
    
    # 配置日志级别
    log_level = config.get("log_level", "INFO")
    logging.getLogger("ContentWorkflow").setLevel(getattr(logging, log_level))
    
    return config

# ======== 主程序 ========

def process_content_request(user_input: str, config_path: str = None) -> Dict[str, Any]:
    """处理内容请求，返回最终状态"""
    try:
        # 初始化配置
        config = initialize_from_config(config_path)
        
        # 构建工作流
        workflow = build_workflow().compile()
        
        # 创建初始状态
        initial_state = WorkflowState(user_input=user_input)
        
        # 执行工作流
        logger.info(f"\n===== 开始处理: '{user_input}' =====")
        result = workflow.invoke(initial_state)
        logger.info(f"===== 处理完成 =====\n")
        
        # 直接返回结果字典，不调用to_dict方法
        # 在较新版本的LangGraph中，result已经是字典形式
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        else:
            # 尝试将结果转换为字典
            try:
                return dict(result)
            except:
                # 如果无法转换，则创建一个新的WorkflowState并手动填充信息
                final_state = WorkflowState()
                
                # 尝试从结果中提取信息
                if 'dialog_info' in result:
                    final_state.dialog_info = result['dialog_info']
                if 'document_info' in result:
                    final_state.document_info = result['document_info']
                if 'messages' in result:
                    final_state.messages = result['messages']
                
                return asdict(final_state)
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        error_state = WorkflowState(user_input=user_input)
        error_state.messages.append({
            "role": "system",
            "content": f"处理请求时发生错误: {str(e)}"
        })
        error_state.errors.append(str(e))
        return asdict(error_state)

# ======== 命令行界面 ========

def main():
    """主函数，处理命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="内容生成工作流")
    parser.add_argument("--input", "-i", type=str, help="用户输入文本")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    parser.add_argument("--interactive", "-t", action="store_true", help="交互模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger("ContentWorkflow").setLevel(logging.DEBUG)
    
    if args.interactive:
        # 交互模式
        print("=== 内容生成工作流交互模式 ===")
        print("输入 'exit' 或 'quit' 退出")
        print("================================")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入内容描述: ").strip()
                
                # 检查是否退出
                if user_input.lower() in ['exit', 'quit']:
                    print("\n=== 退出交互模式 ===")
                    break
                
                # 处理请求
                result = process_content_request(user_input, args.config)
                
                # 打印结果摘要
                print("\n=== 处理结果 ===")
                
                # 安全地获取结果字段
                def get_nested_value(data, *keys, default="未指定"):
                    current = data
                    for key in keys:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            return default
                    return current
                
                theme = get_nested_value(result, 'dialog_info', 'theme')
                pages = get_nested_value(result, 'dialog_info', 'pages')
                doc_format = get_nested_value(result, 'document_info', 'format')
                file_path = get_nested_value(result, 'document_info', 'file_path')
                preview_url = get_nested_value(result, 'document_info', 'preview_url')
                
                print(f"主题: {theme}")
                print(f"页数: {pages}")
                print(f"文档格式: {doc_format}")
                print(f"文件路径: {file_path}")
                print(f"预览URL: {preview_url}")
                
                # 打印消息历史
                print("\n处理历史:")
                messages = get_nested_value(result, 'messages', default=[])
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            print(f"- {msg['role']}: {msg['content']}")
                
                print("\n" + "-"*40)
            
            except KeyboardInterrupt:
                print("\n\n=== 用户中断，退出交互模式 ===")
                break
            except Exception as e:
                print(f"\n发生错误: {str(e)}")
    
    elif args.input:
        # 单次处理模式
        result = process_content_request(args.input, args.config)
        
        # 打印结果
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        # 没有输入，运行演示
        test_inputs = [
            "我需要一份详细的财务报告，包括最近三个月的收入和支出分析",
            "帮我生成一个项目进度报告，要包含甘特图和资源使用情况",
            "我想要一份包含视频概述的销售策略文档"
        ]
        
        for input_text in test_inputs:
            # 处理请求
            result = process_content_request(input_text, args.config)
            
            # 打印最终消息
            print("最终状态:")
            
            # 安全地获取结果字段
            def get_nested_value(data, *keys, default="未指定"):
                current = data
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return default
                return current
            
            # 打印主要信息
            theme = get_nested_value(result, 'dialog_info', 'theme')
            pages = get_nested_value(result, 'dialog_info', 'pages')
            doc_format = get_nested_value(result, 'document_info', 'format')
            file_path = get_nested_value(result, 'document_info', 'file_path')
            preview_url = get_nested_value(result, 'document_info', 'preview_url')
            
            print(f"- 主题: {theme}")
            print(f"- 页数: {pages}")
            print(f"- 文档格式: {doc_format}")
            print(f"- 文件路径: {file_path}")
            print(f"- 预览URL: {preview_url}")
            
            # 打印消息历史
            print("\n对话历史:")
            messages = get_nested_value(result, 'messages', default=[])
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        print(f"- {msg['role']}: {msg['content']}")
            
            print("\n" + "-"*50 + "\n")

# ======== 创建配置文件示例 ========

def create_config_example(config_path: str = "./config.yaml"):
    """创建配置文件示例"""
    try:
        # 检查文件是否已存在
        if os.path.exists(config_path):
            logger.warning(f"配置文件 {config_path} 已存在，跳过创建")
            return
        
        config = {
            "log_level": "INFO",
            "tools": [
                {
                    "name": "llm_generate_text",
                    "type": "llm",
                    "enabled": True,
                    "description": "使用LLM生成文本",
                    "config": {
                        "api_key": "your-api-key-here",
                        "model": "gpt-3.5-turbo"
                    }
                },
                {
                    "name": "vector_db_query",
                    "type": "vector_db",
                    "enabled": True,
                    "description": "查询向量数据库",
                    "config": {
                        "api_key": "your-api-key-here",
                        "url": "https://your-vector-db-endpoint.com"
                    }
                },
                {
                    "name": "generate_image",
                    "type": "image_gen",
                    "enabled": True,
                    "description": "生成图像",
                    "config": {
                        "api_key": "your-api-key-here"
                    }
                }
            ]
        }
        
        # 确保目录存在
        config_dir = os.path.dirname(os.path.abspath(config_path))
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        # 保存配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                try:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    # 如果没有安装yaml库，退回到JSON格式
                    json_path = config_path.rsplit('.', 1)[0] + '.json'
                    logger.warning(f"yaml模块未安装，使用JSON格式保存到 {json_path}")
                    with open(json_path, 'w', encoding='utf-8') as json_f:
                        json.dump(config, json_f, indent=2, ensure_ascii=False)
            else:
                # 默认使用JSON
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"创建配置文件示例: {config_path}")
    
    except Exception as e:
        logger.error(f"创建配置文件示例失败: {str(e)}")
        logger.info("尝试在当前目录创建config.json")
        try:
            with open("config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info("创建config.json成功")
        except Exception as e2:
            logger.error(f"创建备用配置文件失败: {str(e2)}")

# ======== 路由器 ========

def router(state: WorkflowState) -> str:
    """根据当前状态决定下一步执行哪个节点"""
    current_step = state.current_step
    
    # 检查是否有错误
    if state.errors and len(state.errors) > 5:
        # 如果错误太多，提前结束工作流
        logger.warning(f"工作流发生太多错误 ({len(state.errors)}个)，提前结束")
        return "END"
    
    # 基于当前步骤路由到下一步
    step_to_next = {
        "初始化": "process_dialog",
        "对话处理完成": "create_content_structure",
        "内容结构创建完成": "generate_outline",
        "纲要生成完成": "create_document",
        "文档生成完成": "segment_pages",
        "页面拆分完成": "create_media_content",
        "媒体内容创建完成": "assemble_layout",
        "布局组装完成": "synthesize_file",
        "文件合成完成": "integrate_with_systems",
        "系统集成完成": "update_vector_db",
        "向量数据库更新完成": "END"
    }
    
    next_step = step_to_next.get(current_step, "END")
    logger.debug(f"路由: {current_step} -> {next_step}")
    return next_step

# ======== 构建工作流程图 ========

def build_workflow() -> StateGraph:
    """构建工作流图"""
    try:
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("process_dialog", process_dialog)
        workflow.add_node("create_content_structure", create_content_structure)
        workflow.add_node("generate_outline", generate_outline)
        workflow.add_node("create_document", create_document)
        workflow.add_node("segment_pages", segment_pages)
        workflow.add_node("create_media_content", create_media_content)
        workflow.add_node("assemble_layout", assemble_layout)
        workflow.add_node("synthesize_file", synthesize_file)
        workflow.add_node("integrate_with_systems", integrate_with_systems)
        workflow.add_node("update_vector_db", update_vector_db)
        
        # 设置边（路由）
        # 从每个节点设置条件边
        workflow.add_conditional_edges(
            "process_dialog",
            lambda x: router(x),
            {
                "create_content_structure": "create_content_structure",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "create_content_structure",
            lambda x: router(x),
            {
                "generate_outline": "generate_outline",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "generate_outline",
            lambda x: router(x),
            {
                "create_document": "create_document",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "create_document",
            lambda x: router(x),
            {
                "segment_pages": "segment_pages",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "segment_pages",
            lambda x: router(x),
            {
                "create_media_content": "create_media_content",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "create_media_content",
            lambda x: router(x),
            {
                "assemble_layout": "assemble_layout",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "assemble_layout",
            lambda x: router(x),
            {
                "synthesize_file": "synthesize_file",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "synthesize_file",
            lambda x: router(x),
            {
                "integrate_with_systems": "integrate_with_systems",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "integrate_with_systems",
            lambda x: router(x),
            {
                "update_vector_db": "update_vector_db",
                "END": END
            }
        )
        
        workflow.add_conditional_edges(
            "update_vector_db",
            lambda x: router(x),
            {
                "END": END
            }
        )
        
        # 设置入口点
        workflow.set_entry_point("process_dialog")
        
        return workflow
    except Exception as e:
        logger.error(f"构建工作流图失败: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 创建配置文件示例
        create_config_example()
        
        # 运行主程序
        main()
    except Exception as e:
        logger.error(f"程序启动失败: {str(e)}")
        print(f"\n发生致命错误: {str(e)}")
        print("请检查日志或联系开发者获取帮助") 