"""
集成模块 - 将多源检索器和增强记忆管理器整合到对话系统中
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Integration")

try:
    from enhanced_memory import get_memory_manager, MemoryManager, MemoryItem
    from multi_source_retriever import (
        create_sample_knowledge_base, MultiSourceRetriever, 
        KnowledgeSource, AdvancedRetrieverFactory
    )
    from api_config import get_llm
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

# 全局检索器实例
GLOBAL_RETRIEVERS = {}

# 获取或创建检索器
def get_retriever(session_id: str) -> MultiSourceRetriever:
    """获取或创建会话的检索器"""
    if session_id not in GLOBAL_RETRIEVERS:
        # 创建检索器
        GLOBAL_RETRIEVERS[session_id] = create_sample_knowledge_base()
    return GLOBAL_RETRIEVERS[session_id]

# 集成检索和记忆的主函数
async def retrieve_with_memory(
    session_id: str,
    query: str,
    state: Dict[str, Any],
    top_k: int = 5,
    include_memories: bool = True,
    memory_weight: float = 0.3,
    knowledge_weight: float = 0.7
) -> Dict[str, Any]:
    """
    集成记忆和知识检索
    
    Args:
        session_id: 会话ID
        query: 用户查询
        state: 当前状态
        top_k: 返回的结果数量
        include_memories: 是否包含记忆
        memory_weight: 记忆权重
        knowledge_weight: 知识库权重
        
    Returns:
        包含检索结果的字典
    """
    # 获取检索器和记忆管理器
    retriever = get_retriever(session_id)
    memory_manager = get_memory_manager(session_id)
    
    # 异步并行检索知识和记忆
    knowledge_task = retriever.aretrieve_documents(query, state)
    memory_task = memory_manager.get_relevant_memories(query, top_k) if include_memories else asyncio.create_task(asyncio.sleep(0))
    
    # 等待两个任务完成
    knowledge_results, memory_results = await asyncio.gather(knowledge_task, memory_task)
    
    # 处理知识结果
    knowledge_items = []
    for i, doc in enumerate(knowledge_results):
        item = {
            "content": doc.page_content,
            "source": doc.metadata.get("source_name", "未知"),
            "source_type": "knowledge",
            "relevance": doc.metadata.get("combined_score", 1.0) * knowledge_weight,
            "metadata": doc.metadata
        }
        knowledge_items.append(item)
    
    # 处理记忆结果
    memory_items = []
    if include_memories and memory_results:
        for i, memory in enumerate(memory_results):
            item = {
                "content": memory.content,
                "source": "记忆",
                "source_type": "memory",
                "relevance": memory.importance * memory_weight,
                "metadata": memory.metadata,
                "memory_id": memory.id
            }
            memory_items.append(item)
    
    # 合并并排序结果
    combined_results = knowledge_items + memory_items
    combined_results.sort(key=lambda x: x["relevance"], reverse=True)
    
    # 截取top_k个结果
    final_results = combined_results[:top_k]
    
    # 构建结果字典
    result = {
        "retrieval_results": final_results,
        "knowledge_count": len(knowledge_items),
        "memory_count": len(memory_items),
        "query": query,
        "timestamp": datetime.now().isoformat()
    }
    
    return result

# 将检索结果格式化为上下文
def format_retrieval_results(results: Dict[str, Any], include_metadata: bool = False) -> str:
    """将检索结果格式化为上下文文本"""
    if not results or "retrieval_results" not in results:
        return ""
    
    items = results["retrieval_results"]
    if not items:
        return ""
    
    formatted = []
    for i, item in enumerate(items):
        content = item["content"]
        source = item["source"]
        
        formatted_item = f"[{i+1}] {content}"
        
        if include_metadata:
            source_type = item["source_type"]
            relevance = f"{item['relevance']:.2f}" if "relevance" in item else "未知"
            formatted_item += f"\n来源: {source} ({source_type}), 相关度: {relevance}"
        
        formatted.append(formatted_item)
    
    return "\n\n".join(formatted)

# 更新记忆
async def update_memory_with_conversation(
    session_id: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """更新记忆系统与当前对话"""
    # 获取记忆管理器
    memory_manager = get_memory_manager(session_id)
    
    # 获取消息历史
    messages = state.get("messages", [])
    if not messages:
        return state
    
    # 只处理最后一条消息
    last_message = messages[-1]
    if not last_message:
        return state
    
    # 确定消息类型
    is_user = last_message.get("role") == "human"
    
    # 添加到记忆
    memory_item = memory_manager.add_conversation_memory(last_message, is_user)
    
    # 定期进行记忆维护
    # 为避免频繁维护，可以根据消息数量或时间间隔来决定是否维护
    message_count = len(messages)
    if message_count % 5 == 0:  # 每5条消息维护一次
        await memory_manager.maintenance()
    
    # 更新状态
    if not state.get("memory_stats"):
        state["memory_stats"] = {}
    
    state["memory_stats"] = memory_manager.get_stats()
    
    return state

# 获取记忆上下文
async def get_memory_context(
    session_id: str,
    limit: int = 5,
    include_metadata: bool = False
) -> str:
    """获取记忆上下文"""
    # 获取记忆管理器
    memory_manager = get_memory_manager(session_id)
    
    # 获取最近记忆
    recent_memories = await memory_manager.get_recent_memories(limit)
    
    # 格式化记忆
    return memory_manager.format_for_context(recent_memories, include_metadata)

# 异步测试函数
async def test_integration():
    """测试集成功能"""
    session_id = "test-session-001"
    
    # 创建一个模拟状态
    state = {
        "session_id": session_id,
        "messages": [
            {"role": "human", "content": "你好，我想了解智能灯泡。"},
            {"role": "ai", "content": "您好！智能灯泡是可以通过手机APP或智能音箱控制的照明设备，支持调节亮度、颜色等功能。有什么具体问题吗？"},
            {"role": "human", "content": "智能灯泡多少钱？"}
        ],
        "current_intent": {"name": "query_product", "confidence": 0.9},
        "slots": {"product": "智能灯泡"}
    }
    
    # 更新记忆
    state = await update_memory_with_conversation(session_id, state)
    
    # 测试检索
    query = "智能灯泡价格"
    results = await retrieve_with_memory(session_id, query, state)
    
    print(f"查询: {query}")
    print(f"找到 {len(results['retrieval_results'])} 条结果")
    print(f"知识项: {results['knowledge_count']}, 记忆项: {results['memory_count']}")
    
    # 打印格式化结果
    formatted_results = format_retrieval_results(results, include_metadata=True)
    print("\n检索结果:")
    print(formatted_results)
    
    # 获取记忆上下文
    memory_context = await get_memory_context(session_id)
    print("\n记忆上下文:")
    print(memory_context)
    
    # 记忆统计
    print("\n记忆统计:")
    memory_manager = get_memory_manager(session_id)
    print(memory_manager.get_stats())

# 主函数
if __name__ == "__main__":
    asyncio.run(test_integration()) 