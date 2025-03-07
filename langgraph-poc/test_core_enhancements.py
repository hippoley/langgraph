"""
测试核心能力增强
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestCoreEnhancements")

try:
    from multi_source_retriever import create_sample_knowledge_base, MultiSourceRetriever
    from enhanced_memory import get_memory_manager, MemoryManager
    from integration import retrieve_with_memory, update_memory_with_conversation, get_memory_context
    from enhanced_agent import should_pause, handle_breakpoint, process_human_input
    from api_config import get_llm
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

# 测试多源检索
async def test_multi_source_retrieval():
    """测试多源检索功能"""
    print("\n=== 测试多源检索 ===")
    
    # 创建样本检索器
    retriever = create_sample_knowledge_base()
    
    # 测试查询
    queries = [
        "智能灯泡多少钱？",
        "公司成立于哪一年？",
        "如何重置智能设备？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        
        # 执行检索
        docs = await retriever.aretrieve_documents(query)
        
        # 显示结果
        print(f"找到 {len(docs)} 个结果:")
        for i, doc in enumerate(docs[:2]):  # 只显示前2个结果
            print(f"[{i+1}] 内容: {doc.page_content[:100]}...")
            print(f"    来源: {doc.metadata.get('source_name', '未知')}")
            if "combined_score" in doc.metadata:
                print(f"    分数: {doc.metadata['combined_score']:.4f}")

# 测试记忆管理
async def test_memory_management():
    """测试记忆管理功能"""
    print("\n=== 测试记忆管理 ===")
    
    # 创建记忆管理器
    session_id = "test-session-memory"
    memory_manager = get_memory_manager(session_id)
    
    # 清除之前的记忆
    memory_manager.clear()
    
    # 添加一些测试记忆
    print("\n添加测试记忆...")
    memory_manager.add_memory("用户询问智能灯泡的价格", importance=0.8)
    memory_manager.add_memory("助手回答智能灯泡售价99元", importance=0.7)
    memory_manager.add_memory("用户询问智能灯泡的功能", importance=0.6)
    memory_manager.add_memory("助手解释智能灯泡支持1600万种颜色和APP控制", importance=0.9)
    memory_manager.add_memory("用户表示对智能灯泡很满意", importance=0.5)
    
    # 测试获取最近记忆
    recent = await memory_manager.get_recent_memories(3)
    print("\n最近记忆:")
    for memory in recent:
        print(f"- {memory.content} (重要性: {memory.importance})")
    
    # 测试记忆维护
    print("\n执行记忆维护...")
    await memory_manager.maintenance()
    
    # 获取记忆统计
    stats = memory_manager.get_stats()
    print("\n记忆统计:")
    print(json.dumps(stats, indent=2))
    
    # 测试获取相关记忆
    query = "智能灯泡的价格是多少？"
    relevant = await memory_manager.get_relevant_memories(query, limit=2)
    print(f"\n与查询 '{query}' 相关的记忆:")
    for memory in relevant:
        print(f"- {memory.content}")

# 测试集成功能
async def test_integration():
    """测试集成功能"""
    print("\n=== 测试集成功能 ===")
    
    # 创建会话ID
    session_id = "test-session-integration"
    
    # 创建一个模拟状态
    state = {
        "session_id": session_id,
        "messages": [
            {"role": "human", "content": "我想了解智能灯泡"},
            {"role": "ai", "content": "智能灯泡是一种可以通过手机APP或语音控制的照明设备，支持调节亮度、颜色等功能。有什么具体问题吗？"},
            {"role": "human", "content": "它多少钱？"}
        ],
        "current_intent": {"name": "query_product", "confidence": 0.9},
        "slots": {"product": "智能灯泡"}
    }
    
    # 更新记忆
    print("\n更新记忆...")
    state = await update_memory_with_conversation(session_id, state)
    
    # 执行检索
    query = "智能灯泡价格"
    print(f"\n执行检索，查询: '{query}'")
    results = await retrieve_with_memory(session_id, query, state)
    
    # 显示结果
    print(f"找到 {len(results['retrieval_results'])} 条结果")
    print(f"知识项: {results['knowledge_count']}, 记忆项: {results['memory_count']}")
    
    # 获取记忆上下文
    memory_context = await get_memory_context(session_id)
    print("\n记忆上下文:")
    print(memory_context)

# 测试断点功能
async def test_breakpoints():
    """测试断点功能"""
    print("\n=== 测试断点功能 ===")
    
    # 创建一个模拟状态
    state = {
        "session_id": "test-session-breakpoint",
        "messages": [
            {"role": "human", "content": "生成销售报表"},
            {"role": "ai", "content": "好的，我将为您生成销售报表。请问您需要哪个时间段的报表？"},
            {"role": "human", "content": "上个季度"}
        ],
        "current_intent": {"name": "query_sales_report", "confidence": 0.9},
        "slots": {"time_period": "上个季度"},
        "status": "knowledge_retrieved",
        "retrieval_results": [],
        "confidence": 0.85,
        "breakpoints": {
            "knowledge_retrieved": True  # 在知识检索后设置断点
        },
        "error": None,
        "next_step": "report_handler",
        "human_input_required": False,
        "human_input": None
    }
    
    # 检查是否应该暂停
    should_stop = should_pause(state)
    print(f"是否应该暂停: {should_stop}")
    
    if should_stop:
        # 处理断点
        print("\n处理断点...")
        paused_state = handle_breakpoint(state)
        print(f"状态已更新: {paused_state['status']}")
        print(f"需要人工干预: {paused_state['human_input_required']}")
        
        # 模拟人工输入
        print("\n模拟人工输入...")
        paused_state["human_input"] = "/continue"
        processed_state = process_human_input(paused_state)
        print(f"处理后状态: {processed_state['status']}")
        print(f"下一步: {processed_state.get('next_step', '未知')}")

# 主函数
async def main():
    """主测试函数"""
    try:
        # 测试多源检索
        await test_multi_source_retrieval()
        
        # 测试记忆管理
        await test_memory_management()
        
        # 测试集成功能
        await test_integration()
        
        # 测试断点功能
        await test_breakpoints()
        
        print("\n所有测试完成!")
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 