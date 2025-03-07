"""
增强记忆管理 - 支持长短期记忆分离和上下文压缩
"""

import os
import uuid
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable, TypedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque, defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedMemory")

try:
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS
    from api_config import get_llm
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

# 记忆类型定义
MEMORY_TYPES = {
    "SHORT_TERM": "short_term",
    "LONG_TERM": "long_term",
    "EPISODIC": "episodic",
    "SEMANTIC": "semantic",
    "PROCEDURAL": "procedural"
}

# 记忆项定义
@dataclass
class MemoryItem:
    """记忆项，表示单个记忆"""
    id: str                           # 唯一标识符
    content: str                      # 内容
    type: str                         # 记忆类型
    created_at: str                   # 创建时间
    last_accessed: str                # 最后访问时间
    importance: float = 0.5           # 重要性（0-1）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    access_count: int = 0             # 访问次数
    
    @classmethod
    def create(cls, content: str, memory_type: str, importance: float = 0.5, metadata: Dict[str, Any] = None):
        """创建新的记忆项"""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            type=memory_type,
            created_at=now,
            last_accessed=now,
            importance=importance,
            metadata=metadata or {},
            access_count=0
        )
    
    def access(self):
        """访问记忆项，更新访问时间和次数"""
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1
        return self

# 记忆块定义
@dataclass
class MemoryChunk:
    """记忆块，表示相关记忆的集合"""
    id: str                           # 唯一标识符
    items: List[MemoryItem]           # 记忆项列表
    summary: str                      # 摘要
    created_at: str                   # 创建时间
    last_accessed: str                # 最后访问时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    @classmethod
    def create(cls, items: List[MemoryItem], summary: str = "", metadata: Dict[str, Any] = None):
        """创建新的记忆块"""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            items=items,
            summary=summary or "记忆块",
            created_at=now,
            last_accessed=now,
            metadata=metadata or {}
        )
    
    def access(self):
        """访问记忆块，更新访问时间"""
        self.last_accessed = datetime.now().isoformat()
        # 同时更新所有记忆项
        for item in self.items:
            item.access()
        return self
    
    def add_item(self, item: MemoryItem):
        """添加记忆项"""
        self.items.append(item)
        self.last_accessed = datetime.now().isoformat()
        return self

# 全局记忆管理器存储
MEMORY_MANAGERS = {}

# 记忆管理器
class MemoryManager:
    """记忆管理器，管理短期和长期记忆"""
    
    def __init__(
        self,
        short_term_capacity: int = 10,
        long_term_capacity: int = 100,
        importance_threshold: float = 0.7,
        compress_threshold: int = 5
    ):
        """初始化记忆管理器"""
        # 短期记忆（最近的对话）
        self.short_term_memories: deque[MemoryItem] = deque(maxlen=short_term_capacity)
        
        # 长期记忆（重要的信息）
        self.long_term_memories: List[MemoryItem] = []
        
        # 记忆块（压缩后的记忆）
        self.memory_chunks: List[MemoryChunk] = []
        
        # 配置
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.importance_threshold = importance_threshold
        self.compress_threshold = compress_threshold
        
        # 向量存储（用于相似性搜索）
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        
        # 统计信息
        self.stats = {
            "short_term_added": 0,
            "long_term_added": 0,
            "compressed_chunks": 0,
            "items_removed": 0
        }
    
    def add_memory(self, content: str, memory_type: str = MEMORY_TYPES["SHORT_TERM"], 
                  importance: float = 0.5, metadata: Dict[str, Any] = None) -> MemoryItem:
        """添加新记忆"""
        # 创建记忆项
        memory_item = MemoryItem.create(
            content=content, 
            memory_type=memory_type, 
            importance=importance,
            metadata=metadata or {}
        )
        
        # 根据类型添加到相应的记忆存储
        if memory_type == MEMORY_TYPES["SHORT_TERM"]:
            self.short_term_memories.append(memory_item)
            self.stats["short_term_added"] += 1
        else:
            self.long_term_memories.append(memory_item)
            self.stats["long_term_added"] += 1
        
        return memory_item
    
    def add_conversation_memory(self, message: Dict[str, Any], is_user: bool = True) -> Optional[MemoryItem]:
        """添加对话记忆"""
        if not message or "content" not in message:
            return None
        
        content = message.get("content", "")
        if not content:
            return None
        
        # 确定记忆类型并添加元数据
        memory_type = MEMORY_TYPES["SHORT_TERM"]
        
        metadata = {
            "role": "user" if is_user else "assistant",
            "timestamp": message.get("timestamp", datetime.now().isoformat()),
            "message_id": message.get("id", str(uuid.uuid4())),
            "is_conversation": True
        }
        
        # 对于用户消息，可能需要评估重要性
        initial_importance = 0.5  # 默认重要性
        
        # 添加记忆
        memory_item = self.add_memory(
            content=content,
            memory_type=memory_type,
            importance=initial_importance,
            metadata=metadata
        )
        
        return memory_item
    
    async def get_recent_memories(self, limit: int = 5) -> List[MemoryItem]:
        """获取最近的记忆"""
        # 从短期记忆中获取最近的记忆
        recent_memories = list(self.short_term_memories)
        
        # 按时间排序（最新的在前）
        recent_memories.sort(key=lambda x: x.last_accessed, reverse=True)
        
        # 访问这些记忆
        for memory in recent_memories[:limit]:
            memory.access()
        
        return recent_memories[:limit]
    
    async def get_important_memories(self, threshold: float = None, limit: int = 5) -> List[MemoryItem]:
        """获取重要的记忆"""
        if threshold is None:
            threshold = self.importance_threshold
        
        # 从长期记忆中筛选重要的记忆
        important_memories = [memory for memory in self.long_term_memories if memory.importance >= threshold]
        
        # 按重要性排序
        important_memories.sort(key=lambda x: x.importance, reverse=True)
        
        # 访问这些记忆
        for memory in important_memories[:limit]:
            memory.access()
        
        return important_memories[:limit]
    
    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """获取与查询相关的记忆"""
        # 如果向量存储为空，则初始化
        await self._ensure_vector_store()
        
        # 如果仍然为空，返回最近的记忆
        if not self.vector_store:
            return await self.get_recent_memories(limit)
        
        try:
            # 搜索相关记忆
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=limit)
            
            # 解析结果
            relevant_memories = []
            for doc, score in docs_and_scores:
                memory_id = doc.metadata.get("memory_id")
                
                # 尝试在长期和短期记忆中查找
                memory = self._find_memory_by_id(memory_id)
                
                if memory:
                    # 访问这个记忆
                    memory.access()
                    relevant_memories.append(memory)
            
            return relevant_memories
        except Exception as e:
            logger.error(f"获取相关记忆时出错: {str(e)}")
            return await self.get_recent_memories(limit)
    
    async def consolidate_memories(self, memories: List[MemoryItem]) -> MemoryChunk:
        """将多个记忆项合并为一个记忆块"""
        if not memories:
            raise ValueError("没有提供记忆项进行合并")
        
        # 提取记忆内容
        memory_contents = []
        for memory in memories:
            role = memory.metadata.get("role", "unknown")
            content = memory.content
            memory_contents.append(f"{role}: {content}")
        
        combined_content = "\n".join(memory_contents)
        
        # 生成摘要
        summary = await self._generate_summary(combined_content)
        
        # 创建记忆块
        memory_chunk = MemoryChunk.create(
            items=memories,
            summary=summary,
            metadata={
                "source_ids": [memory.id for memory in memories],
                "content_length": len(combined_content)
            }
        )
        
        # 添加到记忆块列表
        self.memory_chunks.append(memory_chunk)
        self.stats["compressed_chunks"] += 1
        
        return memory_chunk
    
    async def _generate_summary(self, text: str) -> str:
        """生成文本摘要"""
        try:
            llm = get_llm(temperature=0)
            
            prompt = f"""
            请为以下对话内容生成一个简短的摘要（不超过50个字），捕捉关键信息:
            
            {text}
            
            摘要:
            """
            
            response = await llm.ainvoke(prompt)
            summary = response.content.strip()
            
            return summary
        except Exception as e:
            logger.error(f"生成摘要时出错: {str(e)}")
            # 创建一个基础摘要
            words = text.split()
            if len(words) > 10:
                return " ".join(words[:10]) + "..."
            return text
    
    async def _assess_importance(self, memory_item: MemoryItem) -> float:
        """评估记忆的重要性"""
        try:
            llm = get_llm(temperature=0)
            
            prompt = f"""
            请评估以下内容的重要性，返回0到1之间的数值（1表示非常重要，0表示不重要）:
            
            内容: {memory_item.content}
            
            创建时间: {memory_item.created_at}
            类型: {memory_item.type}
            访问次数: {memory_item.access_count}
            
            请只返回一个0到1之间的数值作为重要性评分，不要添加其他文字。
            """
            
            response = await llm.ainvoke(prompt)
            importance_str = response.content.strip()
            
            # 提取数值
            try:
                importance = float(importance_str)
                # 确保在0-1范围内
                importance = max(0, min(1, importance))
                return importance
            except ValueError:
                logger.warning(f"无法从LLM响应中提取重要性: {importance_str}")
                return memory_item.importance
        except Exception as e:
            logger.error(f"评估重要性时出错: {str(e)}")
            return memory_item.importance
    
    async def maintenance(self):
        """执行记忆维护任务"""
        try:
            # 1. 压缩短期记忆
            if len(self.short_term_memories) >= self.compress_threshold:
                await self._compress_short_term_memories()
            
            # 2. 将重要的短期记忆转移到长期记忆
            await self._transfer_important_memories()
            
            # 3. 如果长期记忆超过容量，压缩低重要性的长期记忆
            if len(self.long_term_memories) > self.long_term_capacity:
                await self._compress_long_term_memories()
            
            # 4. 重建向量存储
            await self._rebuild_vector_store()
        except Exception as e:
            logger.error(f"记忆维护时出错: {str(e)}")
    
    async def _compress_short_term_memories(self):
        """压缩短期记忆"""
        if len(self.short_term_memories) < self.compress_threshold:
            return
        
        # 按对话分组，寻找需要压缩的记忆组
        conversation_groups = defaultdict(list)
        
        for memory in self.short_term_memories:
            if memory.metadata.get("is_conversation"):
                # 使用会话ID（如果有）或时间戳的小时部分作为分组键
                group_key = memory.metadata.get("conversation_id", memory.created_at.split("T")[0])
                conversation_groups[group_key].append(memory)
        
        # 对大于压缩阈值的组进行压缩
        for group_key, memories in conversation_groups.items():
            if len(memories) >= self.compress_threshold:
                # 按时间排序
                memories.sort(key=lambda x: x.created_at)
                
                # 创建记忆块
                memory_chunk = await self.consolidate_memories(memories)
                
                # 从短期记忆中移除被压缩的记忆
                for memory in memories:
                    if memory in self.short_term_memories:
                        self.short_term_memories.remove(memory)
                        self.stats["items_removed"] += 1
                
                # 添加一个指向记忆块的新记忆
                self.add_memory(
                    content=memory_chunk.summary,
                    memory_type=MEMORY_TYPES["SHORT_TERM"],
                    importance=max([memory.importance for memory in memories]),
                    metadata={
                        "is_chunk_reference": True,
                        "chunk_id": memory_chunk.id,
                        "original_count": len(memories)
                    }
                )
    
    async def _transfer_important_memories(self):
        """将重要的短期记忆转移到长期记忆"""
        # 评估所有短期记忆的重要性
        for memory in list(self.short_term_memories):
            # 跳过已被评估过的记忆
            if memory.metadata.get("importance_assessed"):
                continue
            
            # 评估重要性
            importance = await self._assess_importance(memory)
            memory.importance = importance
            memory.metadata["importance_assessed"] = True
            
            # 如果超过阈值，转移到长期记忆
            if importance >= self.importance_threshold:
                # 创建一个新的记忆项添加到长期记忆（保留原始ID）
                long_term_memory = MemoryItem(
                    id=memory.id,
                    content=memory.content,
                    type=MEMORY_TYPES["LONG_TERM"],
                    created_at=memory.created_at,
                    last_accessed=memory.last_accessed,
                    importance=importance,
                    metadata={**memory.metadata, "source": "short_term"},
                    access_count=memory.access_count
                )
                
                self.long_term_memories.append(long_term_memory)
                self.stats["long_term_added"] += 1
    
    async def _compress_long_term_memories(self):
        """压缩长期记忆"""
        if len(self.long_term_memories) <= self.long_term_capacity:
            return
        
        # 按重要性排序（低重要性在前）
        self.long_term_memories.sort(key=lambda x: x.importance)
        
        # 需要移除的数量
        to_remove = len(self.long_term_memories) - self.long_term_capacity
        
        # 找出低重要性的记忆
        low_importance_memories = self.long_term_memories[:to_remove]
        
        # 将它们分成小组用于压缩
        chunk_size = min(len(low_importance_memories), self.compress_threshold)
        for i in range(0, len(low_importance_memories), chunk_size):
            chunk = low_importance_memories[i:i+chunk_size]
            if len(chunk) > 1:  # 只有当有多个记忆时才压缩
                memory_chunk = await self.consolidate_memories(chunk)
                
                # 从长期记忆中移除被压缩的记忆
                for memory in chunk:
                    if memory in self.long_term_memories:
                        self.long_term_memories.remove(memory)
                        self.stats["items_removed"] += 1
    
    async def _ensure_vector_store(self):
        """确保向量存储已初始化"""
        if self.vector_store is None:
            await self._rebuild_vector_store()
    
    async def _rebuild_vector_store(self):
        """重建向量存储"""
        # 收集所有记忆项
        all_memories = list(self.short_term_memories) + self.long_term_memories
        
        if not all_memories:
            self.vector_store = None
            return
        
        # 创建文档
        documents = []
        for memory in all_memories:
            doc = Document(
                page_content=memory.content,
                metadata={
                    "memory_id": memory.id,
                    "type": memory.type,
                    "importance": memory.importance,
                    "created_at": memory.created_at
                }
            )
            documents.append(doc)
        
        # 创建向量存储
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
    
    def _find_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """通过ID查找记忆"""
        if not memory_id:
            return None
        
        # 在短期记忆中查找
        for memory in self.short_term_memories:
            if memory.id == memory_id:
                return memory
        
        # 在长期记忆中查找
        for memory in self.long_term_memories:
            if memory.id == memory_id:
                return memory
        
        return None
    
    def format_for_context(self, memories: List[MemoryItem], include_metadata: bool = False) -> str:
        """将记忆格式化为上下文文本"""
        if not memories:
            return ""
        
        formatted_memories = []
        for memory in memories:
            formatted = memory.content
            if include_metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in memory.metadata.items() if k != "is_conversation"])
                formatted += f" [类型: {memory.type}, 重要性: {memory.importance:.2f}, {metadata_str}]"
            formatted_memories.append(formatted)
        
        return "\n\n".join(formatted_memories)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            **self.stats,
            "short_term_count": len(self.short_term_memories),
            "long_term_count": len(self.long_term_memories),
            "memory_chunks_count": len(self.memory_chunks)
        }
        return stats
    
    def clear(self):
        """清除所有记忆"""
        self.short_term_memories.clear()
        self.long_term_memories.clear()
        self.memory_chunks.clear()
        self.vector_store = None
        
        # 重置统计信息
        self.stats = {
            "short_term_added": 0,
            "long_term_added": 0,
            "compressed_chunks": 0,
            "items_removed": 0
        }

# 获取记忆管理器
def get_memory_manager(session_id: str) -> MemoryManager:
    """获取或创建记忆管理器"""
    if session_id not in MEMORY_MANAGERS:
        MEMORY_MANAGERS[session_id] = MemoryManager()
    return MEMORY_MANAGERS[session_id]

# 异步测试函数
async def test_memory_manager():
    """测试记忆管理器"""
    memory_manager = MemoryManager()
    
    # 添加一些测试记忆
    memory_manager.add_memory("用户询问智能灯泡的价格", importance=0.8)
    memory_manager.add_memory("助手回答智能灯泡售价99元", importance=0.7)
    memory_manager.add_memory("用户询问智能灯泡的功能", importance=0.6)
    memory_manager.add_memory("助手解释智能灯泡支持1600万种颜色和APP控制", importance=0.9)
    memory_manager.add_memory("用户表示对智能灯泡很满意", importance=0.5)
    
    # 测试获取最近记忆
    recent = await memory_manager.get_recent_memories(3)
    print("最近记忆:")
    for memory in recent:
        print(f"- {memory.content} (重要性: {memory.importance})")
    
    # 测试获取重要记忆
    important = await memory_manager.get_important_memories(threshold=0.7, limit=2)
    print("\n重要记忆:")
    for memory in important:
        print(f"- {memory.content} (重要性: {memory.importance})")
    
    # 测试记忆维护
    await memory_manager.maintenance()
    print("\n维护后统计信息:")
    print(json.dumps(memory_manager.get_stats(), indent=2))
    
    # 测试获取相关记忆
    query = "智能灯泡的价格是多少？"
    relevant = await memory_manager.get_relevant_memories(query, limit=2)
    print(f"\n与查询 '{query}' 相关的记忆:")
    for memory in relevant:
        print(f"- {memory.content}")
    
    # 测试记忆合并
    chunk = await memory_manager.consolidate_memories(list(memory_manager.short_term_memories)[:3])
    print(f"\n记忆块摘要: {chunk.summary}")

# 主函数
if __name__ == "__main__":
    asyncio.run(test_memory_manager()) 