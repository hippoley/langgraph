"""
多源检索模块 - 支持多知识源融合和高级检索策略
"""

import os
import re
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiSourceRetriever")

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.vectorstores import VectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS
    from langchain.schema import BaseRetriever as LangchainBaseRetriever
    from api_config import get_llm
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    logger.error("请确保已安装所有必要的依赖")
    raise

# 知识源定义
@dataclass
class KnowledgeSource:
    """知识源定义"""
    name: str                                  # 知识源名称
    type: str                                  # 类型: vectorstore, api, database
    description: str                           # 描述
    retriever: Optional[BaseRetriever] = None  # 检索器
    weight: float = 1.0                        # 权重
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def __post_init__(self):
        if self.weight < 0 or self.weight > 1:
            raise ValueError("权重必须在0-1之间")

# 多源检索器
class MultiSourceRetriever(BaseRetriever):
    """多源检索器，支持从多个知识源检索并融合结果"""
    
    sources: List[KnowledgeSource]
    top_k: int = 5
    rerank_strategy: str = "weighted"  # weighted, semantic, hybrid
    llm_reranker: bool = False
    use_filter: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self, 
        sources: List[KnowledgeSource], 
        top_k: int = 5,
        rerank_strategy: str = "weighted",
        llm_reranker: bool = False,
        use_filter: bool = True
    ):
        """初始化多源检索器"""
        self.sources = sources
        self.top_k = top_k
        self.rerank_strategy = rerank_strategy
        self.llm_reranker = llm_reranker
        self.use_filter = use_filter
        super().__init__()
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """同步检索文档"""
        return asyncio.run(self.aretrieve_documents(query))
    
    async def aretrieve_documents(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """异步检索文档"""
        # 查询优化
        optimized_query = await self._optimize_query(query, context)
        logger.info(f"优化后的查询: {optimized_query}")
        
        # 从多个源并行检索
        results = await self._retrieve_from_all_sources(optimized_query)
        
        # 重排序
        reranked_results = await self._rerank_results(results, query, context)
        
        # 返回前top_k个结果
        return reranked_results[:self.top_k]
    
    async def _optimize_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """查询优化 - 结合上下文增强查询"""
        if not context or not self.use_filter:
            return query
        
        # 从上下文提取相关信息
        current_intent = context.get("current_intent", {}).get("name", "") if context.get("current_intent") else ""
        slots = context.get("slots", {})
        
        # 构建增强查询提示
        prompt_template = """
        原始查询: {query}
        
        用户当前意图: {intent}
        
        已知信息: {slots}
        
        请重写为更明确、更具体的查询，以便从知识库中获取最相关的信息。保持简洁，聚焦核心问题。
        
        优化查询:
        """
        
        try:
            llm = get_llm(temperature=0)
            slots_str = ", ".join([f"{k}: {v}" for k, v in slots.items()]) if slots else "无"
            
            prompt = prompt_template.format(
                query=query,
                intent=current_intent or "未知",
                slots=slots_str
            )
            
            response = await llm.ainvoke(prompt)
            optimized_query = response.content.strip()
            
            # 确保查询不为空
            if not optimized_query:
                return query
            
            return optimized_query
        except Exception as e:
            logger.warning(f"查询优化失败: {str(e)}")
            return query
    
    async def _retrieve_from_all_sources(self, query: str) -> List[Tuple[Document, KnowledgeSource]]:
        """从所有知识源并行检索文档"""
        tasks = []
        
        for source in self.sources:
            if source.retriever:
                task = self._retrieve_from_source(query, source)
                tasks.append(task)
        
        # 并行执行所有检索任务
        if not tasks:
            logger.warning("没有可用的知识源")
            return []
        
        all_results = await asyncio.gather(*tasks)
        
        # 合并结果
        merged_results = []
        for source_results, source in all_results:
            for doc in source_results:
                # 将源信息添加到文档的元数据中
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["source_name"] = source.name
                doc.metadata["source_type"] = source.type
                doc.metadata["source_weight"] = source.weight
                merged_results.append((doc, source))
        
        return merged_results
    
    async def _retrieve_from_source(
        self, query: str, source: KnowledgeSource
    ) -> Tuple[List[Document], KnowledgeSource]:
        """从单个知识源检索文档"""
        try:
            # 对于普通检索器，包装成异步调用
            if isinstance(source.retriever, LangchainBaseRetriever):
                # 创建异步任务
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(
                    None, lambda: source.retriever.get_relevant_documents(query)
                )
                return docs, source
            # 对于已经是异步检索器的情况
            elif hasattr(source.retriever, "aget_relevant_documents"):
                docs = await source.retriever.aget_relevant_documents(query)
                return docs, source
            else:
                logger.warning(f"不支持的检索器类型: {type(source.retriever)}")
                return [], source
        except Exception as e:
            logger.error(f"从源 {source.name} 检索时出错: {str(e)}")
            return [], source
    
    async def _rerank_results(
        self, 
        results: List[Tuple[Document, KnowledgeSource]], 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """重排序检索结果"""
        if not results:
            return []
        
        # 提取文档
        docs = [item[0] for item in results]
        
        if self.rerank_strategy == "weighted":
            # 加权重排序
            return await self._weighted_rerank(results, query)
        elif self.rerank_strategy == "semantic":
            # 语义相似度重排序
            return await self._semantic_rerank(docs, query)
        elif self.rerank_strategy == "hybrid":
            # 混合重排序
            return await self._hybrid_rerank(results, query)
        elif self.rerank_strategy == "llm" and self.llm_reranker:
            # LLM重排序
            return await self._llm_rerank(docs, query, context)
        else:
            # 默认简单合并
            return docs
    
    async def _weighted_rerank(
        self, results: List[Tuple[Document, KnowledgeSource]], query: str
    ) -> List[Document]:
        """基于权重的重排序"""
        # 根据源的权重和文档相关性评分计算综合分数
        weighted_results = []
        
        for doc, source in results:
            # 获取文档的相关性评分，如果没有则默认为1.0
            relevance_score = getattr(doc, "relevance_score", 1.0)
            
            # 计算综合分数
            combined_score = relevance_score * source.weight
            
            # 将分数添加到文档的元数据中
            doc.metadata["combined_score"] = combined_score
            weighted_results.append((doc, combined_score))
        
        # 按综合分数排序
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的文档
        return [item[0] for item in weighted_results]
    
    async def _semantic_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """基于语义相似度的重排序"""
        try:
            # 使用OpenAI Embeddings
            embeddings = OpenAIEmbeddings()
            
            # 获取查询的嵌入向量
            query_embedding = await embeddings.aembed_query(query)
            
            # 计算每个文档与查询的相似度
            results = []
            for doc in docs:
                # 获取文档的嵌入向量
                doc_embedding = await embeddings.aembed_documents([doc.page_content])
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, doc_embedding[0])
                
                # 将相似度添加到文档的元数据中
                doc.metadata["semantic_similarity"] = similarity
                results.append((doc, similarity))
            
            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 返回排序后的文档
            return [item[0] for item in results]
        except Exception as e:
            logger.error(f"语义重排序失败: {str(e)}")
            return docs
    
    async def _hybrid_rerank(
        self, results: List[Tuple[Document, KnowledgeSource]], query: str
    ) -> List[Document]:
        """混合重排序策略"""
        try:
            # 先提取文档
            docs = [item[0] for item in results]
            
            # 先做加权排序
            weighted_docs = await self._weighted_rerank(results, query)
            
            # 再做语义排序
            semantic_docs = await self._semantic_rerank(weighted_docs, query)
            
            return semantic_docs
        except Exception as e:
            logger.error(f"混合重排序失败: {str(e)}")
            return [item[0] for item in results]
    
    async def _llm_rerank(
        self, 
        docs: List[Document], 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """使用LLM进行重排序"""
        if not docs:
            return []
        
        # 如果文档过多，只取前10个进行LLM重排序
        if len(docs) > 10:
            docs = docs[:10]
        
        # 构建LLM重排序提示
        prompt_template = """
        查询: {query}
        
        以下是一些潜在的相关文档:
        
        {documents}
        
        根据与查询的相关性，对这些文档进行排名（1为最相关）。
        返回格式应为JSON数组，包含排序后的索引，例如：[2, 0, 1, 3]表示第3篇文档最相关，然后是第1篇、第2篇和第4篇。
        
        相关性排名 (JSON格式):
        """
        
        # 准备文档内容
        doc_texts = []
        for i, doc in enumerate(docs):
            doc_text = f"文档 {i+1}:\n{doc.page_content}\n"
            if doc.metadata:
                source = doc.metadata.get("source_name", "未知")
                doc_text += f"来源: {source}\n"
            doc_texts.append(doc_text)
        
        formatted_docs = "\n".join(doc_texts)
        
        try:
            llm = get_llm(temperature=0)
            
            prompt = prompt_template.format(
                query=query,
                documents=formatted_docs
            )
            
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            
            # 提取JSON数组
            match = re.search(r'\[.*?\]', content)
            if not match:
                logger.warning(f"无法从LLM响应中提取排名: {content}")
                return docs
            
            ranking_str = match.group(0)
            ranking = json.loads(ranking_str)
            
            # 验证排名
            if not all(isinstance(i, int) for i in ranking) or not all(0 <= i < len(docs) for i in ranking):
                logger.warning(f"LLM返回的排名无效: {ranking}")
                return docs
            
            # 根据排名重新排序文档
            reranked_docs = [docs[i] for i in ranking if i < len(docs)]
            
            # 如果重排序后的文档数量少于原始文档，添加剩余文档
            if len(reranked_docs) < len(docs):
                missing_indices = set(range(len(docs))) - set(ranking)
                for i in missing_indices:
                    if i < len(docs):
                        reranked_docs.append(docs[i])
            
            return reranked_docs
        except Exception as e:
            logger.error(f"LLM重排序失败: {str(e)}")
            return docs
    
    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

# 高级检索工厂
class AdvancedRetrieverFactory:
    """创建高级检索器的工厂"""
    
    @staticmethod
    def create_vectorstore_source(
        name: str, 
        documents: List[Document],
        description: str = "",
        embedding=None,
        weight: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> KnowledgeSource:
        """创建基于向量存储的知识源"""
        if embedding is None:
            embedding = OpenAIEmbeddings()
        
        # 创建向量存储
        vectorstore = FAISS.from_documents(documents, embedding)
        
        # 创建检索器
        retriever = vectorstore.as_retriever()
        
        return KnowledgeSource(
            name=name,
            type="vectorstore",
            description=description,
            retriever=retriever,
            weight=weight,
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_multi_source_retriever(
        sources: List[KnowledgeSource],
        top_k: int = 5,
        rerank_strategy: str = "weighted",
        llm_reranker: bool = False,
        use_filter: bool = True
    ) -> MultiSourceRetriever:
        """创建多源检索器"""
        return MultiSourceRetriever(
            sources=sources,
            top_k=top_k,
            rerank_strategy=rerank_strategy,
            llm_reranker=llm_reranker,
            use_filter=use_filter
        )

# 示例：创建样本知识库
def create_sample_knowledge_base() -> MultiSourceRetriever:
    """创建样例多源知识库"""
    # 创建文本拆分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    # 示例1：产品信息
    product_docs = []
    product_info = """
    我们公司提供多种智能家居产品：
    
    智能灯泡 - ¥99/个，支持16百万种颜色，可通过APP控制，支持语音助手。
    
    智能插座 - ¥79/个，可远程控制电源开关，监控用电量，设置定时开关。
    
    智能门锁 - ¥1299/套，支持指纹、密码、NFC卡片和APP多种解锁方式，内置安全警报系统。
    
    智能音箱 - ¥399/台，高品质音响，内置智能助手，可控制其他智能设备，支持多种音乐服务。
    
    智能摄像头 - ¥249/台，1080p高清画质，夜视功能，移动侦测，双向语音。
    
    所有产品均支持我们的智能家居平台，可相互连接形成完整的智能家居系统。目前所有产品提供2年质保服务。
    """
    product_chunks = text_splitter.split_text(product_info)
    for chunk in product_chunks:
        product_docs.append(Document(page_content=chunk, metadata={"source": "产品信息", "category": "产品"}))
    
    # 示例2：公司信息
    company_docs = []
    company_info = """
    智慧家科技有限公司成立于2018年，总部位于深圳市南山区科技园。
    
    我们是一家专注于智能家居解决方案的科技公司，致力于通过创新技术提升用户的居家体验。公司拥有一支由50名工程师组成的研发团队，专注于硬件设计、软件开发和人工智能算法。
    
    公司目前估值约2亿元，已完成A轮融资5000万。年营收约1.2亿元，同比增长35%。
    
    我们的使命是将复杂技术简单化，让每个家庭都能轻松享受科技带来的便利。
    
    主要客户包括房地产开发商、酒店集团和个人消费者。我们的产品已进入超过10万户家庭，用户满意度达98%。
    """
    company_chunks = text_splitter.split_text(company_info)
    for chunk in company_chunks:
        company_docs.append(Document(page_content=chunk, metadata={"source": "公司信息", "category": "公司"}))
    
    # 示例3：常见问题
    faq_docs = []
    faq_info = """
    Q: 如何重置我的智能设备？
    A: 大多数设备可以通过长按电源键5秒进行重置。具体请参考产品说明书或联系客服。
    
    Q: 智能设备需要连接什么WiFi频段？
    A: 我们的设备支持2.4GHz WiFi网络，部分新款产品同时支持5GHz频段。
    
    Q: 产品保修期是多久？
    A: 标准保修期为购买日起2年，可在官网延长保修服务。
    
    Q: APP支持哪些手机系统？
    A: 我们的APP支持iOS 10.0及以上和Android 6.0及以上的系统。
    
    Q: 如何添加多个家庭成员使用同一套智能家居系统？
    A: 在APP中进入"家庭管理"，选择"添加成员"，通过手机号或邮箱邀请其他用户加入。
    """
    faq_chunks = text_splitter.split_text(faq_info)
    for chunk in faq_chunks:
        faq_docs.append(Document(page_content=chunk, metadata={"source": "常见问题", "category": "FAQ"}))
    
    # 创建嵌入模型
    embeddings = OpenAIEmbeddings()
    
    # 创建多个知识源
    product_source = AdvancedRetrieverFactory.create_vectorstore_source(
        name="产品信息",
        documents=product_docs,
        description="包含公司所有产品的详细信息",
        embedding=embeddings,
        weight=0.9
    )
    
    company_source = AdvancedRetrieverFactory.create_vectorstore_source(
        name="公司信息",
        documents=company_docs,
        description="公司背景、历史和财务信息",
        embedding=embeddings,
        weight=0.7
    )
    
    faq_source = AdvancedRetrieverFactory.create_vectorstore_source(
        name="常见问题",
        documents=faq_docs,
        description="用户常见问题及解答",
        embedding=embeddings,
        weight=1.0
    )
    
    # 创建多源检索器
    multi_source_retriever = AdvancedRetrieverFactory.create_multi_source_retriever(
        sources=[product_source, company_source, faq_source],
        top_k=5,
        rerank_strategy="hybrid",
        llm_reranker=True
    )
    
    return multi_source_retriever

# 异步测试函数
async def test_retriever():
    """测试检索器"""
    retriever = create_sample_knowledge_base()
    
    # 测试查询
    query = "智能灯泡多少钱？"
    print(f"查询: {query}")
    
    # 测试检索
    docs = await retriever.aretrieve_documents(query)
    
    print(f"检索到 {len(docs)} 个文档:")
    for i, doc in enumerate(docs):
        print(f"文档 {i+1}:")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"来源: {doc.metadata.get('source_name', '未知')}")
        if "combined_score" in doc.metadata:
            print(f"分数: {doc.metadata['combined_score']:.4f}")
        if "semantic_similarity" in doc.metadata:
            print(f"语义相似度: {doc.metadata['semantic_similarity']:.4f}")
        print("-" * 40)

# 主函数
if __name__ == "__main__":
    asyncio.run(test_retriever()) 