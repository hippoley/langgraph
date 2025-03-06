"""
增强的RAG模块 - 实现上下文感知检索和多源知识融合
"""

import os
import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedRAG")

# 尝试导入依赖
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
except ImportError:
    logger.warning("未安装必要的依赖，将使用模拟实现")

@dataclass
class KnowledgeSource:
    """知识源定义"""
    name: str                # 知识源名称
    description: str         # 知识源描述
    source_type: str         # 知识源类型（文件、数据库、API等）
    location: str            # 知识源位置
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type,
            "location": self.location,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeSource':
        """从字典创建知识源"""
        return cls(
            name=data["name"],
            description=data["description"],
            source_type=data["source_type"],
            location=data["location"],
            metadata=data.get("metadata", {})
        )

@dataclass
class RetrievalResult:
    """检索结果"""
    content: str             # 内容
    source: str              # 来源
    relevance: float         # 相关度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "source": self.source,
            "relevance": self.relevance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """从字典创建检索结果"""
        return cls(
            content=data["content"],
            source=data["source"],
            relevance=data["relevance"],
            metadata=data.get("metadata", {})
        )

class EnhancedRetriever:
    """增强检索器 - 实现上下文感知检索"""
    
    def __init__(self, 
                 embedding_model=None, 
                 llm=None,
                 cache_dir: str = "./knowledge_base"):
        self.embedding_model = embedding_model
        self.llm = llm or ChatOpenAI(temperature=0)
        self.cache_dir = cache_dir
        self.vector_stores = {}
        self.knowledge_sources = {}
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def add_knowledge_source(self, source: KnowledgeSource, documents: List[Dict[str, str]]) -> bool:
        """添加知识源"""
        try:
            # 注册知识源
            self.knowledge_sources[source.name] = source
            
            # 转换为Document对象
            docs = [
                Document(
                    page_content=doc["content"],
                    metadata={
                        "source": source.name,
                        **doc.get("metadata", {})
                    }
                )
                for doc in documents
            ]
            
            # 创建向量存储
            if 'FAISS' in globals():
                vector_store = FAISS.from_documents(
                    docs, 
                    self.embedding_model or OpenAIEmbeddings()
                )
                
                # 保存向量存储
                vector_path = os.path.join(self.cache_dir, f"{source.name}.faiss")
                vector_store.save_local(vector_path)
                
                # 注册向量存储
                self.vector_stores[source.name] = vector_store
                
                logger.info(f"已添加知识源: {source.name}，包含{len(docs)}个文档")
                return True
            else:
                # 模拟实现
                self.vector_stores[source.name] = {
                    "documents": docs,
                    "source": source.name
                }
                logger.info(f"已添加知识源(模拟): {source.name}，包含{len(docs)}个文档")
                return True
                
        except Exception as e:
            logger.error(f"添加知识源失败: {str(e)}")
            return False
    
    def load_knowledge_source(self, source_name: str) -> bool:
        """加载知识源"""
        try:
            vector_path = os.path.join(self.cache_dir, f"{source_name}.faiss")
            
            if os.path.exists(vector_path) and 'FAISS' in globals():
                # 加载向量存储
                vector_store = FAISS.load_local(
                    vector_path, 
                    self.embedding_model or OpenAIEmbeddings()
                )
                
                # 注册向量存储
                self.vector_stores[source_name] = vector_store
                
                logger.info(f"已加载知识源: {source_name}")
                return True
            else:
                logger.warning(f"知识源不存在或FAISS未安装: {source_name}")
                return False
                
        except Exception as e:
            logger.error(f"加载知识源失败: {str(e)}")
            return False
    
    def _enhance_query(self, query: str, context: List[Dict[str, str]] = None) -> str:
        """增强查询"""
        if not context:
            return query
        
        # 使用LLM增强查询
        system_prompt = """
        你是一个专业的查询增强助手。请根据用户的原始查询和对话上下文，生成一个增强的查询。
        增强的查询应该:
        1. 包含原始查询的核心意图
        2. 添加上下文中的关键信息
        3. 消除歧义
        4. 保持简洁（不超过100个字符）
        
        只返回增强后的查询文本，不要添加任何解释或其他内容。
        """
        
        # 构建上下文文本
        context_text = "\n".join([
            f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
            for msg in context[-3:]  # 只使用最近的3条消息
        ])
        
        user_prompt = f"""
        原始查询: {query}
        
        对话上下文:
        {context_text}
        
        请生成增强查询:
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            enhanced_query = response.content.strip()
            logger.info(f"查询增强: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"查询增强失败: {str(e)}")
            return query
    
    def _mock_retrieval(self, query: str, source_name: str, top_k: int) -> List[RetrievalResult]:
        """模拟检索（当依赖不可用时）"""
        if source_name not in self.vector_stores:
            return []
        
        docs = self.vector_stores[source_name]["documents"]
        
        # 简单的关键词匹配
        results = []
        for doc in docs:
            # 计算简单的相关度分数
            query_words = set(query.lower().split())
            content_words = set(doc.page_content.lower().split())
            common_words = query_words.intersection(content_words)
            
            if common_words:
                relevance = len(common_words) / len(query_words)
                
                results.append(RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance=relevance,
                    metadata=doc.metadata
                ))
        
        # 按相关度排序
        results.sort(key=lambda x: x.relevance, reverse=True)
        
        return results[:top_k]
    
    def retrieve(self, 
                query: str, 
                source_names: List[str] = None, 
                top_k: int = 3,
                context: List[Dict[str, str]] = None) -> List[RetrievalResult]:
        """检索文档"""
        # 增强查询
        enhanced_query = self._enhance_query(query, context)
        
        # 如果未指定知识源，使用所有知识源
        if not source_names:
            source_names = list(self.vector_stores.keys())
        
        all_results = []
        
        for source_name in source_names:
            if source_name not in self.vector_stores:
                logger.warning(f"知识源不存在: {source_name}")
                continue
            
            try:
                if 'FAISS' in globals():
                    # 使用FAISS检索
                    vector_store = self.vector_stores[source_name]
                    docs_with_scores = vector_store.similarity_search_with_score(
                        enhanced_query, 
                        k=top_k
                    )
                    
                    # 转换为检索结果
                    for doc, score in docs_with_scores:
                        # 转换分数为0-1范围的相关度
                        relevance = 1.0 / (1.0 + score)
                        
                        all_results.append(RetrievalResult(
                            content=doc.page_content,
                            source=doc.metadata.get("source", "unknown"),
                            relevance=relevance,
                            metadata=doc.metadata
                        ))
                else:
                    # 使用模拟检索
                    results = self._mock_retrieval(enhanced_query, source_name, top_k)
                    all_results.extend(results)
                    
            except Exception as e:
                logger.error(f"检索失败: {str(e)}")
        
        # 按相关度排序
        all_results.sort(key=lambda x: x.relevance, reverse=True)
        
        # 返回前K个结果
        return all_results[:top_k]
    
    def rerank_results(self, 
                      query: str, 
                      results: List[RetrievalResult],
                      context: List[Dict[str, str]] = None) -> List[RetrievalResult]:
        """重新排序检索结果"""
        if not results:
            return []
        
        # 使用LLM重新排序
        system_prompt = """
        你是一个专业的检索结果排序助手。请根据用户的查询和对话上下文，对检索结果进行重新排序。
        
        对于每个检索结果，请评估其与查询的相关性，返回0-10的分数，其中:
        - 10分: 完全匹配查询意图，包含所有必要信息
        - 7-9分: 高度相关，包含大部分必要信息
        - 4-6分: 中度相关，包含一些相关信息
        - 1-3分: 低度相关，仅包含少量相关信息
        - 0分: 完全不相关
        
        请以JSON格式返回评分结果，格式为:
        {
            "scores": [
                {"index": 0, "score": 8, "reason": "简短理由"},
                {"index": 1, "score": 5, "reason": "简短理由"},
                ...
            ]
        }
        """
        
        # 构建上下文文本
        context_text = ""
        if context:
            context_text = "\n".join([
                f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
                for msg in context[-3:]  # 只使用最近的3条消息
            ])
        
        # 构建检索结果文本
        results_text = "\n\n".join([
            f"[结果 {i}]\n{result.content}"
            for i, result in enumerate(results)
        ])
        
        user_prompt = f"""
        查询: {query}
        
        对话上下文:
        {context_text}
        
        检索结果:
        {results_text}
        
        请评估每个结果与查询的相关性:
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # 解析JSON响应
            import json
            result_data = json.loads(response.content)
            scores = result_data.get("scores", [])
            
            # 重新排序结果
            scored_results = []
            for score_item in scores:
                index = score_item.get("index")
                score = score_item.get("score")
                
                if index < len(results):
                    result = results[index]
                    # 更新相关度
                    result.relevance = score / 10.0
                    # 添加评分理由
                    result.metadata["rerank_reason"] = score_item.get("reason", "")
                    
                    scored_results.append(result)
            
            # 按新的相关度排序
            scored_results.sort(key=lambda x: x.relevance, reverse=True)
            
            return scored_results
            
        except Exception as e:
            logger.error(f"重新排序失败: {str(e)}")
            return results
    
    def generate_prompt(self, 
                       query: str, 
                       results: List[RetrievalResult],
                       context: List[Dict[str, str]] = None) -> str:
        """生成提示"""
        # 构建上下文文本
        context_text = ""
        if context:
            context_text = "\n".join([
                f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
                for msg in context[-3:]  # 只使用最近的3条消息
            ])
        
        # 构建检索结果文本
        results_text = "\n\n".join([
            f"[来源: {result.source}] (相关度: {result.relevance:.2f})\n{result.content}"
            for result in results
        ])
        
        # 构建完整提示
        prompt = f"""
        用户查询: {query}
        
        对话上下文:
        {context_text}
        
        相关信息:
        {results_text}
        
        请根据上述信息回答用户的查询。如果提供的信息不足以回答查询，请明确指出。
        回答应该:
        1. 准确反映检索到的信息
        2. 与用户查询直接相关
        3. 简洁明了
        4. 如果引用了特定来源的信息，请注明来源
        """
        
        return prompt
    
    def save_state(self, file_path: str) -> bool:
        """保存状态"""
        try:
            state = {
                "knowledge_sources": {
                    name: source.to_dict()
                    for name, source in self.knowledge_sources.items()
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存状态到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存状态失败: {str(e)}")
            return False
    
    @classmethod
    def load_state(cls, 
                  file_path: str, 
                  embedding_model=None, 
                  llm=None,
                  cache_dir: str = "./knowledge_base") -> 'EnhancedRetriever':
        """加载状态"""
        retriever = cls(
            embedding_model=embedding_model,
            llm=llm,
            cache_dir=cache_dir
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 恢复知识源
            for name, source_data in state.get("knowledge_sources", {}).items():
                source = KnowledgeSource.from_dict(source_data)
                retriever.knowledge_sources[name] = source
                
                # 尝试加载向量存储
                retriever.load_knowledge_source(name)
            
            logger.info(f"已从{file_path}加载状态")
            return retriever
            
        except Exception as e:
            logger.error(f"加载状态失败: {str(e)}")
            return retriever

# 创建示例知识库
def create_sample_knowledge_base(retriever: EnhancedRetriever) -> None:
    """创建示例知识库"""
    # 财务报表知识
    financial_docs = [
        {
            "content": "财务报表是反映企业财务状况、经营成果和现金流量的会计报表。主要包括资产负债表、利润表和现金流量表。",
            "metadata": {"type": "definition", "category": "finance"}
        },
        {
            "content": "资产负债表反映企业在某一特定日期的财务状况，包括资产、负债和所有者权益三大部分。",
            "metadata": {"type": "definition", "category": "finance"}
        },
        {
            "content": "利润表反映企业在一定会计期间的经营成果，包括收入、费用和利润。",
            "metadata": {"type": "definition", "category": "finance"}
        },
        {
            "content": "现金流量表反映企业在一定会计期间的现金和现金等价物的流入和流出情况。",
            "metadata": {"type": "definition", "category": "finance"}
        },
        {
            "content": "2023年第一季度财务报表显示，公司总收入为1200万元，比去年同期增长15%。净利润为300万元，增长10%。",
            "metadata": {"type": "report", "category": "finance", "period": "2023Q1"}
        },
        {
            "content": "2023年第二季度财务报表显示，公司总收入为1350万元，比去年同期增长12%。净利润为320万元，增长8%。",
            "metadata": {"type": "report", "category": "finance", "period": "2023Q2"}
        }
    ]
    
    # 项目报表知识
    project_docs = [
        {
            "content": "项目报表是反映项目进度、资源使用和预算执行情况的报告。通常包括项目概况、进度报告、资源报告和风险报告等部分。",
            "metadata": {"type": "definition", "category": "project"}
        },
        {
            "content": "项目进度报告显示项目各阶段的完成情况，包括已完成的任务、正在进行的任务和计划中的任务。",
            "metadata": {"type": "definition", "category": "project"}
        },
        {
            "content": "项目资源报告显示项目资源的分配和使用情况，包括人力资源、设备资源和材料资源等。",
            "metadata": {"type": "definition", "category": "project"}
        },
        {
            "content": "项目A的最新进度报告显示，项目已完成75%，预计将在下个月底完成。目前有8名团队成员全职参与该项目。",
            "metadata": {"type": "report", "category": "project", "project": "A"}
        },
        {
            "content": "项目B的资源报告显示，目前有5名开发人员和2名测试人员参与该项目。项目预算使用了60%，进度完成了65%。",
            "metadata": {"type": "report", "category": "project", "project": "B"}
        }
    ]
    
    # 销售报表知识
    sales_docs = [
        {
            "content": "销售报表是反映企业销售情况的报告，通常包括销售额、客户数量、产品销量等数据。",
            "metadata": {"type": "definition", "category": "sales"}
        },
        {
            "content": "销售额是指企业在一定时期内销售产品或提供服务所取得的收入总额。",
            "metadata": {"type": "definition", "category": "sales"}
        },
        {
            "content": "客户数量是指企业在一定时期内的客户总数，包括新客户和老客户。",
            "metadata": {"type": "definition", "category": "sales"}
        },
        {
            "content": "2023年7月销售报表显示，公司总销售额为450万元，比上月增长5%。新增客户20个，产品A销量最高，达到1000台。",
            "metadata": {"type": "report", "category": "sales", "period": "2023-07"}
        },
        {
            "content": "2023年8月销售报表显示，公司总销售额为480万元，比上月增长6.7%。新增客户25个，产品B销量增长最快，增长了30%。",
            "metadata": {"type": "report", "category": "sales", "period": "2023-08"}
        }
    ]
    
    # 人力资源报表知识
    hr_docs = [
        {
            "content": "人力资源报表是反映企业人力资源状况的报告，通常包括员工数量、招聘情况、离职情况和培训情况等数据。",
            "metadata": {"type": "definition", "category": "hr"}
        },
        {
            "content": "员工数量是指企业在职员工的总数，通常按部门、职位等进行分类统计。",
            "metadata": {"type": "definition", "category": "hr"}
        },
        {
            "content": "招聘情况是指企业在一定时期内的招聘活动情况，包括招聘人数、招聘渠道和招聘效果等。",
            "metadata": {"type": "definition", "category": "hr"}
        },
        {
            "content": "2023年第二季度人力资源报表显示，公司总员工数为500人，其中技术部门200人，销售部门150人，行政部门50人，其他部门100人。",
            "metadata": {"type": "report", "category": "hr", "period": "2023Q2"}
        },
        {
            "content": "2023年第二季度招聘情况：共招聘新员工30人，其中技术岗位15人，销售岗位10人，行政岗位5人。离职员工10人，离职率为2%。",
            "metadata": {"type": "report", "category": "hr", "period": "2023Q2"}
        }
    ]
    
    # 添加知识源
    retriever.add_knowledge_source(
        KnowledgeSource(
            name="financial_knowledge",
            description="财务报表相关知识",
            source_type="document",
            location="financial_docs"
        ),
        financial_docs
    )
    
    retriever.add_knowledge_source(
        KnowledgeSource(
            name="project_knowledge",
            description="项目报表相关知识",
            source_type="document",
            location="project_docs"
        ),
        project_docs
    )
    
    retriever.add_knowledge_source(
        KnowledgeSource(
            name="sales_knowledge",
            description="销售报表相关知识",
            source_type="document",
            location="sales_docs"
        ),
        sales_docs
    )
    
    retriever.add_knowledge_source(
        KnowledgeSource(
            name="hr_knowledge",
            description="人力资源报表相关知识",
            source_type="document",
            location="hr_docs"
        ),
        hr_docs
    )
    
    # 保存状态
    retriever.save_state("./knowledge_base/retriever_state.json") 