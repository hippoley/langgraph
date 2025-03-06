"""
业务指标向量化模块 - 实现业务指标的向量化和动态路由
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BusinessMetrics")

# 尝试导入向量化依赖
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("未安装sentence_transformers，将使用模拟向量化")
    SentenceTransformer = None

@dataclass
class BusinessMetric:
    """业务指标定义"""
    name: str                # 指标名称
    description: str         # 指标描述
    category: str            # 指标类别
    keywords: List[str] = field(default_factory=list)  # 关键词
    examples: List[str] = field(default_factory=list)  # 示例查询
    data_source: Optional[str] = None  # 数据源
    vector: Optional[List[float]] = None  # 向量表示
    related_intents: List[str] = field(default_factory=list)  # 相关意图
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "keywords": self.keywords,
            "examples": self.examples,
            "data_source": self.data_source,
            "vector": self.vector,
            "related_intents": self.related_intents
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessMetric':
        """从字典创建业务指标"""
        return cls(
            name=data["name"],
            description=data["description"],
            category=data["category"],
            keywords=data.get("keywords", []),
            examples=data.get("examples", []),
            data_source=data.get("data_source"),
            vector=data.get("vector"),
            related_intents=data.get("related_intents", [])
        )

class BusinessMetricVectorizer:
    """业务指标向量化器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        # 尝试加载模型
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"已加载向量化模型: {model_name}")
            except Exception as e:
                logger.error(f"加载向量化模型失败: {str(e)}")
    
    def vectorize(self, text: str) -> List[float]:
        """将文本向量化"""
        if self.model:
            try:
                vector = self.model.encode(text)
                return vector.tolist()
            except Exception as e:
                logger.error(f"向量化失败: {str(e)}")
        
        # 如果模型不可用或向量化失败，返回模拟向量
        return self._mock_vectorize(text)
    
    def _mock_vectorize(self, text: str) -> List[float]:
        """模拟向量化（仅用于测试）"""
        import hashlib
        import random
        
        # 使用文本的哈希值作为随机种子
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        random.seed(hash_value)
        
        # 生成384维的模拟向量
        vector = [random.uniform(-1, 1) for _ in range(384)]
        
        # 归一化
        norm = sum(v**2 for v in vector) ** 0.5
        vector = [v / norm for v in vector]
        
        return vector
    
    def vectorize_metric(self, metric: BusinessMetric) -> BusinessMetric:
        """向量化业务指标"""
        # 构建完整描述文本
        text = f"{metric.name}: {metric.description}"
        if metric.keywords:
            text += f" Keywords: {', '.join(metric.keywords)}"
        if metric.examples:
            text += f" Examples: {'; '.join(metric.examples)}"
        
        # 向量化
        vector = self.vectorize(text)
        metric.vector = vector
        
        return metric

class BusinessMetricRegistry:
    """业务指标注册表"""
    
    def __init__(self, vectorizer: Optional[BusinessMetricVectorizer] = None):
        self.metrics: Dict[str, BusinessMetric] = {}
        self.vectorizer = vectorizer or BusinessMetricVectorizer()
        self.vectors: List[List[float]] = []
        self.metric_names: List[str] = []
    
    def register_metric(self, metric: BusinessMetric) -> None:
        """注册业务指标"""
        # 向量化
        if not metric.vector:
            metric = self.vectorizer.vectorize_metric(metric)
        
        # 注册
        self.metrics[metric.name] = metric
        self.vectors.append(metric.vector)
        self.metric_names.append(metric.name)
        
        logger.info(f"已注册业务指标: {metric.name}")
    
    def get_metric(self, name: str) -> Optional[BusinessMetric]:
        """获取业务指标"""
        return self.metrics.get(name)
    
    def find_similar_metrics(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """查找与查询最相似的业务指标"""
        if not self.metrics:
            return []
        
        # 向量化查询
        query_vector = self.vectorizer.vectorize(query)
        
        # 计算相似度
        similarities = []
        for i, metric_vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, metric_vector)
            similarities.append((self.metric_names[i], similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前K个
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        try:
            # 使用numpy计算
            a = np.array(vec1)
            b = np.array(vec2)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except:
            # 手动计算
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a ** 2 for a in vec1) ** 0.5
            norm_b = sum(b ** 2 for b in vec2) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
    
    def route_query_to_metric(self, query: str, threshold: float = 0.6) -> Optional[str]:
        """将查询路由到最相似的业务指标"""
        similar_metrics = self.find_similar_metrics(query, top_k=1)
        
        if not similar_metrics:
            return None
        
        metric_name, similarity = similar_metrics[0]
        
        if similarity >= threshold:
            return metric_name
        
        return None
    
    def save_to_file(self, file_path: str) -> None:
        """保存到文件"""
        data = {
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()}
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存业务指标到: {file_path}")
        except Exception as e:
            logger.error(f"保存业务指标失败: {str(e)}")
    
    @classmethod
    def load_from_file(cls, file_path: str, vectorizer: Optional[BusinessMetricVectorizer] = None) -> 'BusinessMetricRegistry':
        """从文件加载"""
        registry = cls(vectorizer=vectorizer)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, metric_data in data.get("metrics", {}).items():
                metric = BusinessMetric.from_dict(metric_data)
                registry.register_metric(metric)
            
            logger.info(f"已从{file_path}加载{len(registry.metrics)}个业务指标")
        except Exception as e:
            logger.error(f"加载业务指标失败: {str(e)}")
        
        return registry

# 预定义的业务指标
def create_default_metrics() -> BusinessMetricRegistry:
    """创建默认的业务指标"""
    registry = BusinessMetricRegistry()
    
    # 财务报表指标
    financial_report = BusinessMetric(
        name="financial_report",
        description="财务报表，包含公司的收入、支出、利润等财务数据",
        category="finance",
        keywords=["财务", "报表", "收入", "支出", "利润", "财报", "季度报告", "年度报告"],
        examples=[
            "查看上个季度的财务报表",
            "我想了解公司的财务状况",
            "显示2023年第一季度的财报"
        ],
        data_source="finance_db",
        related_intents=["query_financial_report"]
    )
    
    # 项目报表指标
    project_report = BusinessMetric(
        name="project_report",
        description="项目报表，包含项目进度、资源使用、预算执行等信息",
        category="project",
        keywords=["项目", "报表", "进度", "资源", "预算", "里程碑", "项目状态"],
        examples=[
            "查看A项目的进度报表",
            "显示所有项目的资源使用情况",
            "我想了解B项目的预算执行情况"
        ],
        data_source="project_db",
        related_intents=["query_project_report"]
    )
    
    # 销售报表指标
    sales_report = BusinessMetric(
        name="sales_report",
        description="销售报表，包含销售额、客户数量、产品销量等销售数据",
        category="sales",
        keywords=["销售", "报表", "销售额", "客户", "产品", "销量", "业绩"],
        examples=[
            "查看本月的销售报表",
            "显示各区域的销售情况",
            "我想了解产品A的销售情况"
        ],
        data_source="sales_db",
        related_intents=["query_sales_report"]
    )
    
    # 人力资源报表指标
    hr_report = BusinessMetric(
        name="hr_report",
        description="人力资源报表，包含员工数量、招聘、离职、培训等人力资源数据",
        category="hr",
        keywords=["人力", "资源", "报表", "员工", "招聘", "离职", "培训", "HR"],
        examples=[
            "查看部门的人员配置情况",
            "显示本季度的招聘情况",
            "我想了解员工培训的情况"
        ],
        data_source="hr_db",
        related_intents=["query_hr_report"]
    )
    
    # 注册指标
    registry.register_metric(financial_report)
    registry.register_metric(project_report)
    registry.register_metric(sales_report)
    registry.register_metric(hr_report)
    
    return registry 