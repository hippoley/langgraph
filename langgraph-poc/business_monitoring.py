"""
业务指标监测模块 - 动态监测和分析业务指标
"""

import os
import json
import time
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BusinessMonitoring")

# 数据存储路径
METRICS_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "metrics")
os.makedirs(METRICS_STORAGE_DIR, exist_ok=True)

class MetricType(Enum):
    """指标类型"""
    COUNTER = auto()      # 计数器，用于记录事件发生次数
    GAUGE = auto()        # 计量器，用于记录当前值
    HISTOGRAM = auto()    # 直方图，用于统计数值分布
    TIMER = auto()        # 计时器，用于测量持续时间


class AlertLevel(Enum):
    """告警级别"""
    INFO = auto()         # 信息级别
    WARNING = auto()      # 警告级别
    ERROR = auto()        # 错误级别
    CRITICAL = auto()     # 严重级别


@dataclass
class MetricAlert:
    """指标告警配置"""
    metric_name: str                    # 指标名称
    condition: str                      # 触发条件表达式
    level: AlertLevel                   # 告警级别
    message_template: str               # 告警消息模板
    cooldown_seconds: int = 300         # 冷却时间（秒）
    last_triggered: Optional[float] = None  # 上次触发时间
    enabled: bool = True                # 是否启用


@dataclass
class MetricValue:
    """指标值记录"""
    name: str                          # 指标名称
    value: Union[int, float, List]     # 指标值
    timestamp: float                   # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str                          # 指标名称
    description: str                   # 描述
    type: MetricType                   # 类型
    unit: str = ""                     # 单位
    tags: List[str] = field(default_factory=list)  # 标签
    aggregation: str = "last"          # 聚合方式 (last, sum, avg, min, max)


class MetricAggregator:
    """指标聚合器"""
    
    @staticmethod
    def aggregate(metric_values: List[MetricValue], 
                 method: str = "last", 
                 window_seconds: Optional[int] = None) -> Union[int, float, List]:
        """聚合指标值
        
        Args:
            metric_values: 指标值列表
            method: 聚合方法 (last, sum, avg, min, max, count)
            window_seconds: 时间窗口（秒），None表示全部
            
        Returns:
            聚合后的值
        """
        if not metric_values:
            return 0
        
        # 按时间戳排序
        sorted_values = sorted(metric_values, key=lambda x: x.timestamp)
        
        # 如果有时间窗口，筛选数据
        if window_seconds is not None:
            now = time.time()
            window_start = now - window_seconds
            sorted_values = [v for v in sorted_values if v.timestamp >= window_start]
        
        # 没有数据返回0
        if not sorted_values:
            return 0
        
        # 提取数值
        values = [v.value for v in sorted_values]
        
        # 根据方法聚合
        if method == "last":
            return values[-1]
        elif method == "sum":
            return sum(values)
        elif method == "avg":
            return sum(values) / len(values)
        elif method == "min":
            return min(values)
        elif method == "max":
            return max(values)
        elif method == "count":
            return len(values)
        else:
            logger.warning(f"未知聚合方法: {method}，使用最后一个值")
            return values[-1]


class BusinessMetricsMonitor:
    """业务指标监测器"""
    
    def __init__(self):
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_values: Dict[str, List[MetricValue]] = {}
        self.alerts: Dict[str, MetricAlert] = {}
        self.alert_handlers: List[Callable[[MetricAlert, Any], None]] = []
        self.retention_days: int = 30  # 数据保留天数
        self.auto_snapshot: bool = True  # 自动快照
        self.snapshot_interval: int = 3600  # 快照间隔（秒）
        self._last_snapshot_time: float = 0  # 上次快照时间
        self._background_task = None  # 后台任务
        
    def define_metric(self, definition: MetricDefinition) -> None:
        """定义新指标
        
        Args:
            definition: 指标定义
        """
        self.metric_definitions[definition.name] = definition
        if definition.name not in self.metric_values:
            self.metric_values[definition.name] = []
        logger.info(f"已定义指标: {definition.name} ({definition.type.name})")
    
    def record_metric(self, name: str, value: Union[int, float, List], 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """记录指标值
        
        Args:
            name: 指标名称
            value: 指标值
            metadata: 元数据
        """
        if name not in self.metric_definitions:
            logger.warning(f"未定义的指标: {name}")
            return
        
        # 创建指标值记录
        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # 添加到历史记录
        if name not in self.metric_values:
            self.metric_values[name] = []
        self.metric_values[name].append(metric_value)
        
        # 检查告警
        self._check_alerts(name, value, metric_value.metadata)
        
        # 检查是否需要快照
        if self.auto_snapshot and time.time() - self._last_snapshot_time > self.snapshot_interval:
            self.create_snapshot()
    
    def get_metric_value(self, name: str, 
                        method: str = "last", 
                        window_seconds: Optional[int] = None) -> Union[int, float, List, None]:
        """获取指标当前值
        
        Args:
            name: 指标名称
            method: 聚合方法
            window_seconds: 时间窗口（秒）
            
        Returns:
            聚合后的指标值
        """
        if name not in self.metric_values or not self.metric_values[name]:
            return None
        
        return MetricAggregator.aggregate(
            self.metric_values[name], 
            method=method,
            window_seconds=window_seconds
        )
    
    def get_metric_history(self, name: str, 
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> List[MetricValue]:
        """获取指标历史记录
        
        Args:
            name: 指标名称
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            指标值列表
        """
        if name not in self.metric_values:
            return []
        
        values = self.metric_values[name]
        
        # 筛选时间范围
        if start_time is not None:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time is not None:
            values = [v for v in values if v.timestamp <= end_time]
        
        return sorted(values, key=lambda x: x.timestamp)
    
    def configure_alert(self, alert: MetricAlert) -> None:
        """配置指标告警
        
        Args:
            alert: 告警配置
        """
        if alert.metric_name not in self.metric_definitions:
            logger.warning(f"未定义的指标: {alert.metric_name}，无法配置告警")
            return
        
        self.alerts[f"{alert.metric_name}:{alert.condition}"] = alert
        logger.info(f"已配置告警: {alert.metric_name} {alert.condition} ({alert.level.name})")
    
    def add_alert_handler(self, handler: Callable[[MetricAlert, Any], None]) -> None:
        """添加告警处理器
        
        Args:
            handler: 处理器函数，接收告警配置和触发值
        """
        self.alert_handlers.append(handler)
    
    def _check_alerts(self, metric_name: str, value: Any, metadata: Dict[str, Any]) -> None:
        """检查是否触发告警
        
        Args:
            metric_name: 指标名称
            value: 当前值
            metadata: 元数据
        """
        # 查找该指标的所有告警
        for alert_id, alert in self.alerts.items():
            if alert.metric_name != metric_name or not alert.enabled:
                continue
            
            # 检查冷却时间
            if alert.last_triggered is not None:
                elapsed = time.time() - alert.last_triggered
                if elapsed < alert.cooldown_seconds:
                    continue
            
            # 评估条件
            try:
                # 创建条件评估环境
                env = {
                    "value": value,
                    "metadata": metadata,
                    "now": time.time(),
                    "metric": metric_name
                }
                
                # 评估条件
                result = eval(alert.condition, {"__builtins__": {}}, env)
                
                if result:
                    # 触发告警
                    alert.last_triggered = time.time()
                    
                    # 格式化消息
                    message = alert.message_template.format(
                        value=value,
                        metric=metric_name,
                        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        **metadata
                    )
                    
                    # 调用处理器
                    for handler in self.alert_handlers:
                        try:
                            handler(alert, value)
                        except Exception as e:
                            logger.error(f"调用告警处理器时出错: {str(e)}")
                    
                    logger.warning(f"触发告警 [{alert.level.name}]: {message}")
            
            except Exception as e:
                logger.error(f"评估告警条件时出错: {str(e)}, 条件: {alert.condition}")
    
    def create_snapshot(self) -> str:
        """创建指标快照
        
        Returns:
            快照文件路径
        """
        try:
            # 创建快照数据
            snapshot = {
                "timestamp": time.time(),
                "metrics": {},
                "definitions": {}
            }
            
            # 保存指标定义
            for name, definition in self.metric_definitions.items():
                snapshot["definitions"][name] = asdict(definition)
                snapshot["definitions"][name]["type"] = definition.type.name
            
            # 保存最近值
            for name, values in self.metric_values.items():
                if not values:
                    continue
                
                # 对于每种聚合方式都保存一个值
                snapshot["metrics"][name] = {
                    "last": MetricAggregator.aggregate(values, "last"),
                    "sum": MetricAggregator.aggregate(values, "sum", 3600),
                    "avg": MetricAggregator.aggregate(values, "avg", 3600),
                    "min": MetricAggregator.aggregate(values, "min", 3600),
                    "max": MetricAggregator.aggregate(values, "max", 3600),
                    "count": MetricAggregator.aggregate(values, "count", 3600),
                    "last_timestamp": values[-1].timestamp
                }
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_snapshot_{timestamp}.json"
            filepath = os.path.join(METRICS_STORAGE_DIR, filename)
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            
            # 更新最后快照时间
            self._last_snapshot_time = time.time()
            
            logger.info(f"已创建指标快照: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建指标快照时出错: {str(e)}")
            return ""
    
    def load_snapshot(self, filepath: str) -> bool:
        """加载指标快照
        
        Args:
            filepath: 快照文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载指标定义
            for name, def_data in data.get("definitions", {}).items():
                if "type" in def_data:
                    # 转换类型字符串为枚举
                    type_str = def_data.pop("type")
                    metric_type = MetricType[type_str]
                    
                    # 创建定义
                    definition = MetricDefinition(
                        name=name,
                        description=def_data.get("description", ""),
                        type=metric_type,
                        unit=def_data.get("unit", ""),
                        tags=def_data.get("tags", []),
                        aggregation=def_data.get("aggregation", "last")
                    )
                    
                    self.define_metric(definition)
            
            logger.info(f"已从快照加载{len(data.get('definitions', {}))}个指标定义")
            return True
            
        except Exception as e:
            logger.error(f"加载指标快照时出错: {str(e)}")
            return False
    
    def clean_old_data(self) -> int:
        """清理旧数据
        
        Returns:
            清理的数据点数量
        """
        if self.retention_days <= 0:
            return 0
        
        cutoff_time = time.time() - (self.retention_days * 86400)
        total_cleaned = 0
        
        for name, values in self.metric_values.items():
            original_count = len(values)
            self.metric_values[name] = [v for v in values if v.timestamp >= cutoff_time]
            cleaned = original_count - len(self.metric_values[name])
            total_cleaned += cleaned
        
        logger.info(f"已清理{total_cleaned}个过期的数据点")
        return total_cleaned
    
    async def _background_monitoring(self) -> None:
        """后台监控任务"""
        while True:
            try:
                # 清理旧数据
                self.clean_old_data()
                
                # 创建快照
                if self.auto_snapshot and time.time() - self._last_snapshot_time > self.snapshot_interval:
                    self.create_snapshot()
                
                # 等待下一次执行
                await asyncio.sleep(60)  # 每分钟执行一次
                
            except asyncio.CancelledError:
                logger.info("后台监控任务已取消")
                break
            except Exception as e:
                logger.error(f"后台监控任务出错: {str(e)}")
                await asyncio.sleep(10)  # 出错后等待短暂时间再重试
    
    def start_background_monitoring(self) -> None:
        """启动后台监控任务"""
        if self._background_task is not None:
            logger.warning("后台监控任务已在运行")
            return
        
        loop = asyncio.get_event_loop()
        self._background_task = loop.create_task(self._background_monitoring())
        logger.info("已启动后台监控任务")
    
    def stop_background_monitoring(self) -> None:
        """停止后台监控任务"""
        if self._background_task is None:
            return
        
        self._background_task.cancel()
        self._background_task = None
        logger.info("已停止后台监控任务")


# 默认监控实例
_default_monitor = None

def get_default_monitor() -> BusinessMetricsMonitor:
    """获取默认监控实例"""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = BusinessMetricsMonitor()
    return _default_monitor

def create_default_metrics_monitor() -> BusinessMetricsMonitor:
    """创建包含默认指标的监控器"""
    monitor = BusinessMetricsMonitor()
    
    # 定义基础指标
    monitor.define_metric(MetricDefinition(
        name="session_count",
        description="活跃会话数量",
        type=MetricType.GAUGE,
        unit="sessions"
    ))
    
    monitor.define_metric(MetricDefinition(
        name="message_count",
        description="消息总数",
        type=MetricType.COUNTER,
        unit="messages"
    ))
    
    monitor.define_metric(MetricDefinition(
        name="error_count",
        description="错误数量",
        type=MetricType.COUNTER,
        unit="errors"
    ))
    
    monitor.define_metric(MetricDefinition(
        name="intent_success_rate",
        description="意图识别成功率",
        type=MetricType.GAUGE,
        unit="percent"
    ))
    
    monitor.define_metric(MetricDefinition(
        name="response_time",
        description="响应时间",
        type=MetricType.HISTOGRAM,
        unit="seconds"
    ))
    
    monitor.define_metric(MetricDefinition(
        name="human_interventions",
        description="人工干预次数",
        type=MetricType.COUNTER,
        unit="interventions"
    ))
    
    # 为各个意图定义指标
    for intent in ["calculate", "query_weather", "translate", "chitchat", "help",
                  "query_financial_report", "query_project_report", 
                  "query_sales_report", "query_hr_report"]:
        monitor.define_metric(MetricDefinition(
            name=f"intent_{intent}_count",
            description=f"{intent}意图触发次数",
            type=MetricType.COUNTER,
            unit="triggers",
            tags=[intent, "intent"]
        ))
    
    # 配置默认告警
    monitor.configure_alert(MetricAlert(
        metric_name="error_count",
        condition="value > 10",
        level=AlertLevel.WARNING,
        message_template="错误数量过高: {value}个"
    ))
    
    monitor.configure_alert(MetricAlert(
        metric_name="response_time",
        condition="value > 5.0",
        level=AlertLevel.WARNING,
        message_template="响应时间过长: {value:.2f}秒"
    ))
    
    monitor.configure_alert(MetricAlert(
        metric_name="intent_success_rate",
        condition="value < 70.0",
        level=AlertLevel.ERROR,
        message_template="意图识别成功率过低: {value:.1f}%"
    ))
    
    return monitor 