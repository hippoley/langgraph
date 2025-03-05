#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph 监控和可观测性模块

此模块提供了对 LangGraph 应用的监控和可观测性功能，包括：
1. 跟踪和记录智能体行为
2. 性能指标收集
3. 可视化工具
4. 错误和异常监控
5. 审计日志
"""

import os
import json
import time
import datetime
import logging
import threading
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langgraph_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("langgraph_monitor")

# 事件类型
class EventType(Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_CALL = "llm_call"
    LLM_RESULT = "llm_result"
    HUMAN_INPUT = "human_input"
    ERROR = "error"
    STATE_CHANGE = "state_change"
    CUSTOM = "custom"

# 监控事件
class MonitoringEvent:
    def __init__(
        self,
        event_type: EventType,
        session_id: str,
        event_id: str = None,
        timestamp: float = None,
        data: Dict[str, Any] = None,
        parent_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = event_type
        self.session_id = session_id
        self.timestamp = timestamp or time.time()
        self.data = data or {}
        self.parent_id = parent_id
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "timestamp_human": datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
            "data": self.data,
            "parent_id": self.parent_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringEvent':
        """从字典创建事件"""
        return cls(
            event_type=EventType(data["event_type"]),
            session_id=data["session_id"],
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            data=data["data"],
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {})
        )

# 监控器类
class LangGraphMonitor:
    def __init__(self, storage_dir: str = None):
        self.events: List[MonitoringEvent] = []
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitoring_data")
        self.lock = threading.Lock()
        self.callbacks: Dict[EventType, List[Callable]] = {}
        
        # 确保存储目录存在
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        # 初始化性能指标
        self.performance_metrics = {
            "llm_calls": 0,
            "tool_calls": 0,
            "node_transitions": 0,
            "errors": 0,
            "response_times": [],
            "token_usage": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            }
        }
    
    def record_event(self, event: MonitoringEvent) -> str:
        """记录监控事件"""
        with self.lock:
            self.events.append(event)
            
            # 更新性能指标
            if event.event_type == EventType.LLM_CALL:
                self.performance_metrics["llm_calls"] += 1
            elif event.event_type == EventType.TOOL_CALL:
                self.performance_metrics["tool_calls"] += 1
            elif event.event_type in [EventType.NODE_ENTER, EventType.NODE_EXIT]:
                self.performance_metrics["node_transitions"] += 1
            elif event.event_type == EventType.ERROR:
                self.performance_metrics["errors"] += 1
            
            # 记录响应时间
            if event.event_type == EventType.LLM_RESULT and "response_time" in event.data:
                self.performance_metrics["response_times"].append(event.data["response_time"])
            
            # 记录token使用情况
            if event.event_type == EventType.LLM_RESULT and "token_usage" in event.data:
                token_usage = event.data["token_usage"]
                self.performance_metrics["token_usage"]["prompt"] += token_usage.get("prompt", 0)
                self.performance_metrics["token_usage"]["completion"] += token_usage.get("completion", 0)
                self.performance_metrics["token_usage"]["total"] += token_usage.get("total", 0)
            
            # 保存事件到文件
            self._save_event(event)
            
            # 触发回调
            self._trigger_callbacks(event)
            
            return event.event_id
    
    def _save_event(self, event: MonitoringEvent):
        """保存事件到文件"""
        try:
            # 创建会话目录
            session_dir = os.path.join(self.storage_dir, event.session_id)
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            
            # 保存事件到文件
            event_file = os.path.join(session_dir, f"{event.event_id}.json")
            with open(event_file, 'w', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存事件失败: {str(e)}")
    
    def register_callback(self, event_type: EventType, callback: Callable):
        """注册事件回调"""
        with self.lock:
            if event_type not in self.callbacks:
                self.callbacks[event_type] = []
            self.callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event: MonitoringEvent):
        """触发事件回调"""
        if event.event_type in self.callbacks:
            for callback in self.callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"执行回调失败: {str(e)}")
    
    def get_events(self, session_id: str = None, event_type: EventType = None, start_time: float = None, end_time: float = None) -> List[MonitoringEvent]:
        """获取事件列表"""
        with self.lock:
            filtered_events = self.events
            
            if session_id:
                filtered_events = [e for e in filtered_events if e.session_id == session_id]
            
            if event_type:
                filtered_events = [e for e in filtered_events if e.event_type == event_type]
            
            if start_time:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
            
            if end_time:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
            
            return filtered_events
    
    def get_event_by_id(self, event_id: str) -> Optional[MonitoringEvent]:
        """通过ID获取事件"""
        with self.lock:
            for event in self.events:
                if event.event_id == event_id:
                    return event
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.lock:
            metrics = self.performance_metrics.copy()
            
            # 计算平均响应时间
            if metrics["response_times"]:
                metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
            else:
                metrics["avg_response_time"] = 0
            
            return metrics
    
    def visualize_event_timeline(self, session_id: str, save_path: str = None) -> Figure:
        """可视化事件时间线"""
        events = self.get_events(session_id=session_id)
        if not events:
            logger.warning(f"会话 {session_id} 没有事件")
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, "No events found", ha='center', va='center')
            return fig
        
        # 按时间排序
        events.sort(key=lambda e: e.timestamp)
        
        # 准备数据
        event_types = [e.event_type.value for e in events]
        timestamps = [e.timestamp - events[0].timestamp for e in events]  # 相对时间
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 为不同事件类型设置不同颜色
        colors = {
            EventType.AGENT_START.value: 'green',
            EventType.AGENT_END.value: 'red',
            EventType.NODE_ENTER.value: 'blue',
            EventType.NODE_EXIT.value: 'cyan',
            EventType.TOOL_CALL.value: 'orange',
            EventType.TOOL_RESULT.value: 'yellow',
            EventType.LLM_CALL.value: 'purple',
            EventType.LLM_RESULT.value: 'magenta',
            EventType.HUMAN_INPUT.value: 'brown',
            EventType.ERROR.value: 'black',
            EventType.STATE_CHANGE.value: 'gray',
            EventType.CUSTOM.value: 'pink'
        }
        
        # 绘制事件点
        for i, (event_type, timestamp) in enumerate(zip(event_types, timestamps)):
            ax.scatter(timestamp, i, color=colors.get(event_type, 'gray'), s=100, label=event_type)
            ax.text(timestamp, i, event_type, fontsize=8, ha='right', va='bottom')
        
        # 添加连接线
        ax.plot(timestamps, range(len(timestamps)), 'k-', alpha=0.3)
        
        # 设置图表属性
        ax.set_yticks([])
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Event Timeline for Session {session_id}')
        
        # 添加图例（去重）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def visualize_performance(self, save_path: str = None) -> Figure:
        """可视化性能指标"""
        metrics = self.get_performance_metrics()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制调用次数
        calls = ['LLM Calls', 'Tool Calls', 'Node Transitions', 'Errors']
        values = [metrics['llm_calls'], metrics['tool_calls'], metrics['node_transitions'], metrics['errors']]
        
        ax1.bar(calls, values, color=['purple', 'orange', 'blue', 'red'])
        ax1.set_title('API Calls and Events')
        ax1.set_ylabel('Count')
        
        # 绘制Token使用情况
        token_labels = ['Prompt', 'Completion', 'Total']
        token_values = [metrics['token_usage']['prompt'], metrics['token_usage']['completion'], metrics['token_usage']['total']]
        
        ax2.bar(token_labels, token_values, color=['green', 'blue', 'gray'])
        ax2.set_title('Token Usage')
        ax2.set_ylabel('Tokens')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def export_events_to_csv(self, session_id: str = None, output_path: str = None) -> str:
        """导出事件到CSV文件"""
        events = self.get_events(session_id=session_id)
        if not events:
            logger.warning(f"没有找到事件{'用于会话 ' + session_id if session_id else ''}")
            return None
        
        # 准备数据
        data = []
        for event in events:
            event_dict = event.to_dict()
            # 展平数据字段
            flat_data = {
                "event_id": event_dict["event_id"],
                "event_type": event_dict["event_type"],
                "session_id": event_dict["session_id"],
                "timestamp": event_dict["timestamp"],
                "timestamp_human": event_dict["timestamp_human"],
                "parent_id": event_dict["parent_id"]
            }
            
            # 添加数据字段
            for key, value in event_dict["data"].items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    flat_data[f"data_{key}"] = value
            
            data.append(flat_data)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到CSV
        if output_path is None:
            output_path = os.path.join(self.storage_dir, f"events{'_' + session_id if session_id else ''}.csv")
        
        df.to_csv(output_path, index=False)
        return output_path
    
    def clear_events(self, session_id: str = None):
        """清除事件"""
        with self.lock:
            if session_id:
                self.events = [e for e in self.events if e.session_id != session_id]
            else:
                self.events = []

# 创建全局监控器实例
monitor = LangGraphMonitor()

# 辅助函数：记录LLM调用
def record_llm_call(session_id: str, prompt: str, metadata: Dict[str, Any] = None) -> str:
    """记录LLM调用"""
    event = MonitoringEvent(
        event_type=EventType.LLM_CALL,
        session_id=session_id,
        data={"prompt": prompt},
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录LLM结果
def record_llm_result(session_id: str, parent_id: str, response: str, response_time: float, token_usage: Dict[str, int] = None, metadata: Dict[str, Any] = None) -> str:
    """记录LLM结果"""
    event = MonitoringEvent(
        event_type=EventType.LLM_RESULT,
        session_id=session_id,
        parent_id=parent_id,
        data={
            "response": response,
            "response_time": response_time,
            "token_usage": token_usage or {}
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录工具调用
def record_tool_call(session_id: str, tool_name: str, tool_input: Any, metadata: Dict[str, Any] = None) -> str:
    """记录工具调用"""
    event = MonitoringEvent(
        event_type=EventType.TOOL_CALL,
        session_id=session_id,
        data={
            "tool_name": tool_name,
            "tool_input": tool_input
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录工具结果
def record_tool_result(session_id: str, parent_id: str, result: Any, execution_time: float, metadata: Dict[str, Any] = None) -> str:
    """记录工具结果"""
    event = MonitoringEvent(
        event_type=EventType.TOOL_RESULT,
        session_id=session_id,
        parent_id=parent_id,
        data={
            "result": result,
            "execution_time": execution_time
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录节点进入
def record_node_enter(session_id: str, node_name: str, state: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> str:
    """记录节点进入"""
    event = MonitoringEvent(
        event_type=EventType.NODE_ENTER,
        session_id=session_id,
        data={
            "node_name": node_name,
            "state": state or {}
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录节点退出
def record_node_exit(session_id: str, node_name: str, next_node: str = None, state: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> str:
    """记录节点退出"""
    event = MonitoringEvent(
        event_type=EventType.NODE_EXIT,
        session_id=session_id,
        data={
            "node_name": node_name,
            "next_node": next_node,
            "state": state or {}
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录错误
def record_error(session_id: str, error_message: str, error_type: str = None, traceback: str = None, metadata: Dict[str, Any] = None) -> str:
    """记录错误"""
    event = MonitoringEvent(
        event_type=EventType.ERROR,
        session_id=session_id,
        data={
            "error_message": error_message,
            "error_type": error_type,
            "traceback": traceback
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录状态变化
def record_state_change(session_id: str, old_state: Dict[str, Any], new_state: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
    """记录状态变化"""
    event = MonitoringEvent(
        event_type=EventType.STATE_CHANGE,
        session_id=session_id,
        data={
            "old_state": old_state,
            "new_state": new_state
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录人类输入
def record_human_input(session_id: str, input_text: str, metadata: Dict[str, Any] = None) -> str:
    """记录人类输入"""
    event = MonitoringEvent(
        event_type=EventType.HUMAN_INPUT,
        session_id=session_id,
        data={"input_text": input_text},
        metadata=metadata
    )
    return monitor.record_event(event)

# 辅助函数：记录自定义事件
def record_custom_event(session_id: str, event_name: str, data: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> str:
    """记录自定义事件"""
    event = MonitoringEvent(
        event_type=EventType.CUSTOM,
        session_id=session_id,
        data={
            "event_name": event_name,
            **(data or {})
        },
        metadata=metadata
    )
    return monitor.record_event(event)

# 示例：如何使用监控模块
if __name__ == "__main__":
    # 创建会话ID
    session_id = f"demo-session-{int(time.time())}"
    
    # 记录智能体启动
    record_custom_event(session_id, "agent_start", {"agent_type": "advanced_agent"})
    
    # 记录LLM调用
    llm_call_id = record_llm_call(session_id, "你好，请告诉我今天的天气。")
    time.sleep(1)  # 模拟LLM处理时间
    record_llm_result(
        session_id, 
        llm_call_id, 
        "今天天气晴朗，温度25°C。", 
        1.2,
        {"prompt": 10, "completion": 15, "total": 25}
    )
    
    # 记录工具调用
    tool_call_id = record_tool_call(session_id, "get_weather", {"location": "北京"})
    time.sleep(0.5)  # 模拟工具执行时间
    record_tool_result(
        session_id,
        tool_call_id,
        {"temperature": 25, "condition": "晴朗", "humidity": 40},
        0.5
    )
    
    # 记录节点转换
    record_node_enter(session_id, "weather_node", {"query": "今天天气"})
    time.sleep(0.3)
    record_node_exit(session_id, "weather_node", "response_node", {"weather_data": {"temperature": 25}})
    
    # 记录人类输入
    record_human_input(session_id, "谢谢，再见！")
    
    # 记录智能体结束
    record_custom_event(session_id, "agent_end")
    
    # 可视化事件时间线
    fig = monitor.visualize_event_timeline(session_id, "timeline.png")
    plt.show()
    
    # 可视化性能指标
    fig = monitor.visualize_performance("performance.png")
    plt.show()
    
    # 导出事件到CSV
    csv_path = monitor.export_events_to_csv(session_id)
    print(f"事件已导出到: {csv_path}")
    
    # 打印性能指标
    metrics = monitor.get_performance_metrics()
    print("性能指标:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False)) 