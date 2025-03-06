"""
LangGraph 可视化和调试工具模块

提供了一系列工具用于:
1. 可视化工作流程图和状态转换
2. 调试信息的收集和展示
3. 性能分析和监控
4. 交互式调试界面
"""

import os
import json
import logging
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional

class GraphVisualizer:
    """工作流程图可视化工具"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_attrs = {}
        self.edge_attrs = {}
    
    def add_node(self, node_id: str, **attrs):
        """添加节点"""
        self.graph.add_node(node_id)
        self.node_attrs[node_id] = attrs
    
    def add_edge(self, source: str, target: str, **attrs):
        """添加边"""
        self.graph.add_edge(source, target)
        self.edge_attrs[(source, target)] = attrs
    
    def draw(self, save_path: str = "workflow_graph.png"):
        """绘制工作流程图"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color='lightblue',
                             node_size=2000)
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        # 添加标签
        labels = {node: self.node_attrs[node].get('label', node) 
                 for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels)
        
        plt.title("工作流程图")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

class DebugInfo:
    """调试信息收集和展示"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.events = []
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('langgraph_debug')
        logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        if not os.path.exists('logs'):
            os.makedirs('logs')
        fh = logging.FileHandler('logs/debug.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """记录日志"""
        log_func = getattr(self.logger, level.lower())
        log_func(message)
        
        # 记录事件
        self.events.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'level': level,
            'message': message,
            'context': kwargs
        })
    
    def get_events(self, level: Optional[str] = None) -> List[Dict]:
        """获取事件列表"""
        if level:
            return [e for e in self.events if e['level'] == level]
        return self.events
    
    def clear(self):
        """清除事件记录"""
        self.events = []

class PerformanceAnalyzer:
    """性能分析工具"""
    
    def __init__(self):
        self.operations = {}
        self.current_operations = {}
    
    def start_operation(self, operation_name: str):
        """开始记录操作"""
        self.current_operations[operation_name] = datetime.datetime.now()
    
    def end_operation(self, operation_name: str):
        """结束记录操作"""
        if operation_name in self.current_operations:
            start_time = self.current_operations[operation_name]
            duration = (datetime.datetime.now() - start_time).total_seconds()
            
            if operation_name not in self.operations:
                self.operations[operation_name] = []
            
            self.operations[operation_name].append({
                'start_time': start_time,
                'duration': duration
            })
            
            del self.current_operations[operation_name]
    
    def get_metrics(self) -> Dict[str, List[Dict]]:
        """获取性能指标"""
        return self.operations
    
    def get_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        total_duration = 0
        total_operations = 0
        max_duration = 0
        min_duration = float('inf')
        
        for op_list in self.operations.values():
            for op in op_list:
                duration = op['duration']
                total_duration += duration
                total_operations += 1
                max_duration = max(max_duration, duration)
                min_duration = min(min_duration, duration)
        
        return {
            'total_operations': total_operations,
            'total_duration': total_duration,
            'avg_duration': total_duration / total_operations if total_operations > 0 else 0,
            'max_duration': max_duration,
            'min_duration': min_duration if min_duration != float('inf') else 0
        }
    
    def create_timeline(self, save_path: str = "timeline.html"):
        """创建时间线可视化"""
        df = []
        for op_name, op_list in self.operations.items():
            for op in op_list:
                df.append(dict(
                    Task=op_name,
                    Start=op['start_time'],
                    Finish=op['start_time'] + datetime.timedelta(seconds=op['duration'])
                ))
        
        fig = ff.create_gantt(df, 
                            index_col='Task',
                            show_colorbar=True,
                            group_tasks=True,
                            showgrid_x=True,
                            showgrid_y=True)
        
        fig.write_html(save_path)

class InteractiveDebugger:
    """交互式调试工具"""
    
    def __init__(self):
        self.state_history = []
        self.breakpoints = set()
        self.watches = {}
    
    def update_state(self, state: Dict[str, Any]):
        """更新状态"""
        self.state_history.append({
            'timestamp': datetime.datetime.now(),
            'state': state
        })
    
    def add_breakpoint(self, condition: str):
        """添加断点"""
        self.breakpoints.add(condition)
    
    def add_watch(self, name: str, path: str):
        """添加监视"""
        self.watches[name] = path
    
    def get_state_diff(self, index1: int, index2: int) -> Dict:
        """获取状态差异"""
        if not (0 <= index1 < len(self.state_history) and 0 <= index2 < len(self.state_history)):
            return {}
        
        state1 = self.state_history[index1]['state']
        state2 = self.state_history[index2]['state']
        
        diff = {}
        for key in set(state1.keys()) | set(state2.keys()):
            if key not in state1:
                diff[key] = {'type': 'added', 'value': state2[key]}
            elif key not in state2:
                diff[key] = {'type': 'removed', 'value': state1[key]}
            elif state1[key] != state2[key]:
                diff[key] = {
                    'type': 'modified',
                    'old_value': state1[key],
                    'new_value': state2[key]
                }
        
        return diff

def create_debug_report(save_dir: str = "debug_reports") -> str:
    """生成调试报告"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f"debug_report_{timestamp}.html")
    
    # 获取调试信息
    events = debug_info.get_events()
    metrics = performance_analyzer.get_metrics()
    summary = performance_analyzer.get_summary()
    
    # 转换datetime对象为字符串
    def convert_datetime(obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return obj
    
    # 处理metrics中的datetime
    formatted_metrics = {}
    for op_name, op_list in metrics.items():
        formatted_metrics[op_name] = []
        for op in op_list:
            formatted_metrics[op_name].append({
                'start_time': convert_datetime(op['start_time']),
                'duration': op['duration']
            })
    
    # 创建HTML报告
    html_content = f"""
    <html>
    <head>
        <title>LangGraph调试报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
            .event {{ margin: 5px 0; padding: 5px; background-color: #f5f5f5; }}
            .metric {{ margin: 5px 0; }}
            pre {{ background-color: #f8f8f8; padding: 10px; }}
        </style>
    </head>
    <body>
        <h1>LangGraph调试报告</h1>
        <div class="section">
            <h2>生成时间</h2>
            <p>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="section">
            <h2>性能摘要</h2>
            <div class="metric">总操作数: {summary['total_operations']}</div>
            <div class="metric">总执行时间: {summary['total_duration']:.2f} 秒</div>
            <div class="metric">平均执行时间: {summary['avg_duration']:.2f} 秒</div>
            <div class="metric">最长操作: {summary['max_duration']:.2f} 秒</div>
            <div class="metric">最短操作: {summary['min_duration']:.2f} 秒</div>
        </div>
        
        <div class="section">
            <h2>事件日志</h2>
            {''.join(f'<div class="event">[{e["timestamp"]}] {e["level"]}: {e["message"]}</div>' for e in events)}
        </div>
        
        <div class="section">
            <h2>性能指标详情</h2>
            <pre>{json.dumps(formatted_metrics, indent=2, ensure_ascii=False)}</pre>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 生成时间线
    timeline_path = os.path.join(save_dir, f"timeline_{timestamp}.html")
    
    # 准备时间线数据
    df = []
    for op_name, op_list in metrics.items():
        for op in op_list:
            start_time = op['start_time']
            df.append(dict(
                Task=op_name,
                Start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                Finish=(start_time + datetime.timedelta(seconds=op['duration'])).strftime("%Y-%m-%d %H:%M:%S")
            ))
    
    if df:  # 只在有数据时创建时间线
        fig = ff.create_gantt(df, 
                            index_col='Task',
                            show_colorbar=True,
                            group_tasks=True,
                            showgrid_x=True,
                            showgrid_y=True)
        
        fig.write_html(timeline_path)
    
    return report_path

# 创建全局实例
debug_info = DebugInfo()
performance_analyzer = PerformanceAnalyzer()
interactive_debugger = InteractiveDebugger()

def setup_debugging(workflow_graph):
    """设置调试工具"""
    # 这里可以添加更多的调试设置
    pass

# 示例用法
if __name__ == "__main__":
    # 创建示例工作流程图
    visualizer = GraphVisualizer()
    
    # 添加一些节点和边
    visualizer.add_node("start", label="开始")
    visualizer.add_node("process", label="处理")
    visualizer.add_node("decision", label="决策")
    visualizer.add_node("end", label="结束")
    
    visualizer.add_edge("start", "process")
    visualizer.add_edge("process", "decision")
    visualizer.add_edge("decision", "end")
    
    # 绘制图形
    visualizer.draw()
    
    # 记录一些调试信息
    debug_info.log("工作流程开始", level="INFO")
    debug_info.log("处理节点执行", level="DEBUG")
    debug_info.log("发现潜在问题", level="WARNING")
    
    # 记录一些性能指标
    performance_analyzer.start_operation("data_processing")
    time.sleep(1)  # 模拟操作
    performance_analyzer.end_operation("data_processing")
    
    # 生成调试报告
    report_path = create_debug_report()
    print(f"调试报告已生成: {report_path}") 