from visualization import (
    GraphVisualizer, 
    debug_info, 
    performance_analyzer, 
    interactive_debugger,
    create_debug_report
)
import time

def demo_chat_process():
    """演示聊天过程中的可视化和调试功能"""
    print("\n=== 开始演示可视化和调试功能 ===\n")
    
    # 1. 显示工作流程图
    print("1. 创建聊天流程图...")
    visualizer = GraphVisualizer()
    
    # 添加主要步骤
    visualizer.add_node("用户输入", label="接收用户消息")
    visualizer.add_node("理解意图", label="分析用户意图")
    visualizer.add_node("生成回复", label="生成AI回复")
    visualizer.add_node("更新状态", label="更新对话状态")
    
    # 添加步骤之间的连接
    visualizer.add_edge("用户输入", "理解意图")
    visualizer.add_edge("理解意图", "生成回复")
    visualizer.add_edge("生成回复", "更新状态")
    visualizer.add_edge("更新状态", "用户输入")
    
    # 保存流程图
    visualizer.draw("chat_workflow.png")
    print("流程图已保存为 chat_workflow.png")
    
    # 2. 模拟一次对话过程
    print("\n2. 模拟一次对话过程...")
    
    # 记录开始
    debug_info.log("开始新的对话", level="INFO")
    
    # 模拟用户输入处理
    print("\n处理用户输入: '今天天气怎么样？'")
    performance_analyzer.start_operation("用户输入处理")
    time.sleep(1)  # 模拟处理时间
    debug_info.log("收到用户输入: '今天天气怎么样？'", level="DEBUG")
    performance_analyzer.end_operation("用户输入处理")
    
    # 模拟意图理解
    print("分析用户意图...")
    performance_analyzer.start_operation("意图分析")
    time.sleep(0.5)  # 模拟处理时间
    debug_info.log("识别到意图: 查询天气", level="INFO")
    performance_analyzer.end_operation("意图分析")
    
    # 模拟回复生成
    print("生成回复...")
    performance_analyzer.start_operation("回复生成")
    time.sleep(1.5)  # 模拟处理时间
    debug_info.log("生成回复: '今天天气晴朗，气温25度'", level="INFO")
    performance_analyzer.end_operation("回复生成")
    
    # 更新状态
    print("更新对话状态...")
    performance_analyzer.start_operation("状态更新")
    time.sleep(0.3)  # 模拟处理时间
    
    # 更新交互式调试器状态
    interactive_debugger.update_state({
        "当前意图": "查询天气",
        "用户输入": "今天天气怎么样？",
        "AI回复": "今天天气晴朗，气温25度",
        "对话轮次": 1
    })
    
    debug_info.log("对话状态已更新", level="DEBUG")
    performance_analyzer.end_operation("状态更新")
    
    # 3. 生成调试报告
    print("\n3. 生成调试报告...")
    report_path = create_debug_report()
    print(f"调试报告已生成: {report_path}")
    
    # 4. 显示性能统计
    print("\n4. 性能统计:")
    summary = performance_analyzer.get_summary()
    print(f"- 总操作数: {summary['total_operations']}")
    print(f"- 总执行时间: {summary['total_duration']:.2f} 秒")
    print(f"- 平均执行时间: {summary['avg_duration']:.2f} 秒")
    print(f"- 最长操作: {summary['max_duration']:.2f} 秒")
    
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
    demo_chat_process() 