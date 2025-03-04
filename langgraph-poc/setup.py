import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """安装所有依赖项"""
    print("正在安装依赖项...")
    
    # 获取当前目录
    current_dir = Path(__file__).parent.absolute()
    
    # 安装requirements.txt中的依赖
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                          os.path.join(current_dir, "requirements.txt")])
    
    # 确保langgraph-cli正确安装并添加到PATH
    try:
        # 检查langgraph命令是否可用
        subprocess.check_call(["langgraph", "--version"], 
                             stderr=subprocess.DEVNULL, 
                             stdout=subprocess.DEVNULL)
        print("langgraph-cli已成功安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("正在安装langgraph-cli...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langgraph-cli"])
        
        # 提示用户可能需要重启终端
        print("\n安装完成！")
        print("如果在运行'langgraph dev'时仍然遇到问题，请尝试以下步骤：")
        print("1. 关闭并重新打开您的终端/命令提示符")
        print("2. 确保Python的Scripts目录在您的PATH环境变量中")
        
        # 显示Python的Scripts目录路径
        scripts_dir = os.path.join(sys.prefix, "Scripts")
        print(f"\nPython Scripts目录: {scripts_dir}")
        print("请确保此目录已添加到您的PATH环境变量中")

if __name__ == "__main__":
    install_dependencies()
    print("\n所有依赖项已安装完成！")
    print("现在您可以尝试运行：")
    print("1. 直接运行Python文件: python simple_agent.py")
    print("2. 或使用LangGraph CLI: langgraph dev") 