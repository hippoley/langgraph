#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph Fix and Startup Tool
This tool resolves common LangGraph issues, cleans temporary files, and starts the server
"""

import os
import sys
import platform
import subprocess
import time
import shutil
from pathlib import Path

# 颜色输出函数
def print_colored(text, color="default"):
    """打印彩色文本"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "default": "\033[0m",
    }
    
    # Windows命令行可能不支持ANSI颜色代码
    if platform.system() == "Windows":
        try:
            import colorama
            colorama.init()
        except ImportError:
            # 如果没有colorama，则不使用颜色
            print(text)
            return
    
    color_code = colors.get(color, colors["default"])
    end_code = colors["default"]
    print(f"{color_code}{text}{end_code}")

def clear_screen():
    """清除屏幕"""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

# 清理临时文件
def cleanup_langgraph_files():
    """清理LangGraph临时文件和锁定文件"""
    print_colored("Cleaning LangGraph temporary files...", "blue")
    
    # 需要清理的目录
    directories_to_clean = [
        ".langgraph_api",
        ".langgraph",
        ".langgraph_cli",
        "langgraph-poc/.langgraph_api",
        "langgraph-poc/.langgraph",
        "langgraph-poc/.langgraph_cli"
    ]
    
    # 需要删除的文件模式
    file_patterns = [
        "*.pckl",
        "*.pckl.tmp",
        "*.lock",
        "*.db",
        "*.db-journal"
    ]
    
    # 计数器
    deleted_files = 0
    deleted_dirs = 0
    
    # 检查并清理每个目录
    for directory in directories_to_clean:
        dir_path = Path(directory)
        if dir_path.exists():
            print_colored(f"Checking directory: {dir_path}", "cyan")
            
            # 尝试删除匹配的文件
            for pattern in file_patterns:
                for file_path in dir_path.glob(pattern):
                    try:
                        file_path.unlink()
                        print_colored(f"  Deleted file: {file_path}", "green")
                        deleted_files += 1
                    except Exception as e:
                        print_colored(f"  Cannot delete file {file_path}: {str(e)}", "red")
            
            # 如果目录为空，尝试删除它
            if not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    print_colored(f"Deleted empty directory: {dir_path}", "green")
                    deleted_dirs += 1
                except Exception as e:
                    print_colored(f"Cannot delete directory {dir_path}: {str(e)}", "red")
    
    # 总结
    if deleted_files > 0 or deleted_dirs > 0:
        print_colored(f"Cleanup complete! Deleted {deleted_files} files and {deleted_dirs} directories.", "green")
    else:
        print_colored("No files found to clean.", "yellow")
    
    return deleted_files > 0

def kill_langgraph_processes():
    """终止所有LangGraph相关进程"""
    print_colored("Checking and terminating LangGraph processes...", "blue")
    
    if platform.system() == "Windows":
        try:
            os.system("taskkill /f /im langgraph-cli.exe 2>nul")
            os.system("taskkill /f /im langgraph.exe 2>nul")
            os.system("taskkill /f /im python.exe /fi \"WINDOWTITLE eq langgraph*\" 2>nul")
            print_colored("Attempted to terminate LangGraph processes", "green")
        except Exception as e:
            print_colored(f"Error terminating processes: {str(e)}", "red")
    else:
        try:
            os.system("pkill -f langgraph-cli")
            os.system("pkill -f 'python.*langgraph'")
            print_colored("Attempted to terminate LangGraph processes", "green")
        except Exception as e:
            print_colored(f"Error terminating processes: {str(e)}", "red")
    
    # 等待进程完全终止
    print_colored("Waiting for processes to terminate...", "blue")
    time.sleep(2)

# 查找langgraph-cli可执行文件
def find_langgraph_cli():
    """查找langgraph-cli可执行文件的路径"""
    print_colored("Looking for langgraph-cli...", "blue")
    
    # 检查是否已在PATH中
    try:
        if platform.system() == "Windows":
            # 在Windows上，使用where命令
            result = subprocess.run(["where", "langgraph-cli"], 
                                   capture_output=True, 
                                   text=True, 
                                   check=False)
            if result.returncode == 0:
                path = result.stdout.strip().split('\n')[0]
                print_colored(f"Found langgraph-cli in PATH: {path}", "green")
                return path
        else:
            # 在Linux/Mac上，使用which命令
            result = subprocess.run(["which", "langgraph-cli"], 
                                   capture_output=True, 
                                   text=True, 
                                   check=False)
            if result.returncode == 0:
                path = result.stdout.strip()
                print_colored(f"Found langgraph-cli in PATH: {path}", "green")
                return path
    except Exception as e:
        print_colored(f"Error looking for langgraph-cli: {str(e)}", "red")
    
    # 检查Python脚本目录
    try:
        import site
        script_dirs = site.getsitepackages()
        script_dirs.append(site.getusersitepackages())
        
        for script_dir in script_dirs:
            if platform.system() == "Windows":
                cli_path = os.path.join(script_dir, "Scripts", "langgraph-cli.exe")
                if os.path.exists(cli_path):
                    print_colored(f"Found langgraph-cli in Python scripts directory: {cli_path}", "green")
                    return cli_path
            else:
                cli_path = os.path.join(script_dir, "bin", "langgraph-cli")
                if os.path.exists(cli_path):
                    print_colored(f"Found langgraph-cli in Python scripts directory: {cli_path}", "green")
                    return cli_path
    except Exception as e:
        print_colored(f"Error checking Python script directories: {str(e)}", "red")
    
    print_colored("langgraph-cli not found, will attempt to install", "yellow")
    return None

# 检查并安装依赖
def check_dependencies():
    """检查并安装必要的依赖"""
    print_colored("Checking dependencies...", "blue")
    
    # 检查colorama
    try:
        import colorama
        print_colored("colorama is installed", "green")
    except ImportError:
        print_colored("Installing colorama...", "yellow")
        subprocess.run([sys.executable, "-m", "pip", "install", "colorama"], check=False)
    
    # 检查langgraph-api
    try:
        import langgraph_api
        print_colored("langgraph-api is installed", "green")
    except ImportError:
        print_colored("Installing langgraph-api...", "yellow")
        subprocess.run([sys.executable, "-m", "pip", "install", "langgraph-api"], check=False)
    
    # 检查langgraph
    try:
        import langgraph
        print_colored("langgraph is installed", "green")
    except ImportError:
        print_colored("Installing langgraph...", "yellow")
        subprocess.run([sys.executable, "-m", "pip", "install", "langgraph"], check=False)
    
    # 检查langgraph-cli
    langgraph_cli_path = find_langgraph_cli()
    if not langgraph_cli_path:
        print_colored("Installing langgraph-cli...", "yellow")
        subprocess.run([sys.executable, "-m", "pip", "install", "langgraph-cli[inmem]"], check=False)
        langgraph_cli_path = find_langgraph_cli()
    
    return langgraph_cli_path

# 启动LangGraph服务器
def start_langgraph_with_cors(langgraph_cli_path=None):
    """启动带有CORS支持的LangGraph服务器"""
    print_colored("Preparing to start LangGraph server...", "blue")
    
    # 清理临时文件
    cleanup_langgraph_files()
    
    # 设置CORS环境变量
    os.environ["LANGGRAPH_API_CORS_ALLOW_ORIGINS"] = "*"
    os.environ["LANGGRAPH_API_CORS_ALLOW_CREDENTIALS"] = "true"
    os.environ["LANGGRAPH_API_CORS_ALLOW_METHODS"] = "*"
    os.environ["LANGGRAPH_API_CORS_ALLOW_HEADERS"] = "*"
    
    print_colored("CORS environment variables set", "green")
    
    # 确定命令
    if not langgraph_cli_path:
        langgraph_cli_path = "langgraph-cli"  # 使用默认命令
    
    # 构建命令
    if platform.system() == "Windows":
        cmd = [langgraph_cli_path, "dev"]
    else:
        cmd = [langgraph_cli_path, "dev"]
    
    # 启动服务器
    try:
        print_colored("Starting LangGraph server...", "magenta")
        print_colored("Once the server is running, you can access:", "cyan")
        print_colored("  - Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024", "cyan")
        print_colored("  - API: http://localhost:2024", "cyan")
        print_colored("  - API docs: http://localhost:2024/docs", "cyan")
        print_colored("\nPress Ctrl+C to stop the server\n", "yellow")
        
        # 使用subprocess.Popen启动进程
        process = subprocess.Popen(cmd)
        
        # 等待一段时间，确保服务器有时间启动
        time.sleep(3)
        
        # 检查进程是否仍在运行
        if process.poll() is None:
            print_colored("Server started successfully!", "green")
        else:
            print_colored(f"Server failed to start, exit code: {process.returncode}", "red")
            print_colored("Trying alternative startup method...", "yellow")
            
            # 备用方法：直接使用os.system
            if platform.system() == "Windows":
                os.system(f"start cmd /c {langgraph_cli_path} dev")
            else:
                os.system(f"{langgraph_cli_path} dev &")
        
        # 等待用户按Ctrl+C
        try:
            process.wait()
        except KeyboardInterrupt:
            print_colored("\nStopping server...", "yellow")
            process.terminate()
            process.wait(timeout=5)
            print_colored("Server stopped", "green")
    
    except Exception as e:
        print_colored(f"Error starting server: {str(e)}", "red")
        print_colored("Trying alternative startup method...", "yellow")
        
        # 备用方法
        try:
            if platform.system() == "Windows":
                os.system(f"start cmd /c {langgraph_cli_path} dev")
            else:
                os.system(f"{langgraph_cli_path} dev &")
            print_colored("Server started using alternative method", "green")
        except Exception as e2:
            print_colored(f"Alternative startup method also failed: {str(e2)}", "red")
            print_colored("Please try the following troubleshooting steps:", "yellow")
            print_colored("1. Ensure all dependencies are installed: pip install -r requirements.txt", "default")
            print_colored("2. Ensure langgraph-cli is properly installed: pip install -U \"langgraph-cli[inmem]\"", "default")
            print_colored("3. Try restarting your computer", "default")
            print_colored("4. Try running a Python file directly: python simple_agent.py", "default")

def create_batch_file():
    """创建Windows批处理文件"""
    batch_file_path = "Start_LangGraph.bat"
    
    if os.path.exists(batch_file_path):
        overwrite = input("Batch file already exists. Overwrite? (y/n): ").lower().strip() == 'y'
        if not overwrite:
            print_colored("Batch file creation cancelled", "yellow")
            return
    
    print_colored("Creating batch file...", "blue")
    
    batch_content = """@echo off
chcp 65001 > nul
title LangGraph Server

echo [94m=== LangGraph Server Tool ===[0m
echo [96mThis tool will start the LangGraph server with CORS support[0m
echo.

REM Run Python fix script
python langgraph_fix.py --start

pause
"""
    
    try:
        with open(batch_file_path, "w", encoding="utf-8") as f:
            f.write(batch_content)
        print_colored(f"Batch file created: {batch_file_path}", "green")
    except Exception as e:
        print_colored(f"Failed to create batch file: {str(e)}", "red")

def show_menu():
    """显示主菜单"""
    clear_screen()
    print_colored("=== LangGraph Fix and Startup Tool ===", "magenta")
    print_colored("Select an operation:", "cyan")
    print_colored("1. Clean temporary files", "default")
    print_colored("2. Terminate LangGraph processes", "default")
    print_colored("3. Check and install dependencies", "default")
    print_colored("4. Start LangGraph server (with CORS support)", "default")
    print_colored("5. Execute all (clean + terminate + check + start)", "default")
    print_colored("6. Create quick start batch file", "default")
    print_colored("0. Exit", "default")
    print()
    
    choice = input("Enter option [0-6]: ").strip()
    return choice

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--start":
        # 直接启动服务器模式
        kill_langgraph_processes()
        langgraph_cli_path = check_dependencies()
        start_langgraph_with_cors(langgraph_cli_path)
        return
    
    # 交互式菜单模式
    while True:
        choice = show_menu()
        
        if choice == "0":
            print_colored("Exiting program", "yellow")
            break
        
        elif choice == "1":
            cleanup_langgraph_files()
            input("\nPress Enter to continue...")
        
        elif choice == "2":
            kill_langgraph_processes()
            input("\nPress Enter to continue...")
        
        elif choice == "3":
            check_dependencies()
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            langgraph_cli_path = check_dependencies()
            start_langgraph_with_cors(langgraph_cli_path)
            # 启动服务器后退出循环
            break
        
        elif choice == "5":
            kill_langgraph_processes()
            cleanup_langgraph_files()
            langgraph_cli_path = check_dependencies()
            start_langgraph_with_cors(langgraph_cli_path)
            # 启动服务器后退出循环
            break
        
        elif choice == "6":
            create_batch_file()
            input("\nPress Enter to continue...")
        
        else:
            print_colored("Invalid option, please try again", "red")
            time.sleep(1)

if __name__ == "__main__":
    main() 