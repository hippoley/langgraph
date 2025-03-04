import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """检查包是否已安装"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """安装包"""
    print(f"Installing {package_name}...")
    subprocess.run([sys.executable, "-m", "pip", "install", package_name, 
                   "--trusted-host", "pypi.org", 
                   "--trusted-host", "files.pythonhosted.org", 
                   "--trusted-host", "pypi.python.org"], 
                  check=True)
    print(f"{package_name} installed successfully")

def main():
    """主函数"""
    print(f"Python version: {sys.version}")
    
    # 检查并安装必要的包
    required_packages = [
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "python-dotenv",
        "langgraph-cli",
        "langgraph-api"
    ]
    
    for package in required_packages:
        package_name = package.replace("-", "_")
        if not check_package(package_name):
            try:
                install_package(package)
            except Exception as e:
                print(f"Error installing {package}: {e}")
                if package in ["langgraph-cli", "langgraph-api"]:
                    print(f"Trying to install {package} with alternative method...")
                    try:
                        if package == "langgraph-cli":
                            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "langgraph-cli[inmem]",
                                          "--trusted-host", "pypi.org", 
                                          "--trusted-host", "files.pythonhosted.org", 
                                          "--trusted-host", "pypi.python.org"], 
                                         check=True)
                        else:
                            subprocess.run([sys.executable, "-m", "pip", "install", package,
                                          "--trusted-host", "pypi.org", 
                                          "--trusted-host", "files.pythonhosted.org", 
                                          "--trusted-host", "pypi.python.org"], 
                                         check=True)
                        print(f"{package} installed successfully with alternative method")
                    except Exception as e2:
                        print(f"Error installing {package} with alternative method: {e2}")
                        print(f"Please install {package} manually")
    
    # 检查 langgraph-cli 是否已安装
    if check_package("langgraph_cli"):
        print("langgraph-cli is installed")
        
        # 尝试运行 langgraph dev
        print("\nTrying to run LangGraph development server...")
        try:
            # 获取 Python 的 Scripts 目录
            scripts_dir = os.path.join(sys.prefix, "Scripts")
            langgraph_path = os.path.join(scripts_dir, "langgraph.exe" if sys.platform == "win32" else "langgraph")
            
            if os.path.exists(langgraph_path):
                print(f"Found langgraph executable: {langgraph_path}")
                print("Running langgraph dev...")
                print("If the browser doesn't open automatically, please visit: http://localhost:3000")
                subprocess.run([langgraph_path, "dev"], check=True)
            else:
                print(f"langgraph executable not found at {langgraph_path}")
                print("Trying to run with python -m...")
                subprocess.run([sys.executable, "-m", "langgraph_cli.cli", "dev"], check=True)
        except Exception as e:
            print(f"Error running LangGraph development server: {e}")
            print("\nPlease try running one of the following commands manually:")
            print("1. langgraph dev")
            print("2. python -m langgraph_cli.cli dev")
    else:
        print("langgraph-cli is NOT installed")
        print("Please install it with: pip install langgraph-cli[inmem]")

if __name__ == "__main__":
    main() 