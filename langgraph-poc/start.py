import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, shell=True, capture_output=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            stdout=subprocess.PIPE if capture_output else None, 
            stderr=subprocess.PIPE if capture_output else None,
            text=True
        )
        return result
    except Exception as e:
        print(f"Error executing command: {e}")
        # Create a mock CompletedProcess object with error code
        class FailedProcess:
            def __init__(self):
                self.returncode = 1
                self.stdout = ""
                self.stderr = str(e)
        return FailedProcess()

def install_dependencies():
    """Install all dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt file not found")
        return False
    
    try:
        # Install dependencies from requirements.txt, disable SSL verification
        print("Installing dependencies, SSL verification disabled...")
        result = run_command("pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org")
        if result.returncode != 0:
            print("Failed to install dependencies")
            return False
        
        # Ensure langgraph-cli is installed, disable SSL verification
        print("Installing langgraph-cli...")
        result = run_command("pip install langgraph-cli --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org")
        if result.returncode != 0:
            print("Failed to install langgraph-cli")
            return False
            
        # Install langgraph-cli[inmem], which includes langgraph-api, disable SSL verification
        print("Installing langgraph-api...")
        result = run_command('pip install -U "langgraph-cli[inmem]" --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org')
        if result.returncode != 0:
            # If installing langgraph-cli[inmem] fails, try installing langgraph-api directly
            print("Failed to install langgraph-cli[inmem], trying to install langgraph-api directly...")
            result = run_command('pip install langgraph-api --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org')
            if result.returncode != 0:
                print("Failed to install langgraph-api")
                return False
        
        # Verify langgraph-api is installed
        verify_result = run_command("pip show langgraph-api", capture_output=True)
        if verify_result.returncode != 0:
            print("Failed to install langgraph-api, although the installation command seemed to succeed")
            print("Please try manually running: pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org langgraph-api")
            return False
        else:
            print("langgraph-api installation successfully verified")
        
        print("All dependencies installed successfully")
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False

def start_langgraph():
    """Start the LangGraph development server"""
    print("\nStarting LangGraph development server...")
    
    # Check if langgraph-api is installed
    result = run_command("pip show langgraph-api", capture_output=True)
    if result.returncode != 0:
        print("langgraph-api not installed, installing now...")
        result = run_command('pip install -U "langgraph-cli[inmem]" --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org')
        
        # If installing langgraph-cli[inmem] fails, try installing langgraph-api directly
        if result.returncode != 0:
            print("Failed to install langgraph-cli[inmem], trying to install langgraph-api directly...")
            result = run_command('pip install langgraph-api --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org')
        
        # Verify installation again
        verify_result = run_command("pip show langgraph-api", capture_output=True)
        if verify_result.returncode != 0:
            print("Failed to install langgraph-api, cannot start LangGraph development server")
            print("Please try manually running: pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org langgraph-api")
            return False
        else:
            print("langgraph-api installed successfully")
    
    # Get Python's Scripts directory
    scripts_dir = os.path.join(sys.prefix, "Scripts")
    langgraph_path = os.path.join(scripts_dir, "langgraph.exe" if sys.platform == "win32" else "langgraph")
    
    # Try running langgraph command with full path
    if os.path.exists(langgraph_path):
        print(f"Found langgraph executable: {langgraph_path}")
        try:
            # Display output directly in console, don't capture
            result = run_command(f'"{langgraph_path}" dev', capture_output=False)
            if result.returncode == 0:
                return True
            print("Failed to run langgraph command with full path")
        except Exception as e:
            print(f"Error executing langgraph command: {e}")
    
    # Try running using python -m method
    print("Trying to run using python -m method...")
    try:
        # Display output directly in console, don't capture
        result = run_command(f"{sys.executable} -m langgraph_cli.cli dev", capture_output=False)
        if result.returncode == 0:
            return True
        print("Failed to run using python -m method")
    except Exception as e:
        print(f"Error running with python -m method: {e}")
    
    # Try running langgraph command directly
    try:
        # Display output directly in console, don't capture
        result = run_command("langgraph dev", capture_output=False)
        if result.returncode == 0:
            return True
        print("Failed to run langgraph command directly")
    except Exception as e:
        print(f"Error running langgraph command directly: {e}")
    
    # If all attempts fail
    print("\nCannot start LangGraph development server.")
    print("Please try the following steps:")
    print("1. Close and reopen your terminal/command prompt")
    print('2. Manually run: pip install -U "langgraph-cli[inmem]" --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org')
    print("3. Then run: langgraph dev")
    print(f"\nPython Scripts directory: {scripts_dir}")
    print("Please ensure this directory is added to your PATH environment variable")
    return False

def run_python_file(file_name):
    """Run the specified Python file"""
    print(f"\nRunning {file_name}...")
    success, output = run_command(f"{sys.executable} {file_name}", capture_output=False)
    if not success:
        print(f"Failed to run {file_name}")
    return success

def main():
    """Main function"""
    print("LangGraph POC Startup Tool")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        input("Press Enter to exit...")
        return
    
    # Display menu
    while True:
        print("\nPlease select an operation:")
        print("1. Start LangGraph development server (langgraph dev)")
        print("2. Run simple agent (simple_agent.py)")
        print("3. Run prebuilt agent (simple_agent_prebuilt.py)")
        print("4. Run advanced agent (advanced_agent.py)")
        print("5. Run multi-agent system (multi_agent_system.py)")
        print("6. Run streaming and breakpoints example (streaming_breakpoints.py)")
        print("0. Exit")
        
        choice = input("\nEnter option number: ")
        
        if choice == "1":
            start_langgraph()
            # Exit loop after starting server, as it will occupy the console
            break
        elif choice == "2":
            run_python_file("simple_agent.py")
        elif choice == "3":
            run_python_file("simple_agent_prebuilt.py")
        elif choice == "4":
            run_python_file("advanced_agent.py")
        elif choice == "5":
            run_python_file("multi_agent_system.py")
        elif choice == "6":
            run_python_file("streaming_breakpoints.py")
        elif choice == "0":
            print("Thank you for using LangGraph POC!")
            break
        else:
            print("Invalid option, please try again.")
        
        # Pause after each operation
        if choice not in ["0", "1"]:
            input("\nOperation complete, press Enter to continue...")

if __name__ == "__main__":
    main() 