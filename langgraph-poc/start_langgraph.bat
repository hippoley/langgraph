@echo off
chcp 65001 > nul
title LangGraph All-in-One Tool

echo === LangGraph All-in-One Tool ===
echo === English: This tool helps you install dependencies, clean temporary files and start LangGraph server ===
echo === 中文: 此工具帮助您安装依赖、清理临时文件并启动LangGraph服务器 ===
echo.

:MENU
echo === MENU / 菜单 ===
echo 1. Install dependencies / 安装依赖
echo 2. Clean temporary files / 清理临时文件
echo 3. Start server / 启动服务器
echo 4. Execute all operations / 执行所有操作
echo 0. Exit / 退出
echo.

set /p choice=Please enter option number / 请输入选项数字 [0-4]: 

if "%choice%"=="0" goto EXIT
if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto CLEAN
if "%choice%"=="3" goto START
if "%choice%"=="4" goto ALL
echo Invalid option, please try again / 无效选项，请重新选择
goto MENU

:INSTALL
echo.
echo === Installing Dependencies / 安装依赖 ===
echo.

echo === English: Installing packages from requirements.txt... ===
echo === 中文: 正在从requirements.txt安装依赖包... ===
pip install -r requirements.txt

echo.
echo === English: Installing LangGraph CLI and API... ===
echo === 中文: 正在安装LangGraph命令行工具和API... ===
pip install -U "langgraph-cli[inmem]" --trusted-host pypi.org --trusted-host files.pythonhosted.org

echo.
echo === English: Installing additional required packages... ===
echo === 中文: 正在安装额外所需的依赖包... ===
pip install colorama langchain_core langchain_openai

echo.
echo === English: All dependencies installed successfully! ===
echo === 中文: 所有依赖安装完成! ===
echo.
pause
goto MENU

:CLEAN
echo.
echo === Cleaning Temporary Files / 清理临时文件 ===
echo.

echo === English: Removing LangGraph temporary files... ===
echo === 中文: 正在删除LangGraph临时文件... ===

if exist ".langgraph_api\*.pckl.tmp" (
    del /q /f ".langgraph_api\*.pckl.tmp" 2>nul
    echo === English: Temporary files deleted ===
    echo === 中文: 临时文件已删除 ===
) else (
    echo === English: No temporary files found ===
    echo === 中文: 没有找到临时文件 ===
)

if exist ".langgraph_api\*.lock" (
    del /q /f ".langgraph_api\*.lock" 2>nul
    echo === English: Lock files deleted ===
    echo === 中文: 锁定文件已删除 ===
) else (
    echo === English: No lock files found ===
    echo === 中文: 没有找到锁定文件 ===
)

if exist ".langgraph\*.pckl.tmp" (
    del /q /f ".langgraph\*.pckl.tmp" 2>nul
    echo === English: Temporary files deleted ===
    echo === 中文: 临时文件已删除 ===
)

if exist ".langgraph\*.lock" (
    del /q /f ".langgraph\*.lock" 2>nul
    echo === English: Lock files deleted ===
    echo === 中文: 锁定文件已删除 ===
)

if exist ".langgraph_cli\*.pckl.tmp" (
    del /q /f ".langgraph_cli\*.pckl.tmp" 2>nul
    echo === English: Temporary files deleted ===
    echo === 中文: 临时文件已删除 ===
)

if exist ".langgraph_cli\*.lock" (
    del /q /f ".langgraph_cli\*.lock" 2>nul
    echo === English: Lock files deleted ===
    echo === 中文: 锁定文件已删除 ===
)

echo.
echo === English: Cleanup completed! ===
echo === 中文: 清理完成! ===
echo.
pause
goto MENU

:START
echo.
echo === Starting LangGraph Server / 启动LangGraph服务器 ===
echo.

echo === English: Setting up CORS environment variables for cross-origin access... ===
echo === 中文: 正在设置CORS环境变量以允许跨域访问... ===
set LANGGRAPH_API_CORS_ALLOW_ORIGINS=*
set LANGGRAPH_API_CORS_ALLOW_CREDENTIALS=true
set LANGGRAPH_API_CORS_ALLOW_METHODS=*
set LANGGRAPH_API_CORS_ALLOW_HEADERS=*
set LANGGRAPH_API_PORT=2024
set LANGGRAPH_STUDIO_PORT=3000

echo === English: CORS environment variables set successfully ===
echo === 中文: CORS环境变量设置成功 ===
echo.

echo === English: Attempting to start LangGraph server... ===
echo === 中文: 正在尝试启动LangGraph服务器... ===
echo.

echo === Method 1: Using langgraph-cli / 方法1: 使用langgraph-cli ===
langgraph dev

if %ERRORLEVEL% NEQ 0 (
    echo === English: Method 1 failed, trying alternative method... ===
    echo === 中文: 方法1失败，尝试备用方法... ===
    echo.
    
    echo === Method 2: Using Python module / 方法2: 使用Python模块 ===
    python -m langgraph.cli dev
    
    if %ERRORLEVEL% NEQ 0 (
        echo === English: Method 2 failed, trying final method... ===
        echo === 中文: 方法2失败，尝试最后方法... ===
        echo.
        
        echo === Method 3: Using Python script / 方法3: 使用Python脚本 ===
        start cmd /c "python langgraph_fix.py --start"
        
        echo === English: Server started in a new window ===
        echo === 中文: 服务器已在新窗口中启动 ===
        echo === English: Please visit https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 to access Studio UI ===
        echo === 中文: 请访问 https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 查看Studio UI ===
    )
)

echo.
pause
goto MENU

:ALL
echo.
echo === Executing All Operations / 执行所有操作 ===
echo === English: This will install dependencies, clean files and start the server ===
echo === 中文: 这将安装依赖、清理文件并启动服务器 ===
echo.

call :INSTALL
call :CLEAN
call :START
goto MENU

:EXIT
echo.
echo === English: Thank you for using LangGraph All-in-One Tool! ===
echo === 中文: 感谢使用LangGraph一体化工具! ===
echo.
exit /b 0

netstat -ano | findstr :8080
netstat -ano | findstr :3001 