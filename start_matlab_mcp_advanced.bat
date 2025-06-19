@echo off
setlocal enabledelayedexpansion
title MATLAB MCP Auto Starter

echo =====================================
echo      MATLAB MCP Auto Starter
echo =====================================
echo.

:: 获取当前脚本目录
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: 检查MATLAB是否已安装
echo [1/6] Checking MATLAB installation...
where matlab >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] MATLAB is not found in PATH.
    echo Please ensure MATLAB is installed and added to system PATH.
    echo.
    pause
    exit /b 1
)
echo [✓] MATLAB found in PATH.

:: 检查Python虚拟环境
echo [2/6] Checking Python virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv
    echo Please run the following command first:
    echo   uv venv
    echo.
    pause
    exit /b 1
)
echo [✓] Virtual environment found.

:: 检查main.py文件
echo [3/6] Checking MCP server file...
if not exist "main.py" (
    echo [ERROR] main.py not found in current directory.
    echo Please ensure you're running this script from the MatlabMCP directory.
    echo.
    pause
    exit /b 1
)
echo [✓] MCP server file found.

:: 检查是否已有MATLAB引擎共享
echo [4/6] Checking for existing MATLAB shared engines...
call .venv\Scripts\activate.bat
python -c "import matlab.engine; names = matlab.engine.find_matlab(); print(f'Found {len(names)} shared engines: {names}')" 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Some MATLAB engines may already be shared.
    echo [INFO] Will proceed with starting new session if needed.
)

:: 创建MATLAB启动脚本
echo [5/6] Preparing MATLAB startup script...
echo matlab.engine.shareEngine; > temp_matlab_startup.m
echo fprintf('MATLAB engine shared successfully!\n'); >> temp_matlab_startup.m
echo fprintf('Engine name: %%s\n', matlab.engine.engineName); >> temp_matlab_startup.m
echo fprintf('You can minimize this MATLAB window but do not close it.\n'); >> temp_matlab_startup.m
echo fprintf('MCP Server is now starting...\n'); >> temp_matlab_startup.m

:: 启动MATLAB并执行引擎分享
echo [INFO] Starting MATLAB and sharing engine...
echo [INFO] This may take 10-30 seconds depending on your system...
start "" matlab -r "run('temp_matlab_startup.m')"

:: 等待MATLAB启动和引擎分享
echo [INFO] Waiting for MATLAB to start and share engine...
timeout /t 20 /nobreak >nul

:: 验证MATLAB引擎是否成功共享
echo [INFO] Verifying MATLAB engine sharing...
python -c "import matlab.engine; names = matlab.engine.find_matlab(); exit(0 if names else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] No shared MATLAB engines detected.
    echo [WARNING] MCP Server may fail to connect.
    echo [WARNING] Make sure MATLAB started successfully and shared its engine.
    echo.
)

:: 启动MCP服务器
echo [6/6] Starting MCP Server...
echo =====================================
echo MCP Server is now running...
echo Press Ctrl+C to stop the server
echo =====================================
echo.

python main.py

:: 清理临时文件
if exist temp_matlab_startup.m del temp_matlab_startup.m

echo.
echo =====================================
echo MCP Server stopped.
echo =====================================
pause 