@echo off
echo Starting MATLAB MCP System...
echo.

:: 检查MATLAB是否已安装
where matlab >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: MATLAB is not found in PATH. Please ensure MATLAB is installed and added to PATH.
    pause
    exit /b 1
)

:: 获取当前脚本目录
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: 检查Python虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv
    echo Please run: uv venv
    pause
    exit /b 1
)

:: 创建MATLAB启动脚本
echo Creating MATLAB startup script...
echo matlab.engine.shareEngine > temp_matlab_startup.m
echo fprintf('MATLAB engine shared successfully!\n'); >> temp_matlab_startup.m
echo fprintf('You can now close this MATLAB window or keep it running.\n'); >> temp_matlab_startup.m

:: 启动MATLAB并执行引擎分享
echo Starting MATLAB and sharing engine...
start "" matlab -r "run('temp_matlab_startup.m')"

:: 等待MATLAB启动和引擎分享
echo Waiting for MATLAB to start and share engine...
timeout /t 15 /nobreak

:: 激活虚拟环境并启动MCP服务器
echo Starting MCP Server...
call .venv\Scripts\activate.bat
python main.py

:: 清理临时文件
if exist temp_matlab_startup.m del temp_matlab_startup.m

echo.
echo MCP Server stopped.
pause 