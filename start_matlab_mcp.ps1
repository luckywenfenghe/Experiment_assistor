# MATLAB MCP Auto Starter (PowerShell版本)
Write-Host "====================================="
Write-Host "    MATLAB MCP Auto Starter (PS)"
Write-Host "====================================="
Write-Host ""

# 设置执行策略（如果需要）
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force -ErrorAction SilentlyContinue
} catch {
    Write-Warning "Could not set execution policy. This may cause issues."
}

# 获取脚本目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 检查MATLAB是否已安装
Write-Host "[1/6] Checking MATLAB installation..."
$matlabPath = Get-Command matlab -ErrorAction SilentlyContinue
if ($null -eq $matlabPath) {
    Write-Host "[ERROR] MATLAB is not found in PATH." -ForegroundColor Red
    Write-Host "Please ensure MATLAB is installed and added to system PATH." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[✓] MATLAB found in PATH." -ForegroundColor Green

# 检查Python虚拟环境
Write-Host "[2/6] Checking Python virtual environment..."
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found at .venv" -ForegroundColor Red
    Write-Host "Please run the following command first:" -ForegroundColor Red
    Write-Host "  uv venv" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[✓] Virtual environment found." -ForegroundColor Green

# 检查main.py文件
Write-Host "[3/6] Checking MCP server file..."
if (-not (Test-Path "main.py")) {
    Write-Host "[ERROR] main.py not found in current directory." -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the MatlabMCP directory." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[✓] MCP server file found." -ForegroundColor Green

# 激活虚拟环境
Write-Host "[4/6] Activating virtual environment..."
& ".venv\Scripts\Activate.ps1"

# 检查是否已有MATLAB引擎共享
Write-Host "[5/6] Checking for existing MATLAB shared engines..."
try {
    $result = python -c "import matlab.engine; names = matlab.engine.find_matlab(); print(f'Found {len(names)} shared engines')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] $result" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[INFO] Could not check existing engines, will proceed anyway." -ForegroundColor Yellow
}

# 创建MATLAB启动脚本
Write-Host "[INFO] Preparing MATLAB startup script..."
@"
matlab.engine.shareEngine;
fprintf('MATLAB engine shared successfully!\n');
fprintf('Engine name: %s\n', matlab.engine.engineName);
fprintf('You can minimize this MATLAB window but do not close it.\n');
fprintf('MCP Server is now starting...\n');
"@ | Out-File -FilePath "temp_matlab_startup.m" -Encoding ASCII

# 启动MATLAB并执行引擎分享
Write-Host "[INFO] Starting MATLAB and sharing engine..."
Write-Host "[INFO] This may take 10-30 seconds depending on your system..."
Start-Process matlab -ArgumentList "-r", "run('temp_matlab_startup.m')" -WindowStyle Normal

# 等待MATLAB启动和引擎分享
Write-Host "[INFO] Waiting for MATLAB to start and share engine..."
Start-Sleep -Seconds 20

# 验证MATLAB引擎是否成功共享
Write-Host "[INFO] Verifying MATLAB engine sharing..."
try {
    python -c "import matlab.engine; names = matlab.engine.find_matlab(); exit(0 if names else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] No shared MATLAB engines detected." -ForegroundColor Yellow
        Write-Host "[WARNING] MCP Server may fail to connect." -ForegroundColor Yellow
        Write-Host "[WARNING] Make sure MATLAB started successfully and shared its engine." -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "[WARNING] Could not verify engine sharing." -ForegroundColor Yellow
}

# 启动MCP服务器
Write-Host "[6/6] Starting MCP Server..."
Write-Host "====================================="
Write-Host "MCP Server is now running..."
Write-Host "Press Ctrl+C to stop the server"
Write-Host "====================================="
Write-Host ""

try {
    python main.py
} finally {
    # 清理临时文件
    if (Test-Path "temp_matlab_startup.m") {
        Remove-Item "temp_matlab_startup.m" -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "====================================="
Write-Host "MCP Server stopped."
Write-Host "====================================="
Read-Host "Press Enter to exit" 