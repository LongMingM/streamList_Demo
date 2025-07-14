@echo off
echo 正在启动数据可视化与机器学习预测分析工具...

REM 检查Docker是否安装
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker未安装或未运行，请先安装Docker并确保其正常运行。
    pause
    exit /b
)

REM 构建并启动Docker容器
echo 正在构建并启动Docker容器...
docker build -t streamlit-ml-app .
if %errorlevel% neq 0 (
    echo Docker镜像构建失败，请检查错误信息。
    pause
    exit /b
)

echo 正在启动应用...
docker run -d -p 8501:8501 --name streamlit-ml-app streamlit-ml-app
if %errorlevel% neq 0 (
    echo 容器启动失败，可能是端口被占用或容器已存在。
    echo 尝试停止并移除旧容器...
    docker stop streamlit-ml-app >nul 2>&1
    docker rm streamlit-ml-app >nul 2>&1
    echo 重新启动容器...
    docker run -d -p 8501:8501 --name streamlit-ml-app streamlit-ml-app
)

echo 应用已启动，正在打开浏览器...
timeout /t 3 >nul
start http://localhost:8501

echo 应用运行中，请勿关闭此窗口。按任意键停止应用...
pause

echo 正在停止并删除容器...
docker stop streamlit-ml-app
docker rm streamlit-ml-app

echo 应用已停止。
pause 