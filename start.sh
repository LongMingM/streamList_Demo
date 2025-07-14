#!/bin/bash

echo "正在启动数据可视化与机器学习预测分析工具..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "Docker未安装或未运行，请先安装Docker并确保其正常运行。"
    exit 1
fi

# 构建并启动Docker容器
echo "正在构建Docker镜像..."
docker build -t streamlit-ml-app .
if [ $? -ne 0 ]; then
    echo "Docker镜像构建失败，请检查错误信息。"
    exit 1
fi

echo "正在启动应用..."
docker run -d -p 8501:8501 --name streamlit-ml-app streamlit-ml-app
if [ $? -ne 0 ]; then
    echo "容器启动失败，可能是端口被占用或容器已存在。"
    echo "尝试停止并移除旧容器..."
    docker stop streamlit-ml-app &> /dev/null
    docker rm streamlit-ml-app &> /dev/null
    echo "重新启动容器..."
    docker run -d -p 8501:8501 --name streamlit-ml-app streamlit-ml-app
fi

echo "应用已启动，请访问 http://localhost:8501"

# 尝试自动打开浏览器
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501
elif command -v open &> /dev/null; then
    open http://localhost:8501
fi

echo "按 Ctrl+C 停止应用..."
trap "docker stop streamlit-ml-app; docker rm streamlit-ml-app; echo '应用已停止。'; exit 0" INT

# 保持脚本运行
while true; do
    sleep 1
done 