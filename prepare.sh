#!/bin/bash

# 定义目录变量
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PROJECT_ROOT="$DIR"
export PYTHONPATH="$PROJECT_ROOT/models:$PYTHONPATH"

# 更新系统包并安装必要的 Python 开发包
apt-get update && apt-get install -y pybind11-dev || {
    echo "安装 pybind11-dev 失败。"
    exit 1
}

# 升级 pip 和安装 Python 依赖
pip3 install --upgrade pip wheel || {
    echo "pip 升级失败。"
    exit 1
}
pip3 install transformers_stream_generator einops tiktoken accelerate gradio transformers==4.45.2 pybind11[global] dfss gradio || {
    echo "Python 包安装失败。"
    exit 1
}

# 构建模型
pushd "$PROJECT_ROOT/models" || {
    echo "无法切换到模型目录。"
    exit 1
}

# 使用 find 命令查找并构建项目
find . -type d -name "python_demo" -print0 | while IFS= read -r -d '' demo_dir; do
    build_dir="$demo_dir/build"
    mkdir -p "$build_dir"
    pushd "$build_dir" || {
        echo "无法切换到构建目录 $build_dir。"
        continue
    }
    cmake .. && make || {
        echo "构建失败在 $build_dir。"
        popd
        continue
    }
    cp *cpython* .. || echo "复制构建文件失败。"
    popd
done
