# 🍎 MuJoCo macOS 官方使用指南

基于官方文档整理：https://mujoco.readthedocs.io/en/stable/python.html

## ✅ 官方确认的要求

### 1. **必须使用 mjpython**
根据官方文档：
> "On macOS, `launch_passive` requires that the user script is executed via a special `mjpython` launcher."

### 2. **必须调用 viewer.sync()**
自 v2.3.4 起：
> "The user now needs to call viewer.sync() in order for the viewer to pick up changes to the physics state."

### 3. **必须使用 viewer.lock()**
> "User code must ensure that it is holding the viewer lock before modifying any physics or visualization state."

## 🚀 正确的启动方式

### 方式1：直接运行
```bash
mjpython mujoco_viewer_server.py --port 8888
```

### 方式2：使用macOS专用脚本
```bash
./start_macos.sh
```

## 📋 检查清单

1. **确认 mjpython 已安装**：
   ```bash
   which mjpython
   # 应该显示: /opt/miniconda3/bin/mjpython
   ```

2. **确认 MuJoCo Python 包已安装**：
   ```bash
   pip show mujoco
   ```

3. **运行测试**：
   ```bash
   mjpython -c "import mujoco; print(mujoco.__version__)"
   ```

## ⚠️ 常见错误

### ❌ 错误：使用普通 python
```bash
python mujoco_viewer_server.py  # 在macOS上会失败
```

### ✅ 正确：使用 mjpython
```bash
mjpython mujoco_viewer_server.py  # 正确方式
```

## 🔧 环境变量（可选）

如果遇到渲染问题，可以设置：
```bash
export MUJOCO_GL=glfw
```

## 📚 官方文档链接

- [MuJoCo Python文档](https://mujoco.readthedocs.io/en/stable/python.html)
- [launch_passive API](https://github.com/google-deepmind/mujoco/discussions/780)
- [macOS特定说明](https://mujoco.readthedocs.io/en/stable/programming/index.html)

## 💡 提示

mjpython 是 python 的完全替代品，支持所有相同的命令行参数：
- 运行脚本：`mjpython script.py`
- 启动IPython：`mjpython -m IPython`
- 安装包：`mjpython -m pip install package`