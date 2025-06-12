# 🚀 MuJoCo MCP Remote 系统启动指南

## 🔍 问题诊断

从Claude Desktop日志分析发现，当前连接的是**内置模式**MuJoCo MCP服务器，而不是我们的**remote模式**服务器。

### 当前问题：
- Claude Desktop配置指向了错误的服务器
- 缺少MuJoCo Viewer Server进程
- 需要正确的启动顺序

## 🛠️ 完整启动流程

### 第1步：启动MuJoCo Viewer Server
```bash
# 在终端1中运行
cd /Users/robert/workspace/04-code/03-mujoco/mujoco-mcp
python mujoco_viewer_server.py --port 8888
```

**预期输出：**
```
INFO - MuJoCo Viewer Server listening on port 8888
```

### 第2步：验证Viewer Server运行
```bash
# 在另一个终端中测试连接
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 8888))
sock.close()
print('✅ Viewer Server运行正常' if result == 0 else '❌ Viewer Server未运行')
"
```

### 第3步：更新Claude Desktop配置
编辑 `~/.config/claude-desktop/config.json` (或相应位置)：

```json
{
  "mcpServers": {
    "mujoco-mcp-remote": {
      "command": "/opt/miniconda3/bin/python",
      "args": ["/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp/mcp_server_remote.py"],
      "cwd": "/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp/src",
        "MUJOCO_MCP_LOG_LEVEL": "INFO",
        "PATH": "/opt/miniconda3/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      }
    }
  }
}
```

### 第4步：重启Claude Desktop
```bash
# 完全退出Claude Desktop
# Cmd + Q (在macOS上)

# 等待5秒

# 重新启动Claude Desktop
```

### 第5步：验证连接
在Claude Desktop中运行：
```
What MCP servers are available?
```

应该显示 `mujoco-mcp-remote` 服务器。

## 🤖 自动化启动脚本

我将创建一个自动化启动脚本来简化这个过程。

## 🔧 故障排除

### 常见问题：

1. **端口被占用**
   ```bash
   lsof -i :8888
   # 如果有进程占用，杀死它或使用不同端口
   ```

2. **Python路径错误**
   ```bash
   which python
   # 确保使用正确的Python解释器
   ```

3. **权限问题**
   ```bash
   chmod +x mujoco_viewer_server.py
   chmod +x mcp_server_remote.py
   ```

4. **依赖缺失**
   ```bash
   pip install mujoco mcp numpy
   ```

## 📊 系统状态检查

运行以下命令检查系统状态：
```bash
python debug_mcp_inspector.py
# 选择选项3检查Viewer Server状态
```

## 🎯 预期行为

启动成功后，您应该能够：
1. 创建物理场景（pendulum, robotic_arm等）
2. 看到MuJoCo GUI窗口自动弹出
3. 实时控制和观察仿真
4. 使用自然语言命令控制系统