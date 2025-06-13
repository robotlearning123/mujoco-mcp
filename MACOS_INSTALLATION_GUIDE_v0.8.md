# MuJoCo-MCP v0.8 macOS 安装指南

## 快速安装

### 1. 安装包
\```bash
cd /path/to/mujoco-mcp
pip install -e .
\```

### 2. 配置Claude Desktop

编辑Claude Desktop配置文件：
\```bash
# 配置文件路径 (选择一个)
~/Library/Application Support/Claude/claude_desktop_config.json
# 或
~/.config/claude/claude_desktop_config.json
\```

添加以下配置：
\```json
{
  "mcpServers": {
    "mujoco-mcp": {
      "command": "/opt/miniconda3/bin/python",
      "args": ["-m", "mujoco_mcp"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
\```

**重要**: 请将`/opt/miniconda3/bin/python`替换为你的实际Python路径。
获取Python路径：
\```bash
which python
\```

### 3. 重启Claude Desktop

### 4. 测试
在Claude Desktop中输入：
\```
Create a pendulum simulation
\```

## 故障排除

### 常见问题
1. **Python路径错误**: 使用`which python`获取正确路径
2. **包未安装**: 确保在项目目录运行`pip install -e .`
3. **MuJoCo未安装**: 运行`pip install mujoco`

### 调试
查看Claude Desktop日志：
\```bash
tail -f ~/Library/Logs/Claude/mcp-server-mujoco-mcp.log
\```
