#!/bin/bash
# MuJoCo-MCP 安装脚本
# 此脚本帮助用户安装MuJoCo-MCP及其依赖项

set -e  # 遇到错误时终止脚本

# 显示彩色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印信息
echo -e "${GREEN}=== MuJoCo-MCP 安装脚本 ===${NC}"
echo ""
echo "此脚本将：
1. 检查Python环境
2. 安装MuJoCo依赖
3. 安装MCP (Model Context Protocol)
4. 以开发模式安装MuJoCo-MCP
"

# 确认是否继续
read -p "是否继续安装? [Y/n] " response
response=${response:-Y}  # 默认为Y
if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "安装已取消"
    exit 0
fi

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}=== 检查Python环境 ===${NC}"
# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "检测到Python版本: $python_version"

# 检查Python版本是否>=3.8
if [[ $(echo "$python_version" | awk -F. '{ print ($1 >= 3 && $2 >= 8) }') != 1 ]]; then
    echo -e "${RED}错误: 需要Python 3.8或更高版本${NC}"
    exit 1
fi

# 检查pip是否已安装
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}错误: 未找到pip3${NC}"
    echo "请安装pip: https://pip.pypa.io/en/stable/installation/"
    exit 1
fi

echo -e "${GREEN}=== 创建虚拟环境 ===${NC}"
echo -e "${YELLOW}推荐在虚拟环境中安装${NC}"

# 询问是否创建虚拟环境
read -p "创建虚拟环境? [Y/n] " create_venv
create_venv=${create_venv:-Y}  # 默认为Y

if [[ "$create_venv" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    # 检查venv模块
    python3 -c "import venv" 2>/dev/null || { 
        echo -e "${RED}错误: Python venv模块未安装${NC}"; 
        echo "请先安装venv模块"; 
        exit 1; 
    }
    
    # 创建虚拟环境
    echo "在 $REPO_ROOT/venv 创建虚拟环境..."
    python3 -m venv "$REPO_ROOT/venv"
    
    # 激活虚拟环境
    echo "激活虚拟环境..."
    source "$REPO_ROOT/venv/bin/activate"
    
    echo -e "${GREEN}虚拟环境已创建并激活${NC}"
else
    echo "跳过创建虚拟环境"
fi

echo -e "${GREEN}=== 升级pip和安装wheel ===${NC}"
pip3 install --upgrade pip wheel

echo -e "${GREEN}=== 安装MuJoCo依赖 ===${NC}"
echo "安装MuJoCo..."
pip3 install mujoco>=2.3.0

# 检查MuJoCo是否安装成功
if python3 -c "import mujoco; print(f'MuJoCo {mujoco.__version__} 已安装')" &>/dev/null; then
    echo -e "${GREEN}MuJoCo安装成功${NC}"
else
    echo -e "${RED}Warning: MuJoCo安装可能有问题${NC}"
    echo "请参考MuJoCo文档: https://github.com/deepmind/mujoco"
fi

echo -e "${GREEN}=== 安装Model Context Protocol (MCP) ===${NC}"
pip3 install model-context-protocol>=0.1.0

# 检查MCP是否安装成功
if python3 -c "import mcp; print('MCP已安装')" &>/dev/null; then
    echo -e "${GREEN}MCP安装成功${NC}"
else
    echo -e "${RED}Warning: MCP安装可能有问题${NC}"
fi

echo -e "${GREEN}=== 安装MuJoCo-MCP ===${NC}"
cd "$REPO_ROOT"
pip3 install -e .

# 检查MuJoCo-MCP是否安装成功
if python3 -c "import mujoco_mcp; print('MuJoCo-MCP已安装')" &>/dev/null; then
    echo -e "${GREEN}MuJoCo-MCP安装成功${NC}"
else
    echo -e "${RED}Warning: MuJoCo-MCP安装可能有问题${NC}"
fi

echo -e "${GREEN}=== 安装可选依赖 ===${NC}"
# 询问是否安装Anthropic API
read -p "安装Anthropic API用于LLM示例? [Y/n] " install_anthropic
install_anthropic=${install_anthropic:-Y}  # 默认为Y

if [[ "$install_anthropic" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    pip3 install anthropic
    echo -e "${GREEN}Anthropic API已安装${NC}"
else
    echo "跳过安装Anthropic API"
fi

echo -e "${GREEN}=== 运行验证 ===${NC}"
# 询问是否运行验证脚本
read -p "运行项目验证脚本? [Y/n] " run_verify
run_verify=${run_verify:-Y}  # 默认为Y

if [[ "$run_verify" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    python3 "$REPO_ROOT/tools/project_verify.py"
else
    echo "跳过项目验证"
fi

echo ""
echo -e "${GREEN}=== 安装完成 ===${NC}"
echo ""
echo "要开始使用MuJoCo-MCP，请尝试运行演示："
echo "python $REPO_ROOT/examples/demo.py"
echo ""
echo "或者运行LLM集成示例："
echo "python $REPO_ROOT/examples/comprehensive_llm_example.py"
echo ""

if [[ "$create_venv" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "注意: 虚拟环境已激活。要在新的终端中使用，请运行:"
    echo "source $REPO_ROOT/venv/bin/activate"
fi 