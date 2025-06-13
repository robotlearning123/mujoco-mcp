# MuJoCo-MCP 发布前检查清单

## 1. 代码质量检查 ✓

- [ ] 所有测试通过 (`pytest`)
- [ ] 代码覆盖率 > 80%
- [ ] 无 linting 错误 (`ruff check .`)
- [ ] 类型检查通过 (`mypy src/`)
- [ ] 安全扫描通过 (`bandit -r src/`)

## 2. 包结构检查 ✓

- [ ] `pyproject.toml` 完整且正确
- [ ] `README.md` 包含完整安装和使用说明
- [ ] `LICENSE` 文件存在（MIT）
- [ ] `CHANGELOG.md` 更新到最新版本
- [ ] 版本号正确（遵循语义化版本）

## 3. 依赖管理 ✓

- [ ] 最小化依赖列表
- [ ] 依赖版本范围合理
- [ ] 无冲突依赖
- [ ] Python 版本要求明确 (>=3.10)

## 4. 本地测试 ✓

```bash
# 运行所有测试脚本
chmod +x test_local_install.sh
./test_local_install.sh

python test_mcp_compliance.py
python test_e2e_integration.py
python test_performance_benchmark.py
```

## 5. 多环境测试 ✓

### 操作系统
- [ ] macOS (12.0+)
- [ ] Windows 10/11
- [ ] Ubuntu 20.04/22.04

### Python 版本
- [ ] Python 3.10
- [ ] Python 3.11
- [ ] Python 3.12

### MCP 客户端
- [ ] Claude Desktop
- [ ] Cursor
- [ ] MCP Inspector

## 6. 安装测试 ✓

```bash
# 创建干净虚拟环境
python -m venv test_clean
source test_clean/bin/activate

# 从 wheel 安装
pip install dist/mujoco_mcp-*.whl

# 测试基本功能
python -m mujoco_mcp --version
python -c "import mujoco_mcp; print(mujoco_mcp.__version__)"

# 测试 MCP 服务器启动
timeout 10 python -m mujoco_mcp
```

## 7. 文档检查 ✓

- [ ] API 文档完整
- [ ] 使用示例清晰
- [ ] 配置说明详细
- [ ] 故障排查指南
- [ ] 贡献指南

## 8. 性能验证 ✓

- [ ] 启动时间 < 5秒
- [ ] 工具响应时间 < 100ms (p95)
- [ ] 内存使用 < 500MB
- [ ] CPU 使用合理

## 9. MCP 协议合规性 ✓

- [ ] JSON-RPC 2.0 格式正确
- [ ] 标准错误码实现
- [ ] 能力协商正确
- [ ] MCP Inspector 兼容

## 10. 发布准备 ✓

- [ ] 创建 GitHub Release
- [ ] 更新版本标签
- [ ] 准备发布说明
- [ ] 测试 TestPyPI 上传

```bash
# 构建发布包
python -m build

# 上传到 TestPyPI
python -m twine upload --repository testpypi dist/*

# 测试从 TestPyPI 安装
pip install -i https://test.pypi.org/simple/ mujoco-mcp
```

## 11. 最终确认 ✓

- [ ] 所有检查项完成
- [ ] 团队成员审核通过
- [ ] 无已知严重问题
- [ ] 准备正式发布

## 发布命令

```bash
# 正式发布到 PyPI
python -m twine upload dist/*

# 验证安装
pip install mujoco-mcp
```

---

**注意**: 发布前必须完成所有检查项！