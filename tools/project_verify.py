#!/usr/bin/env python3
"""
MuJoCo-MCP Project Structure Verification Tool

此脚本检查项目是否完整、结构是否合理
Usage: python tools/project_verify.py
"""

import os
import sys
import importlib
import importlib.util
import pkgutil
import subprocess
import argparse
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, repo_root)


class ProjectVerifier:
    """MuJoCo-MCP Project Structure Verification Tool"""
    
    def __init__(self, verbose: bool = False):
        """
        初始化验证工具
        
        Args:
            verbose: Whether to output detailed information
        """
        self.verbose = verbose
        self.repo_root = repo_root
        self.src_dir = os.path.join(self.repo_root, "src")
        self.tests_dir = os.path.join(self.repo_root, "tests")
        self.examples_dir = os.path.join(self.repo_root, "examples")
        self.results = {
            "success": [],
            "warnings": [],
            "errors": []
        }
    
    def log(self, message: str):
        """
        输出日志信息（仅在verbose模式下）
        
        Args:
            message: 日志消息
        """
        if self.verbose:
            print(message)
    
    def add_result(self, result_type: str, message: str):
        """
        添加检查结果
        
        Args:
            result_type: 结果类型 (success/warning/error)
            message: 结果消息
        """
        self.results[result_type].append(message)
        
        if result_type == "success":
            prefix = "✅ "
        elif result_type == "warning":
            prefix = "⚠️ "
        else:  # error
            prefix = "❌ "
            
        print(f"{prefix} {message}")
    
    def verify_project_structure(self) -> bool:
        """
        验证项目结构
        
        Returns:
            验证是否通过
        """
        print("\n=== 验证项目结构 ===\n")
        
        # 1. 验证必需文件存在
        required_files = [
            "README.md",
            "pyproject.toml",
            "setup.py",
            "src/mujoco_mcp/__init__.py",
            "src/mujoco_mcp/server.py",
            "src/mujoco_mcp/simulation.py"
        ]
        
        for file_path in required_files:
            full_path = os.path.join(self.repo_root, file_path)
            if os.path.exists(full_path):
                self.add_result("success", f"必需文件存在: {file_path}")
            else:
                self.add_result("errors", f"缺少必需文件: {file_path}")
        
        # 2. 验证高级功能文件存在
        advanced_files = [
            "src/mujoco_mcp/enhanced_auth_manager.py",
            "src/mujoco_mcp/enhanced_simulation.py"
        ]
        
        for file_path in advanced_files:
            full_path = os.path.join(self.repo_root, file_path)
            if os.path.exists(full_path):
                self.add_result("success", f"高级功能文件存在: {file_path}")
            else:
                self.add_result("warnings", f"缺少高级功能文件: {file_path}")
        
        # 3. 验证测试文件存在
        test_files = [
            "tests/test_server.py",
            "tests/test_simulation.py",
            "tests/test_enhanced_auth_manager.py"
        ]
        
        for file_path in test_files:
            full_path = os.path.join(self.repo_root, file_path)
            if os.path.exists(full_path):
                self.add_result("success", f"测试文件存在: {file_path}")
            else:
                self.add_result("warnings", f"缺少测试文件: {file_path}")
        
        # 4. 验证示例文件存在
        example_files = [
            "examples/demo.py",
            "examples/comprehensive_llm_example.py"
        ]
        
        for file_path in example_files:
            full_path = os.path.join(self.repo_root, file_path)
            if os.path.exists(full_path):
                self.add_result("success", f"示例文件存在: {file_path}")
            else:
                self.add_result("warnings", f"缺少示例文件: {file_path}")
        
        # 检查目录结构是否符合Python包标准
        self.verify_package_structure()
        
        return len(self.results["errors"]) == 0
    
    def verify_package_structure(self):
        """验证包结构是否符合Python包标准"""
        
        # 检查src布局
        if os.path.exists(self.src_dir):
            self.add_result("success", "项目使用src布局")
        else:
            self.add_result("warnings", "项目未使用src布局")
        
        # 检查__init__.py文件
        src_mujoco_mcp = os.path.join(self.src_dir, "mujoco_mcp")
        if os.path.exists(src_mujoco_mcp):
            init_file = os.path.join(src_mujoco_mcp, "__init__.py")
            if os.path.exists(init_file):
                self.add_result("success", "包含__init__.py文件")
                
                # 检查版本号是否在__init__.py中定义
                with open(init_file, 'r') as f:
                    content = f.read()
                    if "__version__" in content:
                        self.add_result("success", "__init__.py中定义了__version__")
                    else:
                        self.add_result("warnings", "__init__.py中未定义__version__")
            else:
                self.add_result("errors", "缺少mujoco_mcp/__init__.py文件")
    
    def verify_imports(self) -> bool:
        """
        验证是否可以正确导入核心模块
        
        Returns:
            验证是否通过
        """
        print("\n=== 验证模块导入 ===\n")
        
        import_modules = [
            "mujoco_mcp",
            "mujoco_mcp.server",
            "mujoco_mcp.simulation"
        ]
        
        advanced_modules = [
            "mujoco_mcp.enhanced_auth_manager",
            "mujoco_mcp.enhanced_simulation"
        ]
        
        # 检查核心模块
        for module_name in import_modules:
            try:
                module = importlib.import_module(module_name)
                self.add_result("success", f"成功导入模块: {module_name}")
            except ImportError as e:
                self.add_result("errors", f"无法导入模块: {module_name} - {str(e)}")
        
        # 检查高级功能模块
        for module_name in advanced_modules:
            try:
                module = importlib.import_module(module_name)
                self.add_result("success", f"成功导入高级模块: {module_name}")
            except ImportError as e:
                self.add_result("warnings", f"无法导入高级模块: {module_name} - {str(e)}")
        
        return len(self.results["errors"]) == 0
    
    def verify_dependencies(self) -> bool:
        """
        验证依赖项是否正确定义
        
        Returns:
            验证是否通过
        """
        print("\n=== 验证项目依赖 ===\n")
        
        try:
            # 检查pyproject.toml是否存在
            pyproject_path = os.path.join(self.repo_root, "pyproject.toml")
            if not os.path.exists(pyproject_path):
                self.add_result("errors", "缺少pyproject.toml文件")
                return False
            
            # 读取pyproject.toml内容
            with open(pyproject_path, 'r') as f:
                content = f.read()
                
                # 检查必要依赖
                required_deps = ["mujoco", "numpy", "model-context-protocol"]
                for dep in required_deps:
                    if dep in content:
                        self.add_result("success", f"pyproject.toml包含依赖: {dep}")
                    else:
                        self.add_result("errors", f"pyproject.toml缺少依赖: {dep}")
            
            return len(self.results["errors"]) == 0
            
        except Exception as e:
            self.add_result("errors", f"验证依赖项时出错: {str(e)}")
            return False
    
    def run_tests(self) -> bool:
        """
        运行项目测试
        
        Returns:
            测试是否通过
        """
        print("\n=== 运行项目测试 ===\n")
        
        try:
            # 检查测试目录是否存在
            if not os.path.exists(self.tests_dir):
                self.add_result("warnings", "测试目录不存在，跳过测试运行")
                return True
            
            # 使用pytest运行测试
            print("运行测试...")
            result = subprocess.run(
                ["python", "-m", "pytest", self.tests_dir, "-v"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            # 检查测试结果
            if result.returncode == 0:
                self.add_result("success", "All测试通过")
                if self.verbose:
                    print(result.stdout)
                return True
            else:
                self.add_result("errors", "测试失败")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            self.add_result("errors", f"运行测试时出错: {str(e)}")
            return False
    
    def verify_all(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        运行All验证
        
        Returns:
            验证是否通过，详细结果
        """
        structure_ok = self.verify_project_structure()
        imports_ok = self.verify_imports()
        deps_ok = self.verify_dependencies()
        
        # 只有在前面的检查都通过的情况下才运行测试
        if structure_ok and imports_ok and deps_ok:
            tests_ok = self.run_tests()
        else:
            self.add_result("warnings", "由于前面的验证失败，跳过测试运行")
            tests_ok = False
        
        all_ok = structure_ok and imports_ok and deps_ok and tests_ok
        
        # 打印摘要信息
        print("\n=== 验证摘要 ===\n")
        print(f"成功: {len(self.results['success'])}")
        print(f"Warning: {len(self.results['warnings'])}")
        print(f"错误: {len(self.results['errors'])}")
        
        if all_ok:
            print("\n✅ 验证通过：项目结构完整且可正常工作")
        else:
            print("\n⚠️ 验证未完全通过：请查看上面的错误和Warning")
        
        return all_ok, self.results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MuJoCo-MCP项目结构验证工具")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细信息")
    args = parser.parse_args()
    
    # 创建验证器并运行All验证
    verifier = ProjectVerifier(verbose=args.verbose)
    success, _ = verifier.verify_all()
    
    # 设置退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 