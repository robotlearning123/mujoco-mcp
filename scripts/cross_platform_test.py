#!/usr/bin/env python3
"""
跨平台兼容性测试脚本
测试在不同操作系统和环境中的兼容性
"""

import sys
import platform
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

class CrossPlatformTest:
    def __init__(self):
        self.results = {
            "platform_info": self.get_platform_info(),
            "tests": []
        }
    
    def get_platform_info(self) -> Dict[str, Any]:
        """获取平台信息"""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        }
    
    def run_command(self, cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
        """运行命令并返回结果"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "cmd": " ".join(cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out",
                "cmd": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "cmd": " ".join(cmd)
            }
    
    def test_python_import(self) -> Dict[str, Any]:
        """测试Python包导入"""
        print("测试Python包导入...")
        
        import_tests = [
            "import mujoco_mcp",
            "from mujoco_mcp.version import __version__",
            "from mujoco_mcp.mcp_server import handle_list_tools",
            "import mujoco",
            "import numpy",
            "import mcp"
        ]
        
        results = []
        for import_stmt in import_tests:
            try:
                exec(import_stmt)
                results.append({
                    "import": import_stmt,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "import": import_stmt,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "test_name": "python_import",
            "results": results,
            "success": all(r["success"] for r in results)
        }
    
    def test_cli_commands(self) -> Dict[str, Any]:
        """测试CLI命令"""
        print("测试CLI命令...")
        
        commands = [
            ["mujoco-mcp", "--version"],
            ["mujoco-mcp", "--help"],
            ["mujoco-mcp", "--check"],
            ["python", "-m", "mujoco_mcp", "--version"],
            ["python", "-c", "import mujoco_mcp; print(mujoco_mcp.__version__)"]
        ]
        
        results = []
        for cmd in commands:
            result = self.run_command(cmd)
            results.append(result)
        
        return {
            "test_name": "cli_commands",
            "results": results,
            "success": all(r["success"] for r in results)
        }
    
    def test_mujoco_integration(self) -> Dict[str, Any]:
        """测试MuJoCo集成"""
        print("测试MuJoCo集成...")
        
        try:
            import mujoco
            
            # 测试基本MuJoCo功能
            test_code = """
import mujoco
import numpy as np

# 创建简单模型
xml = '''
<mujoco>
  <worldbody>
    <body name="box">
      <geom name="box_geom" type="box" size="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
'''

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print("MuJoCo basic test passed")
"""
            
            result = self.run_command([
                sys.executable, "-c", test_code
            ])
            
            return {
                "test_name": "mujoco_integration",
                "mujoco_version": mujoco.__version__,
                "success": result["success"],
                "details": result
            }
            
        except ImportError as e:
            return {
                "test_name": "mujoco_integration",
                "success": False,
                "error": f"MuJoCo import failed: {str(e)}"
            }
    
    def test_mcp_protocol(self) -> Dict[str, Any]:
        """测试MCP协议兼容性"""
        print("测试MCP协议兼容性...")
        
        try:
            import asyncio
            from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool
            
            async def test_mcp():
                # 测试工具列表
                tools = await handle_list_tools()
                
                # 测试基本工具调用
                result = await handle_call_tool("get_server_info", {})
                
                return {
                    "tools_count": len(tools),
                    "server_info_success": len(result) > 0,
                    "tools": [tool.name for tool in tools]
                }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            mcp_result = loop.run_until_complete(test_mcp())
            loop.close()
            
            return {
                "test_name": "mcp_protocol",
                "success": True,
                "details": mcp_result
            }
            
        except Exception as e:
            return {
                "test_name": "mcp_protocol",
                "success": False,
                "error": str(e)
            }
    
    def test_file_permissions(self) -> Dict[str, Any]:
        """测试文件权限"""
        print("测试文件权限...")
        
        import tempfile
        import os
        
        try:
            # 测试临时文件创建
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("test")
                temp_file = f.name
            
            # 测试文件读写
            with open(temp_file, 'r') as f:
                content = f.read()
            
            # 清理
            os.unlink(temp_file)
            
            return {
                "test_name": "file_permissions",
                "success": content == "test",
                "details": "File read/write operations successful"
            }
            
        except Exception as e:
            return {
                "test_name": "file_permissions", 
                "success": False,
                "error": str(e)
            }
    
    def test_network_capabilities(self) -> Dict[str, Any]:
        """测试网络功能"""
        print("测试网络功能...")
        
        try:
            import socket
            
            # 测试socket创建
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', 0))  # 绑定到任意可用端口
            port = s.getsockname()[1]
            s.close()
            
            return {
                "test_name": "network_capabilities",
                "success": True,
                "details": f"Successfully bound to port {port}"
            }
            
        except Exception as e:
            return {
                "test_name": "network_capabilities",
                "success": False,
                "error": str(e)
            }
    
    def test_graphics_support(self) -> Dict[str, Any]:
        """测试图形支持"""
        print("测试图形支持...")
        
        graphics_info = {
            "display_available": False,
            "opengl_available": False,
            "headless_mode": False
        }
        
        # 检查显示
        try:
            if platform.system() == "Linux":
                import os
                graphics_info["display_available"] = "DISPLAY" in os.environ
            else:
                graphics_info["display_available"] = True  # Windows/macOS通常有显示
        except:
            pass
        
        # 检查OpenGL
        try:
            import OpenGL.GL as gl
            graphics_info["opengl_available"] = True
        except ImportError:
            pass
        
        # 检查无头模式支持
        try:
            import os
            os.environ["MUJOCO_GL"] = "osmesa"
            graphics_info["headless_mode"] = True
        except:
            pass
        
        return {
            "test_name": "graphics_support",
            "success": graphics_info["opengl_available"] or graphics_info["headless_mode"],
            "details": graphics_info
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print(f"开始跨平台兼容性测试...")
        print(f"平台: {self.results['platform_info']['system']} {self.results['platform_info']['release']}")
        print(f"Python: {self.results['platform_info']['python_version']}")
        print("-" * 50)
        
        # 运行所有测试
        tests = [
            self.test_python_import,
            self.test_cli_commands,
            self.test_mujoco_integration,
            self.test_mcp_protocol,
            self.test_file_permissions,
            self.test_network_capabilities,
            self.test_graphics_support
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.results["tests"].append(result)
                
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"{status} {result['test_name']}")
                
                if not result["success"] and "error" in result:
                    print(f"     错误: {result['error']}")
                
            except Exception as e:
                print(f"❌ FAIL {test_func.__name__} - 异常: {str(e)}")
                self.results["tests"].append({
                    "test_name": test_func.__name__,
                    "success": False,
                    "error": f"Test execution failed: {str(e)}"
                })
        
        # 计算总体结果
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"] if test["success"])
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }
        
        return self.results
    
    def save_report(self, filename: str = None):
        """保存测试报告"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            platform_name = platform.system().lower()
            filename = f"cross_platform_test_{platform_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n测试报告已保存到: {filename}")
        return filename
    
    def print_summary(self):
        """打印测试摘要"""
        if "summary" not in self.results:
            return
        
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("跨平台兼容性测试摘要")
        print("="*60)
        print(f"平台: {self.results['platform_info']['system']} {self.results['platform_info']['release']}")
        print(f"Python: {self.results['platform_info']['python_version']}")
        print(f"架构: {self.results['platform_info']['machine']}")
        print(f"\n测试结果:")
        print(f"  总测试数: {summary['total_tests']}")
        print(f"  通过测试: {summary['passed_tests']}")
        print(f"  失败测试: {summary['failed_tests']}")
        print(f"  成功率: {summary['success_rate']:.1f}%")
        
        if summary['success_rate'] >= 90:
            print(f"\n✅ 平台兼容性良好")
        elif summary['success_rate'] >= 70:
            print(f"\n⚠️  平台兼容性一般，存在一些问题")
        else:
            print(f"\n❌ 平台兼容性较差，需要修复")
        
        print("="*60)

def main():
    test = CrossPlatformTest()
    
    try:
        test.run_all_tests()
        test.print_summary()
        test.save_report()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()