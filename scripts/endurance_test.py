#!/usr/bin/env python3
"""
MuJoCo-MCP 耐久性测试
测试长时间运行的稳定性和内存泄漏
"""

import asyncio
import time
import psutil
import json
import argparse
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('endurance_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("endurance_test")

class EnduranceTest:
    def __init__(self, duration_hours: int = 1):
        self.duration = timedelta(hours=duration_hours)
        self.start_time = None
        self.end_time = None
        self.running = False
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "memory_samples": [],
            "response_times": [],
            "errors": []
        }
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """优雅停止测试"""
        logger.info(f"接收到信号 {signum}，正在停止测试...")
        self.running = False
    
    async def test_tool_call(self, tool_name: str, args: dict) -> Dict:
        """测试单个工具调用"""
        try:
            from mujoco_mcp.mcp_server import handle_call_tool
            
            start_time = time.time()
            result = await handle_call_tool(tool_name, args)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            return {
                "success": True,
                "response_time": response_time,
                "result_length": len(str(result[0].text)) if result else 0
            }
            
        except Exception as e:
            logger.error(f"工具调用失败: {tool_name} - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def collect_memory_stats(self) -> Dict:
        """收集内存统计信息"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "timestamp": time.time(),
            "rss_mb": memory_info.rss / 1024 / 1024,  # 转换为 MB
            "vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads()
        }
    
    async def run_test_cycle(self):
        """运行一个测试周期"""
        test_cases = [
            ("get_server_info", {}),
            ("create_scene", {"scene_type": "pendulum"}),
            ("step_simulation", {"model_id": "test", "steps": 10}),
            ("get_state", {"model_id": "test"}),
            ("reset_simulation", {"model_id": "test"}),
            ("close_viewer", {"model_id": "test"})
        ]
        
        for tool_name, args in test_cases:
            if not self.running:
                break
                
            self.stats["total_calls"] += 1
            result = await self.test_tool_call(tool_name, args)
            
            if result["success"]:
                self.stats["successful_calls"] += 1
                self.stats["response_times"].append(result["response_time"])
            else:
                self.stats["failed_calls"] += 1
                self.stats["errors"].append({
                    "timestamp": time.time(),
                    "tool": tool_name,
                    "error": result.get("error", "Unknown error")
                })
            
            # 短暂延迟避免过载
            await asyncio.sleep(0.1)
    
    async def monitor_resources(self):
        """资源监控协程"""
        while self.running:
            memory_stats = self.collect_memory_stats()
            self.stats["memory_samples"].append(memory_stats)
            
            # 每分钟记录一次状态
            if len(self.stats["memory_samples"]) % 600 == 0:  # 600 * 0.1s = 60s
                logger.info(f"运行状态 - 调用次数: {self.stats['total_calls']}, "
                          f"内存: {memory_stats['rss_mb']:.1f}MB, "
                          f"CPU: {memory_stats['cpu_percent']:.1f}%")
            
            await asyncio.sleep(0.1)
    
    async def run_endurance_test(self):
        """运行耐久性测试"""
        self.start_time = datetime.now()
        self.end_time = self.start_time + self.duration
        self.running = True
        
        logger.info(f"开始耐久性测试，计划运行 {self.duration}")
        logger.info(f"开始时间: {self.start_time}")
        logger.info(f"预计结束: {self.end_time}")
        
        # 启动资源监控
        monitor_task = asyncio.create_task(self.monitor_resources())
        
        try:
            while self.running and datetime.now() < self.end_time:
                await self.run_test_cycle()
                
                # 检查是否需要停止
                if datetime.now() >= self.end_time:
                    logger.info("达到预定测试时间")
                    break
                    
        except Exception as e:
            logger.error(f"测试过程中发生错误: {str(e)}")
        finally:
            self.running = False
            monitor_task.cancel()
            
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        actual_duration = datetime.now() - self.start_time
        logger.info(f"测试结束，实际运行时间: {actual_duration}")
    
    def analyze_results(self) -> Dict:
        """分析测试结果"""
        if not self.stats["response_times"]:
            return {"error": "没有成功的响应时间数据"}
        
        import statistics
        
        memory_samples = self.stats["memory_samples"]
        if memory_samples:
            initial_memory = memory_samples[0]["rss_mb"]
            final_memory = memory_samples[-1]["rss_mb"]
            max_memory = max(sample["rss_mb"] for sample in memory_samples)
            avg_memory = statistics.mean(sample["rss_mb"] for sample in memory_samples)
        else:
            initial_memory = final_memory = max_memory = avg_memory = 0
        
        response_times = self.stats["response_times"]
        
        analysis = {
            "test_summary": {
                "total_calls": self.stats["total_calls"],
                "successful_calls": self.stats["successful_calls"], 
                "failed_calls": self.stats["failed_calls"],
                "success_rate": (self.stats["successful_calls"] / max(1, self.stats["total_calls"])) * 100,
                "total_errors": len(self.stats["errors"])
            },
            "performance": {
                "avg_response_time_ms": statistics.mean(response_times),
                "median_response_time_ms": statistics.median(response_times),
                "p95_response_time_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                "max_response_time_ms": max(response_times),
                "min_response_time_ms": min(response_times)
            },
            "memory_analysis": {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "max_memory_mb": max_memory,
                "avg_memory_mb": avg_memory,
                "memory_growth_mb": final_memory - initial_memory,
                "potential_memory_leak": final_memory > initial_memory * 1.1  # 增长超过10%认为可能有泄漏
            },
            "stability": {
                "no_crashes": len(self.stats["errors"]) == 0,
                "error_rate": (len(self.stats["errors"]) / max(1, self.stats["total_calls"])) * 100
            }
        }
        
        return analysis
    
    def save_results(self, filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"endurance_test_report_{timestamp}.json"
        
        analysis = self.analyze_results()
        
        report = {
            "test_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "planned_duration_hours": self.duration.total_seconds() / 3600,
                "actual_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            },
            "raw_stats": self.stats,
            "analysis": analysis
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"测试报告已保存到: {filename}")
        return filename
    
    def print_summary(self):
        """打印测试摘要"""
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("耐久性测试摘要")
        print("="*60)
        
        summary = analysis["test_summary"]
        print(f"总调用次数: {summary['total_calls']}")
        print(f"成功调用: {summary['successful_calls']}")
        print(f"失败调用: {summary['failed_calls']}")
        print(f"成功率: {summary['success_rate']:.2f}%")
        
        if "performance" in analysis:
            perf = analysis["performance"]
            print(f"\n性能指标:")
            print(f"  平均响应时间: {perf['avg_response_time_ms']:.2f}ms")
            print(f"  P95响应时间: {perf['p95_response_time_ms']:.2f}ms")
            print(f"  最大响应时间: {perf['max_response_time_ms']:.2f}ms")
        
        if "memory_analysis" in analysis:
            mem = analysis["memory_analysis"]
            print(f"\n内存分析:")
            print(f"  初始内存: {mem['initial_memory_mb']:.1f}MB")
            print(f"  最终内存: {mem['final_memory_mb']:.1f}MB")
            print(f"  最大内存: {mem['max_memory_mb']:.1f}MB")
            print(f"  内存增长: {mem['memory_growth_mb']:.1f}MB")
            if mem['potential_memory_leak']:
                print(f"  ⚠️  检测到潜在内存泄漏")
            else:
                print(f"  ✅ 无明显内存泄漏")
        
        if analysis.get("stability", {}).get("no_crashes"):
            print(f"\n✅ 测试期间无崩溃")
        else:
            print(f"\n❌ 检测到 {len(self.stats['errors'])} 个错误")
        
        print("="*60)

async def main():
    parser = argparse.ArgumentParser(description="MuJoCo-MCP 耐久性测试")
    parser.add_argument("--duration", type=float, default=1.0, 
                       help="测试持续时间(小时)")
    parser.add_argument("--output", type=str, 
                       help="输出报告文件名")
    
    args = parser.parse_args()
    
    test = EnduranceTest(duration_hours=args.duration)
    
    try:
        await test.run_endurance_test()
    finally:
        test.print_summary()
        test.save_results(args.output)

if __name__ == "__main__":
    asyncio.run(main())