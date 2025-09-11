#!/usr/bin/env python3
"""
Performance tests for MuJoCo MCP server
Tests response times and resource usage
"""

import asyncio
import time
import statistics
from typing import List, Dict
import psutil
import sys

from test_mcp_functionality import MCPTester

class PerformanceTester(MCPTester):
    """Performance testing for MCP server"""
    
    def __init__(self):
        super().__init__()
        self.performance_data = []
        
    def measure_response_time(self, test_name: str):
        """Decorator to measure response times"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                self.performance_data.append({
                    "test": test_name,
                    "response_time_ms": response_time,
                    "success": result
                })
                
                print(f"  {test_name}: {response_time:.1f}ms")
                return result
            return wrapper
        return decorator
        
    async def test_response_times(self) -> Dict:
        """Test response times for all tools"""
        print("🏃 Testing response times...")
        
        # Warm up
        await self.test_server_info()
        
        # Measure response times
        tests = [
            ("server_info", self.test_server_info),
            ("list_tools", self.test_list_tools),
            ("create_scene", self.test_create_scene),
            ("step_simulation", self.test_step_simulation),
            ("get_state", self.test_get_state),
            ("reset_simulation", self.test_reset_simulation)
        ]
        
        for test_name, test_func in tests:
            # Run test multiple times
            times = []
            for _ in range(5):
                start_time = time.time()
                await test_func()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                await asyncio.sleep(0.1)
                
            avg_time = statistics.mean(times)
            print(f"  {test_name}: {avg_time:.1f}ms avg")
            
        return {"success": True}
        
    async def test_concurrent_requests(self) -> Dict:
        """Test concurrent request handling"""
        print("🔄 Testing concurrent requests...")
        
        # Create multiple concurrent requests
        concurrent_count = 5
        tasks = []
        
        start_time = time.time()
        
        for i in range(concurrent_count):
            task = asyncio.create_task(self.test_server_info())
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        success_count = sum(1 for r in results if r is True)
        
        print(f"  {concurrent_count} concurrent requests: {total_time*1000:.1f}ms total")
        print(f"  Success rate: {success_count}/{concurrent_count}")
        
        return {
            "success": success_count >= concurrent_count * 0.8,
            "concurrent_count": concurrent_count,
            "success_count": success_count,
            "total_time_ms": total_time * 1000
        }
        
    async def test_memory_usage(self) -> Dict:
        """Test memory usage during operations"""
        print("💾 Testing memory usage...")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform operations
        await self.test_create_scene()
        await self.test_step_simulation()
        await self.test_get_state()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (< 100 MB)
        return {
            "success": memory_increase < 100,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase
        }
        
    async def test_stress_operations(self) -> Dict:
        """Stress test with many operations"""
        print("🏋️ Running stress test...")
        
        operations = 50
        success_count = 0
        start_time = time.time()
        
        for i in range(operations):
            if i % 10 == 0:
                print(f"  Progress: {i}/{operations}")
                
            # Alternate between different operations
            if i % 3 == 0:
                result = await self.test_create_scene()
            elif i % 3 == 1:
                result = await self.test_step_simulation()
            else:
                result = await self.test_get_state()
                
            if result:
                success_count += 1
                
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.05)
            
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = operations / total_time
        
        print(f"  {operations} operations in {total_time:.1f}s")
        print(f"  {ops_per_second:.1f} ops/second")
        print(f"  Success rate: {success_count}/{operations}")
        
        return {
            "success": success_count >= operations * 0.9,
            "total_operations": operations,
            "success_count": success_count,
            "total_time_s": total_time,
            "ops_per_second": ops_per_second
        }
        
    async def run_performance_tests(self) -> Dict:
        """Run all performance tests"""
        print("🚀 Starting MuJoCo MCP performance tests...")
        
        # Start server
        if not await self.start_server():
            return {"success": False, "error": "Could not start MCP server"}
            
        try:
            results = {}
            
            # Run performance tests
            results["response_times"] = await self.test_response_times()
            results["concurrent_requests"] = await self.test_concurrent_requests()
            results["memory_usage"] = await self.test_memory_usage()
            results["stress_test"] = await self.test_stress_operations()
            
            # Overall success
            all_success = all(r.get("success", False) for r in results.values())
            
            return {
                "success": all_success,
                "results": results
            }
            
        finally:
            await self.stop_server()

async def main():
    """Main performance test function"""
    tester = PerformanceTester()
    
    try:
        results = await tester.run_performance_tests()
        
        print(f"\\n⚡ Performance Test Results:")
        
        if results["success"]:
            print("✅ Overall: PASSED")
        else:
            print("❌ Overall: FAILED")
            
        # Detailed results
        for test_name, test_result in results.get("results", {}).items():
            status = "✅" if test_result.get("success", False) else "❌"
            print(f"  {status} {test_name}")
            
        return 0 if results["success"] else 1
        
    except Exception as e:
        print(f"❌ Performance test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())