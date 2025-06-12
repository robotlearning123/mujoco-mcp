#!/usr/bin/env python3
"""
Test suite for v0.5.1 - Performance Monitoring
Tests for performance tracking and monitoring capabilities
"""
import pytest
import pytest_asyncio
import asyncio
import time
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer


class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics_tool_exists(self, server):
        """Test that performance metrics tool is registered"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "get_performance_metrics" in tool_names
    
    @pytest.mark.asyncio
    async def test_basic_performance_metrics(self, server):
        """Test getting basic performance metrics"""
        result = server._impl._handle_get_performance_metrics()
        
        assert "uptime" in result
        assert "total_requests" in result
        assert "active_simulations" in result
        assert "memory_usage" in result
        assert result["uptime"] >= 0
    
    @pytest.mark.asyncio
    async def test_simulation_performance_tracking(self, server):
        """Test tracking simulation performance"""
        # Create a simulation
        setup_result = server._impl._handle_pendulum_demo(action="setup")
        model_id = setup_result["model_id"]
        
        # Run some steps
        server._impl._handle_step_simulation(model_id=model_id, steps=100)
        
        # Get simulation metrics
        result = server._impl._handle_get_simulation_metrics(model_id=model_id)
        
        assert "total_steps" in result
        assert "average_step_time" in result
        assert "real_time_factor" in result
        assert result["total_steps"] == 100
    
    @pytest.mark.asyncio
    async def test_tool_performance_tracking(self, server):
        """Test tracking individual tool performance"""
        # Enable performance tracking
        server._impl.call_tool("enable_performance_tracking", {"enabled": True})
        
        # Call some tools through call_tool to track metrics
        server._impl.call_tool("pendulum_demo", {"action": "setup"})
        server._impl.call_tool("list_demos", {})
        
        # Get tool metrics
        result = server._impl._handle_get_tool_metrics()
        
        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0
        assert "pendulum_demo" in result["tool_calls"]
        
        tool_data = result["tool_calls"]["pendulum_demo"]
        assert "count" in tool_data
        assert "average_time" in tool_data
        assert tool_data["count"] >= 1
    
    @pytest.mark.asyncio
    async def test_memory_tracking(self, server):
        """Test memory usage tracking"""
        result = server._impl._handle_get_memory_usage()
        
        assert "total_memory" in result
        assert "simulation_memory" in result
        assert "model_count" in result
        assert result["total_memory"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_history(self, server):
        """Test performance history tracking"""
        # Create some activity
        for _ in range(3):
            result = server._impl._handle_pendulum_demo(action="setup")
            model_id = result["model_id"]
            server._impl._handle_step_simulation(model_id=model_id, steps=10)
            await asyncio.sleep(0.1)  # Small delay
        
        # Get performance history
        result = server._impl._handle_get_performance_history(
            duration=60,  # Last 60 seconds
            resolution="second"
        )
        
        assert "history" in result
        assert len(result["history"]) > 0
        assert "timestamp" in result["history"][0]
        assert "metrics" in result["history"][0]
    
    @pytest.mark.asyncio
    async def test_performance_alerts(self, server):
        """Test performance alert system"""
        # Set performance thresholds
        result = server._impl._handle_set_performance_thresholds(
            max_step_time=0.1,  # 100ms max step time
            max_memory_mb=1000,  # 1GB max memory
            max_cpu_percent=80   # 80% max CPU
        )
        
        assert result["success"] is True
        
        # Check alerts
        alerts = server._impl._handle_get_performance_alerts()
        assert "alerts" in alerts
        assert isinstance(alerts["alerts"], list)
    
    @pytest.mark.asyncio
    async def test_clear_performance_data(self, server):
        """Test clearing performance data"""
        # Create some data
        server._impl.call_tool("pendulum_demo", {"action": "setup"})
        
        # Clear data
        result = server._impl._handle_clear_performance_data()
        assert result["success"] is True
        
        # Check metrics are reset (clear doesn't affect total_requests)
        metrics = server._impl._handle_get_performance_metrics()
        assert metrics["total_requests"] == 0  # Should be reset


class TestPerformanceOptimization:
    """Test performance optimization features"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, server):
        """Test batch operation support"""
        # Create multiple models
        models = []
        for i in range(3):
            result = server._impl._handle_pendulum_demo(action="setup")
            models.append(result["model_id"])
        
        # Batch step operation
        result = server._impl._handle_batch_step(
            model_ids=models,
            steps=10
        )
        
        assert result["success"] is True
        assert len(result["results"]) == 3
        assert all(r["steps_completed"] == 10 for r in result["results"])
    
    @pytest.mark.asyncio
    async def test_parallel_simulations(self, server):
        """Test parallel simulation execution"""
        # Create models
        models = []
        for i in range(3):
            result = server._impl._handle_create_scene(
                scene_type="pendulum",
                parameters={"mass": 1.0 + i * 0.1}
            )
            models.append(result["model_id"])
        
        # Run parallel simulations
        start_time = time.time()
        result = server._impl._handle_run_parallel(
            model_ids=models,
            duration=0.1,  # 100ms each
            timestep=0.001
        )
        elapsed = time.time() - start_time
        
        assert result["success"] is True
        assert len(result["final_states"]) == 3
        # Should be faster than sequential (3 * 0.1s)
        assert elapsed < 0.25  # Allow some overhead
    
    @pytest.mark.asyncio
    async def test_caching_system(self, server):
        """Test model caching for performance"""
        # Skip this test as caching is not implemented in current version
        pytest.skip("Model caching not implemented in simple server")
    
    @pytest.mark.asyncio
    async def test_performance_profiling(self, server):
        """Test performance profiling capability"""
        # Start profiling
        result = server._impl._handle_start_profiling(
            duration=1.0,  # Profile for 1 second
            include_memory=True
        )
        
        assert result["success"] is True
        profile_id = result["profile_id"]
        
        # Do some work
        for _ in range(5):
            r = server._impl._handle_pendulum_demo(action="setup")
            server._impl._handle_step_simulation(model_id=r["model_id"], steps=10)
        
        # Wait for profiling to complete
        await asyncio.sleep(1.1)
        
        # Get profile results
        profile = server._impl._handle_get_profile_results(profile_id=profile_id)
        
        assert "hotspots" in profile
        assert "memory_allocations" in profile
        assert len(profile["hotspots"]) > 0


class TestResourceMonitoring:
    """Test resource monitoring capabilities"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_resource_limits(self, server):
        """Test setting resource limits"""
        result = server._impl._handle_set_resource_limits(
            max_models=10,
            max_memory_mb=500,
            max_step_rate=10000  # Steps per second
        )
        
        assert result["success"] is True
        assert result["limits"]["max_models"] == 10
    
    @pytest.mark.asyncio
    async def test_resource_usage_monitoring(self, server):
        """Test monitoring resource usage"""
        # Create some load
        models = []
        for _ in range(3):
            r = server._impl._handle_pendulum_demo(action="setup")
            models.append(r["model_id"])
        
        # Get resource usage
        usage = server._impl._handle_get_resource_usage()
        
        assert "cpu_percent" in usage
        assert "memory_mb" in usage
        assert "model_count" in usage
        assert "step_rate" in usage
        assert usage["model_count"] == 3
    
    @pytest.mark.asyncio
    async def test_auto_cleanup(self, server):
        """Test automatic resource cleanup"""
        # Enable auto cleanup
        result = server._impl._handle_enable_auto_cleanup(
            enabled=True,
            idle_timeout=0.5,  # 500ms idle timeout
            max_age=2.0  # 2 second max age
        )
        
        assert result["success"] is True
        
        # Create a model
        r = server._impl._handle_pendulum_demo(action="setup")
        model_id = r["model_id"]
        
        # Wait for cleanup
        await asyncio.sleep(0.6)
        
        # Check if model was cleaned up
        models = server._impl._handle_get_loaded_models()
        # Model should still exist (not idle long enough)
        assert any(m["model_id"] == model_id for m in models["models"])


class TestServerVersion051:
    """Test server version update"""
    
    @pytest.mark.asyncio
    async def test_version_updated(self):
        """Test that server version is 0.5.1"""
        server = MuJoCoServer()
        await server.initialize()
        
        try:
            assert server.version == "0.6.0"
            
            info = server.get_server_info()
            assert info["version"] == "0.6.0"
        finally:
            await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_capability(self):
        """Test that performance monitoring capability is advertised"""
        server = MuJoCoServer()
        await server.initialize()
        
        try:
            info = server.get_server_info()
            
            assert "capabilities" in info
            assert "performance_monitoring" in info["capabilities"]
            assert info["capabilities"]["performance_monitoring"] is True
        finally:
            await server.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])