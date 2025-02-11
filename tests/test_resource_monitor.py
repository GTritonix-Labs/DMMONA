import os
import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from src.resource_monitor import ResourceMonitor

class TestResourceMonitor(unittest.TestCase):
    def setUp(self):
        # Use a small moving average window for testing.
        self.monitor = ResourceMonitor(interval=0, moving_avg_window=3)
    
    def test_get_current_metrics(self):
        """Test that current metrics contain 'cpu' and 'memory' as floats."""
        metrics = self.monitor.get_current_metrics()
        self.assertIn("cpu", metrics)
        self.assertIn("memory", metrics)
        self.assertIsInstance(metrics["cpu"], float)
        self.assertIsInstance(metrics["memory"], float)
    
    def test_log_and_history(self):
        """Test that logging metrics adds an entry to history."""
        initial_length = len(self.monitor.history)
        self.monitor.log_metrics()
        self.assertEqual(len(self.monitor.history), initial_length + 1)
    
    def test_forecast_resources_empty_history(self):
        """
        Test forecast_resources when history is empty.
        The forecast should match the current metrics.
        """
        self.monitor.clear_history()
        metrics = self.monitor.get_current_metrics()
        forecast = self.monitor.forecast_resources()
        self.assertEqual(forecast["cpu"], metrics["cpu"])
        self.assertEqual(forecast["memory"], metrics["memory"])
    
    def test_forecast_resources_moving_average(self):
        """
        Test forecast_resources using a preset history.
        It computes the moving average of the last entries.
        """
        self.monitor.clear_history()
        # Manually set history to known values.
        self.monitor.history = [
            {"timestamp": "t1", "cpu": 10.0, "memory": 14.0},
            {"timestamp": "t2", "cpu": 20.0, "memory": 15.0},
            {"timestamp": "t3", "cpu": 30.0, "memory": 16.0}
        ]
        forecast = self.monitor.forecast_resources()
        expected_cpu = (10.0 + 20.0 + 30.0) / 3
        expected_memory = (14.0 + 15.0 + 16.0) / 3
        self.assertAlmostEqual(forecast["cpu"], expected_cpu, places=2)
        self.assertAlmostEqual(forecast["memory"], expected_memory, places=2)

if __name__ == "__main__":
    unittest.main()
