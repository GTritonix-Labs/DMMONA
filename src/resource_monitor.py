import psutil
import time
from datetime import datetime
import csv

class ResourceMonitor:
    """
    ResourceMonitor monitors system resources (CPU and memory usage).

    Features:
      - Logs each measurement with a timestamp.
      - Supports a configurable monitoring interval.
      - Provides forecasting via a simple moving average or a custom function.
      - Can clear history and export the logged metrics to CSV.
      - Stores the last measured metrics to ensure consistency when history is empty.
    """
    def __init__(self, interval=1, moving_avg_window=3, forecast_fn=None):
        """
        Initializes the ResourceMonitor.
        
        Args:
            interval (int, optional): Time interval (in seconds) for measuring CPU usage. Defaults to 1.
            moving_avg_window (int, optional): Number of recent entries to average for forecasting. Defaults to 3.
            forecast_fn (callable, optional): A custom forecasting function. If None, uses the default moving average.
        """
        self.interval = interval
        self.moving_avg_window = moving_avg_window
        self.history = []  # Each entry is a dict with keys: 'timestamp', 'cpu', and 'memory'
        self.last_metrics = None  # Stores the last computed metrics for consistency
        self.forecast_fn = forecast_fn if forecast_fn is not None else self._default_forecast

    def get_current_metrics(self):
        """
        Retrieve current system resource metrics.
        
        Returns:
            dict: Contains:
                  - 'timestamp': current time in ISO format.
                  - 'cpu': CPU usage percentage (float).
                  - 'memory': Memory used in GB (float).
        """
        cpu_usage = psutil.cpu_percent(interval=self.interval)
        memory_usage = psutil.virtual_memory().used / (1024 ** 3)
        timestamp = datetime.now().isoformat()
        metrics = {"timestamp": timestamp, "cpu": cpu_usage, "memory": memory_usage}
        self.last_metrics = metrics
        return metrics

    def log_metrics(self):
        """
        Retrieves current resource metrics and logs them into history.
        
        Returns:
            dict: The current resource metrics including the timestamp.
        """
        metrics = self.get_current_metrics()
        self.history.append(metrics)
        return metrics

    def _default_forecast(self):
        """
        Default forecasting using a simple moving average of the last few measurements.
        
        Returns:
            dict: Forecasted resource metrics.
        """
        if len(self.history) >= self.moving_avg_window:
            recent = self.history[-self.moving_avg_window:]
            avg_cpu = sum(entry["cpu"] for entry in recent) / self.moving_avg_window
            avg_memory = sum(entry["memory"] for entry in recent) / self.moving_avg_window
            return {"timestamp": datetime.now().isoformat(), "cpu": avg_cpu, "memory": avg_memory}
        elif self.last_metrics is not None:
            return self.last_metrics
        else:
            return self.get_current_metrics()

    def forecast_resources(self):
        """
        Forecast resource metrics using the specified forecasting function.
        
        Returns:
            dict: The forecasted resource metrics.
        """
        return self.forecast_fn()

    def clear_history(self):
        """
        Clears the logged resource metrics history.
        """
        self.history = []
    
    def export_history(self, file_path="resource_history.csv"):
        """
        Exports the logged resource metrics history to a CSV file.
        
        Args:
            file_path (str): Destination file path for the CSV export.
        """
        if not self.history:
            print("No history to export.")
            return

        fieldnames = ["timestamp", "cpu", "memory"]
        try:
            with open(file_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for entry in self.history:
                    writer.writerow(entry)
            print(f"History successfully exported to {file_path}")
        except Exception as e:
            print("Error exporting history:", e)

# For standalone testing.
if __name__ == "__main__":
    monitor = ResourceMonitor(interval=1, moving_avg_window=3)
    print("Starting Resource Monitor Test:")
    for i in range(5):
        metrics = monitor.log_metrics()
        print(f"Iteration {i+1} at {metrics['timestamp']}: CPU: {metrics['cpu']}%, Memory: {metrics['memory']:.2f} GB")
        time.sleep(1)
    forecast = monitor.forecast_resources()
    print("\nForecasted Resources based on moving average:")
    print(f"At {forecast['timestamp']}: CPU: {forecast['cpu']}%, Memory: {forecast['memory']:.2f} GB")
    monitor.export_history("resource_history.csv")
