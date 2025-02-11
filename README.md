# DMMONA – Multi-Objective Meta-Optimizer with Neural Architecture Adaptation

**Version:** 1.0.0

## Overview
DMMONA is a Python framework that optimizes ML model training on resource-constrained desktops. It continuously monitors system resources and uses a reinforcement learning (RL) meta controller to dynamically adjust hyperparameters, model architecture, and computation precision. This ensures efficient, adaptive, and stable training even on everyday hardware.

## Key Features
- **Real-Time Resource Monitoring:** Tracks CPU and memory usage with `psutil` and forecasts availability via a moving average.
- **RL-Based Meta Controller:** A lightweight PyTorch model that generates adjustment signals from resource metrics.
- **Dynamic Architecture Adaptation:** Automatically prunes or expands network layers based on meta signals.
- **Adaptive Precision Switching:** Chooses the optimal precision mode (fp32, mixed, or quantized) according to current resources.
- **Modular Design:** Easily integrates into existing ML pipelines.
- **Centralized Logging:** Detailed logs are output to the console and saved to a log file.

## Project Structure
```
DMMONA/
├── README.md                   # Project overview and usage instructions.
├── requirements.txt            # Python dependencies.
├── config/
│   └── config.yaml             # User-configurable parameters.
├── src/
│   ├── __init__.py             # Package initializer.
│   ├── main.py                 # Entry point: loads config and starts training.
│   ├── resource_monitor.py     # Monitors system resources and forecasts availability.
│   ├── meta_controller.py      # RL-based meta controller that outputs adjustment signals.
│   ├── architecture_adaptation.py  # Simulates dynamic model architecture adaptation.
│   ├── adaptive_precision.py       # Selects computation precision mode based on resources.
│   ├── training_scheduler.py       # Coordinates the training loop.
│   └── logger.py               # Sets up centralized logging.
├── tests/                      # Unit tests for each module.
│   ├── test_resource_monitor.py
│   ├── test_meta_controller.py
│   ├── test_architecture_adaptation.py
│   └── test_adaptive_precision.py
├── docs/
│   └── design_documentation.md # Detailed design documentation.
└── notebooks/
    └── exploration.ipynb       # Notebook for experiments and prototyping.
```

## Installation

### Prerequisites
- Python 3.7 or higher
- Git (optional, for cloning the repository)

### Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/dmmona.git
   cd dmmona
   ```

2. **Create & Activate a Virtual Environment:**

   - **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure the Project:**
   Edit `config/config.yaml` to set your dataset paths, training parameters, resource limits, and adaptation thresholds.

2. **Launch the Training Process:**
   ```bash
   python src/main.py -c config/config.yaml
   ```
   The training loop will monitor system resources, adapt the model architecture, and adjust precision settings while logging progress.

3. **Monitor Logs:**
   - Real-time logs appear in the console.
   - Detailed logs are saved to `dmmona.log` (configured in `src/logger.py`).


Users can then install DMMONA via:
```bash
pip install dmmona
```

## Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository.**
2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/your-username/dmmona.git
   cd dmmona
   ```
3. **Create a Branch:**
   ```bash
   git checkout -b feature-or-bugfix-name
   ```
4. **Commit & Push Your Changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-or-bugfix-name
   ```
5. **Submit a Pull Request on GitHub.**

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
