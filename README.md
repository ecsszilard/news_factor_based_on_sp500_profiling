# Advanced News Factor Trading System

A sophisticated multi-task learning system for analyzing news impact on stock prices using attention mechanisms and company embeddings.  
The system performs joint optimization across company embedding space and keyword impact patterns to enable cross-task learning between relevance detection and price prediction.

## Features

- **Multi-task Learning**: Joint optimization of price prediction, volatility estimation, and news relevance  
- **Company Embeddings**: Implicit clustering of companies based on similar price reactions  
- **Attention Mechanisms**: Bi-directional attention between news content and company features  
- **Impact Pattern Analysis**: Discovery of keywords with similar market effects  
- **Automated Trading**: Signal generation and execution based on news analysis  

---

## Prerequisites

### System Requirements

- Windows 10/11 with WSL2  
- NVIDIA GPU with updated drivers (545.x or newer)  
- Docker Desktop  
- Git  

### Installation Steps

1. **Install Docker Desktop**  
   - Download and install from: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)  
   - Restart your computer after installation  

2. **Enable WSL2 (if not already enabled)**  
   ```powershell
   wsl --install
   ```
   Restart your computer.

---

## Quick Start

### Clone and Setup

```bash
git clone <repository-url>
cd advanced-news-trading-system

# Build and run with docker-compose
docker-compose up -d --build
```

### Enter Development Environment

```bash
docker-compose exec tensorflow-gpu bash

# Verify GPU access
python -c "import tensorflow as tf; print('GPU Available:', tf.test.is_gpu_available())"

# Run the main system
python main.py
```

---

## Project Structure

```
advanced-news-trading-system/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .devcontainer/
│   └── devcontainer.json
├── AdvancedTradingSystem.py
├── AttentionBasedNewsFactorModel.py
├── BiDirectionalAttentionLayer.py
├── CompanyEmbeddingSystem.py
├── ImprovedTokenizer.py
├── NewsDataProcessor.py
├── PerformanceAnalyzer.py
└── main.py
```

---

## Development with VS Code

1. **Install VS Code Extensions**  
   - Remote - Containers  
   - Python  
   - Jupyter  

2. **Open in Container**  
   ```bash
   code .
   ```
   Use Command Palette (Ctrl+Shift+P) → *Remote-Containers: Reopen in Container*

### Alternative: Jupyter Notebook Access

```bash
docker-compose exec tensorflow-gpu jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

Access at: [http://localhost:8888](http://localhost:8888)

---

## GPU Testing

```python
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available())
print("CUDA build:", tf.test.is_built_with_cuda())

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Available GPUs: {len(gpus)}")
for gpu in gpus:
    print(f"GPU: {gpu}")
```

---

## System Components

### Core Models
- **AttentionBasedNewsFactorModel**: Multi-task neural network with attention mechanisms  
- **CompanyEmbeddingSystem**: Learns company representations based on news reactions  
- **BiDirectionalAttentionLayer**: Custom attention layer for news-company interactions  

### Data Processing
- **ImprovedTokenizer**: Advanced tokenization with BERT-like preprocessing  
- **NewsDataProcessor**: Prepares training data from news and price information  

### Trading System
- **AdvancedTradingSystem**: Generates trading signals and executes trades  
- **PerformanceAnalyzer**: Evaluates system performance and generates reports  

---

## Usage Example

```python
from CompanyEmbeddingSystem import CompanyEmbeddingSystem
from AttentionBasedNewsFactorModel import AttentionBasedNewsFactorModel
from AdvancedTradingSystem import AdvancedTradingSystem

company_system = CompanyEmbeddingSystem('sp500_companies.csv')
news_model = AttentionBasedNewsFactorModel(company_system)
trading_system = AdvancedTradingSystem(company_system, news_model)

test_news = "Tesla reports breakthrough in battery technology"
companies = ['TSLA', 'AAPL', 'F', 'GM']
impact_analysis = trading_system.analyze_news_impact(test_news, companies)

signals = trading_system.generate_trading_signals(impact_analysis)
executed_trades = trading_system.execute_trades(signals)
```

---

## Docker Commands Reference

### Container Management

```bash
docker-compose up -d --build      # Build and start services
docker-compose down               # Stop services
docker-compose logs tensorflow-gpu
docker-compose exec tensorflow-gpu bash
docker-compose restart
```

### Development Commands

```bash
docker-compose exec tensorflow-gpu pip install package-name
docker-compose exec tensorflow-gpu watch -n 1 nvidia-smi
docker cp file.py container-name:/app/
docker cp container-name:/app/output.txt ./
```

### Cleanup Commands

```bash
docker-compose down --remove-orphans
docker system prune -a
docker-compose down --rmi all
```

---

## Troubleshooting

### GPU Not Detected

```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

- Check Docker Desktop GPU settings: *Docker Desktop > Settings > Resources > WSL Integration*  

### Memory Issues

```bash
docker stats tensorflow-gpu
```

Adjust memory limits in `docker-compose.yml` if needed.

### Permission Issues

```bash
docker-compose exec tensorflow-gpu chown -R $(id -u):$(id -g) /app
```

On Windows, add Docker directories to antivirus exclusions:  
- `%USERPROFILE%\.docker`  
- `C:\ProgramData\Docker`  

---

## Performance Monitoring

### System Metrics

```bash
nvidia-smi -l 1
docker-compose exec tensorflow-gpu htop
```

TensorBoard (if enabled): [http://localhost:6006](http://localhost:6006)

### Trading Performance

```python
from PerformanceAnalyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(trading_system)
report = analyzer.generate_performance_report('performance_report.json')
print(f"Portfolio Value: ${report['portfolio_value']:.2f}")
print(f"Total Trades: {report['total_trades']}")
```

---

## Table of Contents

- Features  
- Prerequisites  
- Installation Steps  
- Quick Start  
- Project Structure  
- Development with VS Code  
- GPU Testing  
- System Components  
- Usage Example  
- Docker Commands Reference  
- Troubleshooting  
- Performance Monitoring  
