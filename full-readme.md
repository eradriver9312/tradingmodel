# High-Frequency Trading System with LSTM Neural Networks

## Overview
A sophisticated high-frequency trading (HFT) system that combines deep learning with technical analysis for automated trading. The system uses LSTM (Long Short-Term Memory) neural networks to predict stock price movements and generate trading signals based on real-time market data.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Input Features](#input-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Components](#model-components)
- [Performance Metrics](#performance-metrics)
- [Trading Logic](#trading-logic)
- [Backend Integration](#backend-integration)
- [Contributing](#contributing)
- [License](#license)

## Features
- Real-time market data integration via Polygon.io API
- LSTM-based price prediction model
- Automated signal generation and trade execution
- Comprehensive technical analysis integration
- Advanced risk management system
- Backtesting capabilities
- Performance visualization and metrics
- Multi-ticker support
- Memory-efficient data processing

## System Architecture

### LSTM Model Structure
- Input Layer: 5 features
- Hidden Layers: 2 LSTM layers (128 units each)
- Output Layer: Single unit (next price prediction)
- Sequence Length: 60 time steps
- Batch Size: 64
- Training Epochs: 50
- Learning Rate: 0.001

### Data Pipeline
1. Data Fetching → 
2. Preprocessing → 
3. Feature Engineering → 
4. Model Training → 
5. Signal Generation → 
6. Trade Execution

## Input Features

### Price and Volume Data
1. **Close Price**
   - Final trading price for each period
   - Primary prediction target
   - Market consensus indicator

2. **Volume**
   - Trading activity measurement
   - Liquidity indicator
   - Trend confirmation tool

### Technical Indicators
3. **SMA (Simple Moving Average) - 20 periods**
   - Trend identification
   - Formula: `SMA = (P1 + P2 + ... + P20) / 20`
   - Trend direction and strength indicator

4. **EMA (Exponential Moving Average) - 20 periods**
   - Weighted moving average
   - More responsive to recent price changes
   - Formula: `EMA = Price * (2 / (1+20)) + Previous EMA * (1 - (2 / (1+20)))`

5. **RSI (Relative Strength Index)**
   - Momentum oscillator (0-100 scale)
   - Overbought/Oversold identification
   - Formula: `RSI = 100 - (100 / (1 + RS))`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ojocodeai/tradingmodel.git
cd tradingmodel
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your Polygon.io API key
```

## Configuration

### Model Parameters (in main.py)
```python
PARAMS = {
    'sequence_length': 60,
    'hidden_size': 128,
    'num_layers': 2,
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 1e-3
}
```

### Trading Parameters
```python
TRADING_PARAMS = {
    'buy_threshold': 1.5,
    'sell_threshold': 0.66,
    'stop_loss_threshold': 0.005
}
```

## Backend Integration

### 1. FastAPI Implementation
- REST API endpoints for historical data
- WebSocket for real-time updates
- Background tasks for model training
- Integrated database storage

### 2. API Endpoints
```plaintext
GET /api/v1/market-data/{symbol}
POST /api/v1/analysis/predict
GET /api/v1/portfolio
POST /api/v1/model/train
GET /api/v1/model/status
```

### 3. Data Flow
- Real-time data streaming
- Model predictions
- Signal generation
- Trade execution
- Performance monitoring

### 4. Database Schema
```sql
-- Market Data Table
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume INTEGER,
    indicators JSONB
);

-- Predictions Table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    predicted_price DECIMAL,
    confidence DECIMAL,
    signals JSONB
);
```

## Performance Metrics

### Model Evaluation
- Direction Accuracy
- Magnitude Correlation
- Timing Accuracy
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

### Trading Performance
- Total Returns
- Win Rate
- Profit Factor
- Maximum Drawdown
- Sharpe Ratio

## Trading Logic

### Entry Conditions
1. Long Position:
   - Predicted price > current price * (1 + buy_threshold)
   - RSI < 70
   - Volume confirmation

2. Short Position:
   - Predicted price < current price * (1 - sell_threshold)
   - RSI > 30
   - Volume confirmation

### Exit Conditions
1. Stop Loss:
   - Position moves against entry by stop_loss_threshold
2. Take Profit:
   - Position reaches profit target
3. Time-based:
   - Position duration exceeds limit


### Evaluating model...
Model Metrics: {'direction_accuracy': 68.81616939364774, 'magnitude_correlation': 0.8765630234685793, 'timing_accuracy': 70.91959557053443}
Trade Performance Metrics: {'total_trades': 875, 'total_profit': 5368.877029418945, 'win_rate': 80.11428571428571, 'average_profit_per_trade': 6.13585946219308, 'maximum_drawdown': -9.289993}

### Evaluating model...
Model Metrics: {'direction_accuracy': 99.19561243144425, 'magnitude_correlation': 0.9994734180557596, 'timing_accuracy': 100.0}
Trade Performance Metrics: {'total_trades': 3, 'total_profit': 5.549201965332031, 'win_rate': 66.66666666666666, 'average_profit_per_trade': 1.849734, 'maximum_drawdown': -2.3209991}

### Evaluating model...
Model Metrics: {'direction_accuracy': 98.91060557513617, 'magnitude_correlation': 0.9995229783731197, 'timing_accuracy': 100.0}
Trade Performance Metrics: {'total_trades': 67, 'total_profit': 868.2642211914062, 'win_rate': 76.11940298507463, 'average_profit_per_trade': 12.9591675, 'maximum_drawdown': -54.866608}


### Example 

$10,000 deposit:

1. Daily Profit: $554.92
   * Total profit percentage = 5.549%
   * On $10,000 deposit: $10,000 × 5.549% = $554.92
   * Simple way: $10,000 × 0.05549 = $554.92

2. Daily Return: 5.55%
   * This is directly from metrics (5.549% rounded to 5.55%)
   * Means for every $100 invested,  made $5.55

3. Average Profit per Trade: $184.97
   * Average profit percentage per trade = 1.849734%
   * On $10,000 deposit: $10,000 × 1.849734% = $184.97
   * Simple way: $10,000 × 0.01849734 = $184.97

4. Maximum Drawdown Risk: $232.10
   * Maximum drawdown percentage = -2.321%
   * On $10,000 deposit: $10,000 × -2.321% = -$232.10
   * Simple way: $10,000 × -0.02321 = -$232.10

Additional Context:
* From 3 total trades, 2 were winners (66.67% win rate)
* To verify total profit: $184.97 (avg profit) × 3 trades = $554.91 (slight difference due to rounding)

This means with $10,000:
* Best case (winning trade): You make around $184.97 per trade
* Worst case (maximum drawdown): You could lose up to $232.10
* Overall performance: $554.92 from these 3 trades

