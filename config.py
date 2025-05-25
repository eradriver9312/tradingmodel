"""
Configuration settings for the LSTM trading model.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import warnings
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("Missing Polygon API KEY. Please check your .env file.")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='lstm_trading_model.log',
    filemode='a',
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress mplfinance warnings for too much data
warnings.filterwarnings("ignore", category=UserWarning, module="mplfinance")

# Tickers to analyze
TICKERS = ['MSFT', 'GOOGL', 'TSLA', 'NVDA', 'TQQQ', 'SQQQ', 'QQQ', 'PSQ', 'QLD']
MARKET_INDEX = 'SPY'
END_DATE = '2025-02-27'

# Training parameters
TRAINING_DAYS = 60  # Number of days for training data

# Model hyperparameters
MODEL_PARAMS = {
    'sequence_length': 60,
    'hidden_size': 256,
    'num_layers': 2,
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'dropout': 0.5,
    'early_stopping_patience': 15
}

# Transaction cost parameters
TRANSACTION_COSTS = {
    'commission_rate': 0.001,  # 0.1% commission
    'slippage_factor': 0.0002  # 0.02% slippage
}

# Stock-specific parameters
STOCK_PROFILES = {
    'MSFT': {
        'base_entry_threshold': 0.0006,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.0005,
        'base_stop_loss': 0.0012,
        'atr_multiplier': 1.0,
        'min_hold_time': 3,
        'max_daily_trades': 25,
        'position_size': 0.2,
        'volatility_threshold': 0.7,
        'volume_threshold': 0.9,
        'trend_threshold': 0.015
    },
    'GOOGL': {
        'base_entry_threshold': 0.0008,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.0006,
        'base_stop_loss': 0.0018,
        'atr_multiplier': 1.1,
        'min_hold_time': 2,
        'max_daily_trades': 28,
        'position_size': 0.2,
        'volatility_threshold': 0.8,
        'volume_threshold': 0.9,
        'trend_threshold': 0.018
    },
    'TSLA': {
        'base_entry_threshold': 0.0025,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.002,
        'base_stop_loss': 0.0035,
        'atr_multiplier': 1.8,
        'min_hold_time': 1,
        'max_daily_trades': 18,
        'position_size': 0.1,
        'volatility_threshold': 1.3,
        'volume_threshold': 1.3,
        'trend_threshold': 0.025
    },
    'NVDA': {
        'base_entry_threshold': 0.002,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.0015,
        'base_stop_loss': 0.003,
        'atr_multiplier': 1.5,
        'min_hold_time': 1,
        'max_daily_trades': 22,
        'position_size': 0.12,
        'volatility_threshold': 1.1,
        'volume_threshold': 1.2,
        'trend_threshold': 0.02
    },
    'TQQQ': {
        'base_entry_threshold': 0.004,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.003,
        'base_stop_loss': 0.006,
        'atr_multiplier': 2.5,
        'min_hold_time': 1,
        'max_daily_trades': 12,
        'position_size': 0.06,
        'volatility_threshold': 2.0,
        'volume_threshold': 1.8,
        'trend_threshold': 0.04
    },
    'SQQQ': {
        'base_entry_threshold': 0.004,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.003,
        'base_stop_loss': 0.006,
        'atr_multiplier': 2.5,
        'min_hold_time': 1,
        'max_daily_trades': 10,
        'position_size': 0.06,
        'volatility_threshold': 2.0,
        'volume_threshold': 2.0,
        'trend_threshold': 0.04
    },
    'QLD': {
        'base_entry_threshold': 0.0025,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.0018,
        'base_stop_loss': 0.0035,
        'atr_multiplier': 1.8,
        'min_hold_time': 1,
        'max_daily_trades': 18,
        'position_size': 0.10,
        'volatility_threshold': 1.5,
        'volume_threshold': 1.4,
        'trend_threshold': 0.03
    },
    'PSQ': {
        'base_entry_threshold': 0.0025,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.0018,
        'base_stop_loss': 0.0035,
        'atr_multiplier': 1.6,
        'min_hold_time': 3,
        'max_daily_trades': 18,
        'position_size': 0.15,
        'volatility_threshold': 1.2,
        'volume_threshold': 1.2,
        'trend_threshold': 0.025
    },
    'QQQ': {
        'base_entry_threshold': 0.0003,
        'short_entry_threshold_factor': 0.2,
        'base_exit_threshold': 0.0002,
        'base_stop_loss': 0.001,
        'atr_multiplier': 0.9,
        'min_hold_time': 1,
        'max_daily_trades': 25,
        'position_size': 0.25,
        'volatility_threshold': 0.3,
        'volume_threshold': 0.0,
        'trend_threshold': 0.006
    }
}

# Default stock profile for any ticker not explicitly listed
DEFAULT_STOCK_PROFILE = {
    'base_entry_threshold': 0.0015,
    'short_entry_threshold_factor': 0.2,
    'base_exit_threshold': 0.001,
    'base_stop_loss': 0.0025,
    'atr_multiplier': 1.3,
    'min_hold_time': 2,
    'max_daily_trades': 18,
    'position_size': 0.15,
    'volatility_threshold': 1.0,
    'volume_threshold': 1.0,
    'trend_threshold': 0.02
}

# Ticker-specific direction confidence
TICKER_CONFIDENCE = {
    'MSFT': {'up': 0.75, 'down': 0.75},  # Balanced
    'GOOGL': {'up': 0.75, 'down': 0.75},  # Balanced
    'NVDA': {'up': 0.75, 'down': 0.75},   # Balanced
    'TSLA': {'up': 0.75, 'down': 0.75},   # Balanced
    'TQQQ': {'up': 0.70, 'down': 0.80},   # Slightly favor down predictions for volatile ETF
    'QQQ': {'up': 0.75, 'down': 0.75},    # Balanced
}

# Default confidence levels - equal for both directions
DEFAULT_CONFIDENCE = {'up': 0.75, 'down': 0.75}

# File paths
MODEL_SAVE_PATH = 'tuneenhancedstructured_models/'