import torch
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Deque, Optional, Tuple
from datetime import datetime, timedelta, time as dt_time
import time
import threading
import os
from collections import deque
import ta
from ta.volatility import BollingerBands
from ta.trend import MACD
import logging
import platform
import sys
import pytz
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from alpaca.trading.stream import TradingStream
from alpaca.trading.client import TradingClient
import mongo as mongo
from dotenv import load_dotenv
import nest_asyncio
import asyncio
import json
import gc
from cachetools import TTLCache, LRUCache
import traceback
import torch.nn.functional as F
import stat
from pathlib import Path
import requests

# Import all the necessary Alpaca components
from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)

from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
    OrderStatus
)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("realtimetrading_execution.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize environment and async support
load_dotenv()

def debug_model_paths():
    """
    Comprehensive debugging for model path issues.
    """
    import os
    import stat
    from pathlib import Path
    
    print("\n===== STARTING COMPREHENSIVE PATH DEBUGGING =====")
    
    # 1. Check the base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory (from __file__): {BASE_DIR}")
    print(f"Base directory exists: {os.path.exists(BASE_DIR)}")
    
    # Alternative base directory calculation
    alt_base_dir = os.getcwd()
    print(f"Alternative base directory (from getcwd): {alt_base_dir}")
    print(f"Alternative base directory exists: {os.path.exists(alt_base_dir)}")
    
    # 2. Check for structured_models directory
    structured_dir = os.path.join(BASE_DIR, "structured_models")
    alt_structured_dir = os.path.join(alt_base_dir, "structured_models")
    
    print(f"\nStructured models directory: {structured_dir}")
    print(f"Structured models directory exists: {os.path.exists(structured_dir)}")
    
    print(f"Alt structured models directory: {alt_structured_dir}")
    print(f"Alt structured models directory exists: {os.path.exists(alt_structured_dir)}")
    
    # Use the directory that exists
    if os.path.exists(structured_dir):
        working_dir = structured_dir
    elif os.path.exists(alt_structured_dir):
        working_dir = alt_structured_dir
    else:
        print("Neither structured models directory exists!")
        
        # Search for any directory with 'model' in the name
        print("\nSearching for model directories in base directory:")
        for item in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, item)
            if os.path.isdir(full_path) and 'model' in item.lower():
                print(f"  Found potential model directory: {item}")
        
        print("\nSearching for model directories in parent directory:")
        parent_dir = os.path.dirname(BASE_DIR)
        for item in os.listdir(parent_dir):
            full_path = os.path.join(parent_dir, item)
            if os.path.isdir(full_path) and 'model' in item.lower():
                print(f"  Found potential model directory: {item}")
        
        working_dir = None
    
    if working_dir:
        print(f"\nWorking with directory: {working_dir}")
        
        # 3. List contents of the structured_models directory
        print("\nContents of structured_models directory:")
        try:
            for item in os.listdir(working_dir):
                full_path = os.path.join(working_dir, item)
                is_dir = os.path.isdir(full_path)
                print(f"  {'[DIR]' if is_dir else '[FILE]'} {item}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
    
    # 4. Check each ticker directory and files
    print("\nChecking each ticker directory and files:")
    for ticker in TICKERS:
        print(f"\n  Ticker: {ticker}")
        
        # Check both possible base directories
        for base in [BASE_DIR, alt_base_dir]:
            ticker_dir = os.path.join(base, "structured_models", ticker)
            print(f"  Directory: {ticker_dir}")
            
            if os.path.exists(ticker_dir):
                print(f"  Directory exists: Yes")
                
                # List contents of ticker directory
                print(f"  Contents of {ticker} directory:")
                try:
                    for item in os.listdir(ticker_dir):
                        full_path = os.path.join(ticker_dir, item)
                        file_size = os.path.getsize(full_path) if os.path.isfile(full_path) else "N/A"
                        permissions = stat.filemode(os.stat(full_path).st_mode)
                        print(f"    {item} (Size: {file_size}, Permissions: {permissions})")
                except Exception as e:
                    print(f"    Error listing directory: {e}")
                
                # Check specific model files
                model_path = os.path.join(ticker_dir, f"enhanced_lstm_model_{ticker}_2025-03-21.pth")
                scaler_path = os.path.join(ticker_dir, f"enhanced_scaler_{ticker}_2025-03-21.pkl")
                
                print(f"  Model path: {model_path}")
                print(f"  Model exists: {os.path.exists(model_path)}")
                
                print(f"  Scaler path: {scaler_path}")
                print(f"  Scaler exists: {os.path.exists(scaler_path)}")
                
                # Try to access the files
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            first_bytes = f.read(10)
                        print(f"  Model file is readable: Yes")
                    except Exception as e:
                        print(f"  Model file read error: {e}")
                
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as f:
                            first_bytes = f.read(10)
                        print(f"  Scaler file is readable: Yes")
                    except Exception as e:
                        print(f"  Scaler file read error: {e}")
            else:
                print(f"  Directory exists: No")
    
    # 5. Look for alternative model paths
    alt_patterns = [
        "lstm_model_*",
        "enhanced_lstm_*",
        "*model*",
        "*scaler*",
        "*lstm*",
        "*.pth"
    ]
    
    print("\nSearching for alternative model files:")
    for base in [BASE_DIR, alt_base_dir]:
        model_parent = os.path.join(base, "structured_models")
        if os.path.exists(model_parent):
            for ticker_dir in os.listdir(model_parent):
                full_ticker_dir = os.path.join(model_parent, ticker_dir)
                if os.path.isdir(full_ticker_dir):
                    print(f"\n  Checking directory: {full_ticker_dir}")
                    try:
                        for item in os.listdir(full_ticker_dir):
                            if any(Path(item).match(pattern) for pattern in alt_patterns):
                                print(f"    Potential model file: {item}")
                    except Exception as e:
                        print(f"    Error listing directory: {e}")
    
    # 6. Check for alternate model directories
    print("\nChecking for alternate model directories:")
    
    alternate_dirs = [
        "tuneenhancedstructured_models",
        "tuned_models",
        "models",
        "enhanced_models",
        "lstm_models"
    ]
    
    for base in [BASE_DIR, alt_base_dir]:
        for alt_dir in alternate_dirs:
            path = os.path.join(base, alt_dir)
            exists = os.path.exists(path)
            print(f"  {path}: {'Exists' if exists else 'Not found'}")
            
            if exists:
                print(f"    Contents:")
                try:
                    for item in os.listdir(path):
                        print(f"      {item}")
                except Exception as e:
                    print(f"      Error listing directory: {e}")
    
    print("\n===== PATH DEBUGGING COMPLETE =====")
    
    # Return working directory and any found model files as a dict
    result = {
        "working_dir": working_dir,
        "base_dir": BASE_DIR,
        "alt_base_dir": alt_base_dir,
        "found_files": {}
    }
    
    if working_dir:
        for ticker in TICKERS:
            result["found_files"][ticker] = []
            for base in [BASE_DIR, alt_base_dir]:
                ticker_dir = os.path.join(base, "structured_models", ticker)
                if os.path.exists(ticker_dir):
                    for item in os.listdir(ticker_dir):
                        if any(Path(item).match(pattern) for pattern in alt_patterns):
                            result["found_files"][ticker].append(os.path.join(ticker_dir, item))
    
    return result

# Initialize Alpaca trading stream with proper error handling
try:
    stream = TradingStream(
        api_key=os.getenv("APCA-API-KEY-ID"),
        secret_key=os.getenv("APCA-API-SECRET-KEY"),
        paper=True,
    )
except Exception as e:
    logger.critical(f"Failed to initialize trading stream: {e}")
    sys.exit(1)

def update_positions():
    """
    Synchronize internal state with Alpaca positions and update MongoDB.
    """
    try:
        positions = trade_client.get_all_positions()
        # the user_id is hardcoded for the time being, but will need to be dynamic eventually
        update_position_cache("3bean", positions)

        with state.lock:
            # Reset positions dictionary to handle closed positions
            state.positions = {ticker: None for ticker in TICKERS}
            
            for position in positions:
                state.positions[position.symbol] = {
                    'type': 'long' if position.side.value == 'long' else 'short',
                    'size': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value)
                }
    except Exception as e:
        logger.error(f"Error updating positions: {e}")

order_cache = {}  # Cache for orders not yet in MongoDB

def cleanup_positions_and_orders():
    """Clean up all existing positions and orders to prevent wash trade conflicts."""
    logger.info("Cleaning up existing positions and orders...")
    
    # First, get all current positions
    try:
        positions = trade_client.get_all_positions()
        for position in positions:
            ticker = position.symbol
            logger.info(f"Closing existing position for {ticker}: {position.qty} shares, side: {position.side.value}")
            
            try:
                # Use the simplest form of close_position without parameters
                trade_client.close_position(ticker)
                logger.info(f"Successfully closed position for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to close position for {ticker}: {e}")
    except Exception as e:
        logger.error(f"Error retrieving positions: {e}")
    
    # Then, cancel all open orders - using the correct approach
    try:
        # Get all orders without filtering parameters
        open_orders = trade_client.get_orders()
        
        # Filter locally for open orders
        for order in open_orders:
            if order.status in ['new', 'accepted', 'partially_filled', 'accepted_for_bidding']:
                try:
                    trade_client.cancel_order_by_id(order_id=order.id)
                    logger.info(f"Canceled order {order.id} for {order.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order.id}: {e}")
    except Exception as e:
        logger.error(f"Error retrieving open orders: {e}")
    
    # Wait a moment for the operations to complete
    time.sleep(1)
    logger.info("Position and order cleanup completed")
    
def handle_trade_updates_sync(data):
    try:
        # Cache the order information before MongoDB update
        order_id = str(data.order.id)
        if hasattr(data.order, 'filled_avg_price') and data.order.filled_avg_price:
            order_cache[order_id] = {
                'symbol': data.order.symbol,
                'price': data.order.filled_avg_price,
                'qty': data.order.filled_qty,
                'side': data.order.side.value
            }
        mongo.update_order_from_websocket(order_update=data, market_price=data.price)
        update_positions()
    except Exception as e:
        logger.error(f"Error handling trade update: {e}")        

async def handle_trade_updates(data):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, handle_trade_updates_sync, data)
    

# Subscribe to trade updates with error handling
try:
    stream.subscribe_trade_updates(handle_trade_updates)
except Exception as e:
    logger.error(f"Failed to subscribe to trade updates: {e}")

def start_stream():
    """
    Start and maintain the Alpaca websocket connection with proper error handling.
    """
    while True:
        try:
            logger.info("Initializing Alpaca trade stream...")
            stream.run()
        except Exception as e:
            logger.error(f"Stream error encountered: {e}")
        finally:
            logger.info("Stream connection terminated, attempting reconnection...")

# Create stream task with error handling
try:
    # Start WebSocket in a separate thread to avoid blocking the main program
    stream_thread = threading.Thread(target=start_stream, daemon=True)
    stream_thread.start()
except Exception as e:
    logger.error(f"Failed to create stream task: {e}")

# Initialize trading client with proper error handling
try:
    API_KEY_ID = os.getenv("APCA-API-KEY-ID")
    API_SECRET_KEY = os.getenv("APCA-API-SECRET-KEY")
    API_SERVER_DOMAIN = os.getenv("APCA-URL")
    paper = True

    if not all([API_KEY_ID, API_SECRET_KEY]):
        raise ValueError("Missing required API credentials")

    trade_client = TradingClient(
        api_key=API_KEY_ID,
        secret_key=API_SECRET_KEY,
        paper=paper,
    )
except Exception as e:
    logger.critical(f"Failed to initialize trading client: {e}")
    sys.exit(1)

# Configuration setup with validation
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    logger.critical("Missing Polygon API key")
    sys.exit(1)

TICKERS = ['TSLA', 'NVDA', 'MSFT', 'GOOGL']
fixed_shares = mongo.get_filtered_fixed_shares(TICKERS)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure model paths with clear patterns from debugging
MODEL_PATHS = {
    ticker: os.path.join(BASE_DIR, f"structured_models/{ticker}/enhanced_lstm_model_{ticker}_2025-03-21.pth")
    for ticker in TICKERS
}
SCALER_PATHS = {
    ticker: os.path.join(BASE_DIR, f"structured_models/{ticker}/enhanced_scaler_{ticker}_2025-03-21.pkl")
    for ticker in TICKERS
}

# Initialize global debugging result
debugging_result = None

SEQUENCE_LENGTH = 60

# Correct number of features to match training pipeline
FEATURE_COUNT = 22  # 1 price feature (close) + 21 technical indicators

# Historical data window for initialization
INITIAL_HISTORY_LENGTH = 100  # Min number of minute bars needed before making predictions

# setup local caching for storing positions
# Cache up to 1000 positions per user with a TTL of 10 minutes
position_cache = TTLCache(maxsize=1000, ttl=600)

def update_position_cache(user_id, positions):
    """
    Inserts or updates multiple positions in the cache.
    Handles both single position objects and lists of positions.
    """
    if not isinstance(positions, list):
        positions = [positions]  # Convert single object to list for consistency

    for position in positions:
        key = f"{user_id}:{position.symbol}"  # Use dot notation (not dict indexing)
        
        position_cache[key] = {
            "symbol": position.symbol,
            "qty": int(position.qty),  # Convert to integer
            "side": position.side.value,  # Convert Enum to string
            "avg_entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "unrealized_pl": float(position.unrealized_pl),
            "cost_basis": float(position.cost_basis),
            "last_synced": time.time()
        }


def get_cached_position(user_id, ticker):
    """
    Retrieves a position from the cache if available.
    Returns None if not found.
    """
    key = f"{user_id}:{ticker}"
    return position_cache.get(key, None)

def process_positions(user_id, positions):
    """
    Takes a list of positions (from Alpaca API) and updates the cache.
    """
    for pos in positions:
        position_data = {
            "user_id": user_id,
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "side": pos["side"].value,  # Convert Enum to string
            "avg_entry_price": pos["avg_entry_price"],
            "current_price": pos["current_price"],
            "unrealized_pl": pos["unrealized_pl"],
            "cost_basis": pos["cost_basis"],
        }
        update_position_cache(position_data)

def should_place_order(user_id, ticker, order_qty, order_type):
    """
    Checks the cache before placing an order.
    Prevents placing redundant or conflicting trades.
    """
    cached_position = get_cached_position(user_id, ticker)
    
    if cached_position:
        print(f"Cached position found: {cached_position}")

        # Example: Prevent buying if already holding the same quantity
        if order_type == "buy" and cached_position["side"] == "long" and cached_position["qty"] >= order_qty:
            print("Skipping order: Already holding position")
            return False

        # Example: Prevent selling more than you hold
        if order_type == "sell" and cached_position["side"] == "long" and cached_position["qty"] < order_qty:
            print("Skipping order: Not enough shares")
            return False

    return True  # Proceed with order placement

# Set process priority on Linux
if platform.system() == 'Linux':
    try:
        os.nice(19)
    except Exception as e:
        logger.warning(f"Failed to set process priority: {e}")

# Updated LSTM model to match the training pipeline's EnhancedLSTMModel
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=FEATURE_COUNT, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better pattern recognition - exactly as in training pipeline
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism - matches training pipeline
        self.attention = torch.nn.Linear(hidden_size*2, 1)
        
        # Output layers
        self.fc1 = torch.nn.Linear(hidden_size*2, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure input has 3 dimensions: (batch, seq_length, input_size)
        if x.dim() == 1:
            # Assuming the 1-D tensor represents a single sequence sample with one time step,
            # and the length of x equals input_size.
            x = x.unsqueeze(0).unsqueeze(0)  # shape becomes (1, 1, input_size)
        elif x.dim() == 2:
            # If it's 2-D, assume it's (seq_length, input_size) and add a batch dimension.
            x = x.unsqueeze(0)  # shape becomes (1, seq_length, input_size)
        
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros, accounting for bidirectionality
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = attention_weights * lstm_out
        context_vector = torch.sum(context_vector, dim=1)
        
        # Process through fully connected layers
        out = self.fc1(context_vector)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

# Function to find model files dynamically based on debug results
def find_model_files(debug_result):
    """Find model files based on debugging results."""
    found_models = {}
    found_scalers = {}
    
    # Prioritize files matching exact expected pattern first
    for ticker in TICKERS:
        found_models[ticker] = None
        found_scalers[ticker] = None
        
        if ticker in debug_result["found_files"]:
            # Try to find exact matches first
            for file_path in debug_result["found_files"][ticker]:
                if f"enhanced_lstm_model_{ticker}" in file_path and file_path.endswith(".pth"):
                    found_models[ticker] = file_path
                elif f"enhanced_scaler_{ticker}" in file_path and file_path.endswith(".pkl"):
                    found_scalers[ticker] = file_path
            
            # If not found, look for any model/scaler file
            if found_models[ticker] is None:
                for file_path in debug_result["found_files"][ticker]:
                    if "model" in file_path.lower() and file_path.endswith(".pth") and found_models[ticker] is None:
                        found_models[ticker] = file_path
            
            if found_scalers[ticker] is None:
                for file_path in debug_result["found_files"][ticker]:
                    if "scaler" in file_path.lower() and file_path.endswith(".pkl") and found_scalers[ticker] is None:
                        found_scalers[ticker] = file_path
    
    return found_models, found_scalers

# Set up device with proper error handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize models and scalers with enhanced error handling and security
models = {}
scalers = {}

class AdaptiveSignalGenerator:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock_profiles = {
            'MSFT': {
                'base_entry_threshold': 0.0008,  # Updated to match training pipeline
                'base_exit_threshold': 0.0006,
                'base_stop_loss': 0.0015,
                'atr_multiplier': 1.2,
                'min_hold_time': 5,
                'max_daily_trades': 20,
                'position_size': 0.2,
                'volatility_threshold': 0.8,
                'volume_threshold': 1.0,
                'trend_threshold': 0.02
            },
            'GOOGL': {
                'base_entry_threshold': 0.001,  # Updated to match training pipeline
                'base_exit_threshold': 0.0007,
                'base_stop_loss': 0.002,
                'atr_multiplier': 1.3,
                'min_hold_time': 3,
                'max_daily_trades': 25,
                'position_size': 0.2,
                'volatility_threshold': 0.9,
                'volume_threshold': 1.0,
                'trend_threshold': 0.02
            },
            'TSLA': {
                'base_entry_threshold': 0.003,  # Updated to match training pipeline
                'base_exit_threshold': 0.002,
                'base_stop_loss': 0.004,
                'atr_multiplier': 2.0,
                'min_hold_time': 2,
                'max_daily_trades': 15,
                'position_size': 0.1,
                'volatility_threshold': 1.5,
                'volume_threshold': 1.5,
                'trend_threshold': 0.03
            },
            'NVDA': {
                'base_entry_threshold': 0.0025,  # Updated to match training pipeline
                'base_exit_threshold': 0.0018,
                'base_stop_loss': 0.0035,
                'atr_multiplier': 1.8,
                'min_hold_time': 2,
                'max_daily_trades': 18,
                'position_size': 0.12,
                'volatility_threshold': 1.3,
                'volume_threshold': 1.3,
                'trend_threshold': 0.025
            }
        }
        
        # Default values for any ticker not explicitly listed
        self.default_profile = {
            'base_entry_threshold': 0.0015,
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
    
        self.params = self.stock_profiles.get(ticker, self.default_profile)
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize trading parameters from stock profile with improved consistency."""
        # Base parameters from stock profile
        self.base_entry_threshold = self.params['base_entry_threshold']
        self.base_exit_threshold = self.params['base_exit_threshold']
        self.base_stop_loss = self.params['base_stop_loss']
        self.atr_multiplier = self.params['atr_multiplier']
        self.min_hold_time = self.params['min_hold_time']
        self.volatility_threshold = self.params['volatility_threshold']
        self.volume_threshold = self.params['volume_threshold']
        self.trend_threshold = self.params['trend_threshold']
        
        # Define trend reversal threshold as half of the trend threshold
        self.trend_reversal_threshold = self.trend_threshold / 2.0
        
        logger.info(f"Initialized threshold for {self.ticker}: entry={self.base_entry_threshold}")
        
        # Tracking state variables
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.current_atr = None
        self.daily_trades = 0
        self.last_trade_date = None
        
        # Active position metrics
        self.current_stop_loss = None
        self.current_take_profit = None
        
        # Performance tracking
        self.winning_trades = 0
        self.total_trades = 0
        self.trade_pnl = []
        
        # Debugging flags
        self.debug_mode = False
        self.last_signal_reason = None
        
        # Initialize adaptive parameters based on recent market conditions
        self.adaptive_thresholds = {
            'entry': self.base_entry_threshold,
            'exit': self.base_exit_threshold,
            'stop_loss': self.base_stop_loss
        }
        
        logger.info(f"Initialized signal generator for {self.ticker} - " +
                    f"Entry threshold: {self.base_entry_threshold*100:.4f}%, " +
                    f"Exit threshold: {self.base_exit_threshold*100:.4f}%, " +
                    f"Trend reversal threshold: {self.trend_reversal_threshold*100:.4f}%")

    def calculate_adaptive_thresholds(self, market_data: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate adaptive thresholds based on current ATR and market conditions.
        This matches how thresholds were calculated in training.
        """
        if market_data is None or market_data.empty or 'volatility_atr' not in market_data.columns:
            return self.base_entry_threshold, self.base_exit_threshold, self.base_stop_loss
        
        try:
            # Get current ATR from market data
            current_atr = market_data['volatility_atr'].iloc[-1]
            if pd.isna(current_atr) or current_atr <= 0:
                return self.base_entry_threshold, self.base_exit_threshold, self.base_stop_loss
            
            # Calculate volatility factor
            last_price = market_data['close'].iloc[-1]
            volatility_factor = (current_atr / last_price) * self.atr_multiplier
            
            # Calculate adaptive thresholds
            entry_threshold = self.base_entry_threshold * (1 + volatility_factor)
            exit_threshold = self.base_exit_threshold * (1 + volatility_factor)
            stop_loss = self.base_stop_loss * (1 + volatility_factor)
            
            logger.debug(f"ADAPTIVE THRESHOLDS - {self.ticker}: entry={entry_threshold:.6f}, " +
                        f"exit={exit_threshold:.6f}, stop_loss={stop_loss:.6f}, " +
                        f"ATR={current_atr:.6f}, volatility_factor={volatility_factor:.4f}")
            
            return entry_threshold, exit_threshold, stop_loss
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return self.base_entry_threshold, self.base_exit_threshold, self.base_stop_loss

    def generate_signal(self, current_price: float, predicted_price: float, timestamp: datetime, 
                    market_data: pd.DataFrame = None) -> Dict:
        """
        Generate trading signals using the same logic as training (abs value comparison)
        with adaptive thresholds based on ATR.
        """
        # Add debugging log at the beginning of the method
        logger.info(f"SIGNAL GENERATOR INPUT - {self.ticker}: current_price={current_price:.4f}, predicted_price={predicted_price:.4f}")
        
        # Reset daily trades if new day
        current_date = timestamp.date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date

        # Check trade frequency limit
        if self.daily_trades >= self.params['max_daily_trades']:
            return None

        # Check minimum hold time
        if self.position and self.entry_time:
            hold_time = (timestamp - self.entry_time).total_seconds() / 60
            if hold_time < self.min_hold_time:
                return None

        # Get price history and calculate recent trends
        price_history = state.feature_buffers[self.ticker]['price_history']
        
        # Calculate adaptive thresholds based on ATR (matching training pipeline)
        entry_threshold, exit_threshold, stop_loss = self.calculate_adaptive_thresholds(market_data)
        
        # Calculate price change percentage directly without bias correction
        # This matches the training code behavior
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Log calculated price change
        logger.info(f"PRICE CHANGE PCT - {self.ticker}: {price_change_pct*100:.4f}%")

        # Check for trend and volume conditions based on market data
        trend_ok = True
        volume_ok = True
        
        if market_data is not None and not market_data.empty:
            # Check trend condition if market data is available with required indicators
            if 'MA5' in market_data.columns and 'SMA_20' in market_data.columns:
                ma5 = market_data['MA5'].iloc[-1]
                ma20 = market_data['SMA_20'].iloc[-1]
                if not pd.isna(ma5) and not pd.isna(ma20) and ma20 > 0:
                    trend_strength = abs((ma5 - ma20) / ma20)
                    trend_ok = trend_strength <= self.trend_threshold
                    logger.debug(f"TREND CHECK - {self.ticker}: strength={trend_strength:.4f}, threshold={self.trend_threshold:.4f}, ok={trend_ok}")
            
            # Check volume condition if market data available
            if 'volume' in market_data.columns and self.volume_threshold > 0:
                current_volume = market_data['volume'].iloc[-1]
                if not pd.isna(current_volume) and 'volume_sma_ratio' in market_data.columns:
                    volume_ratio = market_data['volume_sma_ratio'].iloc[-1]
                    volume_ok = volume_ratio >= self.volume_threshold
                    logger.debug(f"VOLUME CHECK - {self.ticker}: ratio={volume_ratio:.2f}, threshold={self.volume_threshold:.2f}, ok={volume_ok}")

        if self.position is None:
            # Log signal evaluation with unified checks
            logger.info(f"SIGNAL EVALUATION - {self.ticker}: change={price_change_pct*100:.4f}%, " + 
                    f"trend_ok={trend_ok}, volume_ok={volume_ok}, " +
                    f"threshold={entry_threshold*100:.4f}%")

            # Entry conditions using abs() comparison like in training code
            if abs(price_change_pct) > entry_threshold and trend_ok and volume_ok:
                if price_change_pct > 0:
                    # LONG signal processing
                    signal = {
                        'action': 'enter_long',
                        'price': current_price,
                        'timestamp': timestamp,
                        'thresholds': {
                            'stop_loss': stop_loss,
                            'take_profit': exit_threshold
                        }
                    }
                    
                    # Update position tracking
                    self.position = 'long'
                    self.entry_price = current_price
                    self.entry_time = timestamp
                    self.daily_trades += 1
                    
                    # Store current thresholds for this position
                    self.current_stop_loss = stop_loss
                    self.current_take_profit = exit_threshold

                    logger.info(f"LONG SIGNAL: {self.ticker} at {timestamp}, price={current_price:.2f}, " +
                            f"predicted_change={price_change_pct*100:.2f}%, threshold={entry_threshold*100:.2f}%")
                    
                    return signal
                else:
                    # SHORT signal processing
                    signal = {
                        'action': 'enter_short',
                        'price': current_price,
                        'timestamp': timestamp,
                        'thresholds': {
                            'stop_loss': stop_loss,
                            'take_profit': exit_threshold
                        }
                    }
                    
                    # Update position tracking
                    self.position = 'short'
                    self.entry_price = current_price
                    self.entry_time = timestamp
                    self.daily_trades += 1
                    
                    # Store current thresholds for this position
                    self.current_stop_loss = stop_loss
                    self.current_take_profit = exit_threshold

                    logger.info(f"SHORT SIGNAL: {self.ticker} at {timestamp}, price={current_price:.2f}, " +
                            f"predicted_change={price_change_pct*100:.2f}%, threshold={-entry_threshold*100:.2f}%")
                    
                    return signal
            
        else:
            # Position exit logic - use stored thresholds for consistent exit conditions
            exit_threshold = self.current_take_profit if self.current_take_profit is not None else self.base_exit_threshold
            stop_loss = self.current_stop_loss if self.current_stop_loss is not None else self.base_stop_loss
            
            if self.position == 'long':
                # Exit long position conditions
                price_change_from_entry = (current_price - self.entry_price) / self.entry_price
                
                # Stop loss check
                if price_change_from_entry < -stop_loss:
                    signal = {
                        'action': 'exit_long',
                        'price': current_price,
                        'timestamp': timestamp,
                        'reason': 'stop_loss',
                        'profit_pct': price_change_from_entry * 100
                    }
                    
                    logger.info(f"EXIT LONG (STOP LOSS): {self.ticker} at {timestamp}, entry={self.entry_price:.2f}, " +
                            f"exit={current_price:.2f}, profit={price_change_from_entry*100:.2f}%")
                    
                    # Reset position tracking
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.current_stop_loss = None
                    self.current_take_profit = None
                    
                    return signal
                
                # Take profit check
                elif price_change_from_entry > exit_threshold:
                    signal = {
                        'action': 'exit_long',
                        'price': current_price,
                        'timestamp': timestamp,
                        'reason': 'take_profit',
                        'profit_pct': price_change_from_entry * 100
                    }
                    
                    logger.info(f"EXIT LONG (TAKE PROFIT): {self.ticker} at {timestamp}, entry={self.entry_price:.2f}, " +
                            f"exit={current_price:.2f}, profit={price_change_from_entry*100:.2f}%")
                    
                    # Reset position tracking
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.current_stop_loss = None
                    self.current_take_profit = None
                    
                    return signal
                
                # Trend reversal check - exit if prediction suggests downward movement
                elif price_change_pct < -self.trend_reversal_threshold:
                    signal = {
                        'action': 'exit_long',
                        'price': current_price,
                        'timestamp': timestamp,
                        'reason': 'trend_reversal',
                        'profit_pct': price_change_from_entry * 100
                    }
                    
                    logger.info(f"EXIT LONG (TREND REVERSAL): {self.ticker} at {timestamp}, entry={self.entry_price:.2f}, " +
                            f"exit={current_price:.2f}, profit={price_change_from_entry*100:.2f}%, prediction={price_change_pct*100:.2f}%")
                    
                    # Reset position tracking
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.current_stop_loss = None
                    self.current_take_profit = None
                    
                    return signal
                    
            elif self.position == 'short':
                # Exit short position conditions
                price_change_from_entry = (self.entry_price - current_price) / self.entry_price
                
                # Stop loss check
                if price_change_from_entry < -stop_loss:
                    signal = {
                        'action': 'exit_short',
                        'price': current_price,
                        'timestamp': timestamp,
                        'reason': 'stop_loss',
                        'profit_pct': price_change_from_entry * 100
                    }
                    
                    logger.info(f"EXIT SHORT (STOP LOSS): {self.ticker} at {timestamp}, entry={self.entry_price:.2f}, " +
                            f"exit={current_price:.2f}, profit={price_change_from_entry*100:.2f}%")
                    
                    # Reset position tracking
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.current_stop_loss = None
                    self.current_take_profit = None
                    
                    return signal
                
                # Take profit check
                elif price_change_from_entry > exit_threshold:
                    signal = {
                        'action': 'exit_short',
                        'price': current_price,
                        'timestamp': timestamp,
                        'reason': 'take_profit',
                        'profit_pct': price_change_from_entry * 100
                    }
                    
                    logger.info(f"EXIT SHORT (TAKE PROFIT): {self.ticker} at {timestamp}, entry={self.entry_price:.2f}, " +
                            f"exit={current_price:.2f}, profit={price_change_from_entry*100:.2f}%")
                    
                    # Reset position tracking
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.current_stop_loss = None
                    self.current_take_profit = None
                    
                    return signal
                
                # Trend reversal check - exit if prediction suggests upward movement
                elif price_change_pct > self.trend_reversal_threshold:
                    signal = {
                        'action': 'exit_short',
                        'price': current_price,
                        'timestamp': timestamp,
                        'reason': 'trend_reversal',
                        'profit_pct': price_change_from_entry * 100
                    }
                    
                    logger.info(f"EXIT SHORT (TREND REVERSAL): {self.ticker} at {timestamp}, entry={self.entry_price:.2f}, " +
                            f"exit={current_price:.2f}, profit={price_change_from_entry*100:.2f}%, prediction={price_change_pct*100:.2f}%")
                    
                    # Reset position tracking
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.current_stop_loss = None
                    self.current_take_profit = None
                    
                    return signal
        
        # No signal generated
        return None

class RobustWebSocketClient:
    def __init__(self):
        # CRITICAL FIX: Changed subscription from T.{ticker} (trades) to AM.{ticker} (minute aggregates)
        self.client = WebSocketClient(
            api_key=API_KEY,
            subscriptions=[f"AM.{ticker}" for ticker in TICKERS],  # Subscribed to minute bars
            max_reconnects=5,
            verbose=True
        )
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds

    def message_handler(self, msgs: List[WebSocketMessage]):
        try:
            state.last_message = time.time()
            for msg in msgs:
                # Check if it's a minute aggregate message (AM) and has expected fields
                if hasattr(msg, 'symbol') and msg.symbol in TICKERS and hasattr(msg, 'close'):
                    # Add log immediately after identifying valid AM bar
                    logger.info(f"Handler received AM bar for {msg.symbol}: close={msg.close}, volume={msg.volume}")
                    
                    # Process the aggregate minute bar
                    processed = preprocess_aggregate_bar({
                        'symbol': msg.symbol,
                        'open': msg.open,
                        'high': msg.high,
                        'low': msg.low,
                        'close': msg.close,
                        'volume': msg.volume,
                        'timestamp': datetime.fromtimestamp(msg.end_timestamp/1000)
                    })
                    
                    # Log the result of preprocessing before lock
                    logger.info(f"Handler: Result for {msg.symbol}: {'ndarray' if processed is not None else 'None'}")
                    
                    with state.lock:
                        # Log after acquiring lock
                        logger.debug(f"Handler: Acquired lock for {msg.symbol} to append data")
                        
                        if processed is not None:
                            # Log before append
                            logger.debug(f"Handler: 'processed' is valid for {msg.symbol}. Attempting append.")
                            try:
                                state.data_window[msg.symbol].append(processed)
                                # Log after successful append
                                logger.info(f"Handler: Appended features for {msg.symbol}. New data_window len: {len(state.data_window[msg.symbol])}")
                            except Exception as append_err:
                                # Log if append fails
                                logger.error(f"Handler: FAILED TO APPEND data for {msg.symbol}: {append_err}", exc_info=True)
                        else:
                            # Log if processed is None
                            logger.warning(f"Handler: 'processed' is None for {msg.symbol}. Not adding to data_window.")
                        
                        # Log before releasing lock
                        logger.debug(f"Handler: Releasing lock for {msg.symbol}")
                        
                # Also add a fallback for unrecognized message types
                elif hasattr(msg, 'symbol') and msg.symbol in TICKERS:
                    logger.debug(f"Handler received non-AM message for {msg.symbol}: {type(msg)}")
                    
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            logger.error(traceback.format_exc())

    def start(self):
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                state.connection_active = True
                logger.info("Starting Polygon WebSocket connection...")
                self.client.run(
                    handle_msg=self.message_handler,
                    close_timeout=5
                )
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.reconnect_attempts += 1
                time.sleep(self.reconnect_delay)
                continue
            
            # Reset counter on successful connection (if we get here, run() has returned)
            self.reconnect_attempts = 0
            logger.warning("WebSocket connection closed - attempting to reconnect...")
            time.sleep(2)  # Wait before reconnection
            
        logger.critical("Maximum reconnection attempts reached. Exiting...")
        sys.exit(1)

class TradingState:
    def __init__(self):
        self.lock = threading.RLock()
        self.data_window = {ticker: deque(maxlen=SEQUENCE_LENGTH * 2) for ticker in TICKERS}
        
        # Enhanced feature buffers to store minute bars instead of individual trades
        self.feature_buffers = {
            ticker: {
                'price_history': deque(maxlen=100),  # Track recent prices
                'last_price': None,  # Most recent price
                'minute_bars': deque(maxlen=INITIAL_HISTORY_LENGTH),  # Store minute bars
                'last_bar_timestamp': None,  # Track the last processed minute
                'market_data': pd.DataFrame(columns=[
                    'open', 'high', 'low', 'close', 'volume', 
                    'timestamp'  # Include timestamp for easier debugging
                ])  # DataFrame for technical indicators
            } for ticker in TICKERS
        }
        
        self.positions = {ticker: None for ticker in TICKERS}
        self.portfolio = {
            'cash': 100000.00,
            'positions': {},
            'total_value': 100000.00,
            'daily_starting_value': 100000.00
        }
        self.trade_history = {
            ticker: {
                'closed': [],
                'open': {},
                'metrics': {
                    'total_profit': 0.0,
                    'winning_trades': 0,
                    'total_trades': 0,
                    'daily_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'current_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'profit_factor': 0.0,
                    'average_trade_duration': 0.0
                }
            } for ticker in TICKERS
        }
        self.signal_generators = {
            ticker: AdaptiveSignalGenerator(ticker) for ticker in TICKERS
        }
        self.last_message = time.time()
        self.connection_active = False
        self.market_hours = {
            'open': dt_time(9, 30),
            'close': dt_time(16, 0)
        }
        self.tz = pytz.timezone('America/New_York')
        self.daily_reset_time = None
        
        # Add recent predictions tracking for monitoring trends
        self.recent_predictions = {ticker: deque(maxlen=20) for ticker in TICKERS}
        
        # Flag to track initialization status
        self.initialized = {ticker: False for ticker in TICKERS}

    def is_market_open(self):
        now = datetime.now(self.tz).time()
        return self.market_hours['open'] <= now < self.market_hours['close']

    def should_reset_daily_metrics(self):
        now = datetime.now(self.tz)
        if (self.daily_reset_time is None or now.date() > self.daily_reset_time.date()):
            self.daily_reset_time = now
            return True
        return False

    def reset_daily_metrics(self):
        self.portfolio['daily_starting_value'] = self.portfolio['total_value']
        for ticker in TICKERS:
            self.trade_history[ticker]['metrics']['daily_pnl'] = 0.0
            self.trade_history[ticker]['metrics']['current_drawdown'] = 0.0
            
    def update_market_data(self, ticker: str):
        """
        Convert minute bars stored in buffer to a DataFrame with technical indicators.
        """
        buffer = self.feature_buffers[ticker]
        
        if len(buffer['minute_bars']) < 26:  # Need at least 26 bars for MACD
            return False
            
        # Create DataFrame from stored minute bars
        bars_data = list(buffer['minute_bars'])
        df = pd.DataFrame(bars_data)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators using ta library
        # Features calculated in main() and dataset class
        df['MA5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD indicators
        macd_indicator = MACD(close=df['close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        
        # Bollinger Bands
        bb_indicator = BollingerBands(close=df['close'])
        df['Bollinger_High'] = bb_indicator.bollinger_hband()
        df['Bollinger_Low'] = bb_indicator.bollinger_lband()

        # Additional technical features from training dataset
        df['volatility_atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['volatility_bbw'] = ta.volatility.bollinger_pband(df['close'])
        df['trend_adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['trend_cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        df['trend_ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
        df['momentum_roc'] = ta.momentum.roc(df['close'])
        df['momentum_kama'] = ta.momentum.kama(df['close'])
        df['momentum_stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['volume_cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        df['volume_em'] = ta.volume.ease_of_movement(df['high'], df['low'], df['close'], df['volume'])
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_distance_from_ma'] = (df['close'] - df['SMA_20']) / df['SMA_20']

        # Handle NaN values consistently with training
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        # Update market data
        buffer['market_data'] = df
        
        # Set initialization status
        if not self.initialized[ticker] and len(df) >= INITIAL_HISTORY_LENGTH:
            self.initialized[ticker] = True
            logger.info(f"Initialization complete for {ticker} with {len(df)} minute bars")
            
        return True


def handle_signal(ticker: str, signal: Dict):
    """
    Process trading signals and execute trades with improved wash trade prevention
    and enhanced error handling for order processing.
    """
    logger.info(f"Processing signal: {ticker} - {signal['action']} at {signal['price']}")

    if not state.is_market_open():
        logger.warning(f"Cannot handle signal for {ticker} outside market hours")
        return

    # Track recent orders to prevent wash trades
    current_time = time.time()
    if hasattr(state, 'recent_orders') == False:
        state.recent_orders = {}
    
    # Check if we've recently acted on this ticker (prevent rapid trading)
    if ticker in state.recent_orders and current_time - state.recent_orders[ticker]['time'] < 60:
        recent_action = state.recent_orders[ticker]['action']
        logger.info(f"Skipping signal for {ticker} - recent {recent_action} action less than 60 seconds ago")
        return

    try:
        # Verify current position with direct API call to ensure accuracy
        current_position = None
        try:
            position = trade_client.get_open_position(ticker)
            current_position = {
                'type': 'long' if position.side.value == 'long' else 'short',
                'size': float(position.qty),
                'entry_price': float(position.avg_entry_price)
            }
        except Exception:
            # No position exists
            pass

        # Update our internal state to match reality
        with state.lock:
            state.positions[ticker] = current_position

        # First, check for any pending orders for this ticker and cancel them
        try:
            open_orders = trade_client.get_orders()
            canceled_orders = False
            
            for order in open_orders:
                if order.symbol == ticker and order.status in ['new', 'accepted', 'partially_filled', 'pending_new']:
                    try:
                        trade_client.cancel_order_by_id(order_id=order.id)
                        logger.info(f"Canceled existing order {order.id} for {ticker}")
                        canceled_orders = True
                    except Exception as cancel_error:
                        logger.warning(f"Could not cancel order {order.id}: {cancel_error}")
            
            # Add a small delay after canceling orders to allow the broker to process
            if canceled_orders:
                time.sleep(2)
                
        except Exception as order_error:
            logger.warning(f"Error retrieving/canceling orders for {ticker}: {order_error}")

        # Now check again for positions after canceling orders
        if 'enter' in signal['action']:
            # Determine the position type from the signal action
            position_type = signal['action'].split('_')[1]  # 'long' or 'short'
            
            logger.info(f"Processing ENTER signal for {ticker}: Type={position_type}, Price={signal['price']}")
            
            # If we have an opposite position, close it first with a dedicated request
            if current_position is not None and current_position['type'] != position_type:
                logger.info(f"Closing opposite {current_position['type']} position for {ticker} before entering new {position_type} position")
                
                try:
                    # Use ClosePositionRequest for more reliable position closing
                    close_request = ClosePositionRequest(percentage="100")
                    
                    try:
                        response = trade_client.close_position(ticker, close_request)
                        logger.info(f"Close position request submitted for {ticker}")
                        
                        # Add delay after closing position to avoid wash trade
                        time.sleep(3)
                        
                        # Verify position is closed by checking current positions
                        position_closed = False
                        for attempt in range(3):
                            try:
                                trade_client.get_open_position(ticker)
                                logger.warning(f"Position still exists after close request. Waiting... (Attempt {attempt+1}/3)")
                                time.sleep(2)  # Increased delay
                            except Exception:
                                logger.info(f"Successfully closed {current_position['type']} position for {ticker}")
                                current_position = None
                                position_closed = True
                                break
                                
                        if not position_closed:
                            logger.warning(f"Could not verify position closure for {ticker}. Skipping signal.")
                            return
                    
                    except Exception as e:
                        error_message = str(e)
                        if "filled" in error_message or "does not exist" in error_message:
                            logger.info(f"Position for {ticker} was already closed or does not exist")
                            current_position = None
                        else:
                            logger.error(f"Failed to close existing position for {ticker}: {e}")
                            return
                    
                except Exception as e:
                    logger.error(f"Failed to close existing position for {ticker}: {e}")
                    return

            # Update recent orders tracking to prevent wash trades
            state.recent_orders[ticker] = {
                'time': current_time,
                'action': signal['action']
            }

            # Position sizing
            if ticker in fixed_shares:
                position_size = fixed_shares[ticker]
            else:
                available_capital = state.portfolio['cash'] * state.signal_generators[ticker].params['position_size']
                position_size = min(
                    available_capital / signal['price'],
                    state.portfolio['cash'] / signal['price']
                )
                position_size = int(position_size)
                if position_size <= 0:
                    logger.warning(f"Insufficient capital for {ticker} position. Skipping.")
                    return

            order_side = OrderSide.BUY if 'long' in signal['action'] else OrderSide.SELL
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=position_size,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            # Execute order with proper verification
            try:
                order = trade_client.submit_order(order_data=market_order_data)
                mongo.insert_order(order=order, customer_id="3bean", signal_price=signal['price'])
                
                # Enhanced logging for short orders to aid troubleshooting
                if 'short' in signal['action']:
                    logger.info(f"SHORT ORDER SUBMITTED: {ticker} - Side: {order_side.name}, Qty: {position_size}, Status: {order.status}")
                    if hasattr(order, 'rejected_reason') and order.rejected_reason:
                        logger.error(f"SHORT ORDER REJECTED: {order.rejected_reason}")
                else:
                    logger.info(f"ORDER SUBMITTED: {ticker} - Side: {order_side.name}, Qty: {position_size}, Status: {order.status}")
                    if hasattr(order, 'rejected_reason') and order.rejected_reason:
                        logger.error(f"LONG ORDER REJECTED: {order.rejected_reason}")
                
                if order.status == OrderStatus.FILLED:
                    with state.lock:
                        state.positions[ticker] = {
                            'type': signal['action'].split('_')[1],
                            'size': position_size,
                            'entry_price': float(order.filled_avg_price) if order.filled_avg_price else signal['price'],
                            'entry_time': order.filled_at if order.filled_at else datetime.now(state.tz)
                        }
                        cost = position_size * float(order.filled_avg_price if order.filled_avg_price else signal['price'])
                        state.portfolio['cash'] -= cost
                    logger.info(f"Entered {signal['action']} {ticker} @ ${float(order.filled_avg_price) if order.filled_avg_price else signal['price']:.2f} | Size: {position_size}")
                    
                elif order.status in [OrderStatus.ACCEPTED, OrderStatus.NEW, OrderStatus.PENDING_NEW]:
                    logger.info(f"Order accepted for {ticker} - waiting for fill")
                else:
                    logger.warning(f"Order status for {ticker}: {order.status}")
                    if hasattr(order, 'rejected_reason') and order.rejected_reason:
                        logger.error(f"Order rejected: {order.rejected_reason}")

            except Exception as e:
                logger.error(f"Error executing entry order for {ticker}: {e}")
                return

        elif 'exit' in signal['action']:
            # Skip if no position exists
            if current_position is None:
                logger.warning(f"Skipping exit for {ticker} - no position found")
                return

            # Verify position type matches exit signal
            expected_type = 'long' if 'long' in signal['action'] else 'short'
            if current_position['type'] != expected_type:
                logger.warning(f"Skipping exit for {ticker} - position type mismatch: expected {expected_type}, found {current_position['type']}")
                return

            # Update recent orders tracking
            state.recent_orders[ticker] = {
                'time': current_time,
                'action': signal['action']
            }

            # Use ClosePositionRequest for more reliable position closing
            try:
                close_request = ClosePositionRequest(percentage="100")
                
                try:
                    response = trade_client.close_position(ticker, close_request)
                    logger.info(f"Position close request submitted for {ticker}")
                    
                    # Record trade metrics - safely handling None values
                    try:
                        with state.lock:
                            # Safely get values with defaults if None
                            filled_qty = float(response.filled_qty) if response.filled_qty else current_position['size']
                            filled_price = float(response.filled_avg_price) if response.filled_avg_price else signal['price']
                            
                            exit_value = filled_qty * filled_price
                            entry_value = filled_qty * current_position['entry_price']
                            pnl = exit_value - entry_value if current_position['type'] == 'long' else entry_value - exit_value
                            
                            state.portfolio['cash'] += exit_value
                            state.trade_history[ticker]['metrics']['total_trades'] += 1
                            state.trade_history[ticker]['metrics']['total_profit'] += pnl
                            state.trade_history[ticker]['metrics']['daily_pnl'] += pnl
                            
                            if pnl > 0:
                                state.trade_history[ticker]['metrics']['winning_trades'] += 1
                                
                            state.positions[ticker] = None
                            
                        logger.info(f"Exited {current_position['type']} {ticker} @ ${filled_price:.2f} | PnL: ${pnl:.2f}")
                    except Exception as calc_error:
                        logger.error(f"Error calculating PnL for {ticker}: {calc_error}")
                        # Still mark position as closed even if PnL calculation fails
                        with state.lock:
                            state.positions[ticker] = None
                
                except Exception as e:
                    error_message = str(e)
                    if "filled" in error_message or "does not exist" in error_message:
                        logger.info(f"Position for {ticker} was already closed")
                        with state.lock:
                            state.positions[ticker] = None
                    else:
                        logger.error(f"Error closing position for {ticker}: {e}")
                
            except Exception as e:
                logger.error(f"Error closing position for {ticker}: {e}")

    except Exception as e:
        logger.error(f"Error handling signal for {ticker}: {e}")


def preprocess_aggregate_bar(bar: Dict) -> Optional[np.ndarray]:
    # --- VERY TOP log ---
    logger.info(f"Preprocess START for {bar.get('symbol', 'UNKNOWN')} at {bar.get('timestamp', 'UNKNOWN')}")
    try:
        ticker = bar['symbol']
        timestamp = bar['timestamp']
        
        # --- Log BEFORE state lock ---
        logger.debug(f"Preprocess: Entering state lock for {ticker}")
        with state.lock:
            # --- Log AFTER state lock ---
            logger.debug(f"Preprocess: Acquired state lock for {ticker}")
            buffer = state.feature_buffers[ticker]
            
            # Store minute bar in buffer
            buffer['minute_bars'].append(bar)
            
            # Update price tracking
            buffer['last_price'] = bar['close']
            buffer['price_history'].append(bar['close'])
            buffer['last_bar_timestamp'] = timestamp
            # --- Log AFTER buffer updates ---
            logger.debug(f"Preprocess: Updated buffers for {ticker}. Bar count: {len(buffer['minute_bars'])}")

            # Update market data DataFrame with technical indicators
            # --- Log BEFORE update_market_data ---
            logger.debug(f"Preprocess: Calling update_market_data for {ticker}")
            update_success = state.update_market_data(ticker)
            # --- Log AFTER update_market_data ---
            logger.debug(f"Preprocess: update_market_data for {ticker} returned {update_success}")

            if not update_success:
                logger.warning(f"Preprocess: state.update_market_data failed for {ticker}. Returning None.")
                # --- Log BEFORE returning None ---
                logger.debug(f"Preprocess END for {ticker}. Returning None (update_market_data failed).")
                return None

            # Skip if we don't have enough data yet (check initialization flag)
            if not state.initialized[ticker]:
                logger.debug(f"Preprocess: Still initializing {ticker}. Returning None.")
                 # --- Log BEFORE returning None ---
                logger.debug(f"Preprocess END for {ticker}. Returning None (not initialized).")
                return None
                
            # Get the market data DataFrame with calculated indicators
            market_data = buffer['market_data']
            # --- Log AFTER getting market_data ---
            logger.debug(f"Preprocess: Retrieved market_data for {ticker}. Shape: {market_data.shape}")

            if market_data.empty or len(market_data) < SEQUENCE_LENGTH: # Check >= SEQUENCE_LENGTH ?
                 logger.warning(f"Preprocess: Market data empty or too short for {ticker}. Len: {len(market_data)}. Required: {SEQUENCE_LENGTH}. Returning None.")
                 # --- Log BEFORE returning None ---
                 logger.debug(f"Preprocess END for {ticker}. Returning None (market_data insufficient).")
                 return None
                
            # Extract latest bar's features
            # --- Log BEFORE iloc ---
            logger.debug(f"Preprocess: Extracting latest bar features for {ticker} using iloc[-1]")
            latest_bar = market_data.iloc[-1]
            # --- Log AFTER iloc ---
            logger.debug(f"Preprocess: Extracted latest bar for {ticker}. Close: {latest_bar.get('close', 'N/A')}")
            
            # Create feature vector in the correct order
            price_feature = [latest_bar['close']]
            technical_features = [
                latest_bar['volume'], latest_bar['SMA_20'], latest_bar['EMA_20'], latest_bar['RSI'], latest_bar['MA5'],
                latest_bar['Bollinger_High'], latest_bar['Bollinger_Low'], latest_bar['MACD'], latest_bar['MACD_Signal'],
                latest_bar['volatility_atr'], latest_bar['trend_adx'], latest_bar['momentum_roc'], latest_bar['momentum_kama'],
                latest_bar['volume_cmf'], latest_bar['volume_em'], latest_bar['volume_sma_ratio'], latest_bar['price_distance_from_ma'],
                latest_bar['volatility_bbw'], latest_bar['momentum_stoch'], latest_bar['trend_cci'], latest_bar['trend_ichimoku_a']
            ]
            # --- Log AFTER feature extraction ---
            logger.debug(f"Preprocess: Created feature lists for {ticker}. Price: {price_feature[0]}, Tech features count: {len(technical_features)}")
            
            # Scale features using loaded scalers
            if ticker not in scalers or 'price_scaler' not in scalers[ticker] or 'feature_scaler' not in scalers[ticker]:
                logger.error(f"Scalers not properly loaded for {ticker}. Returning None.")
                # --- Log BEFORE returning None ---
                logger.debug(f"Preprocess END for {ticker}. Returning None (scalers not loaded).")
                return None
            
            # --- Log BEFORE scaling ---
            logger.debug(f"Preprocess: Scaling features for {ticker}")
            # Scale price feature (as a 2D array)
            scaled_price = scalers[ticker]['price_scaler'].transform(np.array([price_feature]))[0]
            # Scale technical features (as a 2D array)
            scaled_technical = scalers[ticker]['feature_scaler'].transform(np.array([technical_features]))[0]
            # --- Log AFTER scaling ---
            logger.debug(f"Preprocess: Features scaled for {ticker}.")
            
            # Combine price and technical features
            scaled_features = np.concatenate([scaled_price, scaled_technical])
            
            # Ensure we have exactly the expected number of features
            if len(scaled_features) != FEATURE_COUNT:
                logger.error(f"Feature count mismatch: {len(scaled_features)} != {FEATURE_COUNT}. Returning None.")
                # --- Log BEFORE returning None ---
                logger.debug(f"Preprocess END for {ticker}. Returning None (feature count mismatch).")
                return None
            
            # --- Log SUCCESS before returning ---
            logger.info(f"Preprocess END for {ticker}. Returning scaled features (shape {scaled_features.shape}).")
            return scaled_features.astype(np.float32) # Ensure dtype just before return
            
    except Exception as e:
        logger.error(f"Error processing aggregate bar for {bar.get('symbol', 'UNKNOWN')}: {e}")
        logger.error(traceback.format_exc())
        # --- Log EXCEPTION before returning None ---
        logger.warning(f"Preprocess EXCEPTION for {bar.get('symbol', 'UNKNOWN')}. Returning None.")
        return None


def fetch_historical_minute_bars(ticker: str, lookback_days: int = 2) -> bool:
    """
    Fetch historical minute bars from Polygon to initialize the system.
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Format dates for Polygon API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Construct API URL
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
        
        logger.info(f"Fetching historical data for {ticker} from {start_str} to {end_str}")
        
        # Make request
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'results' not in data or not data['results']:
            logger.warning(f"No historical data found for {ticker}")
            return False
            
        # Process results
        logger.info(f"Received {len(data['results'])} historical bars for {ticker}")
        
        # Store in buffer
        with state.lock:
            buffer = state.feature_buffers[ticker]
            
            # Clear any existing data
            buffer['minute_bars'].clear()
            
            # Convert and store minute bars
            for bar in data['results']:
                timestamp = datetime.fromtimestamp(bar['t'] / 1000)
                
                # Removed market hours check to include all historical data
                minute_bar = {
                    'symbol': ticker,
                    'timestamp': timestamp,
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v']
                }
                
                buffer['minute_bars'].append(minute_bar)
                
            # Set last price
            if buffer['minute_bars']:
                buffer['last_price'] = buffer['minute_bars'][-1]['close']
                
            # Update market data
            state.update_market_data(ticker)
                
            logger.info(f"Stored {len(buffer['minute_bars'])} historical bars for {ticker}")
            
            # Return whether we have enough data
            return len(buffer['minute_bars']) >= INITIAL_HISTORY_LENGTH
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return False


def prediction_engine():
    """
    Enhanced prediction engine that properly converts model predictions before signal generation.
    Only runs predictions after sufficient data is available.
    """
    while True:
        try:
            # Validate market and connection status
            if not state.is_market_open():
                logger.info("Market is closed - pausing prediction engine")
                time.sleep(60)  # Sleep longer when market is closed
                continue

            if not state.connection_active:
                logger.warning("WebSocket connection inactive - skipping prediction cycle")
                time.sleep(5)
                continue

            # Process each ticker
            for ticker in TICKERS:
                try:
                    # Skip if not enough data or not initialized
                    with state.lock:
                        if not state.initialized[ticker]:
                            # Check if we can initialize now
                            if len(state.feature_buffers[ticker]['minute_bars']) >= INITIAL_HISTORY_LENGTH:
                                logger.info(f"Initializing {ticker} with {len(state.feature_buffers[ticker]['minute_bars'])} minute bars")
                                state.update_market_data(ticker)
                                state.initialized[ticker] = True
                            else:
                                logger.debug(f"Waiting for initialization - {ticker}: {len(state.feature_buffers[ticker]['minute_bars'])}/{INITIAL_HISTORY_LENGTH} bars")
                                continue
                                
                        if len(state.data_window[ticker]) < SEQUENCE_LENGTH:
                            logger.debug(f"Insufficient data for {ticker}: {len(state.data_window[ticker])}/{SEQUENCE_LENGTH} sequences")
                            continue

                        if not state.feature_buffers[ticker]["last_price"]:
                            logger.debug(f"No last price available for {ticker}")
                            continue
                            
                        # Get current market data for signals
                        market_data = state.feature_buffers[ticker]['market_data']
                        if market_data.empty:
                            logger.debug(f"No market data available for {ticker}")
                            continue

                        # Extract sequence from data_window (deque)
                        data_list = list(state.data_window[ticker])
                        sequence = np.array(data_list[-SEQUENCE_LENGTH:], dtype=np.float32)

                        # Create tensor for model input
                        tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)

                        # Generate prediction with proper interpretation
                        with torch.no_grad():
                            # Get raw model output (percentage change prediction)
                            raw_prediction_pct_change = models[ticker](tensor).item()
                            logger.info(f"RAW MODEL OUTPUT (PCT CHANGE) - {ticker}: {raw_prediction_pct_change:.8f}")
                            
                            # Get current price for conversion to absolute price
                            current_price = state.feature_buffers[ticker]["last_price"]
                            
                            # Convert percentage change to absolute price prediction
                            # THIS IS THE CRITICAL FIX
                            predicted_price_absolute = current_price * (1 + raw_prediction_pct_change)
                            
                            # Log before/after conversion for verification
                            logger.info(f"PREDICTION CONVERSION - {ticker}: current={current_price:.4f}, " +
                                      f"pct_change={raw_prediction_pct_change*100:.4f}%, " +
                                      f"predicted_absolute={predicted_price_absolute:.4f}")
                            
                            # Store raw prediction for monitoring
                            state.recent_predictions[ticker].append(raw_prediction_pct_change)
                        
                        # Clean up memory
                        del tensor
                        torch.cuda.empty_cache()

                    # Generate timestamp outside lock to minimize lock duration
                    timestamp = datetime.now(state.tz)

                    # Generate trading signal using CORRECTLY CONVERTED absolute predicted price
                    signal = state.signal_generators[ticker].generate_signal(
                        current_price=current_price,
                        predicted_price=predicted_price_absolute,  # Use absolute price (not percentage)
                        timestamp=timestamp,
                        market_data=market_data
                    )

                    # Handle signal if generated
                    if signal:
                        logger.info(f"SIGNAL GENERATED - {ticker}: {signal['action']} @ ${signal['price']:.2f}")
                        handle_signal(ticker, signal)

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"GPU memory exhausted for {ticker}")
                    torch.cuda.empty_cache()
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing {ticker}: {e}\n{traceback.format_exc()}"
                    )
                    continue

            # Small delay between prediction cycles
            time.sleep(5)  # Appropriate for minute bars

        except Exception as e:
            logger.error(
                f"Prediction engine error: {e}\n{traceback.format_exc()}"
            )
            time.sleep(5)  # Pause before retrying on outer loop error

class LiveDashboard:
    def __init__(self):
        self.start_time = time.time()
        self.last_trade_count = {ticker: 0 for ticker in TICKERS}
        
    def display(self):
        while True:
            time.sleep(10)  # Update dashboard every 10 seconds
            try:
                with state.lock:
                    # Log initialization status
                    initialized_tickers = [t for t in TICKERS if state.initialized[t]]
                    if len(initialized_tickers) < len(TICKERS):
                        waiting_tickers = [t for t in TICKERS if not state.initialized[t]]
                        waiting_counts = {t: len(state.feature_buffers[t]['minute_bars']) for t in waiting_tickers}
                        print(f"Initialization Status: {len(initialized_tickers)}/{len(TICKERS)} tickers ready")
                        print(f"Waiting: {waiting_counts}")
                    
                    for ticker in TICKERS:
                        metrics = state.trade_history[ticker]['metrics']
                        current_trades = metrics['total_trades']

                        # Only log if new trades occurred or if it's been a while
                        show_update = current_trades != self.last_trade_count[ticker] or time.time() % 120 < 5
                        
                        if show_update:
                            self.last_trade_count[ticker] = current_trades
                            duration = (time.time() - self.start_time) / 60
                            win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
                            profit_min = metrics['daily_pnl'] / duration if duration > 0 else 0
                            
                            active_pos_value = 0
                            position_type = "None"
                            if state.positions[ticker]:
                                pos = state.positions[ticker]
                                position_type = pos['type']
                                current_price = state.feature_buffers[ticker]['last_price']
                                if current_price:
                                    active_pos_value = pos['size'] * current_price

                            # Add recent prediction info
                            recent_preds = list(state.recent_predictions[ticker])
                            recent_pred_summary = ""
                            if recent_preds:
                                avg_pred = sum(recent_preds) / len(recent_preds)
                                recent_pred_summary = f"Avg Prediction: {avg_pred*100:+.2f}%"
                            
                            output = f"""
=== LIVE TRADING DASHBOARD - {ticker} ===
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Uptime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))}

[Status]
- Initialized: {'Yes' if state.initialized[ticker] else 'No'}
- Current Price: ${state.feature_buffers[ticker]['last_price']:.2f}
- Active Position: {position_type}
- {recent_pred_summary}

[Performance]
- Trades: {metrics['total_trades']} 
- Win Rate: {win_rate:.1f}%
- Net Profit: ${metrics['total_profit']:+,.2f}
- Daily PnL: ${metrics['daily_pnl']:+,.2f}
- Profit/Min: ${profit_min:+,.2f}
- Max Drawdown: ${metrics['max_drawdown']:,.2f}

[Portfolio]
- Cash: ${state.portfolio['cash']:,.2f}
- Position Value: ${active_pos_value:,.2f}
- Total Value: ${state.portfolio['cash'] + active_pos_value:,.2f}
================================
"""

                            logger.info(output)
                            # Add direct terminal output
                            print(output)
                            sys.stdout.flush()  # Force output to flush immediately

            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                # Also print to terminal for visibility
                print(f"Dashboard error: {e}")

def connection_monitor():
    """
    Enhanced connection monitoring with improved error handling and reconnection logic.
    """
    while True:
        time.sleep(15)
        try:
            if not state.connection_active:
                logger.warning("Connection inactive - attempting to reconnect...")
                try:
                    ws_client.start()
                except Exception as e:
                    logger.error(f"Reconnection attempt failed: {e}")
            
            if time.time() - state.last_message > 60:  # Increased timeout for minute bars
                logger.error("No messages received for 60s - forcing reconnection")
                try:
                    state.connection_active = False
                    ws_client.client.close()
                    time.sleep(5)  # Wait before reconnection attempt
                    ws_client.start()
                except Exception as e:
                    logger.error(f"Forced reconnection failed: {e}")
            
            
        except Exception as e:
            logger.error(f"Connection monitor error: {e}")

def test_short_capabilities():
    """Test if the broker allows shorting specific tickers with more comprehensive verification."""
    logger.info("Testing short selling capabilities...")
    
    # First, clean up any existing positions and orders
    cleanup_positions_and_orders()
    
    # Now test shorts for each ticker
    for ticker in TICKERS:
        try:
            # Check if asset is shortable
            asset = trade_client.get_asset(ticker)
            shortable = asset.shortable
            borrow_fee = getattr(asset, 'easy_to_borrow', None)
            
            logger.info(f"SHORT TEST - {ticker}: Shortable={shortable}, Easy to borrow={borrow_fee}")
            
            if shortable:
                # Double-check no positions exist for this ticker
                try:
                    position = trade_client.get_open_position(ticker)
                    logger.warning(f"Found existing position for {ticker} despite cleanup. Skipping test.")
                    continue
                except Exception:
                    # No position exists, proceed with test
                    pass
                
                # Test with a small short order using IOC to avoid leaving open orders
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=1,
                    side=OrderSide.SELL,  # This is a short sell
                    time_in_force=TimeInForce.DAY,
                )
                
                try:
                    order = trade_client.submit_order(order_data=order_data)
                    logger.info(f"SHORT TEST - {ticker}: Status={order.status}, " 
                               f"Reason={getattr(order, 'rejected_reason', 'N/A')}")
                    
                    # Verify if the order was actually placed as a short
                    if hasattr(order, 'side') and order.side == OrderSide.SELL:
                        logger.info(f"SHORT TEST - {ticker}: Verified short order was accepted")
                    else:
                        logger.warning(f"SHORT TEST - {ticker}: Order may not be properly recognized as short")
                    
                    # If order was filled, close it immediately
                    if order.status == OrderStatus.FILLED:
                        close_order = MarketOrderRequest(
                            symbol=ticker,
                            qty=1,
                            side=OrderSide.BUY,  # This closes the short position
                            time_in_force=TimeInForce.DAY,
                        )
                        trade_client.submit_order(order_data=close_order)
                        logger.info(f"SHORT TEST - {ticker}: Closed test position")
                except Exception as e:
                    logger.error(f"SHORT TEST - {ticker}: Order error: {str(e)}")
            else:
                logger.warning(f"SHORT TEST - {ticker}: Not shortable according to broker")
                
        except Exception as e:
            logger.error(f"SHORT TEST - {ticker}: API error: {str(e)}")
    
    logger.info("Short selling capability test completed")

if __name__ == "__main__":
    try:
        # Run comprehensive debugging at the start
        debugging_result = debug_model_paths()
        
        # Use debugging result to determine model paths
        if debugging_result:
            found_models, found_scalers = find_model_files(debugging_result)
            
            # Update paths based on debugging results
            for ticker in TICKERS:
                if found_models.get(ticker):
                    MODEL_PATHS[ticker] = found_models[ticker]
                    print(f"Using detected model for {ticker}: {MODEL_PATHS[ticker]}")
                if found_scalers.get(ticker):
                    SCALER_PATHS[ticker] = found_scalers[ticker]
                    print(f"Using detected scaler for {ticker}: {SCALER_PATHS[ticker]}")
                    
            # Also handle directory mismatch if needed
            if debugging_result["working_dir"] and debugging_result["working_dir"] != os.path.join(BASE_DIR, "structured_models"):
                alt_dir = debugging_result["working_dir"]
                for ticker in TICKERS:
                    # Check if models exist in detected directory
                    model_path = os.path.join(alt_dir, ticker, f"enhanced_lstm_model_{ticker}_2025-03-21.pth")
                    scaler_path = os.path.join(alt_dir, ticker, f"enhanced_scaler_{ticker}_2025-03-21.pkl")
                    
                    if os.path.exists(model_path):
                        MODEL_PATHS[ticker] = model_path
                        print(f"Using alternate model path for {ticker}: {MODEL_PATHS[ticker]}")
                    
                    if os.path.exists(scaler_path):
                        SCALER_PATHS[ticker] = scaler_path
                        print(f"Using alternate scaler path for {ticker}: {SCALER_PATHS[ticker]}")
        
        # Look for all possible model files if none detected
        if not any(os.path.exists(path) for path in MODEL_PATHS.values()):
            print("\nStill no valid model files found. Searching for ANY .pth files...")
            for root, dirs, files in os.walk(BASE_DIR):
                for file in files:
                    if file.endswith('.pth'):
                        print(f"Found model file: {os.path.join(root, file)}")
                    elif file.endswith('.pkl'):
                        print(f"Found scaler file: {os.path.join(root, file)}")
        
        # After finding models and scalers, load them
        for ticker in TICKERS:
            try:
                # Check if model path exists
                model_path = MODEL_PATHS[ticker]
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at: {model_path}")
                
                print(f"Loading model for {ticker} from: {model_path}")
                
                # Initialize and load model
                models[ticker] = LSTMModel(
                    input_size=FEATURE_COUNT,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2
                ).to(device)
                
                models[ticker].load_state_dict(
                    torch.load(
                        model_path,
                        map_location=device
                    )
                )
                models[ticker].eval()
                
                # Load scaler
                scaler_path = SCALER_PATHS[ticker]
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
                
                print(f"Loading scaler for {ticker} from: {scaler_path}")
                with open(scaler_path, "rb") as f:
                    scaler_dict = pickle.load(f)
                    
                    # Extract both scalers
                    if isinstance(scaler_dict, dict) and 'feature_scaler' in scaler_dict and 'price_scaler' in scaler_dict:
                        scalers[ticker] = {
                            'price_scaler': scaler_dict['price_scaler'],
                            'feature_scaler': scaler_dict['feature_scaler']
                        }
                    else:
                        # Fallback for backward compatibility
                        scalers[ticker] = {'feature_scaler': scaler_dict, 'price_scaler': None}
                
                logger.info(f"Successfully loaded model and scaler for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to load model/scaler for {ticker}: {e}")
                # Critical error - model is required, exit program
                sys.exit(1)
                
        # Initialize state
        state = TradingState()

        # Log the trading parameters for each ticker
        for ticker in TICKERS:
            if ticker in state.signal_generators:
                logger.info(f"Trading parameters for {ticker}: " +
                          f"Entry threshold: {state.signal_generators[ticker].base_entry_threshold*100:.4f}%, " +
                          f"Exit threshold: {state.signal_generators[ticker].base_exit_threshold*100:.4f}%")

        cleanup_positions_and_orders()

        # Initialize with historical data
        logger.info("Preloading historical minute data for initialization...")
        for ticker in TICKERS:
            fetch_historical_minute_bars(ticker, lookback_days=5)
            logger.info(f"Loaded {len(state.feature_buffers[ticker]['minute_bars'])} historical bars for {ticker}")

        # Initial position synchronization
        logger.info("Performing initial position synchronization...")
        update_positions()
        
        # Initialize WebSocket client
        ws_client = RobustWebSocketClient()
        
        # Validate features
        try:
            for ticker in TICKERS:
                # Create a dummy feature vector matching the 22 features from training pipeline
                dummy_features = [
                    150.0,  # close price
                    1000,   # volume
                    150.0,  # SMA_20
                    150.0,  # EMA_20
                    50.0,   # RSI
                    150.0,  # MA5
                    155.0,  # Bollinger_High
                    145.0,  # Bollinger_Low
                    0.5,    # MACD
                    0.3,    # MACD_Signal
                    0.01,   # volatility_atr
                    20.0,   # trend_adx
                    0.5,    # momentum_roc
                    150.0,  # momentum_kama
                    0.1,    # volume_cmf
                    0.01,   # volume_em
                    1.2,    # volume_sma_ratio
                    0.02,   # price_distance_from_ma
                    0.04,   # volatility_bbw
                    50.0,   # momentum_stoch
                    10.0,   # trend_cci
                    150.0   # trend_ichimoku_a
                ]
                
                # Ensure we have exactly 22 features
                assert len(dummy_features) == FEATURE_COUNT, f"Feature count mismatch: {len(dummy_features)} vs {FEATURE_COUNT}"
                
                # Test feature scaling with separate scalers for price and technical features
                price_feature = [[dummy_features[0]]]
                technical_features = [dummy_features[1:]]
                scaled_price = scalers[ticker]['price_scaler'].transform(price_feature)[0]
                scaled_technical = scalers[ticker]['feature_scaler'].transform(technical_features)[0]
                features = np.concatenate([scaled_price, scaled_technical])
                assert features.shape[0] == FEATURE_COUNT, f"Feature dimension mismatch for {ticker}"
                
                # Test model with dummy features
                test_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                with torch.no_grad():
                    test_output = models[ticker](test_tensor)
                print(f"Model test output for {ticker}: {test_output.item()}")
                
            logger.info("Feature validation passed")
        except Exception as e:
            logger.critical(f"Feature validation failed: {e}")
            logger.critical(traceback.format_exc())
            sys.exit(1)

        # Test short selling capabilities
        test_short_capabilities()

        # Start threads with enhanced error handling
        threads = [
            threading.Thread(target=ws_client.start, name="WS-Client"),
            threading.Thread(target=prediction_engine, name="Prediction-Engine"),
            threading.Thread(target=LiveDashboard().display, name="Live-Dashboard"),
            threading.Thread(target=connection_monitor, name="Connection-Monitor")
        ]

        for t in threads:
            t.daemon = True
            t.start()
            logger.info(f"Started thread: {t.name}")

        # Main loop with improved error handling
        while True:
            time.sleep(1)
            if state.should_reset_daily_metrics():
                state.reset_daily_metrics()
                logger.info("Daily metrics reset completed")
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal - closing gracefully...")
        # Attempt to close positions if needed
        try:
            for ticker in TICKERS:
                if state.positions[ticker]:
                    logger.info(f"Closing position for {ticker}")
                    trade_client.close_position(ticker)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            os._exit(0)
    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}")
        logger.critical(traceback.format_exc())
        os._exit(1)