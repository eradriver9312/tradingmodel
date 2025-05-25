import torch
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Deque
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
import json 


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
        logging.FileHandler("realtimeshorttest_execution.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize environment and async support
load_dotenv()
# nest_asyncio.apply()

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
    //TODO Probably should refactor this to only get the position that changed in the trade update callback 
    """

    # logger.info(f"Updating positions")
    try:
        positions = trade_client.get_all_positions()
        # the user_id is hardcoded for the time being, but will need to be dynamic eventually, once we're handling multiple accounts
        update_position_cache("3bean", positions)


        # what is this? 
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
                    # trade_client.cancel_order(order_id=order.id)
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
    # print("Trade update callback triggered")
    try:
        # logger.info(f"Received trade update: {data} now updating mongo")
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

        # Use cached data if MongoDB update failed
        # if mongo_update_debug is None and order_id in order_cache:
        #     logger.info(f"Using cached order data for {order_id}")
        # print("Mongo update order debug", mongo_update_debug)
        update_positions()
        # logger.info(f"Current positions after update: {state.positions}")
    except Exception as e:
        logger.error(f"Error handling trade update: {e}")        

async def handle_trade_updates(data):
    # print("handle_trade_updates async called")
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

#TICKERS = mongo.get_all_tickers_only()
TICKERS = ['TSLA', 'NVDA', 'MSFT', 'GOOGL']
fixed_shares = mongo.get_filtered_fixed_shares(TICKERS)
#fixed_shares = mongo.get_filtered_fixed_shares(['TSLA', 'NVDA', 'MSFT', 'GOOGL']) 
# MODEL_PATHS = {ticker: f"structured_models/{ticker}/lstm_model_{ticker}_2025-02-14.pth" for ticker in TICKERS}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
MODEL_PATHS = {
    ticker: os.path.join(BASE_DIR, f"tuneenhancedstructured_models/{ticker}/lstm_model_{ticker}_2025-03-10.pth")
    for ticker in TICKERS
}
SCALER_PATHS = {
    ticker: os.path.join(BASE_DIR, f"tuneenhancedstructured_models/{ticker}/scaler_{ticker}_2025-03-10.pkl")
    for ticker in TICKERS
}
SEQUENCE_LENGTH = 60
FEATURE_COUNT = 27

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

#class LSTMModel(torch.nn.Module):
    #def __init__(self, input_size=FEATURE_COUNT, hidden_size=256, num_layers=2):
        #super(LSTMModel, self).__init__()
        #self.hidden_size = hidden_size
        #self.num_layers = num_layers
        #self.lstm = torch.nn.LSTM(
            #input_size=input_size,
            #hidden_size=hidden_size,
            #num_layers=num_layers,
            #batch_first=True
        #)
        #self.fc = torch.nn.Linear(hidden_size, 1)

    #def forward(self, x):
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #out, _ = self.lstm(x, (h0, c0))
        #out = self.fc(out[:, -1, :])
        #return out
#class updated
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=FEATURE_COUNT, hidden_size=256, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # Attention mechanism 
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size // 2, 1)
        )
        
        # Price variance prediction for dynamic range
        self.fc_variance = torch.nn.Linear(hidden_size, 1)
        
        # Main prediction branch
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size // 2, 1)
        
        # Output transformation for combining prediction and variance
        self.output_transform = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attn_weights = torch.nn.functional.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        
        # Apply dropout to the attention output
        context = self.dropout(context)
        
        # Base prediction
        base_pred = self.fc2(self.relu(self.fc1(context)))
        
        # Predict variance (confidence)
        pred_variance = torch.exp(self.fc_variance(context))
        
        # Combine for final prediction
        combined = torch.cat((base_pred, pred_variance), dim=1)
        final_output = self.output_transform(combined)
        
        return final_output


# Set up device with proper error handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize models and scalers with enhanced error handling and security
models = {}
scalers = {}
for ticker in TICKERS:
    try:
        # Validate model path exists
        if not os.path.exists(MODEL_PATHS[ticker]):
            raise FileNotFoundError(f"Model file not found for {ticker}")
        
        # Initialize model
        models[ticker] = LSTMModel(
            input_size=FEATURE_COUNT,
            hidden_size=256,
            dropout=0.5,
            num_layers=2
        ).to(device)
        
        # Load model with security measures
        try:
            models[ticker].load_state_dict(
                torch.load(
                    MODEL_PATHS[ticker],
                    map_location=device,
                    weights_only=True  # Security enhancement
                )
            )
        except Exception as e:
            raise Exception(f"Failed to load model weights for {ticker}: {e}")
        
        models[ticker].eval()
        
        # Load and validate scaler
        if not os.path.exists(SCALER_PATHS[ticker]):
            raise FileNotFoundError(f"Scaler file not found for {ticker}")
            
        with open(SCALER_PATHS[ticker], "rb") as f:
            scalers[ticker] = pickle.load(f)
        
        # Validate scaler features
        if not hasattr(scalers[ticker], 'n_features_in_'):
            raise ValueError(f"Invalid scaler format for {ticker}")
            
        if scalers[ticker].n_features_in_ != FEATURE_COUNT:
            raise ValueError(f"Scaler feature mismatch for {ticker}: expected {FEATURE_COUNT}, got {scalers[ticker].n_features_in_}")

        logger.info(f"Successfully initialized model and scaler for {ticker}")

    except Exception as e:
        logger.critical(f"Initialization failed for {ticker}: {e}")
        sys.exit(1)

class AdaptiveSignalGenerator:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock_profiles = {
            'MSFT': {
                'base_entry_threshold': 0.0006,
                'base_short_entry_threshold': 0.0003,    # Changed from 0.2 to 1.0
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
                'base_short_entry_threshold': 0.0004,  # Add direct SHORT threshold
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
                'base_short_entry_threshold': 0.0012,  # Add direct SHORT threshold
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
                'base_short_entry_threshold': 0.001,  # Add direct SHORT threshold
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
                'short_entry_threshold_factor': 0.8,  # Changed from 0.2 to 1.0
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
                'short_entry_threshold_factor': 0.8,  # Changed from 0.2 to 1.0
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
                'short_entry_threshold_factor': 0.8,  # Changed from 0.2 to 1.0
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
                'short_entry_threshold_factor': 0.8,  # Changed from 0.2 to 1.0
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
                'short_entry_threshold_factor': 0.8,  # Changed from 0.2 to 1.0
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
        # Default values for any ticker not explicitly listed
        self.default_profile = {
            'base_entry_threshold': 0.0015,
            'base_short_entry_threshold': 0.0008,  # Add direct SHORT threshold
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
        
    import traceback

    def _initialize_parameters(self):
        """Initialize trading parameters from stock profile with improved consistency."""
        # Base parameters from stock profile
        self.base_entry_threshold = self.params['base_entry_threshold']
        self.short_entry_threshold = self.params['base_short_entry_threshold']  # Direct SHORT threshold
        self.base_exit_threshold = self.params['base_exit_threshold']
        self.base_stop_loss = self.params['base_stop_loss']
        self.atr_multiplier = self.params['atr_multiplier']
        self.min_hold_time = self.params['min_hold_time']
        self.volatility_threshold = self.params['volatility_threshold']
        self.volume_threshold = self.params['volume_threshold']
        self.trend_threshold = self.params['trend_threshold']
        
        # Define trend reversal threshold as half of the trend threshold
        self.trend_reversal_threshold = self.trend_threshold / 2.0
        
        logger.info(f"Initialized thresholds for {self.ticker}: long={self.base_entry_threshold}, short={self.short_entry_threshold}")
        
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
        
        # Direction confidence for balanced threshold application
        self.direction_confidence = {'up': 0.75, 'down': 0.75}
        
        # Performance tracking
        self.winning_trades = 0
        self.total_trades = 0
        self.trade_pnl = []
        
        # Debugging flags
        self.debug_mode = False
        self.last_signal_reason = None
        
        # Initialize adaptive parameters based on recent market conditions
        self.adaptive_thresholds = {
            'long_entry': self.base_entry_threshold,
            'short_entry': self.short_entry_threshold,
            'exit': self.base_exit_threshold,
            'stop_loss': self.base_stop_loss
        }
        
        logger.info(f"Initialized signal generator for {self.ticker} - " +
                    f"Long threshold: {self.base_entry_threshold*100:.4f}%, " +
                    f"Short threshold: {self.short_entry_threshold*100:.4f}%, " +
                    f"Exit threshold: {self.base_exit_threshold*100:.4f}%, " +
                    f"Trend reversal threshold: {self.trend_reversal_threshold*100:.4f}%")

        
    def generate_signal(self, current_price: float, predicted_price: float, timestamp: datetime, 
                    market_data: pd.DataFrame = None) -> Dict:
        """Generate trading signals with manual SHORT thresholds for balanced long/short generation."""                     
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
        short_trend = 0
        medium_trend = 0
        
        if len(price_history) >= 10:
            # Calculate short-term trend (last 5 bars)
            short_trend = price_history[-1] / price_history[-5] - 1 if price_history[-5] > 0 else 0
            # Calculate medium-term trend (last 10 bars)
            medium_trend = price_history[-1] / price_history[-10] - 1 if price_history[-10] > 0 else 0

        # Log raw prediction for diagnostic purposes
        raw_prediction_change = (predicted_price - current_price) / current_price
        logger.debug(f"PREDICTION ANALYSIS - {self.ticker}: current={current_price:.2f}, prediction={predicted_price:.2f}, raw_change={raw_prediction_change*100:.4f}%")

        # Calculate bias correction - REDUCED to match backtest environment
        # Calculate stronger bias correction to address systematic upward bias
        bias_correction = current_price * 0.01  # 1% bias correction

        # Apply asymmetric bias correction that favors short signals
        corrected_prediction = predicted_price
        if raw_prediction_change > 0:
            # For long positions (positive predictions), reduce prediction
            corrected_prediction = predicted_price - bias_correction
        else:
            # For short positions (negative predictions), ALSO apply negative adjustment
            # This asymmetric approach helps overcome the systematic upward bias
            corrected_prediction = predicted_price - bias_correction * 0.5
        
        # Calculate price movement percentage with corrected prediction
        price_change_pct = (corrected_prediction - current_price) / current_price

        # Enhanced logging to track all prediction processing steps
        logger.debug(f"PREDICTION DEBUG - {self.ticker}: raw={raw_prediction_change*100:.4f}%, " +
                    f"corrected={price_change_pct*100:.4f}%, bias={bias_correction/current_price*100:.4f}%")

        # Enhanced logging for SHORT signal diagnostics
        if price_change_pct < -0.0005:  # Log any significant negative prediction
            logger.info(f"SHORT SIGNAL ANALYSIS - {self.ticker}: price={current_price:.2f}, " +
                    f"prediction={corrected_prediction:.2f}, change={price_change_pct*100:.4f}%, " +
                    f"threshold={-self.short_entry_threshold*100:.4f}%, " +
                    f"meets_threshold={price_change_pct < -self.short_entry_threshold}")

        if self.position is None:
            # Use fixed thresholds instead of volatility-adjusted ones
            long_threshold = self.base_entry_threshold
            short_threshold = self.short_entry_threshold
            
            logger.debug(f"THRESHOLDS - {self.ticker}: long={long_threshold*100:.4f}%, short={short_threshold*100:.4f}%")

            # Check volume condition if market data available
            volume_ok = True
            if market_data is not None and len(market_data) > 0 and self.volume_threshold > 0:
                current_volume = market_data['volume'].iloc[-1]
                avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
                volume_ok = (current_volume / avg_volume) >= self.volume_threshold

            # Use balanced trend checks for long and short positions
            long_trend_ok = True
            short_trend_ok = True  # More permissive for shorts to match backtesting
            
            # Log trend conditions
            logger.debug(f"SIGNAL EVALUATION - {self.ticker}: change={price_change_pct*100:.4f}%, " + 
                    f"long_ok={long_trend_ok}, short_ok={short_trend_ok}, volume_ok={volume_ok}, " +
                    f"short_trend={short_trend*100:.4f}%, threshold_long={long_threshold*100:.4f}%, threshold_short={-short_threshold*100:.4f}%")

            # Track near-miss short signals for debugging
            if price_change_pct < -short_threshold * 0.5 and price_change_pct > -short_threshold:
                logger.debug(f"NEAR SHORT - {self.ticker}: change={price_change_pct*100:.4f}%, " +
                        f"threshold={-short_threshold*100:.4f}%, trend_ok={short_trend_ok}, volume_ok={volume_ok}")
                
            # Entry conditions - check if we should enter a position
            if price_change_pct > long_threshold and volume_ok and long_trend_ok:
                # Long signal processing 
                # Calculate standard stop loss and take profit thresholds
                stop_loss = self.base_stop_loss
                take_profit = self.base_exit_threshold

                # Create the signal for long position
                signal = {
                    'action': 'enter_long',
                    'price': current_price,
                    'timestamp': timestamp,
                    'thresholds': {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                }
                
                # Update position tracking
                self.position = 'long'
                self.entry_price = current_price
                self.entry_time = timestamp
                self.daily_trades += 1

                logger.info(f"LONG SIGNAL: {self.ticker} at {timestamp}, price={current_price:.2f}, " +
                        f"predicted_change={price_change_pct*100:.2f}%, threshold={long_threshold*100:.2f}%")
                
                return signal
            
            elif price_change_pct < -short_threshold and volume_ok and short_trend_ok:
                # Special debug for short signal detection
                logger.info(f"SHORT SIGNAL DETECTED - {self.ticker}: price={current_price:.2f}, " +
                        f"prediction={corrected_prediction:.2f}, change={price_change_pct*100:.4f}%, " +
                        f"threshold={-short_threshold*100:.4f}%")
                
                # Calculate standard stop loss and take profit thresholds
                stop_loss = self.base_stop_loss
                take_profit = self.base_exit_threshold

                # Create the signal for short position
                signal = {
                    'action': 'enter_short',
                    'price': current_price,
                    'timestamp': timestamp,
                    'thresholds': {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                }
                
                # Update position tracking
                self.position = 'short'
                self.entry_price = current_price
                self.entry_time = timestamp
                self.daily_trades += 1

                logger.info(f"SHORT SIGNAL: {self.ticker} at {timestamp}, price={current_price:.2f}, " +
                        f"predicted_change={price_change_pct*100:.2f}%, threshold={-short_threshold*100:.2f}%")
                
                return signal
            
        else:
            # Position exit logic
            if self.position == 'long':
                # Exit long position conditions
                price_change_from_entry = (current_price - self.entry_price) / self.entry_price
                
                # Stop loss check
                if price_change_from_entry < -self.base_stop_loss:
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
                    
                    return signal
                
                # Take profit check
                elif price_change_from_entry > self.base_exit_threshold:
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
                    
                    return signal
                    
            elif self.position == 'short':
                # Exit short position conditions
                price_change_from_entry = (self.entry_price - current_price) / self.entry_price
                
                # Stop loss check
                if price_change_from_entry < -self.base_stop_loss:
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
                    
                    return signal
                
                # Take profit check
                elif price_change_from_entry > self.base_exit_threshold:
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
                    
                    return signal
        
        # No signal generated
        return None

class RobustWebSocketClient:
    def __init__(self):
        self.client = WebSocketClient(
            api_key=API_KEY,
            subscriptions=[f"T.{ticker}" for ticker in TICKERS],
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
                if hasattr(msg, 'symbol') and msg.symbol in TICKERS:
                    processed = preprocess_trade({
                        'symbol': msg.symbol,
                        'price': msg.price,
                        'size': msg.size,
                        'timestamp': datetime.fromtimestamp(msg.timestamp/1000)
                    })
                    with state.lock:
                        if processed is not None:
                            state.data_window[msg.symbol].append(processed)
                            # Update position information after each trade
                        
        except Exception as e:
            logger.error(f"Message processing error: {e}")

    def start(self):
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                state.connection_active = True
                self.client.run(
                    handle_msg=self.message_handler,
                    close_timeout=5
                )
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.reconnect_attempts += 1
                time.sleep(self.reconnect_delay)
                continue
            
            self.reconnect_attempts = 0  # Reset counter on successful connection
            
        logger.critical("Maximum reconnection attempts reached. Exiting...")
        sys.exit(1)

class TradingState:
    def __init__(self):
        self.lock = threading.RLock()
        self.data_window = {ticker: deque(maxlen=SEQUENCE_LENGTH * 2) for ticker in TICKERS}
        self.feature_buffers = {
            ticker: {
                'prices': deque(maxlen=26),
                'volumes': deque(maxlen=20),  # Increased for volume average
                'rsi_window': deque(maxlen=14),
                'last_price': None,
                'price_history': deque(maxlen=100),
                'market_data': pd.DataFrame()  # Added for technical indicators
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
                    logger.info(f"Order submitted for {ticker}: {order_side.name}, {position_size} shares, status: {order.status}")
                
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

def prediction_engine():
    """
    Continuous prediction engine that generates trading signals based on ML model predictions.
    Enhanced with improved short signal generation and expanded prediction range.
    """
    # Initialize recent predictions tracking if not exists
    if not hasattr(state, 'recent_predictions'):
        state.recent_predictions = {ticker: deque(maxlen=20) for ticker in TICKERS}
        
    while True:
        try:
            # Validate market and connection status
            if not state.is_market_open():
                logger.info("Market is closed - pausing prediction engine")
                time.sleep(1)
                continue

            if not state.connection_active:
                logger.warning("WebSocket connection inactive - skipping prediction cycle")
                time.sleep(1)
                continue

            # Process each ticker
            for ticker in TICKERS:
                try:
                    with state.lock:
                        # Check data availability
                        if len(state.data_window[ticker]) < SEQUENCE_LENGTH:
                            logger.debug(
                                f"Insufficient data for {ticker}: "
                                f"{len(state.data_window[ticker])} < {SEQUENCE_LENGTH}"
                            )
                            continue

                        if not state.feature_buffers[ticker]["last_price"]:
                            logger.debug(f"No last price available for {ticker}")
                            continue

                        # Extract sequence from data_window (deque)
                        data_list = list(state.data_window[ticker])
                        sequence = np.array(data_list[-SEQUENCE_LENGTH:], dtype=np.float32)

                        # Create tensor for model input
                        tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)

                        # Generate raw prediction
                        with torch.no_grad():
                            raw_prediction = models[ticker](tensor).item()
                            
                        # Get current market data
                        current_price = state.feature_buffers[ticker]["last_price"]
                        
                        # Apply prediction range expansion
                        prediction = apply_prediction_range_expansion(ticker, current_price, raw_prediction)
                        
                        # Log raw and adjusted predictions
                        raw_change = (raw_prediction - current_price) / current_price
                        adjusted_change = (prediction - current_price) / current_price
                        logger.debug(f"PREDICTION DETAIL - {ticker}: current={current_price:.2f}, "
                                f"raw_pred={raw_prediction:.2f} ({raw_change*100:.4f}%), "
                                f"adjusted_pred={prediction:.2f} ({adjusted_change*100:.4f}%)")
                        
                        # Clean up memory
                        del tensor
                        torch.cuda.empty_cache()

                    # Generate timestamp outside lock to minimize lock duration
                    timestamp = datetime.now(state.tz)

                    # Generate trading signal with enhanced short capability
                    signal = state.signal_generators[ticker].generate_signal(
                        current_price, prediction, timestamp
                    )

                    # Handle signal if generated
                    if signal:
                        logger.info(f"Signal generated for {ticker}: {signal['action']} @ ${signal['price']:.2f}")
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

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)  # 10ms delay between cycles

        except Exception as e:
            logger.error(
                f"Prediction engine error: {e}\n{traceback.format_exc()}"
            )
            time.sleep(1)  # Pause before retrying on outer loop error

def apply_prediction_range_expansion(ticker, current_price, raw_prediction):
    """
    Apply the same post-processing to expand prediction range as used in training.
    This ensures consistency between training and production environments.
    """
    try:
        # Get historical price data for this ticker
        price_history = list(state.feature_buffers[ticker]['price_history'])
        if len(price_history) < 10:  # Need sufficient history
            return raw_prediction
            
        # Calculate true price range from recent history
        true_range = max(price_history) - min(price_history)
        if true_range <= 0.001:  # Avoid division by zero
            return raw_prediction
            
        # Get recent predictions if available
        if len(state.recent_predictions[ticker]) < 5:  # Need sufficient prediction history
            return raw_prediction
            
        # Calculate prediction range from recent predictions
        recent_preds = list(state.recent_predictions[ticker])
        pred_range = max(recent_preds) - min(recent_preds)
        if pred_range <= 0.001:  # Avoid division by zero
            return raw_prediction
        
        # Calculate range ratio
        range_ratio = pred_range / true_range
        
        # Calculate raw change percentage
        raw_change = (raw_prediction - current_price) / current_price
        
        # Apply post-processing using the same logic as in training pipeline
        # Apply modified range expansion that doesn't amplify bullish bias
        if range_ratio < 0.6:
            # Calculate expected range based on historical true range
            expected_range = true_range * 0.7
            
            # Determine if current prediction range is too small
            current_range = abs(raw_prediction - current_price)
            
            if current_range < expected_range * 0.3:  # If using less than 30% of expected range
                # Apply stronger expansion but preserve prediction direction
                direction = np.sign(raw_prediction - current_price)
                # Use a balanced approach - 40% of expected range in the direction of prediction
                adjusted_prediction = current_price + (direction * expected_range * 0.4)
                adjusted_change = (adjusted_prediction - current_price) / current_price
                expansion_factor = 0.4  # For logging
                
                logger.info(f"PREDICTION EXPANSION - {ticker}: " +
                          f"raw={raw_prediction:.2f} ({raw_change*100:.2f}%), " +
                          f"adjusted={adjusted_prediction:.2f} ({adjusted_change*100:.2f}%), " +
                          f"ratio={range_ratio:.2f}, factor={expansion_factor:.2f}")
                
                return adjusted_prediction
            else:
                return raw_prediction
        else:
            # Use traditional approach for larger movements
            pred_mean = np.mean(recent_preds)
            expansion_factor = min((true_range / pred_range) * 0.7, 5.0)
            adjusted_prediction = pred_mean + (raw_prediction - pred_mean) * expansion_factor
            adjusted_change = (adjusted_prediction - current_price) / current_price
            
            logger.info(f"PREDICTION EXPANSION - {ticker}: " +
                      f"raw={raw_prediction:.2f} ({raw_change*100:.2f}%), " +
                      f"adjusted={adjusted_prediction:.2f} ({adjusted_change*100:.2f}%), " +
                      f"ratio={range_ratio:.2f}, factor={expansion_factor:.2f}")
            
            return adjusted_prediction
            
    except Exception as e:
        logger.error(f"Error in prediction expansion for {ticker}: {e}")
        return raw_prediction  # Fall back to raw prediction on error

class LiveDashboard:
    def __init__(self):
        self.start_time = time.time()
        self.last_trade_count = {ticker: 0 for ticker in TICKERS}
        
    def display(self):
        while True:
            time.sleep(5)  # Keep original timing
            try:
                with state.lock:
                    for ticker in TICKERS:
                        metrics = state.trade_history[ticker]['metrics']
                        current_trades = metrics['total_trades']


                        
                        # Only log if new trades occurred
                        if current_trades != self.last_trade_count[ticker]:
                            self.last_trade_count[ticker] = current_trades
                            duration = (time.time() - self.start_time) / 60
                            win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
                            profit_min = metrics['daily_pnl'] / duration if duration > 0 else 0
                            
                            active_pos_value = 0
                            if state.positions[ticker]:
                                pos = state.positions[ticker]
                                current_price = state.feature_buffers[ticker]['last_price']
                                if current_price:
                                    active_pos_value = pos['size'] * current_price

                            output = f"""
=== LIVE TRADING DASHBOARD - {ticker} ===
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Uptime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))}

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

# 2. Update preprocess_trade to calculate more of the training features
def preprocess_trade(trade: Dict) -> List[float]:
    """
    Enhanced preprocess function that calculates features matching the training pipeline.
    Returns a feature vector aligned with what the model was trained on.
    """
    try:
        ticker = trade['symbol']
        price = trade['price']
        volume = trade['size']
        timestamp = trade['timestamp']
        
        buffer = state.feature_buffers[ticker]
        
        # Update price and volume buffers
        buffer['prices'].append(price)
        buffer['volumes'].append(volume)
        buffer['price_history'].append(price)
        buffer['last_price'] = price
        
        # Wait until we have enough data
        if len(buffer['prices']) < 26:  # Need at least 26 points for EMA
            return None
            
        # ---------- CORE FEATURES (match training pipeline) ----------
        
        # 1. Basic price/volume data
        prices_list = list(buffer['prices'])
        volumes_list = list(buffer['volumes'])
        
        # 2. Calculate moving averages - using proper list indexing
        last_n_prices = prices_list[-5:] if len(prices_list) >= 5 else prices_list[:]
        ma5 = sum(last_n_prices) / len(last_n_prices)
        
        last_n_prices = prices_list[-20:] if len(prices_list) >= 20 else prices_list[:]
        sma_20 = sum(last_n_prices) / len(last_n_prices)
        
        # Calculate EMA20 - safer implementation without negative indices
        if len(prices_list) >= 20:
            alpha = 2 / (20 + 1)
            ema_20 = prices_list[-20]  # Start with oldest value in our window
            for i in range(len(prices_list) - 19, len(prices_list)):  # Process remaining 19 values
                ema_20 = prices_list[i] * alpha + ema_20 * (1 - alpha)
        else:
            ema_20 = price
        
        # 3. Calculate RSI - safer implementation
        if len(prices_list) >= 15:
            changes = []
            for i in range(1, len(prices_list)):
                changes.append(prices_list[i] - prices_list[i-1])
                
            gains = [max(0, c) for c in changes]  # Use max instead of conditional
            losses = [max(0, -c) for c in changes]  # Use max instead of conditional
            
            avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else sum(gains) / max(1, len(gains))
            avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else sum(losses) / max(1, len(losses))
            
            rs = avg_gain / max(0.001, avg_loss)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Default when not enough data
            
        # 4. Calculate MACD - safer implementation
        if len(prices_list) >= 26:
            # Calculate EMA12
            alpha_12 = 2 / (12 + 1)
            ema_12 = prices_list[0]  # Start with first value
            for i in range(1, len(prices_list)):
                ema_12 = prices_list[i] * alpha_12 + ema_12 * (1 - alpha_12)
                
            # Calculate EMA26
            alpha_26 = 2 / (26 + 1)
            ema_26 = prices_list[0]  # Start with first value
            for i in range(1, len(prices_list)):
                ema_26 = prices_list[i] * alpha_26 + ema_26 * (1 - alpha_26)
                
            macd = ema_12 - ema_26
            
            # MACD Signal (9-period EMA of MACD)
            macd_signal = macd  # Simplified - in real implementation would track historical MACD values
        else:
            macd = 0
            macd_signal = 0
            
        # 5. Bollinger Bands - safer implementation
        if len(prices_list) >= 20:
            last_20_prices = prices_list[-20:]
            sma_20 = sum(last_20_prices) / 20
            squared_diffs = [(p - sma_20)**2 for p in last_20_prices]
            variance = sum(squared_diffs) / 20
            std_20 = variance**0.5  # Manual standard deviation calculation
            bollinger_high = sma_20 + 2 * std_20
            bollinger_low = sma_20 - 2 * std_20
            bollinger_width = (bollinger_high - bollinger_low) / sma_20 if sma_20 > 0 else 0
        else:
            bollinger_high = price * 1.02
            bollinger_low = price * 0.98
            bollinger_width = 0.04
            
        # 6. Price change and returns - safer implementation
        if len(prices_list) > 1:
            previous_price = prices_list[-2]
            price_change = price / previous_price - 1
            log_return = np.log(price / previous_price) if previous_price > 0 else 0
        else:
            price_change = 0
            log_return = 0
        
        # 7. Volume analysis
        volume_history = volumes_list[-20:] if len(volumes_list) >= 20 else volumes_list[:]
        volume_ma = sum(volume_history) / len(volume_history)
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1
        price_volume = abs(price_change) * volume_ratio
        
        # 8. Price patterns (simplified) - safer implementation
        if len(prices_list) >= 2:
            last_two_prices = prices_list[-2:]
            daily_range = max(last_two_prices) - min(last_two_prices)
            daily_range_ratio = daily_range / price if price > 0 else 0
        else:
            daily_range = 0
            daily_range_ratio = 0
            
        if len(prices_list) > 1:
            body_size = abs(price - prices_list[-2]) / prices_list[-2] if prices_list[-2] > 0 else 0
        else:
            body_size = 0
        
        # 9. Enhanced features
        price_acceleration = 0
        if len(prices_list) >= 3:
            if prices_list[-3] > 0 and prices_list[-2] > 0:  # Prevent division by zero
                return_t1 = prices_list[-1] / prices_list[-2] - 1
                return_t2 = prices_list[-2] / prices_list[-3] - 1
                price_acceleration = return_t1 - return_t2
            
        price_dist_from_mean_5d = (price - ma5) / price if price > 0 else 0
        price_dist_from_mean_20d = (price - sma_20) / price if price > 0 else 0
        
        # 10. Trend and momentum - safer implementation
        price_momentum_5d = 0
        price_momentum_20d = 0
        if len(prices_list) >= 6:
            momentum_sum = 0
            for i in range(1, 6):  # Calculate for last 5 periods
                if i < len(prices_list) and prices_list[-(i+1)] > 0:  # Ensure index is valid
                    momentum_sum += prices_list[-i] / prices_list[-(i+1)] - 1
            price_momentum_5d = momentum_sum
                
        if len(prices_list) >= 21:
            momentum_sum = 0
            for i in range(1, 21):  # Calculate for last 20 periods
                if i < len(prices_list) and prices_list[-(i+1)] > 0:  # Ensure index is valid
                    momentum_sum += prices_list[-i] / prices_list[-(i+1)] - 1
            price_momentum_20d = momentum_sum
            
        trend_strength = (ma5 - sma_20) / sma_20 if sma_20 > 0 else 0
        ma_cross = 1 if ma5 > sma_20 else 0
            
        # 11. Volatility - safer implementation
        volatility = 0.01  # Default
        if len(prices_list) >= 3:  # Need at least a few points for meaningful volatility
            returns = []
            for i in range(1, min(21, len(prices_list))):
                if prices_list[-i-1] > 0:  # Prevent division by zero
                    returns.append(prices_list[-i] / prices_list[-i-1] - 1)
                    
            if returns:  # Make sure we have returns to calculate std
                # Manual std calculation to avoid NumPy issues
                mean_return = sum(returns) / len(returns)
                squared_diffs = [(r - mean_return)**2 for r in returns]
                variance = sum(squared_diffs) / len(returns)
                std_return = variance**0.5
                volatility = std_return * (252**0.5)  # Annualize
        
        # 12. Additional transformations
        log_abs_return = np.log(abs(price_change) + 1e-6)
        return_sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
        
        # Create complete feature vector
        features = [
            price,                  # 1. Current price
            volume,                 # 2. Current volume
            sma_20,                 # 3. 20-period SMA 
            ema_20,                 # 4. 20-period EMA
            rsi,                    # 5. RSI
            ma5,                    # 6. 5-period MA
            bollinger_high,         # 7. Bollinger high
            bollinger_low,          # 8. Bollinger low
            macd,                   # 9. MACD
            macd_signal,            # 10. MACD signal
            volatility,             # 11. Volatility
            trend_strength,         # 12. Trend strength
            body_size,              # 13. Body size
            volume_ratio,           # 14. Volume ratio
            price_volume,           # 15. Price-volume
            price_acceleration,     # 16. Price acceleration
            price_dist_from_mean_5d,  # 17. Price distance from 5d mean
            price_dist_from_mean_20d, # 18. Price distance from 20d mean
            price_momentum_5d,      # 19. 5-day momentum
            price_momentum_20d,     # 20. 20-day momentum
            log_abs_return,         # 21. Log of absolute return
            daily_range_ratio,      # 22. Daily range ratio
            bollinger_width,        # 23. Bollinger width
            ma_cross,               # 24. MA cross
            return_sign,            # 25. Return sign
            log_return,             # 26. Log return
            price_change            # 27. Price change
        ]
        
        # Scale features using the saved scaler
        scaled_features = scalers[ticker].transform([features])[0]
        return scaled_features

    except Exception as e:
        logger.error(f"Error preprocessing trade for {trade['symbol']}: {e}")
        # Log detailed error for debugging
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

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
            
            if time.time() - state.last_message > 30:
                logger.error("No messages received for 30s - forcing reconnection")
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

# if __name__ == "__main__":
#     try:
#         # Initialize state
#         state = TradingState()
        
#         # Add recent predictions tracking for range expansion
#         state.recent_predictions = {ticker: deque(maxlen=20) for ticker in TICKERS}

#         cleanup_positions_and_orders()

#         # Initial position synchronization
#         logger.info("Performing initial position synchronization...")
#         update_positions()
        
#         # Initialize WebSocket client
#         ws_client = RobustWebSocketClient()
        
#         # Validate features
#         try:
#             for ticker in TICKERS:
#                 # Create a dummy feature vector with realistic values
#                 dummy_features = [
#                     150.0,  # price
#                     1000,   # volume
#                     150.0,  # sma_20
#                     150.0,  # ema_20
#                     50.0,   # rsi
#                     150.0,  # ma5
#                     155.0,  # bollinger_high
#                     145.0,  # bollinger_low
#                     0.5,    # macd
#                     0.3,    # macd_signal
#                     0.01,   # volatility
#                     0.005,  # trend_strength
#                     0.003,  # body_size
#                     1.2,    # volume_ratio
#                     0.005,  # price_volume
#                     0.001,  # price_acceleration
#                     0.02,   # price_dist_from_mean_5d
#                     0.03,   # price_dist_from_mean_20d
#                     0.01,   # price_momentum_5d
#                     0.02,   # price_momentum_20d
#                     -5.0,   # log_abs_return
#                     0.01,   # daily_range_ratio
#                     0.02,   # bollinger_width
#                     1.0,    # ma_cross
#                     1.0,    # return_sign
#                     0.001,  # log_return
#                     0.001   # price_change
#                 ]
                
#                 # Ensure we have exactly 27 features
#                 assert len(dummy_features) == FEATURE_COUNT, f"Feature count mismatch: {len(dummy_features)} vs {FEATURE_COUNT}"
                
#                 features = scalers[ticker].transform([dummy_features])
#                 assert features.shape == (1, FEATURE_COUNT), f"Feature dimension mismatch for {ticker}"
#             logger.info("Feature validation passed")
#         except Exception as e:
#             logger.critical(f"Feature validation failed: {e}")
#             sys.exit(1)

#         # Test short selling capabilities
#         test_short_capabilities()

#         # Reset short entry threshold factors to match training values
#         logger.info("Adjusting SHORT entry thresholds to match training configuration...")
#         for ticker in TICKERS:
#             if ticker in state.signal_generators:
#         #         # Set threshold factor to the value used in training
#                 state.signal_generators[ticker].short_entry_threshold_factor = 0.2
#                 state.signal_generators[ticker].short_entry_threshold = state.signal_generators[ticker].base_entry_threshold * 0.2
#                 logger.info(f"Updated {ticker} SHORT threshold: {state.signal_generators[ticker].short_entry_threshold*100:.4f}%")

#         # Start threads with enhanced error handling
#         threads = [
#             threading.Thread(target=ws_client.start, name="WS-Client"),
#             threading.Thread(target=prediction_engine, name="Prediction-Engine"),
#             threading.Thread(target=LiveDashboard().display, name="Live-Dashboard"),
#             threading.Thread(target=connection_monitor, name="Connection-Monitor")
#         ]

#         for t in threads:
#             t.daemon = True
#             t.start()
#             logger.info(f"Started thread: {t.name}")

#         # Main loop with improved error handling
#         while True:
#             time.sleep(1)
#             if state.should_reset_daily_metrics():
#                 state.reset_daily_metrics()
#                 logger.info("Daily metrics reset completed")
            
#     except KeyboardInterrupt:
#         logger.info("Received shutdown signal - closing gracefully...")
#         # Attempt to close positions if needed
#         try:
#             for ticker in TICKERS:
#                 if state.positions[ticker]:
#                     logger.info(f"Closing position for {ticker}")
#                     trade_client.close_position(ticker)
#         except Exception as e:
#             logger.error(f"Error during shutdown: {e}")
#         finally:
#             os._exit(0)
#     except Exception as e:
#         logger.critical(f"Critical error in main loop: {e}")
#         os._exit(1)

if __name__ == "__main__":
    try:
        # Initialize state
        state = TradingState()
        
        # Add recent predictions tracking for range expansion
        state.recent_predictions = {ticker: deque(maxlen=20) for ticker in TICKERS}

        # Update stock profiles to include direct SHORT thresholds
        for ticker in TICKERS:
            if ticker in state.signal_generators:
                # Set manual SHORT thresholds based on existing long thresholds
                if ticker == 'MSFT':
                    state.signal_generators[ticker].params['base_short_entry_threshold'] = 0.0003  # 0.03%
                elif ticker == 'GOOGL':
                    state.signal_generators[ticker].params['base_short_entry_threshold'] = 0.0004  # 0.04%
                elif ticker == 'TSLA':
                    state.signal_generators[ticker].params['base_short_entry_threshold'] = 0.0012  # 0.12%
                elif ticker == 'NVDA':
                    state.signal_generators[ticker].params['base_short_entry_threshold'] = 0.001   # 0.10%
                else:
                    # Default case - set to 50% of long threshold
                    base_threshold = state.signal_generators[ticker].params['base_entry_threshold']
                    state.signal_generators[ticker].params['base_short_entry_threshold'] = base_threshold * 0.5
                
                # Reinitialize parameters to apply the new thresholds
                state.signal_generators[ticker]._initialize_parameters()
                
                logger.info(f"Updated {ticker} thresholds - LONG: {state.signal_generators[ticker].base_entry_threshold*100:.4f}%, "
                           f"SHORT: {state.signal_generators[ticker].short_entry_threshold*100:.4f}%")

        cleanup_positions_and_orders()

        # Initial position synchronization
        logger.info("Performing initial position synchronization...")
        update_positions()
        
        # Initialize WebSocket client
        ws_client = RobustWebSocketClient()
        
        # Validate features
        try:
            for ticker in TICKERS:
                # Create a dummy feature vector with realistic values
                dummy_features = [
                    150.0,  # price
                    1000,   # volume
                    150.0,  # sma_20
                    150.0,  # ema_20
                    50.0,   # rsi
                    150.0,  # ma5
                    155.0,  # bollinger_high
                    145.0,  # bollinger_low
                    0.5,    # macd
                    0.3,    # macd_signal
                    0.01,   # volatility
                    0.005,  # trend_strength
                    0.003,  # body_size
                    1.2,    # volume_ratio
                    0.005,  # price_volume
                    0.001,  # price_acceleration
                    0.02,   # price_dist_from_mean_5d
                    0.03,   # price_dist_from_mean_20d
                    0.01,   # price_momentum_5d
                    0.02,   # price_momentum_20d
                    -5.0,   # log_abs_return
                    0.01,   # daily_range_ratio
                    0.02,   # bollinger_width
                    1.0,    # ma_cross
                    1.0,    # return_sign
                    0.001,  # log_return
                    0.001   # price_change
                ]
                
                # Ensure we have exactly 27 features
                assert len(dummy_features) == FEATURE_COUNT, f"Feature count mismatch: {len(dummy_features)} vs {FEATURE_COUNT}"
                
                features = scalers[ticker].transform([dummy_features])
                assert features.shape == (1, FEATURE_COUNT), f"Feature dimension mismatch for {ticker}"
            logger.info("Feature validation passed")
        except Exception as e:
            logger.critical(f"Feature validation failed: {e}")
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
        os._exit(1)