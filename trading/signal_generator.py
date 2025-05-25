"""
Trading signal generation for the LSTM trading model.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from config import logger, STOCK_PROFILES, DEFAULT_STOCK_PROFILE

class AdaptiveSignalGenerator:
    """Generate trading signals with adaptive thresholds based on market conditions."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        # Use stock-specific profile if available, otherwise use default
        self.params = STOCK_PROFILES.get(ticker, DEFAULT_STOCK_PROFILE)
        self._initialize_parameters()
        
        # For storing prediction confidence levels
        self.direction_confidence = {'up': 0.5, 'down': 0.5}
        
    def _initialize_parameters(self):
        """Initialize trading parameters from stock profile."""
        self.base_entry_threshold = self.params['base_entry_threshold']
        # Use full value for short entry threshold (removed the 0.7 reduction factor)
        self.short_entry_threshold_factor = self.params['short_entry_threshold_factor']
        self.base_exit_threshold = self.params['base_exit_threshold']
        self.base_stop_loss = self.params['base_stop_loss']
        self.atr_multiplier = self.params['atr_multiplier']
        self.min_hold_time = self.params['min_hold_time']
        self.volatility_threshold = self.params['volatility_threshold']
        
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.current_atr = None
        self.daily_trades = 0
        self.last_trade_date = None
        
    def calculate_atr(self, market_data: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = market_data['high'].values
        low = market_data['low'].values
        close = pd.Series(market_data['close'].values)
        
        tr1 = pd.Series(high - low)
        tr2 = pd.Series(abs(high - close.shift()))
        tr3 = pd.Series(abs(low - close.shift()))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        return atr
        
    def calculate_thresholds(self, current_volatility: float, 
                        market_regime: str = 'neutral',
                        prediction_confidence: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate adaptive thresholds based on current volatility, market regime, and prediction confidence."""
        volatility_factor = current_volatility * self.atr_multiplier
        
        # Base thresholds - make them equal for long and short entries
        long_entry_threshold = self.base_entry_threshold * (1 + volatility_factor)
        # Make short entry threshold equal to long threshold for balanced opportunities
        short_entry_threshold = self.base_entry_threshold * (1 + volatility_factor)
        exit_threshold = self.base_exit_threshold * (1 + volatility_factor)
        stop_loss = self.base_stop_loss * (1 + volatility_factor)
        
        # Adjust based on prediction confidence if available
        if prediction_confidence:
            # Scale thresholds inversely by confidence
            # Higher confidence = lower thresholds (easier to trigger)
            long_entry_threshold *= (1.2 - prediction_confidence.get('up', 0.5))
            short_entry_threshold *= (1.2 - prediction_confidence.get('down', 0.5))
        
        # Adjust based on market regime - make shorting easier in downtrends
        if market_regime == 'high_volatility':
            long_entry_threshold *= 1.2   # More conservative for longs in high volatility
            short_entry_threshold *= 0.8   # More aggressive for shorts in high volatility (reduced from 0.9)
            stop_loss *= 1.1              # Wider stop loss in high volatility
        elif market_regime == 'trending_up':
            long_entry_threshold *= 0.9    # More aggressive for longs in uptrend
            short_entry_threshold *= 1.1   # More conservative for shorts in uptrend (reduced from 1.2)
        elif market_regime == 'trending_down':
            long_entry_threshold *= 1.2    # More conservative for longs in downtrend
            short_entry_threshold *= 0.7   # More aggressive for shorts in downtrend (reduced from 0.85)
        
        return {
            'long_entry': long_entry_threshold,
            'short_entry': short_entry_threshold,
            'exit': exit_threshold,
            'stop_loss': stop_loss
        }
        
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime based on price action and indicators."""
        if len(market_data) < 20:
            return 'neutral'
            
        # Calculate volatility regime
        returns = market_data['close'].pct_change().dropna()
        current_volatility = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized
        avg_volatility = returns.iloc[-60:].std() * np.sqrt(252)
        
        # Calculate trend
        ma5 = market_data['close'].rolling(5).mean().iloc[-1]
        ma20 = market_data['close'].rolling(20).mean().iloc[-1]
        trend_strength = (ma5 - ma20) / ma20
        
        # Determine regime
        if current_volatility > avg_volatility * 1.3:
            regime = 'high_volatility'
        elif trend_strength > 0.02:
            regime = 'trending_up'
        elif trend_strength < -0.02:
            regime = 'trending_down'
        else:
            regime = 'neutral'
            
        return regime
        
    def check_trend_condition(self, market_data: pd.DataFrame) -> Dict[str, bool]:
        """Check if trend conditions are favorable, return separate conditions for long and short."""
        if 'MA5' not in market_data.columns or 'SMA_20' not in market_data.columns:
            # Calculate if not present
            market_data['MA5'] = market_data['close'].rolling(5).mean()
            market_data['SMA_20'] = market_data['close'].rolling(20).mean()
        
        ma5 = market_data['MA5'].iloc[-1]
        ma20 = market_data['SMA_20'].iloc[-1]
        
        trend_strength = abs((ma5 - ma20) / ma20)
        trend_direction = np.sign(ma5 - ma20)
        
        # Different trend conditions for long and short
        if self.ticker in ['TSLA', 'NVDA', 'TQQQ', 'SQQQ']:
            trend_threshold = self.params['trend_threshold'] * 1.5
        else:
            trend_threshold = self.params['trend_threshold']
            
        # More permissive trend check for shorts
        long_trend_ok = trend_strength <= trend_threshold
        short_trend_ok = True  # Allow shorts regardless of trend strength
        
        return {'long': long_trend_ok, 'short': short_trend_ok}
            
    def check_volume_condition(self, market_data: pd.DataFrame) -> bool:
        """Check if volume conditions are favorable."""
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # If volume_threshold is 0, disable volume check (always return True)
        if self.params['volume_threshold'] <= 0:
            return True
            
        return volume_ratio >= self.params['volume_threshold']
        
    def update_direction_confidence(self, metrics: Dict[str, float]):
        """Update direction confidence based on recent model metrics."""
        if 'up_direction_accuracy' in metrics and 'down_direction_accuracy' in metrics:
            # Scale between 0.5 and 1.0
            self.direction_confidence['up'] = 0.5 + (metrics['up_direction_accuracy'] / 200)
            self.direction_confidence['down'] = 0.5 + (metrics['down_direction_accuracy'] / 200)
    
    def _create_signal(self, ticker: str, timestamp: pd.Timestamp, price: float) -> Dict:
        """Create a base signal dictionary."""
        return {
            'ticker': ticker,
            'timestamp': timestamp,
            'price': price,
            'action': None,
            'position': None,
            'position_size': self.params['position_size'],
            'thresholds': {
                'long_entry': self.base_entry_threshold,
                'short_entry': self.base_entry_threshold * self.short_entry_threshold_factor,
                'exit': self.base_exit_threshold,
                'stop_loss': self.base_stop_loss
            },
            'market_conditions': {
                'atr': self.current_atr,
                'daily_trades': self.daily_trades
            }
        }
            
    def generate_signals(self, ticker: str, current_price: float, 
                        predicted_price: float, timestamp: pd.Timestamp,
                        market_data: pd.DataFrame,
                        order_book_data: Dict = None,
                        prediction_confidence: Dict[str, float] = None) -> Dict:
        """Generate trading signals with balanced thresholds for long/short."""
        # Reset daily trades if new day
        current_date = timestamp.date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
            
        # Check trade frequency limit
        if self.daily_trades >= self.params['max_daily_trades']:
            return self._create_signal(ticker, timestamp, current_price)
            
        # Check minimum hold time
        if self.position and self.entry_time:
            hold_time = (timestamp - self.entry_time).total_seconds() / 60
            if hold_time < self.params['min_hold_time']:
                return self._create_signal(ticker, timestamp, current_price)
        
        # Variables for trend and volume checks - initialize outside the if blocks
        trend_conditions = None
        volume_ok = False
        
        try:
            # Calculate ATR and market regime
            self.current_atr = self.calculate_atr(market_data)
            market_regime = self.detect_market_regime(market_data)
            
            # Add a stronger bias correction for predicted prices
            # This addresses the persistent upward bias in predictions
            downward_bias_correction = current_price * 0.003  # Increased from 0.3% to 0.6% adjustment
            corrected_prediction = predicted_price - downward_bias_correction
            
            # Use provided prediction confidence or default to internal values
            confidence = prediction_confidence or self.direction_confidence
            
            # Get adaptive thresholds based on current conditions
            thresholds = self.calculate_thresholds(
                self.current_atr, 
                market_regime,
                confidence
            )
            
            # Calculate price change percentage using corrected prediction
            price_change_pct = (corrected_prediction - current_price) / current_price
            
            # Create base signal
            signal = self._create_signal(ticker, timestamp, current_price)
            signal['market_regime'] = market_regime
            signal['thresholds'] = thresholds
            
            # Pre-calculate trend and volume conditions once
            trend_conditions = self.check_trend_condition(market_data)
            volume_ok = self.check_volume_condition(market_data)
            
            # Log prediction and thresholds for debugging
            if abs(price_change_pct) > 0.001:  # Only log meaningful predictions
                logger.debug(f"Prediction for {ticker}: change={price_change_pct:.4f}%, " +
                            f"long_threshold={thresholds['long_entry']:.4f}%, " +
                            f"short_threshold={thresholds['short_entry']:.4f}%")
            
            # Handle entry signals with balanced thresholds
            if self.position is None:
                # Long entry
                if price_change_pct > thresholds['long_entry'] and trend_conditions['long'] and volume_ok:
                    self.position = 'long'
                    self.entry_price = current_price
                    self.entry_time = timestamp
                    signal['action'] = 'enter_long'
                    signal['position'] = 'long'
                    signal['position_size'] = self.params['position_size'] * confidence.get('up', 0.5) * 2
                    self.daily_trades += 1
                    logger.info(f"LONG SIGNAL: {ticker} at {timestamp}, price={current_price:.2f}, " +
                            f"predicted_change={price_change_pct*100:.2f}%, threshold={thresholds['long_entry']*100:.2f}%")
                
                # Short entry - using the same threshold conditions as long entries
                elif price_change_pct < -thresholds['short_entry'] and trend_conditions['short'] and volume_ok:
                    self.position = 'short'
                    self.entry_price = current_price
                    self.entry_time = timestamp
                    signal['action'] = 'enter_short'
                    signal['position'] = 'short'
                    signal['position_size'] = self.params['position_size'] * confidence.get('down', 0.5) * 2
                    self.daily_trades += 1
                    logger.info(f"SHORT SIGNAL: {ticker} at {timestamp}, price={current_price:.2f}, " +
                            f"predicted_change={price_change_pct*100:.2f}%, threshold={-thresholds['short_entry']*100:.2f}%")
                    
                # Force short entry on significant bearish indicators, with a more aggressive trigger
                elif ('RSI' in market_data.columns and 'MACD_Histogram' in market_data.columns and
                    market_data['RSI'].iloc[-1] > 68 and market_data['MACD_Histogram'].iloc[-1] < 0 and 
                    volume_ok and trend_conditions['short']):
                    # RSI overbought and MACD bearish - potential short opportunity
                    self.position = 'short'
                    self.entry_price = current_price
                    self.entry_time = timestamp
                    signal['action'] = 'enter_short'
                    signal['position'] = 'short'
                    signal['position_size'] = self.params['position_size'] * 0.9  # Increased from 0.8
                    signal['forced_signal'] = True  # Flag to indicate this was a forced signal
                    self.daily_trades += 1
                    logger.info(f"FORCED SHORT SIGNAL: {ticker} at {timestamp}, price={current_price:.2f}, " +
                            f"RSI={market_data['RSI'].iloc[-1]:.1f}, MACD_Hist={market_data['MACD_Histogram'].iloc[-1]:.4f}")
            
            # Handle exit signals
            else:
                if self.position == 'long':
                    price_change_from_entry = (current_price - self.entry_price) / self.entry_price
                    if (price_change_from_entry <= -thresholds['stop_loss'] or 
                        price_change_from_entry >= thresholds['exit']):
                        self.position = None
                        self.entry_price = None
                        self.entry_time = None
                        signal['action'] = 'exit_long'
                        signal['position'] = 'long'
                        signal['exit_reason'] = 'stop_loss' if price_change_from_entry <= -thresholds['stop_loss'] else 'take_profit'
                
                elif self.position == 'short':
                    price_change_from_entry = (self.entry_price - current_price) / self.entry_price
                    if (price_change_from_entry <= -thresholds['stop_loss'] or 
                        price_change_from_entry >= thresholds['exit']):
                        self.position = None
                        self.entry_price = None
                        self.entry_time = None
                        signal['action'] = 'exit_short'
                        signal['position'] = 'short'
                        signal['exit_reason'] = 'stop_loss' if price_change_from_entry <= -thresholds['stop_loss'] else 'take_profit'
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return self._create_signal(ticker, timestamp, current_price)
        
        return signal