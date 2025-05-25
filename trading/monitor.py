"""
Multi-ticker monitoring for the LSTM trading model.
"""
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
from config import TICKER_CONFIDENCE, DEFAULT_CONFIDENCE, logger
from trading.signal_generator import AdaptiveSignalGenerator

class MultiTickerMonitor:
    """Monitor multiple tickers and generate trading signals."""

    def __init__(self, signal_generator: AdaptiveSignalGenerator):
        self.tracked_tickers = {}
        self.signal_generator = signal_generator
        
        # Ticker-specific direction confidence - stored in config.py
        self.ticker_confidence = TICKER_CONFIDENCE
        # Default confidence levels
        self.default_confidence = DEFAULT_CONFIDENCE

    def add_ticker(self, ticker: str, initial_price: float):
        """Add a new ticker to monitor."""
        self.tracked_tickers[ticker] = {
            'current_price': initial_price,
            'signals': [],
            'last_update': datetime.now()
        }

    def update_ticker(self, ticker: str, current_price: float,
                     predicted_price: float, timestamp: pd.Timestamp,
                     market_data: pd.DataFrame,
                     order_book_data: Dict = None,
                     prediction_metrics: Dict = None) -> Optional[Dict]:
        """Update ticker information and generate signals."""
        if ticker not in self.tracked_tickers:
            self.add_ticker(ticker, current_price)
    
        self.tracked_tickers[ticker]['current_price'] = current_price
        self.tracked_tickers[ticker]['last_update'] = timestamp
        
        # Get prediction confidence for this ticker
        price_change = predicted_price - current_price
        direction = 'up' if price_change > 0 else 'down'
        
        # Use stored confidence or default
        confidence = self.ticker_confidence.get(ticker, self.default_confidence)
        
        # Update confidence if metrics provided
        if prediction_metrics:
            if 'up_direction_accuracy' in prediction_metrics:
                confidence['up'] = 0.5 + (prediction_metrics['up_direction_accuracy'] / 200)
            if 'down_direction_accuracy' in prediction_metrics:
                confidence['down'] = 0.5 + (prediction_metrics['down_direction_accuracy'] / 200)
    
        signal = self.signal_generator.generate_signals(
            ticker=ticker,
            current_price=current_price,
            predicted_price=predicted_price,
            timestamp=timestamp,
            market_data=market_data,
            order_book_data=order_book_data,
            prediction_confidence=confidence
        )
    
        if signal and signal['action']:
            self.tracked_tickers[ticker]['signals'].append(signal)
            return signal
    
        return None

    def get_signals(self, ticker: str) -> List[Dict]:
        """Retrieve signals for a specific ticker."""
        return self.tracked_tickers.get(ticker, {}).get('signals', [])