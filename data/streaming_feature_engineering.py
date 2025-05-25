"""
Streaming-optimized feature engineering for real-time LSTM trading model.
Handles continuous data streams while maintaining feature calculation consistency.
"""
import numpy as np
import pandas as pd
import ta
from ta.volatility import BollingerBands
from ta.trend import MACD
from scipy.stats import percentileofscore
from typing import Optional, Generator, Dict
from config import logger

# testing directory change to activate CI actions again and again and again...

class StreamingFeatureEngineer:
    def __init__(self, lookback_periods: int = 60):
        """
        Initialize feature engineering with streaming support.
        
        Args:
            lookback_periods: Number of previous periods needed for feature calculation
        """
        self.lookback_periods = lookback_periods
        self.previous_data = pd.DataFrame()
        self.feature_stats = {}  # Store running statistics for normalization
        
    def _update_feature_stats(self, features: pd.DataFrame) -> None:
        """Update running statistics for feature normalization."""
        for col in features.columns:
            if col not in self.feature_stats:
                self.feature_stats[col] = {
                    'mean': features[col].mean(),
                    'std': features[col].std(),
                    'count': len(features)
                }
            else:
                n1 = self.feature_stats[col]['count']
                n2 = len(features)
                m1 = self.feature_stats[col]['mean']
                m2 = features[col].mean()
                
                # Update mean
                new_mean = (n1 * m1 + n2 * m2) / (n1 + n2)
                
                # Update std using Welford's online algorithm
                delta = features[col] - m1
                delta2 = features[col] - m2
                M2 = self.feature_stats[col]['std']**2 * (n1 - 1)
                M2 += np.sum(delta * delta2)
                
                new_std = np.sqrt(M2 / (n1 + n2 - 1))
                
                self.feature_stats[col].update({
                    'mean': new_mean,
                    'std': new_std,
                    'count': n1 + n2
                })

    def process_chunk(self, 
                     chunk: pd.DataFrame, 
                     market_index_chunk: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process a chunk of data with feature engineering.
        
        Args:
            chunk: DataFrame chunk with OHLCV data
            market_index_chunk: Optional market index data for the same period
            
        Returns:
            DataFrame with engineered features
        """
        # Combine with previous data for proper feature calculation
        data = pd.concat([self.previous_data, chunk])
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        enhanced_df = data.copy()
        
        # Basic price features
        enhanced_df['returns'] = enhanced_df['close'].pct_change()
        enhanced_df['log_returns'] = np.log(enhanced_df['close'] / enhanced_df['close'].shift(1))
        
        # Moving averages
        enhanced_df['MA5'] = enhanced_df['close'].rolling(window=5, min_periods=1).mean()
        enhanced_df['SMA_20'] = enhanced_df['close'].rolling(window=20, min_periods=1).mean()
        enhanced_df['EMA_20'] = enhanced_df['close'].ewm(span=20, min_periods=1).mean()
        
        # Momentum
        enhanced_df['RSI'] = ta.momentum.rsi(enhanced_df['close'], window=14)
        
        # Volatility
        bb_indicator = BollingerBands(close=enhanced_df['close'], window=20, window_dev=2)
        enhanced_df['Bollinger_High'] = bb_indicator.bollinger_hband()
        enhanced_df['Bollinger_Low'] = bb_indicator.bollinger_lband()
        enhanced_df['Bollinger_Width'] = (enhanced_df['Bollinger_High'] - enhanced_df['Bollinger_Low']) / enhanced_df['SMA_20']
        
        # MACD
        macd_indicator = MACD(close=enhanced_df['close'])
        enhanced_df['MACD'] = macd_indicator.macd()
        enhanced_df['MACD_Signal'] = macd_indicator.macd_signal()
        enhanced_df['MACD_Histogram'] = enhanced_df['MACD'] - enhanced_df['MACD_Signal']
        
        # Price patterns
        enhanced_df['daily_range'] = enhanced_df['high'] - enhanced_df['low']
        enhanced_df['gap_up'] = (enhanced_df['open'] - enhanced_df['close'].shift(1)) / enhanced_df['close'].shift(1)
        enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open']) / enhanced_df['open']
        
        # Volume
        enhanced_df['volume_ma'] = enhanced_df['volume'].rolling(window=20, min_periods=1).mean()
        enhanced_df['volume_ratio'] = enhanced_df['volume'] / enhanced_df['volume_ma']
        enhanced_df['price_volume'] = enhanced_df['returns'].abs() * enhanced_df['volume_ratio']
        
        # Relative features
        enhanced_df['price_acceleration'] = enhanced_df['returns'].diff()
        enhanced_df['ma_cross'] = (enhanced_df['MA5'] > enhanced_df['SMA_20']).astype(int)
        enhanced_df['bb_position'] = (enhanced_df['close'] - enhanced_df['Bollinger_Low']) / (
            enhanced_df['Bollinger_High'] - enhanced_df['Bollinger_Low']
        )
        
        # Market regime
        enhanced_df['volatility'] = enhanced_df['returns'].rolling(20, min_periods=1).std() * np.sqrt(252)
        
        # Add market features if available
        if market_index_chunk is not None and not market_index_chunk.empty:
            market_features = self._calculate_market_features(enhanced_df, market_index_chunk)
            enhanced_df = pd.concat([enhanced_df, market_features], axis=1)
        
        # Store last lookback_periods rows for next chunk
        self.previous_data = enhanced_df.iloc[-self.lookback_periods:]
        
        # Update feature statistics
        self._update_feature_stats(enhanced_df)
        
        # Return only the rows corresponding to the input chunk
        return enhanced_df.iloc[-len(chunk):]
    
    def _calculate_market_features(self, 
                                 data: pd.DataFrame, 
                                 market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-related features."""
        market_features = pd.DataFrame(index=data.index)
        
        # Ensure market data is aligned
        market_data = market_data.reindex(data.index, method='ffill')
        
        # Calculate market returns
        market_features['market_return'] = market_data['close'].pct_change()
        market_features['market_vol'] = market_features['market_return'].rolling(
            20, min_periods=1
        ).std() * np.sqrt(252)
        
        # Calculate beta
        returns = data['returns']
        covariance = returns.rolling(60, min_periods=30).cov(market_features['market_return'])
        market_variance = market_features['market_return'].rolling(60, min_periods=30).var()
        market_features['beta'] = covariance / market_variance
        
        # Relative strength
        market_features['rs_ratio'] = (
            (1 + data['returns']).rolling(20, min_periods=1).mean() /
            (1 + market_features['market_return']).rolling(20, min_periods=1).mean()
        )
        
        return market_features
    
    def get_feature_stats(self) -> Dict:
        """Return current feature statistics."""
        return self.feature_stats

def stream_enhance_features(data_stream: Generator[pd.DataFrame, None, None],
                          market_data_stream: Optional[Generator[pd.DataFrame, None, None]] = None
                          ) -> Generator[pd.DataFrame, None, None]:
    """
    Stream-process data with feature engineering.
    
    Args:
        data_stream: Generator yielding DataFrame chunks
        market_data_stream: Optional generator yielding market data chunks
        
    Yields:
        DataFrame chunks with engineered features
    """
    engineer = StreamingFeatureEngineer()
    
    for chunk in data_stream:
        market_chunk = next(market_data_stream) if market_data_stream else None
        enhanced_chunk = engineer.process_chunk(chunk, market_chunk)
        yield enhanced_chunk