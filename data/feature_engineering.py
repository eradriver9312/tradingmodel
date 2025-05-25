"""
Feature engineering and data processing for the LSTM trading model.
"""
import numpy as np
import pandas as pd
import ta
from ta.volatility import BollingerBands
from ta.trend import MACD
from scipy.stats import percentileofscore
from typing import Optional
from config import logger

def enhance_features(ticker_df: pd.DataFrame, market_index_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add enhanced features including market regime detection.
    
    Args:
        ticker_df: DataFrame with ticker OHLCV data
        market_index_df: Optional DataFrame with market index data for context
    
    Returns:
        DataFrame with additional features and indicators
    """
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in ticker_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in ticker_df.columns]
        raise ValueError(f"Missing required columns: {missing}")
        
    enhanced_df = ticker_df.copy()
    
    # Add basic technical indicators
    enhanced_df['returns'] = enhanced_df['close'].pct_change()
    enhanced_df['log_returns'] = np.log(enhanced_df['close'] / enhanced_df['close'].shift(1))
    
    # Price moving averages
    enhanced_df['MA5'] = enhanced_df['close'].rolling(window=5).mean()
    enhanced_df['SMA_20'] = enhanced_df['close'].rolling(window=20).mean()
    enhanced_df['EMA_20'] = enhanced_df['close'].ewm(span=20).mean()
    
    # Momentum indicators
    enhanced_df['RSI'] = ta.momentum.rsi(enhanced_df['close'], window=14)
    
    # Volatility indicators
    bb_indicator = BollingerBands(close=enhanced_df['close'])
    enhanced_df['Bollinger_High'] = bb_indicator.bollinger_hband()
    enhanced_df['Bollinger_Low'] = bb_indicator.bollinger_lband()
    enhanced_df['Bollinger_Width'] = (enhanced_df['Bollinger_High'] - enhanced_df['Bollinger_Low']) / enhanced_df['SMA_20']
    
    # Trend indicators
    macd_indicator = MACD(close=enhanced_df['close'])
    enhanced_df['MACD'] = macd_indicator.macd()
    enhanced_df['MACD_Signal'] = macd_indicator.macd_signal()
    enhanced_df['MACD_Histogram'] = enhanced_df['MACD'] - enhanced_df['MACD_Signal']
    
    # Price patterns
    enhanced_df['daily_range'] = enhanced_df['high'] - enhanced_df['low']
    enhanced_df['gap_up'] = (enhanced_df['open'] - enhanced_df['close'].shift(1)) / enhanced_df['close'].shift(1)
    enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open']) / enhanced_df['open']
    
    # Volume analysis
    enhanced_df['volume_ma'] = enhanced_df['volume'].rolling(window=20).mean()
    enhanced_df['volume_ratio'] = enhanced_df['volume'] / enhanced_df['volume_ma']
    enhanced_df['price_volume'] = enhanced_df['returns'].abs() * enhanced_df['volume_ratio']

    # Add relative features rather than absolute values
    enhanced_df['price_acceleration'] = enhanced_df['returns'].diff()
    enhanced_df['ma_cross'] = (enhanced_df['MA5'] > enhanced_df['SMA_20']).astype(int)
    enhanced_df['bb_position'] = (enhanced_df['close'] - enhanced_df['Bollinger_Low']) / (enhanced_df['Bollinger_High'] - enhanced_df['Bollinger_Low'])
    
    # Market regime detection
    enhanced_df['volatility'] = enhanced_df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized
    enhanced_df['volatility_percentile'] = enhanced_df['volatility'].rolling(60).apply(
        lambda x: percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50
    )
    
    # Trend regime
    enhanced_df['trend_strength'] = (enhanced_df['MA5'] - enhanced_df['SMA_20']) / enhanced_df['SMA_20']
    enhanced_df['trend_regime'] = pd.cut(
        enhanced_df['trend_strength'].fillna(0),
        bins=[-float('inf'), -0.02, -0.005, 0.005, 0.02, float('inf')],
        labels=['strong_downtrend', 'downtrend', 'neutral', 'uptrend', 'strong_uptrend']
    )
    
    # Up/down move features
    enhanced_df['up_day'] = (enhanced_df['close'] > enhanced_df['close'].shift(1)).astype(int)
    enhanced_df['down_day'] = (enhanced_df['close'] < enhanced_df['close'].shift(1)).astype(int)
    enhanced_df['up_volume'] = enhanced_df['volume'] * enhanced_df['up_day']
    enhanced_df['down_volume'] = enhanced_df['volume'] * enhanced_df['down_day']

    # Enhanced price distance features
    enhanced_df['price_dist_from_mean_5d'] = (enhanced_df['close'] - enhanced_df['MA5']) / enhanced_df['close']
    enhanced_df['price_dist_from_mean_20d'] = (enhanced_df['close'] - enhanced_df['SMA_20']) / enhanced_df['close']
    
    # Enhanced momentum features
    enhanced_df['price_momentum_5d'] = enhanced_df['returns'].rolling(5).sum()
    enhanced_df['price_momentum_20d'] = enhanced_df['returns'].rolling(20).sum()
    
    # Non-linear transformations
    enhanced_df['log_abs_return'] = np.log(np.abs(enhanced_df['returns']) + 1e-6)
    enhanced_df['return_sign'] = np.sign(enhanced_df['returns'])
    enhanced_df['return_squared'] = enhanced_df['returns'] ** 2
    
    # Range features
    enhanced_df['daily_range_ratio'] = enhanced_df['daily_range'] / enhanced_df['close']
    enhanced_df['range_momentum'] = enhanced_df['daily_range_ratio'].rolling(5).mean()
    
    # Add market index context if available
    if market_index_df is not None and not market_index_df.empty:
        # Align indices
        market_index_df = market_index_df.reindex(enhanced_df.index, method='ffill')
        
        # Add market return features
        enhanced_df['market_return'] = market_index_df['close'].pct_change()
        enhanced_df['market_ma20'] = market_index_df['close'].rolling(20).mean()
        enhanced_df['market_vol'] = market_index_df['close'].pct_change().rolling(20).std()
        
        # Correlation metrics
        enhanced_df['market_correlation'] = enhanced_df['returns'].rolling(20).corr(enhanced_df['market_return'])
        enhanced_df['beta'] = enhanced_df['returns'].rolling(20).cov(enhanced_df['market_return']) / enhanced_df['market_return'].rolling(20).var()
        
        # Relative strength
        enhanced_df['relative_strength'] = enhanced_df['close'] / enhanced_df['close'].shift(20)
        enhanced_df['market_relative_strength'] = market_index_df['close'] / market_index_df['close'].shift(20)
        enhanced_df['rs_ratio'] = enhanced_df['relative_strength'] / enhanced_df['market_relative_strength']
    
    # Drop NaN values created by rolling windows
    enhanced_df.dropna(inplace=True)
    
    return enhanced_df