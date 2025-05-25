"""
PyTorch dataset implementation for the LSTM trading model.
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

class HFTDataset(Dataset):
    """Enhanced Dataset for High-Frequency Trading data preparation."""

    def __init__(self, data: pd.DataFrame, sequence_length: int, 
                target_column: str = 'close',
                scaler: Optional[StandardScaler] = None,
                include_current_price: bool = True,
                relative_normalization: bool = True,
                ticker: str = None):
        """
        Initialize the dataset with enhanced features and sequence normalization.
        
        Args:
            data: DataFrame with market data and features
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict (default: 'close')
            scaler: Optional scaler for feature normalization
            include_current_price: Whether to include current price in output
            relative_normalization: Whether to normalize sequences relative to their first element
            ticker: Ticker symbol for ticker-specific adaptations
        """
        self.sequence_length = sequence_length
        self.include_current_price = include_current_price
        self.scaler = scaler or StandardScaler()
        self.relative_normalization = relative_normalization
        self.ticker = ticker
        
        # Define required feature columns
        feature_cols = [
            'close', 'volume', 'SMA_20', 'EMA_20', 'RSI', 'MA5',
            'Bollinger_High', 'Bollinger_Low', 'MACD', 'MACD_Signal'
        ]
        
        # Add enhanced features if available
        enhanced_features = [
            'volatility', 'trend_strength', 'body_size', 'volume_ratio', 
            'price_volume', 'price_acceleration', 'price_dist_from_mean_5d',
            'price_dist_from_mean_20d', 'price_momentum_5d', 'price_momentum_20d',
            'log_abs_return', 'daily_range_ratio', 'range_momentum'
        ]
        
        for feat in enhanced_features:
            if feat in data.columns:
                feature_cols.append(feat)
                
        # Add market features if available
        market_features = ['market_return', 'market_vol', 'beta', 'rs_ratio']
        for feat in market_features:
            if feat in data.columns:
                feature_cols.append(feat)

        # Validate feature columns
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            from config import logger
            logger.warning(f"Missing feature columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in data.columns]
            
        # Prepare features and labels
        self.features = data[feature_cols].values
        self.target_col_idx = data.columns.get_loc(target_column)
        self.labels = data[target_column].shift(-1).values  # Predict next price
        
        # Store current prices if needed
        if include_current_price:
            self.current_prices = data[target_column].values

        # Remove last row (contains NaN label)
        self.features = self.features[:-1]
        self.labels = self.labels[:-1]
        if include_current_price:
            self.current_prices = self.current_prices[:-1]

        # Apply standard scaling first
        self.features = self.scaler.fit_transform(self.features)
        
        # Apply enhanced normalization for price-related features
        # This helps with the price magnitude prediction issue
        price_col_indices = [i for i, name in enumerate(feature_cols) 
                             if 'close' in name or 'price' in name or 'high' in name or 'low' in name]
        
        if price_col_indices and relative_normalization:
            # Calculate volatility for adaptive scaling
            for idx in price_col_indices:
                price_series = self.features[:, idx]
                volatility = np.std(price_series)
                scaling_factor = min(1.0, max(0.1, volatility * 10))
                
                # Apply more aggressive normalization for low-volatility series
                if volatility < 0.01:
                    # Center around mean but allow wider distribution
                    mean_val = np.mean(price_series)
                    self.features[:, idx] = (price_series - mean_val) * scaling_factor
        
        # Store the original scaled features for sequence creation
        self.scaled_features = self.features.copy()
        
        # Store column names for reference
        self.feature_names = feature_cols

    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
        return max(0, len(self.labels) - self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sequence and its corresponding label with relative normalization."""
        # Extract the sequence
        seq = self.scaled_features[idx:idx + self.sequence_length].copy()
        
        # Apply relative normalization if enabled
        if self.relative_normalization:
            # Store the first values for each feature
            first_values = seq[0].copy().reshape(1, -1)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            # Normalize relative to first element in sequence
            # This highlights relative changes within the sequence
            seq = seq / (first_values + epsilon) - 1.0
        
        y = self.labels[idx + self.sequence_length]
        
        if self.include_current_price:
            current_price = self.current_prices[idx + self.sequence_length]
            return (
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(current_price, dtype=torch.float32)
            )
        else:
            return (
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )