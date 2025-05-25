"""
Utility functions and helper classes for the LSTM trading model.
"""
import gc
import traceback
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from config import logger

@dataclass
class TrainingHistory:
    """Tracks training metrics during model training."""
    loss_history: List[float] = field(default_factory=list)
    validation_loss_history: List[float] = field(default_factory=list)
    direction_accuracy_history: List[float] = field(default_factory=list)
    long_accuracy_history: List[float] = field(default_factory=list)
    short_accuracy_history: List[float] = field(default_factory=list)

    def update(self, epoch: int, loss: float, val_loss: float, 
              direction_accuracy: float = None, 
              long_accuracy: float = None, 
              short_accuracy: float = None):
        """Update training history with new metrics."""
        self.loss_history.append(loss)
        self.validation_loss_history.append(val_loss)
        if direction_accuracy is not None:
            self.direction_accuracy_history.append(direction_accuracy)
        if long_accuracy is not None:
            self.long_accuracy_history.append(long_accuracy)
        if short_accuracy is not None:
            self.short_accuracy_history.append(short_accuracy)

def apply_dynamic_scaling(model, train_loader, device: torch.device):
    """Apply dynamic output scaling based on target distribution."""
    price_ranges = []
    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 3:
                _, batch_y, _ = batch
            else:
                _, batch_y = batch
            price_ranges.append((batch_y.min().item(), batch_y.max().item()))
    
    # Calculate global min and max
    global_min = min([r[0] for r in price_ranges])
    global_max = max([r[1] for r in price_ranges])
    price_range = global_max - global_min
    
    # Set model's output transformation parameters
    if hasattr(model, 'output_transform'):
        # Initialize last layer to produce outputs in appropriate range
        last_layer = model.output_transform[-1]
        with torch.no_grad():
            # Scale the last layer weights to produce wider range
            current_range = last_layer.weight.abs().mean().item()
            target_range = price_range / 10  # Aim for 10% of full range
            scale_factor = target_range / (current_range + 1e-8)
            last_layer.weight.mul_(scale_factor)
    
    logger.info(f"Applied dynamic scaling with price range: {price_range:.2f}, scale factor: {scale_factor:.4f}")

def clean_up_memory():
    """Clean up memory to prevent memory leaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_stratified_dataset(data_df, min_regime_count: int = 50, 
                             min_total_samples: int = 1000,
                             max_sampling_fraction: float = 0.7):
    """
    Create a dataset with balanced market regimes for more even training.
    
    Args:
        data_df: DataFrame with OHLCV data and calculated features
        min_regime_count: Minimum number of samples per regime
        min_total_samples: Minimum total samples in the resulting dataset
        max_sampling_fraction: Maximum reduction as a fraction of original size
    
    Returns:
        Balanced DataFrame with representation from all market regimes
    """
    import pandas as pd
    import numpy as np
    
    # Define market regimes if not already done
    if 'regime' not in data_df.columns:
        # Initialize regime column
        data_df['regime'] = 'neutral'
        
        # Identify high volatility periods (more sensitive thresholds)
        volatility_threshold = data_df['volatility'].quantile(0.65)
        high_vol = data_df['volatility'] >= volatility_threshold
        
        # Identify up and down days (more sensitive thresholds)
        up_day = data_df['returns'] > 0.0008
        down_day = data_df['returns'] < -0.0008
        
        # Combine into regimes
        data_df.loc[high_vol & up_day, 'regime'] = 'high_vol_up'
        data_df.loc[high_vol & down_day, 'regime'] = 'high_vol_down'
        data_df.loc[(~high_vol) & up_day, 'regime'] = 'low_vol_up'
        data_df.loc[(~high_vol) & down_day, 'regime'] = 'low_vol_down'
    
    # Get counts by regime
    regime_counts = data_df['regime'].value_counts()
    logger.info(f"Original regime distribution: {regime_counts.to_dict()}")
    
    # Check if we have sufficient data in any regime
    non_zero_counts = regime_counts[regime_counts > 0]
    if len(non_zero_counts) == 0:
        logger.warning("No valid regimes found in dataset")
        return data_df
        
    # If we have very few samples in any regime, consider all data as neutral
    if non_zero_counts.min() < 10:
        logger.warning(f"Very few samples in some regimes. Classifying all as neutral.")
        data_df['regime'] = 'neutral'
        regime_counts = data_df['regime'].value_counts()
        non_zero_counts = regime_counts
    
    # Find minimum count, but ensure it's at least min_regime_count
    min_count = max(int(non_zero_counts.min() * 0.8), min_regime_count)
    
    # Calculate total samples after balancing
    num_regimes = len(data_df['regime'].unique())
    total_balanced_samples = min_count * num_regimes
    
    # Adjust if total samples would be too small compared to original
    if total_balanced_samples < min_total_samples or total_balanced_samples < max_sampling_fraction * len(data_df):
        # Increase min_count to meet the minimum total requirement
        required_count = max(
            min_total_samples // num_regimes,
            int(max_sampling_fraction * len(data_df) // num_regimes)
        )
        min_count = max(min_count, required_count)
        logger.info(f"Adjusted regime count to {min_count} to maintain sufficient data volume")
    
    # Sample from each regime
    balanced_df = pd.DataFrame()
    for regime in data_df['regime'].unique():
        regime_data = data_df[data_df['regime'] == regime]
        if len(regime_data) == 0:
            continue
            
        if len(regime_data) > min_count:
            # Sample without replacement if enough data
            sampled_data = regime_data.sample(min_count)
        else:
            # Sample with replacement if needed (ensuring we get enough data)
            sampled_data = regime_data.sample(min_count, replace=True)
            logger.info(f"Sampled with replacement for regime '{regime}': {len(regime_data)} → {min_count}")
            
        balanced_df = pd.concat([balanced_df, sampled_data])
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Balanced regime distribution: {balanced_df['regime'].value_counts().to_dict()}")
    logger.info(f"Original dataset: {len(data_df)} samples → Balanced dataset: {len(balanced_df)} samples")
    
    return balanced_df

def backtest_trades_with_costs(signals, df, commission_rate: float = 0.001, 
                             slippage_factor: float = 0.0002):
    """
    Execute backtesting with realistic transaction costs and slippage.
    
    Args:
        signals: List of trading signals
        df: DataFrame with market data
        commission_rate: Trading commission as a percentage
        slippage_factor: Price slippage as a percentage
        
    Returns:
        Tuple of trades list, total commission, and total slippage
    """
    import pandas as pd
    
    trades = []
    entry_signal = None
    total_commission = 0
    total_slippage = 0
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    for signal in signals:
        signal_time = pd.to_datetime(signal['timestamp'])
        
        # Apply slippage to entry and exit prices
        price_with_slippage = signal['price']
        slippage_amount = 0
        
        if signal['action'].startswith('enter_long') or signal['action'].startswith('exit_short'):
            # Buy operations - price moves against us (higher)
            slippage_amount = signal['price'] * slippage_factor
            price_with_slippage = signal['price'] + slippage_amount
        elif signal['action'].startswith('enter_short') or signal['action'].startswith('exit_long'):
            # Sell operations - price moves against us (lower)
            slippage_amount = signal['price'] * slippage_factor
            price_with_slippage = signal['price'] - slippage_amount
        
        # Track slippage
        total_slippage += slippage_amount
        
        # Calculate commission
        commission = price_with_slippage * commission_rate
        total_commission += commission
        
        # Process entry signals
        if signal['action'].startswith('enter_'):
            if entry_signal is None:
                # Use position_size from signal if available
                position_size = signal.get('position_size', 1.0)
                
                entry_signal = {
                    'timestamp': signal_time,
                    'price': price_with_slippage,
                    'action': signal['action'],
                    'commission': commission,
                    'slippage': slippage_amount,
                    'position_size': position_size
                }
        
        # Process exit signals
        elif signal['action'].startswith('exit_') and entry_signal is not None:
            exit_reason = signal.get('exit_reason', 'unspecified')
            duration = (signal_time - entry_signal['timestamp']).total_seconds() / 60
            
            # Calculate profit/loss including transaction costs
            total_trade_commission = entry_signal['commission'] + commission
            total_trade_slippage = entry_signal['slippage'] + slippage_amount
            position_size = entry_signal.get('position_size', 1.0)
            
            if entry_signal['action'] == 'enter_long':
                profit = (price_with_slippage - entry_signal['price'] - total_trade_commission) * position_size
            else:  # Short trade
                profit = (entry_signal['price'] - price_with_slippage - total_trade_commission) * position_size
            
            trade = {
                'entry_time': entry_signal['timestamp'],
                'entry_price': entry_signal['price'],
                'exit_time': signal_time,
                'exit_price': price_with_slippage,
                'position': entry_signal['action'],
                'position_size': position_size,
                'profit': profit,
                'transaction_costs': total_trade_commission,
                'slippage_costs': total_trade_slippage,
                'duration': duration,
                'exit_reason': exit_reason
            }
            trades.append(trade)
            entry_signal = None
    
    # Close any open position at the end of the dataset
    if entry_signal is not None:
        last_time = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        # Apply slippage to last price
        slippage_amount = 0
        price_with_slippage = last_price
        
        if entry_signal['action'] == 'enter_short':
            # For short positions, buying to close (higher price)
            slippage_amount = last_price * slippage_factor
            price_with_slippage = last_price + slippage_amount
        else:
            # For long positions, selling to close (lower price)
            slippage_amount = last_price * slippage_factor
            price_with_slippage = last_price - slippage_amount
        
        # Track slippage
        total_slippage += slippage_amount
        
        # Calculate commission
        commission = price_with_slippage * commission_rate
        total_commission += commission
        
        duration = (last_time - entry_signal['timestamp']).total_seconds() / 60
        
        # Calculate profit/loss
        total_trade_commission = entry_signal['commission'] + commission
        total_trade_slippage = entry_signal['slippage'] + slippage_amount
        position_size = entry_signal.get('position_size', 1.0)
        
        if entry_signal['action'] == 'enter_long':
            profit = (price_with_slippage - entry_signal['price'] - total_trade_commission) * position_size
        else:  # Short trade
            profit = (entry_signal['price'] - price_with_slippage - total_trade_commission) * position_size
        
        trade = {
            'entry_time': entry_signal['timestamp'],
            'entry_price': entry_signal['price'],
            'exit_time': last_time,
            'exit_price': price_with_slippage,
            'position': entry_signal['action'],
            'position_size': position_size,
            'profit': profit,
            'transaction_costs': total_trade_commission,
            'slippage_costs': total_trade_slippage,
            'duration': duration,
            'exit_reason': 'end_of_data'
        }
        trades.append(trade)
    
    return trades, total_commission, total_slippage