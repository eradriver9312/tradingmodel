"""
Visualization functions for the LSTM trading model.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import traceback
from config import logger
from utils import TrainingHistory

def plot_candlestick_analysis(df: pd.DataFrame, signals: Optional[List[Dict]] = None, 
                           trades: Optional[List[Dict]] = None,
                           ticker: str = '') -> None:
    """
    Plot candlestick chart with trade signals and performance metrics.
    
    Args:
        df: DataFrame with OHLCV data
        signals: Optional list of trading signals
        trades: Optional list of executed trades
        ticker: Stock ticker symbol
    """
    try:
        df_plot = df.copy()
        df_plot = df_plot[['open', 'high', 'low', 'close', 'volume']]
        df_plot.index.name = 'Date'

        # Create subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        # Plot candlesticks and volume
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        mpf.plot(df_plot, type='candle', style=s, ax=ax1, volume=False, show_nontrading=False)

        # Plot Volume
        df_plot['volume'].plot(ax=ax2, color='blue', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Plot additional indicator (e.g., RSI)
        if 'RSI' in df.columns:
            df['RSI'].plot(ax=ax3, color='purple')
            ax3.set_ylabel('RSI')
            ax3.grid(True)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        else:
            # Plot close price with SMA
            df['close'].plot(ax=ax3, color='blue')
            if 'SMA_20' in df.columns:
                df['SMA_20'].plot(ax=ax3, color='orange')
            ax3.set_ylabel('Price')
            ax3.grid(True)

        # Add signals if provided
        if signals:
            for signal in signals:
                if signal.get('action') is None:
                    continue
                    
                signal_time = pd.to_datetime(signal['timestamp'])
                price = signal['price']
                
                if 'enter_long' in signal['action']:
                    ax1.scatter(signal_time, price, 
                             marker='^', color='g', s=100, label='Buy Signal')
                elif 'enter_short' in signal['action']:
                    ax1.scatter(signal_time, price, 
                             marker='v', color='r', s=100, label='Sell Signal')
                elif 'exit_long' in signal['action'] or 'exit_short' in signal['action']:
                    ax1.scatter(signal_time, price, 
                             marker='o', color='k', s=100, label='Exit Signal')

        # Plot trades if available
        if trades:
            # Prepare data for trade annotations
            for trade in trades:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                profit = trade.get('profit', 0)
                position_size = trade.get('position_size', 1.0)
                
                # Choose color based on profit
                color = 'g' if profit > 0 else 'r'
                
                # Draw line connecting entry and exit
                ax1.plot([entry_time, exit_time], [entry_price, exit_price], 
                      color=color, linestyle='-', linewidth=1.5 * position_size, alpha=0.7)
                
                # Add profit annotation
                if 'profit' in trade:
                    mid_time = entry_time + (exit_time - entry_time) / 2
                    ax1.annotate(f"${profit:.2f}", 
                             xy=(mid_time, max(entry_price, exit_price)), 
                             xytext=(0, 5), textcoords='offset points',
                             fontsize=8, color=color)

        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        plt.title(f"Candlestick Chart with Trade Signals for {ticker}")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error in plot_candlestick_analysis: {e}")
        traceback.print_exc()

def plot_trading_metrics(metrics: Dict[str, float], ticker: str) -> None:
    """
    Create comprehensive visualization of trading metrics.
    
    Args:
        metrics: Dictionary of trading metrics
        ticker: Stock ticker symbol
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2)

    # Rates Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    rates = ['win_rate', 'loss_rate', 'long_win_rate', 'short_win_rate']
    values = [metrics.get(rate, 0) for rate in rates]
    colors = ['green', 'red', 'lightgreen', 'lightcoral']
    ax1.bar(rates, values, color=colors)
    ax1.set_title('Trading Rates Comparison')
    ax1.set_ylabel('Percentage (%)')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Profit Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    profit_metrics = ['total_profit', 'average_profit_per_trade']
    values = [metrics.get(metric, 0) for metric in profit_metrics]
    ax2.bar(profit_metrics, values, color='blue')
    ax2.set_title('Profit Metrics')
    ax2.set_ylabel('Amount ($)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Risk Metrics
    ax3 = fig.add_subplot(gs[1, 0])
    risk_metrics = ['maximum_drawdown', 'sharpe_ratio', 'profit_factor']
    values = [metrics.get(metric, 0) for metric in risk_metrics]
    ax3.bar(risk_metrics, values, color='purple')
    ax3.set_title('Risk Metrics')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Trade Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    trade_metrics = ['total_trades', 'long_trades', 'short_trades', 'average_trade_duration']
    values = [metrics.get(metric, 0) for metric in trade_metrics]
    ax4.bar(trade_metrics, values, color='orange')
    ax4.set_title('Trade Analysis')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(f'Trading Performance Metrics for {ticker}', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(training_history: TrainingHistory, ticker: str) -> None:
    """
    Plot learning curves showing training metrics over epochs.
    
    Args:
        training_history: Object containing training and validation metrics
        ticker: Stock ticker symbol for plot title
    """
    plt.figure(figsize=(18, 12))
    
    # Create subplot grid
    gs = gridspec.GridSpec(2, 2)
    
    # Plot training and validation loss
    ax1 = plt.subplot(gs[0, 0])
    epochs = range(1, len(training_history.loss_history) + 1)
    ax1.plot(epochs, training_history.loss_history, 'b-', label='Training Loss')
    ax1.plot(epochs, training_history.validation_loss_history, 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')  # Use log scale for better visualization
    
    # Plot direction accuracy if available
    if hasattr(training_history, 'direction_accuracy_history') and training_history.direction_accuracy_history:
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(epochs, training_history.direction_accuracy_history, 'g-', label='Direction Accuracy')
        ax2.set_title('Direction Prediction Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        ax2.legend()
        
    # Plot long/short accuracy if available
    if (hasattr(training_history, 'long_accuracy_history') and training_history.long_accuracy_history and
        hasattr(training_history, 'short_accuracy_history') and training_history.short_accuracy_history):
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(epochs, training_history.long_accuracy_history, 'g-', label='Long Accuracy')
        ax3.plot(epochs, training_history.short_accuracy_history, 'r-', label='Short Accuracy')
        ax3.set_title('Long vs. Short Prediction Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True)
        ax3.legend()
        
        # Plot accuracy ratio (long/short)
        if len(training_history.long_accuracy_history) == len(training_history.short_accuracy_history):
            ax4 = plt.subplot(gs[1, 1])
            ratio = [l/s if s > 0 else 1.0 for l, s in zip(
                training_history.long_accuracy_history, 
                training_history.short_accuracy_history
            )]
            ax4.plot(epochs, ratio, 'b-', label='Long/Short Accuracy Ratio')
            ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
            ax4.set_title('Direction Prediction Balance')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Ratio')
            ax4.grid(True)
            ax4.legend()
    
    plt.suptitle(f'Learning Curves for {ticker}', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                          current_prices: Optional[np.ndarray] = None,
                          ticker: str = '') -> None:
    """
    Plot analysis of prediction performance with focus on directional accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        current_prices: Current prices (for direction calculation)
        ticker: Stock ticker symbol
    """
    plt.figure(figsize=(20, 16))
    
    # Create subplot grid
    gs = gridspec.GridSpec(3, 2)
    
    # Plot 1: True vs Predicted prices
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.3)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_xlabel('True Price')
    ax1.set_ylabel('Predicted Price')
    ax1.set_title('True vs Predicted Prices')
    ax1.grid(True)
    
    # Plot 2: Prediction Error Distribution
    ax2 = plt.subplot(gs[0, 1])
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax2.hist(errors, bins=50, alpha=0.7, color='blue')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution (Mean: {mean_error:.4f}, Std: {std_error:.4f})')
    ax2.grid(True)
    
    # Calculate direction accuracy if current prices are available
    if current_prices is not None:
        # Plot 3: Direction Accuracy
        ax3 = plt.subplot(gs[1, 0])
        
        # Calculate true and predicted directions
        true_directions = np.sign(y_true - current_prices)
        pred_directions = np.sign(y_pred - current_prices)
        
        # Calculate direction matches
        matches = (true_directions == pred_directions)
        
        # Separate into up and down movements
        up_indices = np.where(true_directions > 0)[0]
        down_indices = np.where(true_directions < 0)[0]
        
        # Calculate accuracies
        overall_accuracy = np.mean(matches) * 100
        up_accuracy = np.mean(matches[up_indices]) * 100 if len(up_indices) > 0 else 0
        down_accuracy = np.mean(matches[down_indices]) * 100 if len(down_indices) > 0 else 0
        
        # Create bar chart
        accuracies = [overall_accuracy, up_accuracy, down_accuracy]
        labels = ['Overall', 'Up Movement', 'Down Movement']
        colors = ['blue', 'green', 'red']
        
        ax3.bar(labels, accuracies, color=colors)
        ax3.set_ylabel('Direction Accuracy (%)')
        ax3.set_title(f'Direction Prediction Accuracy')
        ax3.grid(True)
        
        # Add text annotations
        for i, acc in enumerate(accuracies):
            ax3.text(i, acc + 1, f'{acc:.1f}%', ha='center')
            
        # Plot 4: Movement Distribution
        ax4 = plt.subplot(gs[1, 1])
        
        # Count movement types
        up_count = len(up_indices)
        down_count = len(down_indices)
        no_move_count = len(true_directions) - up_count - down_count
        
        # Create pie chart
        labels = ['Up', 'Down', 'No Change']
        sizes = [up_count, down_count, no_move_count]
        colors = ['green', 'red', 'gray']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.axis('equal')
        ax4.set_title('True Price Movement Distribution')
        
        # Plot 5: Prediction Bias
        ax5 = plt.subplot(gs[2, 0])
        
        # Calculate predicted movement types
        pred_up = np.sum(pred_directions > 0)
        pred_down = np.sum(pred_directions < 0)
        pred_no_move = len(pred_directions) - pred_up - pred_down
        
        # Create side-by-side bar chart
        labels = ['Up', 'Down', 'No Change']
        true_counts = [up_count, down_count, no_move_count]
        pred_counts = [pred_up, pred_down, pred_no_move]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax5.bar(x - width/2, true_counts, width, label='True')
        ax5.bar(x + width/2, pred_counts, width, label='Predicted')
        
        ax5.set_xlabel('Movement Direction')
        ax5.set_ylabel('Count')
        ax5.set_title('True vs Predicted Movement Distribution')
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels)
        ax5.legend()
        ax5.grid(True)
        
        # Plot 6: Error by True Direction
        ax6 = plt.subplot(gs[2, 1])
        
        # Separate errors by direction
        up_errors = errors[up_indices]
        down_errors = errors[down_indices]
        
        # Create box plot
        data = [up_errors, down_errors]
        ax6.boxplot(data, labels=['Up Movement', 'Down Movement'])
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_ylabel('Prediction Error')
        ax6.set_title('Error Distribution by True Direction')
        ax6.grid(True)
    
    plt.suptitle(f'Prediction Analysis for {ticker}', fontsize=16)
    plt.tight_layout()
    plt.show()