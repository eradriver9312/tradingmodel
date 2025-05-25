"""
Main execution script for the LSTM trading model.
"""
import os
import gc
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback

# Custom modules
from config import (
    logger, API_KEY, TICKERS, MARKET_INDEX, END_DATE, TRAINING_DAYS, 
    MODEL_PARAMS, TRANSACTION_COSTS, MODEL_SAVE_PATH
)
from data import PolygonDataFetcher, HFTDataset, enhance_features
from models import LSTMModel, DirectionalPredictionLoss, train_model
from trading import AdaptiveSignalGenerator, ModelEvaluator, evaluate_model, MultiTickerMonitor
from utils import TrainingHistory, backtest_trades_with_costs, create_stratified_dataset, clean_up_memory
from visualization import (
    plot_candlestick_analysis, plot_trading_metrics, 
    plot_learning_curves, plot_prediction_analysis
)

def main():
    """Enhanced main function for the high-frequency trading system."""
    # Initialize components
    fetcher = PolygonDataFetcher(API_KEY)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    evaluator = ModelEvaluator()

    # Define date range
    end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
    start_date = end_date - timedelta(days=TRAINING_DAYS)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Fetch market index data for context
    logger.info(f"Fetching market index data from {start_date_str} to {end_date_str}")
    market_index_df = fetcher.fetch_market_index_data(
        index_symbol=MARKET_INDEX,
        start_date=start_date_str,
        end_date=end_date_str
    )
    
    if market_index_df.empty:
        logger.warning(f"No market index data available. Proceeding without market context.")

    for ticker in TICKERS:
        try:
            logger.info(f"Processing {ticker}")
            print(f"\nProcessing {ticker}...")

            # Initialize stock-specific components
            signal_generator = AdaptiveSignalGenerator(ticker)
            monitor = MultiTickerMonitor(signal_generator)

            # Fetch and prepare market data
            logger.info(f"Fetching market data from {start_date_str} to {end_date_str}")
            stock_df = fetcher.fetch_stock_data(ticker, start_date_str, end_date_str)

            if stock_df.empty:
                logger.warning(f"No data fetched for {ticker}. Skipping.")
                continue

            # Apply enhanced feature engineering
            stock_df = enhance_features(stock_df, market_index_df)
            logger.info(f"Enhanced features created for {ticker}")
            
            # Create balanced dataset with stratified sampling
            balanced_df = create_stratified_dataset(stock_df)
            logger.info(f"Created balanced dataset with {len(balanced_df)} samples")
            
            # Check if stratified dataset has sufficient data
            if len(balanced_df) < 200:
                logger.warning(f"WARNING: Stratified dataset for {ticker} has only {len(balanced_df)} samples")
                logger.warning("This may be insufficient for effective model training")
                
                # Fallback to original dataset if samples are extremely low
                if len(balanced_df) < 100:
                    logger.warning(f"Using original dataset instead for {ticker} due to insufficient stratified samples")
                    balanced_df = stock_df
                    logger.info(f"Using original dataset with {len(balanced_df)} samples")

            # Prepare datasets with current price for directional loss
            dataset = HFTDataset(
                data=balanced_df,
                sequence_length=MODEL_PARAMS['sequence_length'],
                include_current_price=True,
                ticker=ticker
            )
            logger.info(f"Dataset created with {len(dataset)} sequences and {len(dataset.feature_names)} features")

            # Split data
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size

            if train_size <= 0 or val_size <= 0 or test_size <= 0:
                logger.warning(f"Insufficient data for {ticker} after preprocessing. Skipping.")
                continue

            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=MODEL_PARAMS['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=MODEL_PARAMS['batch_size']
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=MODEL_PARAMS['batch_size']
            )

            # Initialize model
            input_size = len(dataset.feature_names)
            model = LSTMModel(
                input_size=input_size,
                hidden_size=MODEL_PARAMS['hidden_size'],
                num_layers=MODEL_PARAMS['num_layers'],
                dropout=MODEL_PARAMS['dropout']
            )

            # Initialize training components with direction-specific loss
            criterion = DirectionalPredictionLoss(
                direction_weight=2.5,
                magnitude_weight=0.8,
                short_penalty_multiplier=1.3,  # Penalize missing downward movements more
                bias_correction_weight=0.5
            )
            
            # Set ticker for loss function
            criterion.set_ticker(ticker)
            
            optimizer = optim.Adam(
                model.parameters(), 
                lr=MODEL_PARAMS['learning_rate'],
                weight_decay=5e-3  # L2 regularization
            )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.4,  # More aggressive reduction
                patience=3,   # Reduced patience
                verbose=True,
                min_lr=1e-5   # Set minimum learning rate
            )
            
            training_history = TrainingHistory()

            # Train model
            logger.info("Starting model training...")
            print("Training model with enhanced direction-specific loss...")
            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=MODEL_PARAMS['num_epochs'],
                device=device,
                training_history=training_history,
                early_stopping_patience=MODEL_PARAMS['early_stopping_patience'],
                scheduler=scheduler,
                ticker=ticker
            )

            # Plot learning curves
            logger.info("Plotting learning curves...")
            plot_learning_curves(training_history, ticker)

            # Evaluate model
            logger.info("Evaluating model...")
            y_true, y_pred, current_prices = evaluate_model(
                model, 
                test_loader, 
                device,
                return_predictions=True,
                evaluator=evaluator,
                ticker=ticker
            )
            
            # Plot prediction analysis
            plot_prediction_analysis(y_true, y_pred, current_prices, ticker)
            
            # Calculate metrics from model evaluator
            evaluation_metrics = evaluator.calculate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                current_prices=current_prices,
                ticker=ticker
            )
            
            # Log evaluation metrics
            logger.info(f"Model Evaluation Metrics for {ticker}:")
            for metric, value in evaluation_metrics.items():
                logger.info(f"  {metric}: {value}")
            
            # Update signal generator with direction-specific accuracies
            signal_generator.update_direction_confidence(evaluation_metrics)
            
            # Generate trading signals using stock-specific signal generator
            logger.info("Generating trading signals...")
            
            # Use original (non-balanced) data for signal generation and backtesting
            stock_df_orig = enhance_features(stock_df, market_index_df)
            
            # Prepare test period for signal generation
            test_period = stock_df_orig.iloc[-len(y_true):]
            test_timestamps = test_period.index
            
            print(f"Generating signals for {len(test_timestamps)} timestamps...")
            
            all_signals = []
            for idx, (timestamp, actual, predicted) in enumerate(zip(test_timestamps, y_true, y_pred)):
                # Print progress update for large datasets
                if idx % 1000 == 0:
                    print(f"Processing index {idx}: Current price = {actual:.2f}, Predicted price = {predicted:.2f}")
                
                # Get market data window for signal generation
                market_data_window = stock_df_orig.iloc[max(0, idx-20):idx+1].copy()
                
                if len(market_data_window) == 0:
                    continue
                
                # Apply bias correction if needed
                correction = evaluator.get_adaptive_correction(ticker)
                corrected_prediction = predicted - correction
                
                # Generate signal
                signal = monitor.update_ticker(
                    ticker=ticker,
                    current_price=actual,
                    predicted_price=corrected_prediction,
                    timestamp=timestamp,
                    market_data=market_data_window,
                    prediction_metrics=evaluation_metrics
                )
                
                if signal and signal['action']:
                    if idx < 10 or idx % 500 == 0:  # Limit logging for clarity
                        print(f"Generated signal at {timestamp}: {signal['action']}")
                    all_signals.append(signal)
            
            print(f"Generated {len(all_signals)} signals")
            
            # Add diagnostic information if no signals were generated
            if len(all_signals) == 0:
                print("\nDiagnostic information for signal generation:")
                # Check the first few test samples
                for idx in range(min(5, len(test_timestamps))):
                    timestamp = test_timestamps[idx]
                    actual = y_true[idx]
                    predicted = y_pred[idx]
                    price_change_pct = (predicted - actual) / actual
                    
                    # Get market data window
                    market_data_window = stock_df_orig.iloc[max(0, idx-20):idx+1].copy()
                    
                    # Check trend condition
                    if 'MA5' not in market_data_window.columns:
                        market_data_window['MA5'] = market_data_window['close'].rolling(5).mean()
                    if 'SMA_20' not in market_data_window.columns:
                        market_data_window['SMA_20'] = market_data_window['close'].rolling(20).mean()
                    
                    ma5 = market_data_window['MA5'].iloc[-1]
                    ma20 = market_data_window['SMA_20'].iloc[-1]
                    trend_strength = abs((ma5 - ma20) / ma20)
                    trend_threshold = signal_generator.params['trend_threshold']
                    trend_ok = trend_strength <= trend_threshold
                    
                    # Check volume condition
                    current_volume = market_data_window['volume'].iloc[-1]
                    avg_volume = market_data_window['volume'].rolling(20).mean().iloc[-1]
                    volume_ratio = current_volume / avg_volume
                    volume_threshold = signal_generator.params['volume_threshold']
                    volume_ok = volume_ratio >= volume_threshold
                    
                    # Check signal thresholds
                    thresholds = signal_generator.calculate_thresholds(signal_generator.current_atr)
                    
                    print(f"\nTimestamp {idx}: {timestamp}")
                    print(f"  Current price: {actual:.2f}, Predicted: {predicted:.2f}")
                    print(f"  Price change %: {price_change_pct*100:.4f}% (need ±{thresholds['long_entry']*100:.4f}% for long, ±{thresholds['short_entry']*100:.4f}% for short)")
                    print(f"  Trend check: {'PASS' if trend_ok else 'FAIL'} (strength: {trend_strength:.4f}, threshold: {trend_threshold:.4f})")
                    print(f"  Volume check: {'PASS' if volume_ok else 'FAIL'} (ratio: {volume_ratio:.2f}, threshold: {volume_threshold:.2f})")
                    print(f"  Signal would be generated: {'Yes' if abs(price_change_pct) > thresholds['long_entry'] and trend_ok and volume_ok else 'No'}")
            
            # Perform backtesting with realistic transaction costs
            trades, total_commission, total_slippage = backtest_trades_with_costs(
                all_signals, 
                stock_df_orig,
                commission_rate=TRANSACTION_COSTS['commission_rate'],
                slippage_factor=TRANSACTION_COSTS['slippage_factor']
            )
            print(f"Generated {len(trades)} trades")
            
            # Calculate trade performance metrics
            trade_metrics = evaluator.calculate_trade_performance_metrics(
                trades=trades,
                initial_capital=10000.0,
                include_transaction_costs=True
            )
            
            # Log costs
            logger.info(f"Total commission: ${total_commission:.2f}, Total slippage: ${total_slippage:.2f}")
            
            # Print performance summary
            print("\nTrading Performance Summary:")
            print(f"Total Trades: {trade_metrics['total_trades']}")
            print(f"Win Rate: {trade_metrics['win_rate']:.2f}%")
            print(f"Long Trades: {trade_metrics.get('long_trades', 0)}, Short Trades: {trade_metrics.get('short_trades', 0)}")
            print(f"Long Win Rate: {trade_metrics.get('long_win_rate', 0):.2f}%, Short Win Rate: {trade_metrics.get('short_win_rate', 0):.2f}%")
            print(f"Total Profit: ${trade_metrics['total_profit']:.2f}")
            print(f"Maximum Drawdown: ${trade_metrics['maximum_drawdown']:.2f}")
            print(f"Sharpe Ratio: {trade_metrics['sharpe_ratio']:.2f}")
            print(f"Transaction Costs: ${total_commission + total_slippage:.2f}")
            
            # Generate visualizations
            plot_trading_metrics(trade_metrics, ticker)
            plot_candlestick_analysis(stock_df_orig, signals=all_signals, trades=trades, ticker=ticker)
            
            # Save model and artifacts
            save_path = f'{MODEL_SAVE_PATH}/{ticker}/'
            os.makedirs(save_path, exist_ok=True)
            
            model_filename = f'{save_path}lstm_model_{ticker}_{end_date_str}.pth'
            scaler_filename = f'{save_path}scaler_{ticker}_{end_date_str}.pkl'
            history_filename = f'{save_path}training_history_{ticker}_{end_date_str}.pkl'
            metrics_filename = f'{save_path}performance_metrics_{ticker}_{end_date_str}.pkl'
            
            # Save model
            torch.save(model.state_dict(), model_filename)
            
            # Save scaler
            with open(scaler_filename, 'wb') as f:
                pickle.dump(dataset.scaler, f)
                
            # Save training history
            with open(history_filename, 'wb') as f:
                pickle.dump(training_history, f)
                
            # Save performance metrics
            with open(metrics_filename, 'wb') as f:
                metrics_data = {
                    'evaluation_metrics': evaluation_metrics,
                    'trade_metrics': trade_metrics,
                    'transaction_costs': {
                        'commission': total_commission,
                        'slippage': total_slippage
                    }
                }
                pickle.dump(metrics_data, f)
                
            print(f"\nSaved model to {model_filename}")
            print(f"Saved scaler to {scaler_filename}")
            print(f"Saved training history to {history_filename}")
            print(f"Saved performance metrics to {metrics_filename}")
            
            logger.info(f"Completed processing for {ticker}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            traceback.print_exc()
            continue
            
        finally:
            # Clean up memory
            clean_up_memory()
    
    logger.info("Completed processing all tickers")

if __name__ == "__main__":
    try:
        logger.info("Starting enhanced LSTM trading model training pipeline")
        main()
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        traceback.print_exc()
    finally:
        logger.info("Pipeline execution completed")