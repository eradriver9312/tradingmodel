"""
Model evaluation utilities for the LSTM trading model.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple

class ModelEvaluator:
    def evaluate_direction_accuracy(self,
                                  model: torch.nn.Module,
                                  data_loader: DataLoader,
                                  device: torch.device) -> float:
        """Calculate direction prediction accuracy."""
        correct_directions = 0
        total_predictions = 0
        
        model.eval()
        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs = model(sequences)
                
                # Calculate prediction directions
                pred_directions = (outputs > 0).float()
                true_directions = (targets > 0).float()
                
                correct_directions += (pred_directions == true_directions).sum().item()
                total_predictions += targets.size(0)
        
        return (correct_directions / total_predictions) * 100

    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         current_prices: np.ndarray,
                         ticker: str = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing various performance metrics
        """
        metrics = {}
        
        # Direction accuracy
        correct_directions = np.sum((y_pred > 0) == (y_true > 0))
        metrics['direction_accuracy'] = (correct_directions / len(y_true)) * 100
        
        # Mean absolute error
        metrics['mae'] = np.mean(np.abs(y_pred - y_true))
        
        # Root mean squared error
        metrics['rmse'] = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        # Direction-specific accuracies
        up_mask = y_true > 0
        down_mask = y_true < 0
        
        if np.sum(up_mask) > 0:
            metrics['up_accuracy'] = (
                np.sum((y_pred > 0) & up_mask) / np.sum(up_mask)
            ) * 100
        
        if np.sum(down_mask) > 0:
            metrics['down_accuracy'] = (
                np.sum((y_pred < 0) & down_mask) / np.sum(down_mask)
            ) * 100
        
        # Add ticker-specific metrics if provided
        if ticker:
            metrics['ticker'] = ticker
        
        return metrics