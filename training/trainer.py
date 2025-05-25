"""
Enhanced LSTM model trainer with support for ensemble models and streaming datasets.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import logger
from models.lstm_model import LSTMModel
from models.loss_functions import DirectionalPredictionLoss
from utils import TrainingHistory
from .evaluator import ModelEvaluator

class ModelTrainer:
    def __init__(self,
                 device: torch.device,
                 model_params: Dict,
                 ticker: Optional[str] = None):
        """
        Initialize the model trainer.
        
        Args:
            device: Computation device (CPU/GPU)
            model_params: Dictionary containing model hyperparameters
            ticker: Optional stock ticker symbol
        """
        self.device = device
        self.model_params = model_params
        self.ticker = ticker
        self.evaluator = ModelEvaluator()
        
    def apply_dynamic_scaling(self,
                            model: nn.Module,
                            train_loader: DataLoader) -> None:
        """Apply dynamic scaling to model output layer based on price range."""
        price_values = []
        
        # Collect price values from training data
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                _, targets = batch
                price_values.extend(targets.numpy())
        
        price_range = np.max(price_values) - np.min(price_values)
        scale_factor = 1.0 / max(1.0, price_range)
        
        # Apply scaling to last layer
        for name, param in model.named_parameters():
            if 'fc' in name and 'weight' in name:
                last_layer = param
                last_layer.data.mul_(scale_factor)
        
        logger.info(f"Applied dynamic scaling with price range: {price_range:.2f}, scale factor: {scale_factor:.4f}")

    def train_model(self,
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   num_epochs: int,
                   training_history: Optional[TrainingHistory] = None,
                   early_stopping_patience: int = 10,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> nn.Module:
        """
        Train the LSTM model with enhanced monitoring and early stopping.
        """
        model.to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Apply dynamic scaling for output range
        self.apply_dynamic_scaling(model, train_loader)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # Update training history
            if training_history is not None:
                training_history.update(avg_train_loss, avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}")
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model

    def train_ensemble(self,
                      model_configs: List[Dict],
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      test_loader: DataLoader) -> Tuple[List[nn.Module], List[float]]:
        """
        Train multiple models for ensemble prediction.
        
        Returns:
            Tuple of (trained models list, model weights list)
        """
        ensemble_models = []
        ensemble_weights = []
        
        for i, config in enumerate(model_configs):
            logger.info(f"Training model {i+1}/{len(model_configs)}")
            
            # Initialize model
            model = LSTMModel(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            
            # Initialize training components
            criterion = DirectionalPredictionLoss(
                direction_weight=2.5,
                magnitude_weight=0.8,
                short_penalty_multiplier=3.5,
                bias_correction_weight=1.5
            )
            
            if self.ticker:
                criterion.set_ticker(self.ticker)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=0.01
            )
            
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            # Train model
            model_history = TrainingHistory()
            model = self.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=config['num_epochs'],
                training_history=model_history,
                early_stopping_patience=config['early_stopping_patience'],
                scheduler=scheduler
            )
            
            # Evaluate model
            model.eval()
            direction_accuracy = self.evaluator.evaluate_direction_accuracy(
                model, val_loader, self.device
            )
            
            # Calculate model weight based on validation performance
            model_weight = max(0.1, direction_accuracy - 49)  # Minimum weight of 0.1
            
            ensemble_models.append(model)
            ensemble_weights.append(model_weight)
            
            logger.info(f"Model {i+1} validation accuracy: {direction_accuracy:.2f}%, "
                       f"weight: {model_weight:.2f}")
        
        # Normalize weights
        total_weight = sum(ensemble_weights)
        ensemble_weights = [w / total_weight for w in ensemble_weights]
        
        return ensemble_models, ensemble_weights