"""
SageMaker-specific training script that uses our existing ModelTrainer.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
import argparse

from training.trainer import ModelTrainer
from models.lstm_model import LSTMModel
from models.loss_functions import DirectionalPredictionLoss
from data.StreamingHFTDataset import create_streaming_dataset
from config import logger

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--ticker', type=str, default='AAPL')
    
    return parser.parse_args()

def main(args):
    logger.info(f"Starting training with args: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create streaming datasets from SageMaker channels
    train_dataset = create_streaming_dataset(
        data_path=args.train,  # Points to SM_CHANNEL_TRAIN
        sequence_length=args.sequence_length,
        ticker=args.ticker,
        batch_size=args.batch_size
    )
    
    val_dataset = create_streaming_dataset(
        data_path=args.validation,  # Points to SM_CHANNEL_VALIDATION
        sequence_length=args.sequence_length,
        ticker=args.ticker,
        batch_size=args.batch_size
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model and training components
    model_params = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'sequence_length': args.sequence_length
    }
    
    trainer = ModelTrainer(
        device=device,
        model_params=model_params,
        ticker=args.ticker
    )
    
    model = LSTMModel(
        input_size=train_dataset.feature_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    criterion = DirectionalPredictionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    trained_model = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        early_stopping_patience=10
    )
    
    # Save model artifacts
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(trained_model.state_dict(), model_path)
    
    # Save model parameters
    params_path = os.path.join(args.model_dir, 'model_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f)
    
    logger.info("Training completed successfully")

if __name__ == '__main__':
    args = parse_args()
    main(args)
