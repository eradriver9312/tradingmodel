from training import ModelTrainer
from models.lstm_model import LSTMModel
from data.StreamingHFTDataset import create_streaming_dataset

# Initialize trainer
trainer = ModelTrainer(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    model_params=MODEL_PARAMS,
    ticker='AAPL'
)

# Create streaming dataset and data loaders
dataset = create_streaming_dataset(...)  # As shown in previous example
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32)

# Initialize model and training components
model = LSTMModel(...)
criterion = DirectionalPredictionLoss(...)
optimizer = torch.optim.Adam(...)

# Train model
trained_model = trainer.train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    early_stopping_patience=10
)