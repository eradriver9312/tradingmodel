from data.streaming_feature_engineering import stream_enhance_features
from data.StreamingHFTDataset import create_streaming_dataset

# S3 configuration
s3_config = {
    'bucket': 'your-bucket',
    'base_path': 'processed_data',
    'region': 'us-east-1',
    'aws_access_key_id': 'your-key',
    'aws_secret_access_key': 'your-secret'
}

# Create data stream
raw_data_stream = your_data_source_generator()  # Your data source
market_data_stream = your_market_data_generator()  # Your market data source

# Create enhanced feature stream
enhanced_stream = stream_enhance_features(raw_data_stream, market_data_stream)

# Create streaming dataset
dataset = create_streaming_dataset(
    data_stream=enhanced_stream,
    sequence_length=60,
    s3_config=s3_config,
    ticker='AAPL',
    batch_size=32,
    include_current_price=True
)

# Use in training/inference
for sequences, targets in dataset:
    # Your training/inference code here
    pass