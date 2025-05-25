"""
Streaming-optimized dataset implementation with S3 integration for the LSTM trading model.
Handles continuous data processing and efficient storage management.
"""
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
import boto3
import io
import glob
import os
from typing import Optional, Generator, Dict, Tuple, List
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from config import logger

# testing directory change to activate CI actions

class StreamingHFTDataset(IterableDataset):
    @classmethod
    def from_sagemaker_channel(cls,
                              channel_path: str,
                              sequence_length: int,
                              target_column: str = 'close',
                              batch_size: int = 32,
                              **kwargs) -> 'StreamingHFTDataset':
        """
        Create dataset from SageMaker channel containing processed parquet files.
        
        Args:
            channel_path: Path to SageMaker channel directory
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict
            batch_size: Size of batches to process
            **kwargs: Additional arguments for StreamingHFTDataset
        """
        def parquet_stream() -> Generator[pd.DataFrame, None, None]:
            # Get all parquet files in channel directory
            parquet_files = glob.glob(os.path.join(channel_path, "**/*.parquet"), recursive=True)
            
            for file_path in sorted(parquet_files):
                try:
                    # Read parquet file in chunks to maintain streaming behavior
                    for chunk in pd.read_parquet(file_path, chunksize=batch_size):
                        yield chunk
                except Exception as e:
                    logger.error(f"Error reading parquet file {file_path}: {e}")
                    continue

        return cls(
            data_stream=parquet_stream(),
            sequence_length=sequence_length,
            target_column=target_column,
            batch_size=batch_size,
            **kwargs
        )

    def __init__(self,
                 data_stream: Generator[pd.DataFrame, None, None],
                 sequence_length: int,
                 target_column: str = 'close',
                 s3_output_config: Optional[Dict] = None,
                 scaler: Optional[StandardScaler] = None,
                 include_current_price: bool = True,
                 relative_normalization: bool = True,
                 batch_size: int = 32,
                 ticker: Optional[str] = None):
        """
        Initialize the streaming dataset.
        
        Args:
            data_stream: Generator yielding DataFrame chunks
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict
            s3_output_config: Dict containing S3 configuration:
                {
                    'bucket': str,
                    'base_path': str,
                    'region': str,
                    'aws_access_key_id': str,
                    'aws_secret_access_key': str
                }
            scaler: Optional scaler for feature normalization
            include_current_price: Whether to include current price in output
            relative_normalization: Whether to normalize sequences relative to first element
            batch_size: Size of batches to process
            ticker: Ticker symbol for ticker-specific adaptations
        """
        self.data_stream = data_stream
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.s3_output_config = s3_output_config
        self.scaler = scaler or StandardScaler()
        self.include_current_price = include_current_price
        self.relative_normalization = relative_normalization
        self.batch_size = batch_size
        self.ticker = ticker
        
        # Initialize S3 client if config provided
        self.s3_client = None
        if s3_output_config:
            self.s3_client = boto3.client(
                's3',
                region_name=s3_output_config.get('region', 'us-east-1'),
                aws_access_key_id=s3_output_config.get('aws_access_key_id'),
                aws_secret_access_key=s3_output_config.get('aws_secret_access_key')
            )
        
        # Initialize sequence buffer
        self.sequence_buffer = []
        
        # Track feature names
        self.feature_names = None
        
    def _get_s3_path(self, timestamp: datetime) -> str:
        """Generate S3 path for storing processed data."""
        if not self.s3_output_config:
            return None
            
        date_str = timestamp.strftime('%Y/%m/%d')
        ticker_path = f"{self.ticker}/" if self.ticker else ""
        return f"{self.s3_output_config['base_path']}/{ticker_path}{date_str}"

    def _store_processed_chunk(self, 
                             processed_data: pd.DataFrame,
                             timestamp: datetime) -> None:
        """Store processed data chunk to S3."""
        if not self.s3_client:
            return
            
        s3_path = self._get_s3_path(timestamp)
        if not s3_path:
            return
            
        # Convert to parquet format
        buffer = io.BytesIO()
        processed_data.to_parquet(buffer)
        buffer.seek(0)
        
        # Generate unique filename
        filename = f"processed_data_{timestamp.strftime('%H%M%S')}_{self.ticker}.parquet"
        s3_key = f"{s3_path}/{filename}"
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_output_config['bucket'],
                Key=s3_key,
                Body=buffer.getvalue()
            )
            logger.info(f"Stored processed data to s3://{self.s3_output_config['bucket']}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to store data to S3: {str(e)}")

    def _prepare_sequence(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a sequence for model input."""
        if self.feature_names is None:
            self.feature_names = data.columns.tolist()
        
        # Apply scaling
        scaled_data = self.scaler.transform(data.values)
        
        # Apply relative normalization if enabled
        if self.relative_normalization:
            price_cols = [i for i, name in enumerate(self.feature_names) 
                         if 'close' in name.lower() or 
                            'price' in name.lower() or 
                            'high' in name.lower() or 
                            'low' in name.lower()]
            
            if price_cols:
                reference_prices = scaled_data[:, price_cols[0]].reshape(-1, 1)
                scaled_data[:, price_cols] = scaled_data[:, price_cols] / reference_prices
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = scaled_data[i:(i + self.sequence_length)]
            target = scaled_data[i + self.sequence_length, 
                               self.feature_names.index(self.target_column)]
            
            if self.include_current_price:
                current_price = scaled_data[i + self.sequence_length - 1,
                                         self.feature_names.index(self.target_column)]
                sequences.append(np.append(seq, current_price))
            else:
                sequences.append(seq)
                
            targets.append(target)
        
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

    def __iter__(self):
        """Iterate over the data stream and yield sequences."""
        for chunk in self.data_stream:
            if chunk.empty:
                continue
                
            # Store processed chunk if S3 config provided
            if self.s3_output_config:
                self._store_processed_chunk(chunk, chunk.index[0])
            
            # Prepare sequences from chunk
            sequences, targets = self._prepare_sequence(chunk)
            
            # Yield sequences in batches
            for i in range(0, len(sequences), self.batch_size):
                batch_end = min(i + self.batch_size, len(sequences))
                yield sequences[i:batch_end], targets[i:batch_end]

    @property
    def feature_count(self) -> int:
        """Return the number of features in the dataset."""
        return len(self.feature_names) if self.feature_names else 0

def create_streaming_dataset(
    data_stream: Optional[Generator[pd.DataFrame, None, None]] = None,
    data_path: Optional[str] = None,
    sequence_length: int = 60,
    s3_config: Optional[Dict] = None,
    **kwargs
) -> StreamingHFTDataset:
    """
    Factory function to create a streaming dataset.
    
    Args:
        data_stream: Generator yielding DataFrame chunks (for processing)
        data_path: Path to processed data (for training)
        sequence_length: Number of time steps in each sequence
        s3_config: Optional S3 configuration dictionary
        **kwargs: Additional arguments for StreamingHFTDataset
        
    Returns:
        Configured StreamingHFTDataset instance
    """
    if data_path is not None:
        return StreamingHFTDataset.from_sagemaker_channel(
            channel_path=data_path,
            sequence_length=sequence_length,
            **kwargs
        )
    elif data_stream is not None:
        return StreamingHFTDataset(
            data_stream=data_stream,
            sequence_length=sequence_length,
            s3_output_config=s3_config,
            **kwargs
        )
    else:
        raise ValueError("Either data_stream or data_path must be provided")
