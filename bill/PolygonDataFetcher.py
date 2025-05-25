import time
from datetime import datetime, timedelta
import logging
import requests
import os
import pandas as pd
from tqdm import tqdm


# Suppress mplfinance warnings for too much data
# import warnings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='stock_prediction_hft.log',
    filemode='a',
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# API Key Configuration
API_KEY = os.environ.get("POLYGON_API_KEY")


class PolygonDataFetcher:
    BASE_URL = "https://api.polygon.io/v2"

    def __init__(self, api_key):
        """
        Initialize the PolygonDataFetcher with API key.

        Args:
            api_key (str): Polygon.io API key.
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.validation_metrics = []

    def _make_request(self, endpoint, params=None, retries=3):
        """
        Make an API request with retry logic.

        Args:
            endpoint (str): API endpoint.
            params (dict, optional): Query parameters.
            retries (int): Number of retries.

        Returns:
            dict: JSON response.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 429:
                    wait_time = min(60 * (attempt + 1), 300)  # Max 5 minutes
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                data = response.json()

                # Log the keys of the response
                logger.debug(f"API response keys: {data.keys()}")

                # Optional: Log a sample of the results
                if 'results' in data and len(data['results']) > 0:
                    logger.debug(f"Sample result: {data['results'][0]}")

                return data
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Request failed after {retries} attempts: {e}")
                    raise
                logger.warning(f"Request failed (Attempt {attempt + 1}/{retries}): {e}. Retrying in 30 seconds...")
                time.sleep(30)
        raise Exception(f"Failed after {retries} attempts")

    def fetch_stock_data(self, symbol, start_date, end_date, timespan="minute", multiplier=1):
        """
        Fetch aggregate stock data (e.g., OHLC) for a specific ticker and date range.

        Args:
            symbol (str): Stock ticker symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            timespan (str, optional): Timespan of the aggregates ('minute', 'hour', 'day'). Defaults to "minute".
            multiplier (int, optional): The size of the timespan multiplier. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing aggregated stock data.
        """
        endpoint = f"aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        all_results = []

        with tqdm(desc=f"Fetching {symbol}", unit='rows') as pbar:
            while True:
                data = self._make_request(endpoint, params)
                if 'results' in data:
                    # Check if 'c' (close) is in first result
                    if len(data['results']) > 0 and 'c' not in data['results'][0]:
                        logger.error(f"Data for {symbol} does not contain 'c' (close) field.")
                        return pd.DataFrame()

                    all_results.extend(data['results'])
                    pbar.update(len(data['results']))
                if len(data.get('results', [])) < 50000:
                    break
                last_timestamp = data['results'][-1]['t']
                # Increment start_date to avoid infinite loop
                start_date = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d')
                params['from'] = start_date

        df = pd.DataFrame(all_results)

        # Remove duplicate timestamps
        if 't' in df.columns:
            duplicates_before = len(df) - df.drop_duplicates(subset=['t'], keep='last').shape[0]
            if duplicates_before > 0:
                logger.info(f"Removed {duplicates_before} duplicate timestamps for {symbol}.")
            df = df.drop_duplicates(subset=['t'], keep='last')
        else:
            logger.error("'t' column not found in fetched data.")
            return pd.DataFrame()

        # Rename columns for clarity
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'n': 'transactions'
        })

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df