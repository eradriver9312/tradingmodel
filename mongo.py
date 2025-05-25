# Required Imports
import os
from dotenv import load_dotenv
# Required Imports
from datetime import datetime
from pymongo import MongoClient, errors
import uuid


load_dotenv()


try:
    client = MongoClient(os.getenv("MONGO_URL"), serverSelectionTimeoutMS=5000)
    db = client["trading_db"]
    orders_collection = db["orders"]
    tickers_collection = db["tickers"]
    # Check the connection
    client.admin.command("ping")
    print("Connected to MongoDB successfully.")
except errors.ConnectionFailure as e:
    print(f"MongoDB connection failed: {e}")
    exit(1)

# Utility Functions
def datetime_to_unix(dt):
    """Convert datetime object to Unix timestamp (seconds)."""
    return int(dt.timestamp()) if dt else None

def calculate_transaction_duration(created_at, filled_at):
    """Calculate transaction duration in seconds."""
    if created_at and filled_at:
        return filled_at - created_at
    return None

def calculate_net_profit_loss(filled_qty, filled_avg_price, market_price):
    """Calculate net profit/loss for a filled order. Return a positive value for profit and a negative value for loss."""
    try:
        filled_qty = float(filled_qty)  # Ensure it's a float (to support fractional shares)
        filled_avg_price = float(filled_avg_price)  # Ensure it's a float
        market_price = float(market_price)  # Ensure it's a float
        return round((market_price - filled_avg_price) * filled_qty, 2)
    except (ValueError, TypeError) as e:
        print(f"Error calculating net profit/loss: {e}")
        return None


# Insert Order Function with Error Handling
def insert_order(order, customer_id, signal_price):
    print("insert_order")
    """Insert a new order document into MongoDB."""
    try:
        order_document = {
            "order_id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "customer_id": customer_id,
            "timestamps": {
                "created_at": datetime_to_unix(order.created_at),
                "updated_at": datetime_to_unix(order.updated_at),
                "submitted_at": datetime_to_unix(order.submitted_at),
                "filled_at": datetime_to_unix(order.filled_at),
                "expired_at": datetime_to_unix(order.expired_at),
                "canceled_at": datetime_to_unix(order.canceled_at),
                "failed_at": datetime_to_unix(order.failed_at),
                "expires_at": datetime_to_unix(order.expires_at)
            },
            "asset": {
                "asset_id": str(order.asset_id),
                "symbol": order.symbol,
                "asset_class": order.asset_class.value
            },
            "order_details": {
                "signal_price":signal_price,
                "qty": int(order.qty),
                "filled_qty": int(order.filled_qty),
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "price": None,
                "order_class": order.order_class.value,
                "order_type": order.order_type.value,
                "side": order.side.value,
                "time_in_force": order.time_in_force.value,
                "status": order.status.value,
                "position_qty": None,
                "transaction_duration": None,
                "net_profit_loss": None
            }
        }
        result = orders_collection.insert_one(order_document)
        # print(f"Order inserted with ID: {result.inserted_id}")
    except errors.PyMongoError as e:
        print(f"Failed to insert order: {e}")

# Update Order from Websocket Function with Error Handling
def update_order_from_websocket(order_update, market_price=None):
    """Update an existing order document based on websocket data."""
    try:
        order = order_update.order  # The actual order object

        # print(f"DEBUG: Raw order data -> {order.__dict__}")  
        # print(f"DEBUG: market_price before conversion -> {market_price} (type={type(market_price)})")

        # Build the MongoDB query based on order ID
        query = {"order_id": str(order.id)}

        # Convert timestamps safely
        filled_at_unix = datetime_to_unix(order.filled_at) if order.filled_at else None
        created_at_unix = datetime_to_unix(order.created_at) if order.created_at else None
        transaction_duration = calculate_transaction_duration(created_at_unix, filled_at_unix) if filled_at_unix and created_at_unix else None

        # Force all values to be floats, replacing None with 0.0
        market_price = float(market_price) if isinstance(market_price, (int, float, str)) and market_price not in [None, "None"] else 0.0
        filled_qty = float(order.filled_qty) if isinstance(order.filled_qty, (int, float, str)) and order.filled_qty not in [None, "None"] else 0.0
        filled_avg_price = float(order.filled_avg_price) if isinstance(order.filled_avg_price, (int, float, str)) and order.filled_avg_price not in [None, "None"] else 0.0
        order_price = float(getattr(order, "price", 0.0)) if isinstance(getattr(order, "price", None), (int, float, str)) else 0.0
        position_qty = float(getattr(order_update, "position_qty", 0.0)) if isinstance(getattr(order_update, "position_qty", None), (int, float, str)) else 0.0

        # print(f"DEBUG: filled_qty={filled_qty}, filled_avg_price={filled_avg_price}, market_price={market_price}")
        # print(f"DEBUG: order_price={order_price}, position_qty={position_qty}")

        # Compute net profit/loss only for sell orders that are fully filled
        net_profit_loss = 0.0  # Default value
        if order.side.value == "sell" and order.status.value == "filled":
            net_profit_loss = calculate_net_profit_loss(filled_qty, filled_avg_price, market_price)

        print(f"DEBUG: net_profit_loss={net_profit_loss}")

        # Build the MongoDB update query
        update = {
            "$set": {
                "order_details.status": order.status.value,
                "order_details.filled_qty": filled_qty,
                "order_details.filled_avg_price": filled_avg_price,
                "order_details.price": order_price,
                "order_details.position_qty": position_qty,
                "timestamps.updated_at": datetime_to_unix(order.updated_at) if order.updated_at else None,
                "timestamps.filled_at": filled_at_unix,
                "order_details.transaction_duration": transaction_duration,
                "order_details.net_profit_loss": net_profit_loss,
                "order_details.side": order.side.value
            }
        }

        # print(f"DEBUG: Final update object -> {update}")

        # Perform MongoDB update
        result = orders_collection.update_one(query, update)
        if result.modified_count > 0:
            print(f"Order {order.id} updated successfully.")
        else:
            print(f"No order found with ID {order.id}.")
    except Exception as e:
        print(f"ERROR in update_order_from_websocket: {e}")








def get_all_ticker_objects():
    tickers_cursor = tickers_collection.find({}, {})  # No projection to include _id
    tickers_list = list(tickers_cursor)  # Convert cursor to a list of dictionaries
    return tickers_list

def get_all_tickers_only():
    tickers_cursor = tickers_collection.find(
        {}, {"_id": 0, "ticker": 1}
    )  # Exclude `_id`, include only `ticker`
    tickers = [
        doc["ticker"] for doc in tickers_cursor
    ]  # Extract ticker values into a list
    return tickers

def get_fixed_shares():
    tickers_cursor = tickers_collection.find(
        {}, {"_id": 0, "ticker": 1, "default_shares": 1}
    )  # Fetch only needed fields
    fixed_shares = {
        doc["ticker"]: doc["default_shares"] for doc in tickers_cursor
    }  # Convert to dictionary
    return fixed_shares

def get_filtered_fixed_shares(ticker_list):
    """Fetches default_shares values for a given subset of tickers."""
    tickers_cursor = tickers_collection.find(
        {"ticker": {"$in": ticker_list}},  # Filter by tickers in the provided list
        {"_id": 0, "ticker": 1, "default_shares": 1},  # Project only necessary fields
    )
    filtered_shares = {
        doc["ticker"]: doc["default_shares"] for doc in tickers_cursor
    }  # Convert to dictionary
    return filtered_shares

# Example Usage (Mock Order and Websocket Update)
# class MockOrder:
#     id = uuid.uuid4()
#     client_order_id = uuid.uuid4()
#     created_at = datetime.utcnow()
#     updated_at = datetime.utcnow()
#     submitted_at = datetime.utcnow()
#     filled_at = None
#     expired_at = None
#     canceled_at = None
#     failed_at = None
#     expires_at = datetime.utcnow()
#     asset_id = uuid.uuid4()
#     symbol = "AAPL"
#     asset_class = type("AssetClass", (), {"value": "us_equity"})
#     qty = "10"
#     filled_qty = "0"
#     filled_avg_price = None
#     order_class = type("OrderClass", (), {"value": "simple"})
#     order_type = type("OrderType", (), {"value": "market"})
#     side = type("OrderSide", (), {"value": "buy"})
#     time_in_force = type("TimeInForce", (), {"value": "day"})
#     status = type("OrderStatus", (), {"value": "pending_new"})


# order = MockOrder()
# insert_order(order, "3bean")

# # Mock Websocket Update Data (with made-up "filled" data)
# mock_websocket_update = {
#     "id": order.id,  # Same ID as the inserted order to ensure we update it
#     "status": type("OrderStatus", (), {"value": "filled"}),  # Status set to 'filled'
#     "filled_qty": "10",  # Fully filled
#     "filled_avg_price": "228.50",  # Made-up average filled price
#     "price": "229.00",  # Current market price at the time of update
#     "position_qty": 130.0,  # Updated position quantity
#     "updated_at": datetime.utcnow(),
#     "filled_at": datetime.utcnow()  # Made-up filled timestamp
# }

# # Simulate Websocket Update Call
# update_order_from_websocket(mock_websocket_update, market_price=mock_websocket_update["price"])

