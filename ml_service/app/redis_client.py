# app/redis_client.py

import redis
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# In a real app, these would come from environment variables or a config file.
# For local setup, we can use defaults that match our Docker command.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Create a reusable Redis connection pool.
# decode_responses=True means that the client will automatically decode
# responses from Redis from bytes to UTF-8 strings.
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True
    )
    # Ping the server to check the connection
    redis_client.ping()
    log.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    log.error(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Error: {e}")
    redis_client = None

def store_dataframe_as_json(key: str, df, expiration_seconds: int = 3600):
    """
    Serializes a Pandas DataFrame to JSON and stores it in Redis with an expiration.
    """
    if not redis_client:
        log.error("Redis client is not available. Cannot store DataFrame.")
        return

    log.info(f"Attempting to store DataFrame in Redis under key: '{key}'")
    try:
        data_list = df.to_dict(orient='records')
        json_string = json.dumps(data_list, default=str)
        redis_client.setex(key, expiration_seconds, json_string)
        log.info(f"Successfully stored {len(data_list)} records ({len(json_string)} bytes) for key '{key}'.")
    except Exception as e:
        log.error(f"Failed to store DataFrame for key '{key}'. Error: {e}", exc_info=True)
        raise

def get_dataframe_from_json(key: str):
    """
    Retrieves a JSON string from Redis and deserializes it back into a DataFrame.
    """
    if not redis_client:
        log.error("Redis client is not available. Cannot get DataFrame.")
        return None

    import pandas as pd
    log.info(f"Attempting to retrieve DataFrame from Redis with key: '{key}'")
    try:
        json_string = redis_client.get(key)
        if json_string:
            log.info(f"Found data for key '{key}'. Deserializing...")
            data_list = json.loads(json_string)
            df = pd.DataFrame(data_list)
            log.info(f"Successfully deserialized DataFrame. Shape: {df.shape}")
            return df
        else:
            log.warning(f"No data found in Redis for key: '{key}'")
            return None
    except Exception as e:
        log.error(f"Failed to retrieve or parse DataFrame for key '{key}'. Error: {e}", exc_info=True)
        raise

# ... (existing code)

def delete_key(key: str):
    """Deletes a key from Redis."""
    if not redis_client:
        log.error("Redis client is not available. Cannot delete key.")
        return

    log.info(f"Attempting to delete key from Redis: '{key}'")
    try:
        # The delete command returns the number of keys deleted.
        result = redis_client.delete(key)
        if result > 0:
            log.info(f"Successfully deleted key '{key}'.")
        else:
            log.warning(f"Key '{key}' not found, nothing to delete.")
        return result
    except Exception as e:
        log.error(f"Failed to delete key '{key}'. Error: {e}", exc_info=True)
        raise
