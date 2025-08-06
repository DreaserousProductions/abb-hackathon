# app/services/csv_service.py

import logging
from app.services import data_processing_service
from app.services import model_training_service
from app.services import simulation_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Facade Functions: Delegate to Specialized Services ---

def find_parquet_file(dataset_id: str) -> str | None:
    """Delegates to the data processing service."""
    log.info(f"Master service delegating parquet file search for dataset: {dataset_id}")
    return data_processing_service.find_parquet_file(dataset_id)

def process_csv_to_parquet(temp_csv_path: str, user_id: str) -> dict:
    """Delegates to the data processing service."""
    log.info(f"Master service delegating CSV processing for user: {user_id}")
    return data_processing_service.process_csv_to_parquet(temp_csv_path, user_id)

def validate_and_count_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
    """Delegates to the data processing service."""
    log.info(f"Master service delegating range validation for dataset: {dataset_id}")
    return data_processing_service.validate_and_count_ranges(user_id, dataset_id, ranges)

def create_feature_importance_plot(feature_names: list, importances: list, top_n: int = 20) -> str:
    """Delegates to the model training service."""
    log.info(f"Master service delegating feature importance plot generation")
    return model_training_service.create_feature_importance_plot(feature_names, importances, top_n)

def create_training_history_plot(evals_result: dict) -> str:
    """Delegates to the model training service."""
    log.info(f"Master service delegating training history plot generation")
    return model_training_service.create_training_history_plot(evals_result)

def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
    """Delegates to the model training service."""
    log.info(f"Master service delegating model training for dataset: {dataset_id}")
    return model_training_service.train_model_with_ranges(user_id, dataset_id, ranges)

async def run_simulation_for_websocket(user_id: str, dataset_id: str):
    """Delegates to the simulation service."""
    log.info(f"Master service delegating simulation for dataset: {dataset_id}")
    async for result in simulation_service.run_simulation_for_websocket(user_id, dataset_id):
        yield result