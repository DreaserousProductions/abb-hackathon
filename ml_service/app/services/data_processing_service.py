# app/services/data_processing_service.py

import pandas as pd
import uuid
import os
import glob
from datetime import datetime, timedelta
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import gc  # Added for explicit garbage collection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Define a storage directory for Parquet files ---
PARQUET_STORAGE_DIR = "data_store"
os.makedirs(PARQUET_STORAGE_DIR, exist_ok=True)

def find_parquet_file(dataset_id: str) -> str | None:
    """Finds the full path of a parquet file using its unique dataset_id."""
    search_pattern = os.path.join(PARQUET_STORAGE_DIR, f"*_dataset_{dataset_id}_*.parquet")
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        log.error(f"No parquet file found for dataset_id: {dataset_id}")
        return None
    return matching_files[0]

def process_csv_to_parquet(temp_csv_path: str, user_id: str) -> dict:
    """
    BLAZING FAST single-pass CSV processing that simultaneously:
    1. Extracts metadata (string-based date range, counts, pass rate)
    2. Converts to Parquet chunk by chunk
    3. Minimizes memory usage and disk I/O
    This replaces the previous slow two-pass approach.
    """
    log.info(f"Starting OPTIMIZED single-pass processing for user_id: {user_id} from {temp_csv_path}")

    # Generate unique identifiers
    dataset_id = str(uuid.uuid4())
    temp_parquet_name = f"temp_{uuid.uuid4().hex}.parquet"
    temp_parquet_path = os.path.join(PARQUET_STORAGE_DIR, temp_parquet_name)

    # Metadata tracking variables
    total_records = 0
    pass_count = 0
    num_columns = 0
    time_col_name = None

    # String-based date range tracking (PERFORMANCE OPTIMIZATION)
    min_date_string = None
    max_date_string = None

    # Parquet writer for efficient chunk-by-chunk writing
    parquet_writer = None

    try:
        log.info("Beginning single-pass chunk processing...")

        # SINGLE PASS: Process CSV in chunks, extract metadata AND write Parquet simultaneously
        chunk_size = 20_000  # Process 100K rows at a time for optimal memory usage

        with pd.read_csv(temp_csv_path, chunksize=chunk_size, low_memory=False) as chunk_iterator:
            for chunk_idx, chunk in enumerate(chunk_iterator):
                # === METADATA EXTRACTION (First chunk only) ===
                if chunk_idx == 0:
                    num_columns = len(chunk.columns)
                    # Detect timestamp column
                    time_col_name = next(
                        (col for col in chunk.columns if 'timestamp' in col.lower() or 'date' in col.lower()),
                        'synthetic_timestamp'
                    )
                    log.info(f"Detected timestamp column: '{time_col_name}'")

                # === SYNTHETIC TIMESTAMP GENERATION (if needed) ===
                if time_col_name == 'synthetic_timestamp':
                    # Generate synthetic timestamps based on global row position
                    start_time = datetime(2025, 1, 1, 0, 0, 0)
                    chunk[time_col_name] = [
                        start_time + timedelta(seconds=(total_records + i))
                        for i in range(len(chunk))
                    ]

                # === STRING-BASED DATE RANGE EXTRACTION (CRITICAL OPTIMIZATION) ===
                # Instead of expensive pd.to_datetime on millions of rows, work with strings
                time_col_values = chunk[time_col_name].astype(str)
                # Find lexicographic min/max (works for ISO format timestamps)
                chunk_min_str = time_col_values.min()
                chunk_max_str = time_col_values.max()

                # Update global string-based min/max
                if min_date_string is None or chunk_min_str < min_date_string:
                    min_date_string = chunk_min_str
                if max_date_string is None or chunk_max_str > max_date_string:
                    max_date_string = chunk_max_str

                # Clean up intermediate string values immediately
                del time_col_values, chunk_min_str, chunk_max_str

                # === PASS RATE CALCULATION ===
                if 'Response' in chunk.columns:
                    response_col = pd.to_numeric(chunk['Response'], errors='coerce')
                    pass_count += response_col[response_col == 1].count()
                    del response_col  # Clean up immediately

                # === TYPE CONVERSION FOR PARQUET (per chunk) ===
                # Convert timestamp column to proper datetime (only for this chunk)
                chunk[time_col_name] = pd.to_datetime(chunk[time_col_name], errors='coerce')
                chunk.dropna(subset=[time_col_name], inplace=True)

                # Update total record count
                total_records += len(chunk)

                # === PARQUET WRITING (chunk by chunk) ===
                # Convert chunk to PyArrow table
                table = pa.Table.from_pandas(chunk)

                if parquet_writer is None:
                    # Initialize writer with schema from first chunk
                    parquet_writer = pq.ParquetWriter(temp_parquet_path, table.schema)
                    log.info(f"Initialized ParquetWriter for temporary file: {temp_parquet_path}")

                # Write chunk to Parquet file
                parquet_writer.write_table(table)

                # === CRITICAL MEMORY CLEANUP ===
                # Explicitly delete the PyArrow table to free memory immediately
                del table
                # Clear the chunk DataFrame to free memory
                del chunk
                
                # Force garbage collection every chunk to prevent memory accumulation
                gc.collect()

                if chunk_idx % 10 == 0:  # Log progress every 10 chunks
                    log.info(f"Processed chunk {chunk_idx + 1}, total records so far: {total_records:,}")

        # === FINALIZE PARQUET WRITER ===
        if parquet_writer:
            parquet_writer.close()
            log.info("ParquetWriter closed successfully.")

        # === CONVERT STRING DATES TO DATETIME (only twice, not millions of times) ===
        try:
            overall_min_date = pd.to_datetime(min_date_string)
            overall_max_date = pd.to_datetime(max_date_string)
        except Exception as e:
            log.warning(f"Could not parse date strings '{min_date_string}', '{max_date_string}': {e}")
            overall_min_date = pd.Timestamp.now()
            overall_max_date = pd.Timestamp.now()

        # === CONSTRUCT SMART FILENAME ===
        start_date_str = overall_min_date.strftime('%Y%m%d%H%M%S')
        end_date_str = overall_max_date.strftime('%Y%m%d%H%M%S')
        final_filename = f"user_{user_id}_dataset_{dataset_id}_from_{start_date_str}_to_{end_date_str}.parquet"
        final_parquet_path = os.path.join(PARQUET_STORAGE_DIR, final_filename)

        # === ATOMIC RENAME (instantaneous) ===
        os.rename(temp_parquet_path, final_parquet_path)
        log.info(f"Renamed temporary file to final path: {final_parquet_path}")

        # === CALCULATE FINAL METRICS ===
        pass_rate = (pass_count / total_records) * 100 if total_records > 0 else 0.0
        date_range = {
            "start": overall_min_date.isoformat() if pd.notna(overall_min_date) else None,
            "end": overall_max_date.isoformat() if pd.notna(overall_max_date) else None
        }

        log.info(f"OPTIMIZED processing complete! Dataset ID: {dataset_id}")
        log.info(f"Total records: {total_records:,}, Pass rate: {pass_rate:.2f}%")

        return {
            "datasetId": dataset_id,
            "userId": user_id,
            "totalRecords": int(total_records),
            "numColumns": int(num_columns),
            "passRate": round(float(pass_rate), 2),
            "dateRange": date_range
        }

    except Exception as e:
        log.error(f"Error during optimized processing: {e}", exc_info=True)
        # Clean up temporary file if it exists
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)
        raise

    finally:
        # === RELIABLE CLEANUP ===
        # Always clean up the temporary CSV file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            log.info(f"Cleaned up temporary CSV file: {temp_csv_path}")

        # Ensure parquet writer is closed if something went wrong
        if parquet_writer:
            try:
                parquet_writer.close()
            except:
                pass

        # Final garbage collection to ensure all resources are released
        gc.collect()

def validate_and_count_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
    """
    Validates date ranges by first checking the filename, then estimates counts based on the
    assumption of one record per second without loading the Parquet dataset.
    """
    log.info(f"--- Validating ranges for dataset: {dataset_id} ---")

    parquet_path = find_parquet_file(dataset_id)
    if not parquet_path:
        raise ValueError("Dataset file not found. It may have expired or failed processing.")

    # OPTIMIZATION: Validate against filename before reading the file
    try:
        parts = os.path.basename(parquet_path).split('_')
        file_start_str = parts[-3]  # e.g., 20230101093000
        file_end_str = parts[-1].split('.')[0]  # e.g., 20231231174559

        file_start_date = pd.to_datetime(file_start_str, format='%Y%m%d%H%M%S')
        file_end_date = pd.to_datetime(file_end_str, format='%Y%m%d%H%M%S')

        request_training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
        request_simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

        if request_training_start < file_start_date or request_simulation_end > file_end_date:
            detail = f"Requested ranges are outside the dataset's total time boundary ({file_start_date} to {file_end_date})."
            log.warning(f"Validation failed (filename check): {detail}")
            return {"status": "Invalid", "detail": detail}

        log.info("Filename boundary check passed.")

    except (IndexError, ValueError) as e:
        log.error(f"Could not parse dates from filename '{parquet_path}': {e}")
        raise ValueError("Could not validate filename. File may be malformed.")

    # Parse all date ranges
    training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
    training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
    testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
    testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
    simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
    simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

    # Check for overlaps
    if training_end > testing_start or testing_end > simulation_start:
        return {"status": "Invalid", "detail": "Date ranges cannot overlap."}

    # Calculate counts assuming 1 record per second
    training_count = int((training_end - training_start).total_seconds())
    testing_count = int((testing_end - testing_start).total_seconds())
    simulation_count = int((simulation_end - simulation_start).total_seconds())

    # Calculate monthlyCounts from training start to simulation end
    # Generate all months between training start and simulation end
    current_month = training_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = simulation_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    monthly_counts = {}
    
    while current_month <= end_month:
        # Calculate the start and end of this month
        month_start = current_month
        if current_month.month == 12:
            month_end = current_month.replace(year=current_month.year + 1, month=1) - pd.Timedelta(seconds=1)
        else:
            month_end = current_month.replace(month=current_month.month + 1) - pd.Timedelta(seconds=1)
        
        # Calculate how many seconds from each range fall within this month
        month_total = 0
        
        # Check training range overlap with this month
        training_overlap_start = max(training_start, month_start)
        training_overlap_end = min(training_end, month_end)
        if training_overlap_start < training_overlap_end:
            month_total += int((training_overlap_end - training_overlap_start).total_seconds())
        
        # Check testing range overlap with this month
        testing_overlap_start = max(testing_start, month_start)
        testing_overlap_end = min(testing_end, month_end)
        if testing_overlap_start < testing_overlap_end:
            month_total += int((testing_overlap_end - testing_overlap_start).total_seconds())
        
        # Check simulation range overlap with this month
        simulation_overlap_start = max(simulation_start, month_start)
        simulation_overlap_end = min(simulation_end, month_end)
        if simulation_overlap_start < simulation_overlap_end:
            month_total += int((simulation_overlap_end - simulation_overlap_start).total_seconds())
        
        # Add to monthly counts if there are any records
        if month_total > 0:
            month_key = current_month.strftime('%Y-%m')
            monthly_counts[month_key] = month_total
        
        # Move to next month
        if current_month.month == 12:
            current_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            current_month = current_month.replace(month=current_month.month + 1)

    return {
        "status": "Valid",
        "training": {"count": training_count},
        "testing": {"count": testing_count},
        "simulation": {"count": simulation_count},
        "monthlyCounts": monthly_counts
    }


# def validate_and_count_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     Validates date ranges by first checking the filename, then loads the
#     Parquet file to get exact counts if validation passes.
#     """
#     log.info(f"--- Validating ranges for dataset: {dataset_id} ---")

#     parquet_path = find_parquet_file(dataset_id)
#     if not parquet_path:
#         raise ValueError("Dataset file not found. It may have expired or failed processing.")

#     # OPTIMIZATION: Validate against filename before reading the file
#     try:
#         parts = os.path.basename(parquet_path).split('_')
#         file_start_str = parts[-3]  # e.g., 20230101093000
#         file_end_str = parts[-1].split('.')[0]  # e.g., 20231231174559

#         file_start_date = pd.to_datetime(file_start_str, format='%Y%m%d%H%M%S')
#         file_end_date = pd.to_datetime(file_end_str, format='%Y%m%d%H%M%S')

#         request_training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#         request_simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

#         if request_training_start < file_start_date or request_simulation_end > file_end_date:
#             detail = f"Requested ranges are outside the dataset's total time boundary ({file_start_date} to {file_end_date})."
#             log.warning(f"Validation failed (filename check): {detail}")
#             return {"status": "Invalid", "detail": detail}

#         log.info("Filename boundary check passed. Proceeding to load Parquet file.")

#     except (IndexError, ValueError) as e:
#         log.error(f"Could not parse dates from filename '{parquet_path}': {e}")
#         raise ValueError("Could not validate filename. File may be malformed.")

#     # Load data only if the initial check passes
#     df = pd.read_parquet(parquet_path)
#     time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)

#     if not time_col_name:
#         raise ValueError("Timestamp column not found in the dataset.")

#     df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')

#     if df[time_col_name].isnull().all():
#         return {"status": "Invalid", "detail": "Could not parse valid dates from the dataset's timestamp column."}

#     training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#     training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
#     testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
#     testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
#     simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
#     simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

#     if training_end > testing_start or testing_end > simulation_start:
#         return {"status": "Invalid", "detail": "Date ranges cannot overlap."}

#     training_mask = (df[time_col_name] >= training_start) & (df[time_col_name] < training_end)
#     testing_mask = (df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)
#     simulation_mask = (df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)

#     df['YYYY-MM'] = df[time_col_name].dt.to_period('M').astype(str)
#     monthly_counts = df['YYYY-MM'].value_counts().sort_index().to_dict()

#     return {
#         "status": "Valid",
#         "training": {"count": int(df.loc[training_mask].shape[0])},
#         "testing": {"count": int(df.loc[testing_mask].shape[0])},
#         "simulation": {"count": int(df.loc[simulation_mask].shape[0])},
#         "monthlyCounts": monthly_counts
#     }