# # app/services/simulation_service.py

# import asyncio
# import os
# import logging
# import pandas as pd
# import xgboost as xgb
# from datetime import datetime
# from app.redis_client import get_dataframe_from_json

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

# async def run_simulation_for_websocket(user_id: str, dataset_id: str):
#     """
#     An async generator that loads a model and data, yielding one prediction per second.
#     """
#     simulation_id = f"{user_id}:{dataset_id}"
#     log.info(f"[{simulation_id}] Initializing simulation stream.")

#     model_path = f"models/model_{user_id}_{dataset_id}.ubj"
#     if not os.path.exists(model_path):
#         yield {"error": "Model not found. Please train the model first."}
#         return

#     model = xgb.XGBClassifier()
#     model.load_model(model_path)

#     sim_key = f"user:{user_id}:simulation_data:{dataset_id}"
#     sim_df = get_dataframe_from_json(sim_key)

#     if sim_df is None:
#         yield {"error": "Simulation data not found. Please re-run the training process."}
#         return

#     try:
#         final_features = model.feature_names_in_
#     except AttributeError:
#         final_features = [col for col in sim_df.columns if col not in ['Id', 'synthetic_timestamp', 'Response']]

#     # --- Start of Changes ---
#     # 1. Before the loop, find a potential ID column.
#     id_col_name = next((col for col in sim_df.columns if col.lower() == 'id'), None)

#     if id_col_name:
#         print(f"Using column '{id_col_name}' as the row identifier.")
#     else:
#         print("No 'Id' column found. Falling back to using the DataFrame index as the identifier.")

#     # --- End of Changes ---

#     for index, row in sim_df.iterrows():
#         try:
#             # 2. Determine which identifier to use for the current row.
#             row_identifier = int(row[id_col_name]) if id_col_name else int(index)

#             X_pred = pd.DataFrame([row[final_features]], columns=final_features)
#             prediction_proba = model.predict_proba(X_pred)[0]
#             prediction = int(prediction_proba.argmax())
#             confidence = float(prediction_proba[prediction])
#             actual_response = int(row['Response'])

#             yield {
#                 # 3. Use the determined identifier in the output.
#                 "rowIndex": row_identifier,
#                 "prediction": prediction,
#                 "confidence": round(confidence, 4),
#                 "actual": actual_response,
#                 "isCorrect": prediction == actual_response,
#                 "timestamp": row.get('synthetic_timestamp', str(datetime.now()))
#             }

#             await asyncio.sleep(1)

#         except Exception as row_error:
#             log.error(f"[{simulation_id}] Error processing row {index}: {row_error}")
#             yield {"error": f"Error on row {index}: {row_error}"}

# app/services/simulation_service.py

import asyncio
import os
import logging
import pandas as pd
import xgboost as xgb
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

async def run_simulation_for_websocket(user_id: str, dataset_id: str):
    """
    An async generator that loads a model and data, yielding one prediction per second.
    """
    simulation_id = f"{user_id}:{dataset_id}"
    log.info(f"[{simulation_id}] Initializing simulation stream.")

    model_path = f"models/model_{user_id}_{dataset_id}.ubj"
    if not os.path.exists(model_path):
        yield {"error": "Model not found. Please train the model first."}
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # UPDATED: Load from data_store directory instead of Redis
    data_store_dir = "data_store"
    simulation_filename = f"simulation_data_{user_id}_{dataset_id}.parquet"
    simulation_path = os.path.join(data_store_dir, simulation_filename)
    
    if not os.path.exists(simulation_path):
        yield {"error": "Simulation data not found. Please re-run the training process."}
        return
    
    # Load simulation data from Parquet file
    try:
        sim_df = pd.read_parquet(simulation_path)
        log.info(f"[{simulation_id}] Loaded simulation data from: {simulation_path}")
    except Exception as e:
        yield {"error": f"Error loading simulation data: {str(e)}"}
        return

    try:
        final_features = model.feature_names_in_
    except AttributeError:
        final_features = [col for col in sim_df.columns if col not in ['Id', 'synthetic_timestamp', 'Response']]

    # Find a potential ID column
    id_col_name = next((col for col in sim_df.columns if col.lower() == 'id'), None)

    if id_col_name:
        log.info(f"Using column '{id_col_name}' as the row identifier.")
    else:
        log.info("No 'Id' column found. Falling back to using the DataFrame index as the identifier.")

    for index, row in sim_df.iterrows():
        try:
            # Determine which identifier to use for the current row
            row_identifier = int(row[id_col_name]) if id_col_name else int(index)

            X_pred = pd.DataFrame([row[final_features]], columns=final_features)
            prediction_proba = model.predict_proba(X_pred)[0]
            prediction = int(prediction_proba.argmax())
            confidence = float(prediction_proba[prediction])
            actual_response = int(row['Response'])

            timestamp_value = row.get('synthetic_timestamp')
            if timestamp_value is not None:
                # Convert pandas Timestamp to ISO format string
                timestamp_str = pd.to_datetime(timestamp_value).isoformat()
            else:
                timestamp_str = datetime.now().isoformat()

            yield {
                "rowIndex": row_identifier,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "actual": actual_response,
                "isCorrect": prediction == actual_response,
                "timestamp": timestamp_str
            }

            await asyncio.sleep(1)

        except Exception as row_error:
            log.error(f"[{simulation_id}] Error processing row {index}: {row_error}")
            yield {"error": f"Error on row {index}: {row_error}"}
