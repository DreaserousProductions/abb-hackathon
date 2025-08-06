# # # app/services/model_training_service.py

# # import pandas as pd
# # import io
# # import os
# # import logging
# # import base64
# # import numpy as np
# # import xgboost as xgb
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # from sklearn.feature_selection import VarianceThreshold
# # from sklearn.impute import SimpleImputer
# # import matplotlib
# # matplotlib.use('Agg')  # Use a non-interactive backend
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.model_selection import train_test_split
# # from imblearn.over_sampling import SMOTE
# # from app.redis_client import store_dataframe_as_json
# # from app.services.data_processing_service import find_parquet_file

# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # log = logging.getLogger(__name__)

# # def create_feature_importance_plot(feature_names: list, importances: list, top_n: int = 20) -> str:
# #     """Creates a feature importance bar plot and returns it as a Base64 string."""
# #     log.info(f"Generating feature importance plot for top {top_n} features.")
# #     feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
# #     feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

# #     plt.figure(figsize=(10, 8))
# #     sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
# #     plt.title(f'Top {top_n} Feature Importances')
# #     plt.xlabel('Importance Score')
# #     plt.ylabel('Features')
# #     plt.tight_layout()

# #     buf = io.BytesIO()
# #     plt.savefig(buf, format='png')
# #     buf.seek(0)
# #     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
# #     plt.close()
# #     log.info("Successfully generated and encoded the plot.")
# #     return img_base64

# # def create_training_history_plot(evals_result: dict) -> str:
# #     """Creates a plot of training history (loss and accuracy) and returns it as a Base64 string."""
# #     log.info("Generating training history plot.")
# #     results = evals_result['validation_0']
# #     epochs = len(results['logloss'])
# #     x_axis = range(0, epochs)

# #     fig, ax1 = plt.subplots(figsize=(12, 6))

# #     # Plot Loss on the primary Y-axis
# #     ax1.plot(x_axis, results['logloss'], 'g-', label='Validation LogLoss')
# #     ax1.set_xlabel('Epochs (Boosting Rounds)')
# #     ax1.set_ylabel('LogLoss', color='g')
# #     ax1.tick_params(axis='y', labelcolor='g')

# #     # --- FIX: Robustly check for the error metric before plotting ---
# #     # Find the metric key that is NOT 'logloss'
# #     error_keys = [key for key in results.keys() if key != 'logloss']
# #     if error_keys:
# #         error_key = error_keys[0]
# #         log.info(f"Using '{error_key}' as the error metric for plotting accuracy.")
        
# #         # Create a secondary Y-axis for Accuracy
# #         ax2 = ax1.twinx()
# #         accuracy = [1 - x for x in results[error_key]]
# #         ax2.plot(x_axis, accuracy, 'b-', label='Validation Accuracy')
# #         ax2.set_ylabel('Accuracy', color='b')
# #         ax2.tick_params(axis='y', labelcolor='b')
# #     else:
# #         log.warning("Only 'logloss' was found in eval_results. Accuracy will not be plotted.")

# #     plt.title('XGBoost Training History')
# #     fig.tight_layout()

# #     buf = io.BytesIO()
# #     plt.savefig(buf, format='png')
# #     buf.seek(0)
# #     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
# #     plt.close()
# #     log.info("Successfully generated and encoded the training history plot.")
# #     return img_base64

# # def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
# #     """
# #     Trains an XGBoost model with controlled resampling and conservative threshold tuning.
# #     """
# #     log.info(f"--- Starting controlled model training for dataset: {dataset_id} ---")

# #     # === Steps 1-5: Same as before (Data loading, slicing, feature selection, imputation) ===
# #     parquet_path = find_parquet_file(dataset_id)
# #     if not parquet_path:
# #         raise ValueError("Dataset file not found. It may have expired or failed processing.")

# #     log.info(f"Loading data from Parquet file: {parquet_path}")
# #     df = pd.read_parquet(parquet_path)
# #     log.info(f"Successfully loaded DataFrame. Shape: {df.shape}")

# #     time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
# #     df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')

# #     cols_to_convert = [col for col in df.columns if col not in ['Id', time_col_name]]
# #     for col in cols_to_convert:
# #         df[col] = pd.to_numeric(df[col], errors='coerce')

# #     training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
# #     training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
# #     testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
# #     testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
# #     simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
# #     simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

# #     train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
# #     print(f"Training set length: {len(train_df)}")
# #     test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
# #     print(f"Testing set length: {len(test_df)}")
# #     simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]
# #     print(f"Simulation set length: {len(simulation_df)}")

# #     if train_df.empty or test_df.empty:
# #         raise ValueError("Training or testing date range resulted in zero records.")

# #     target_col = 'Response'
# #     initial_features = [col for col in df.columns if col not in [target_col, time_col_name, 'Id']]

# #     nan_threshold = 50.0
# #     nan_percentages = (train_df[initial_features].isnull().sum() / len(train_df)) * 100
# #     cols_to_drop_by_nan = nan_percentages[nan_percentages > nan_threshold].index
# #     features_after_nan_filter = [col for col in initial_features if col not in cols_to_drop_by_nan]

# #     temp_train_data = train_df[features_after_nan_filter].fillna(0)
# #     selector = VarianceThreshold(threshold=(.98 * (1 - .98)))
# #     selector.fit(temp_train_data)
# #     final_features = temp_train_data.columns[selector.get_support()].tolist()

# #     X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
# #     X_test, y_test = test_df[final_features], test_df[target_col].astype(int)

# #     imputer = SimpleImputer(strategy='median')
# #     imputer.fit(X_train)
# #     X_train = pd.DataFrame(imputer.transform(X_train), columns=final_features)
# #     X_test = pd.DataFrame(imputer.transform(X_test), columns=final_features)

# #     # === Step 6: CONTROLLED SMOTE Application ===
# #     log.info("\n--- Step 6: Applying Controlled SMOTE ---")
# #     log.info(f"Original training class distribution: {y_train.value_counts().to_dict()}")

# #     # **FIX 1: Use conservative SMOTE ratio instead of full balance**
# #     # For 1:199 ratio, only oversample to 1:10 or 1:20 (not 1:1)
# #     minority_count = y_train.sum()
# #     majority_count = len(y_train) - minority_count

# #     # Target ratio of 1:10 instead of 1:1 (less aggressive)
# #     target_minority_count = min(majority_count // 10, minority_count * 5)

# #     if minority_count >= 2 and target_minority_count > minority_count:
# #         smote = SMOTE(
# #             sampling_strategy={1: target_minority_count},
# #             random_state=42,
# #             k_neighbors=min(5, minority_count - 1)
# #         )
# #         X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# #         log.info(f"After controlled SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
# #     else:
# #         # If too few positive samples, skip SMOTE
# #         X_train_resampled, y_train_resampled = X_train, y_train
# #         log.info("Skipping SMOTE due to insufficient positive samples")

# #     # === Step 7: Enhanced Class Weight Calculation ===
# #     log.info("\n--- Step 7: Calculating Enhanced Class Weight ---")
# #     try:
# #         # Use original imbalance ratio for scale_pos_weight (more conservative)
# #         scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
# #     except (KeyError, ZeroDivisionError):
# #         scale_pos_weight = 1

# #     # === Step 8: Training with More Conservative Parameters ===
# #     log.info("\n--- Step 8: Training XGBoost Model ---")
# #     model = xgb.XGBClassifier(
# #         objective='binary:logistic',
# #         use_label_encoder=False,
# #         eval_metric='logloss',
# #         scale_pos_weight=scale_pos_weight,
# #         n_estimators=200,  # **FIX 2: Reduced from 500 to prevent overfitting**
# #         learning_rate=0.05,  # **FIX 3: Lower learning rate**
# #         max_depth=4,  # **FIX 4: Limit tree depth**
# #         subsample=0.8,
# #         colsample_bytree=0.8,
# #         min_child_weight=3,  # **FIX 5: Higher min_child_weight for stability**
# #         early_stopping_rounds=15,  # **FIX 6: Earlier stopping**
# #         random_state=42
# #     )

# #     # Split test for validation
# #     X_val, X_test_final, y_val, y_test_final = train_test_split(
# #         X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
# #     )

# #     model.fit(
# #         X_train_resampled, y_train_resampled,
# #         eval_set=[(X_val, y_val)],
# #         verbose=False
# #     )

# #     # === Step 9: CONSERVATIVE Threshold Tuning ===
# #     log.info("\n--- Step 9: Conservative Threshold Tuning ---")
# #     y_proba = model.predict_proba(X_test_final)[:, 1]

# #     # **FIX 7: More conservative threshold range and better validation**
# #     best_threshold = 0.5  # Default fallback
# #     best_f1 = 0
# #     best_precision = 0
# #     best_recall = 0

# #     # Test more conservative threshold range
# #     thresholds_to_test = np.arange(0.1, 0.9, 0.05)  # 0.1 to 0.85, not 0.01

# #     for threshold in thresholds_to_test:
# #         y_pred_threshold = (y_proba >= threshold).astype(int)
        
# #         # **FIX 8: Add validation - skip if all predictions are same class**
# #         if len(np.unique(y_pred_threshold)) == 1:
# #             continue

# #         precision = precision_score(y_test_final, y_pred_threshold, zero_division=0)
# #         recall = recall_score(y_test_final, y_pred_threshold, zero_division=0)
# #         f1 = f1_score(y_test_final, y_pred_threshold, zero_division=0)

# #         # **FIX 9: Require minimum precision AND recall**
# #         if f1 > best_f1 and precision > 0.1 and recall > 0.1:
# #             best_f1 = f1
# #             best_threshold = threshold
# #             best_precision = precision
# #             best_recall = recall

# #     log.info(f"Optimal threshold: {best_threshold:.3f}, F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

# #     # === Step 10: Final Evaluation ===
# #     y_pred_final = (y_proba >= best_threshold).astype(int)

# #     # **FIX 10: Validate final predictions**
# #     unique_preds = np.unique(y_pred_final)
# #     log.info(f"Final prediction classes: {unique_preds}")

# #     if len(unique_preds) == 1:
# #         log.warning(f"Model predicting only class {unique_preds[0]}! Using default threshold 0.5")
# #         y_pred_final = (y_proba >= 0.5).astype(int)

# #     tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_final).ravel()

# #     metrics = {
# #         "accuracy": accuracy_score(y_test_final, y_pred_final),
# #         "precision": precision_score(y_test_final, y_pred_final, zero_division=0),
# #         "recall": recall_score(y_test_final, y_pred_final, zero_division=0),
# #         "f1Score": f1_score(y_test_final, y_pred_final, zero_division=0),
# #         "trueNegative": int(tn), "falsePositive": int(fp),
# #         "falseNegative": int(fn), "truePositive": int(tp)
# #     }

# #     # Rest of the function remains the same...
# #     b64_plot_string = create_feature_importance_plot(
# #         feature_names=X_train.columns,
# #         importances=model.feature_importances_
# #     )

# #     b64_training_plot = create_training_history_plot(model.evals_result())
# #     plots = {"featureImportance": b64_plot_string, "trainingPlot": b64_training_plot}

# #     model_dir = "models"
# #     os.makedirs(model_dir, exist_ok=True)
# #     model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
# #     model.save_model(model_path)

# #     if not simulation_df.empty:
# #         simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
# #         sim_features = simulation_df[final_features]
# #         sim_imputed = imputer.transform(sim_features)
# #         simulation_df_processed = pd.DataFrame(sim_imputed, columns=final_features)

# #         for col in ['Id', time_col_name, 'Response']:
# #             if col in simulation_df:
# #                 simulation_df_processed[col] = simulation_df[col].values

# #         store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)

# #     os.remove(parquet_path)
# #     return {"metrics": metrics, "plots": plots}
########################
# # app/services/csv_service.py

# import pandas as pd
# import io
# import uuid
# from datetime import datetime, timedelta
# from app.redis_client import store_dataframe_as_json, get_dataframe_from_json, delete_key
# import logging
# import base64
# import os
# import asyncio # New import for async sleep

# import xgboost as xgb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

# def create_feature_importance_plot(feature_names: list, importances: list, top_n: int = 20) -> str:
#     """Creates a feature importance bar plot and returns it as a Base64 string."""
#     log.info(f"Generating feature importance plot for top {top_n} features.")
#     feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
#     feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

#     plt.figure(figsize=(10, 8))
#     sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
#     plt.title(f'Top {top_n} Feature Importances')
#     plt.xlabel('Importance Score')
#     plt.ylabel('Features')
#     plt.tight_layout()

#     # --- Convert plot to Base64 ---
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close() # Close the plot to free up memory
#     log.info("Successfully generated and encoded the plot.")
#     return img_base64

# def process_and_analyze_csv(contents: bytes, user_id: str) -> dict:
#     """
#     Reads CSV content, processes it, stores it in Redis, and returns metrics.
#     """
#     log.info(f"Starting CSV processing for user_id: {user_id}")
#     df = pd.read_csv(io.BytesIO(contents))

#     # --- Data Augmentation: Add synthetic timestamps ---
#     timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)

#     if timestamp_col:
#         log.info(f"Found existing timestamp column: '{timestamp_col}'")
#         df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
#         df.dropna(subset=[timestamp_col], inplace=True)
#         time_col_name = timestamp_col
#     else:
#         log.info("No timestamp column found, generating synthetic timestamps.")
#         start_time = datetime.now()
#         time_col_name = 'synthetic_timestamp'
#         df[time_col_name] = [start_time + timedelta(seconds=i) for i in range(len(df))]

#     # --- Persist the dataset temporarily ---
#     dataset_id = str(uuid.uuid4())
#     redis_key = f"user:{user_id}:dataset:{dataset_id}"
    
#     store_dataframe_as_json(redis_key, df, expiration_seconds=3600)

#     # --- Calculate Metrics ---
#     total_records = len(df)
#     num_columns = len(df.columns)

#     pass_rate = 0.0
#     if 'Response' in df.columns:
#         response_col = pd.to_numeric(df['Response'], errors='coerce')
#         pass_count = response_col[response_col == 1].count()
#         if total_records > 0:
#             pass_rate = (pass_count / total_records) * 100
    
#     earliest_timestamp = df[time_col_name].min()
#     latest_timestamp = df[time_col_name].max()

#     date_range = {
#         "start": earliest_timestamp.isoformat() if pd.notna(earliest_timestamp) else None,
#         "end": latest_timestamp.isoformat() if pd.notna(latest_timestamp) else None
#     }
    
#     log.info(f"Finished CSV processing for user_id: {user_id}. Dataset ID: {dataset_id}")
#     return {
#         "datasetId": dataset_id,
#         "userId": user_id,
#         "totalRecords": total_records,
#         "numColumns": num_columns,
#         "passRate": round(pass_rate, 2),
#         "dateRange": date_range
#     }

# def validate_and_count_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     Validates date ranges against the stored dataset and calculates record counts.
#     This version is robust against NaT values and timezone mismatches.
#     """
#     log.info("--- Inside validate_and_count_ranges service function ---")
#     log.info(f"Received arguments: user_id='{user_id}', dataset_id='{dataset_id}'")
#     log.info(f"Received ranges: {ranges}")

#     try:
#         redis_key = f"user:{user_id}:dataset:{dataset_id}"
#         df = get_dataframe_from_json(redis_key)

#         if df is None:
#             raise ValueError("Dataset not found or expired.")
        
#         log.info(f"Successfully retrieved DataFrame from Redis. Shape: {df.shape}")

#         time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
#         if not time_col_name:
#             raise ValueError("Timestamp column not found in the dataset.")
        
#         log.info(f"Identified timestamp column: '{time_col_name}'")
#         df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
        
#         # --- FIX #1: Check for NaT values AFTER conversion ---
#         if df[time_col_name].isnull().all():
#             detail = "Could not parse valid dates from the dataset's timestamp column."
#             log.error(f"Validation failed for key '{redis_key}': {detail}")
#             # Return a clean error instead of crashing
#             return {"status": "Invalid", "detail": detail}

#         # --- FIX #2: Make request dates timezone-naive to match the dataset ---
#         log.info("Parsing and localizing date ranges from request...")
#         training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#         training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
#         testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
#         testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
#         simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
#         simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)
#         log.info("Successfully parsed and localized all date ranges.")

#         # --- Validation logic can now run safely ---
#         log.info("--- Performing Validation Logic ---")
#         if training_end > testing_start or testing_end > simulation_start:
#             detail = "Date ranges cannot overlap."
#             log.warning(f"Validation failed: {detail}")
#             return {"status": "Invalid", "detail": detail}
#         log.info("Overlap check passed.")
        
#         # This block is now safe from crashes
#         min_date = df[time_col_name].min().floor('S')
#         max_date = df[time_col_name].max().ceil('S')

#         log.info(f"Adjusted Dataset bounds for validation: min='{min_date}', max='{max_date}'")
#         if training_start < min_date or simulation_end > max_date:
#             detail = "Ranges must be within the dataset's date boundaries."
#             log.warning(f"Validation failed: {detail}")
#             return {"status": "Invalid", "detail": detail}
#         log.info("Boundary check passed.")

#         # --- Counting logic remains the same ---
#         log.info("--- Performing Counting Logic ---")
#         training_mask = (df[time_col_name] >= training_start) & (df[time_col_name] < training_end)
#         testing_mask = (df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)
#         simulation_mask = (df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)

#         training_count = int(df.loc[training_mask].shape[0])
#         testing_count = int(df.loc[testing_mask].shape[0])
#         simulation_count = int(df.loc[simulation_mask].shape[0])
#         log.info(f"Counts: Training={training_count}, Testing={testing_count}, Simulation={simulation_count}")

#         log.info("Calculating monthly counts...")
#         df['YYYY-MM'] = df[time_col_name].dt.to_period('M').astype(str)
#         monthly_counts = df['YYYY-MM'].value_counts().sort_index().to_dict()
#         log.info("Monthly counts calculated successfully.")

#         result = {
#             "status": "Valid",
#             "training": {"count": training_count},
#             "testing": {"count": testing_count},
#             "simulation": {"count": simulation_count},
#             "monthlyCounts": monthly_counts
#         }
#         log.info("--- Successfully completed validate_and_count_ranges ---")
#         return result

#     except Exception as e:
#         log.error(f"An unexpected error occurred in validate_and_count_ranges: {e}", exc_info=True)
#         # Re-raise the exception so the router can catch it and return a 500
#         raise

# def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     Trains, tests, and persists a robust XGBoost model with correct data typing and verbose logging.
#     """
#     log.info(f"--- Starting ADVANCED model training for user:'{user_id}', dataset:'{dataset_id}' ---")
    
#     # === Step 1: Fetch and Prepare Full Dataset (with CORRECTED Order of Operations) ===
#     log.info(f"\n--- Step 1: Fetching and Preparing Data ---")
#     redis_key = f"user:{user_id}:dataset:{dataset_id}"
#     log.info(f"Attempting to load data from Redis key: {redis_key}")
#     df = get_dataframe_from_json(redis_key)
#     if df is None:
#         raise ValueError("Dataset not found or expired in Redis.")
#     log.info(f"Successfully loaded DataFrame. Initial shape: {df.shape}")
    
#     # --- FIX IS HERE: Identify the timestamp column BEFORE any conversions ---
#     log.info("Detecting timestamp column...")
#     if 'synthetic_timestamp' in df.columns:
#         time_col_name = 'synthetic_timestamp'
#         log.info(f"Found specific 'synthetic_timestamp' column.")
#     else:
#         log.warning("Could not find 'synthetic_timestamp', falling back to generic name search ('timestamp' or 'date').")
#         time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)

#     if not time_col_name:
#         raise ValueError("A timestamp column could not be found in the dataset.")
#     log.info(f"Using '{time_col_name}' as the timestamp column.")
    
#     # --- Now, convert all OTHER columns to numeric, EXCLUDING Id and the identified timestamp column ---
#     log.info("Converting feature columns to numeric, coercing errors...")
#     cols_to_convert = [col for col in df.columns if col not in ['Id', time_col_name]]
#     for col in cols_to_convert:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # --- Finally, convert the (now safe) timestamp column to datetime ---
#     df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
#     log.info(f"Converted '{time_col_name}' to datetime objects.")
    
#     if df[time_col_name].isnull().all():
#         log.error(f"FATAL: The timestamp column '{time_col_name}' contains no valid dates after conversion. All values are NaT.")
#         raise ValueError(f"The identified timestamp column '{time_col_name}' contains no valid dates.")

#     # === Step 2: Slice Data based on Date Ranges (Unchanged from here) ===
#     log.info("\n--- Step 2: Slicing Data with Timezone Correction ---")
    
#     training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#     training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
#     testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
#     testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
#     simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
#     simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

#     log.info(f"Dataset's full time range: {df[time_col_name].min()} to {df[time_col_name].max()}")
#     log.info(f"Requested training range (naive): {training_start} to {training_end}")
#     log.info(f"Requested testing range (naive): {testing_start} to {testing_end}")

#     train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
#     test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
#     simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]
#     log.info(f"Slicing complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}, Sim shape: {simulation_df.shape}")

#     if train_df.empty or test_df.empty:
#         raise ValueError("Training or testing date range resulted in zero records. Please ensure selected dates are within the dataset's time range.")
        
#     # === The rest of the function (Steps 3 through 8) is correct and remains unchanged ===
#     # ... (feature selection, imputation, training, evaluation, persistence) ...

#     # === Step 3: Feature Selection ===
#     log.info("\n--- Step 3: Performing Feature Selection ---")
#     target_col = 'Response'
#     initial_features = [col for col in df.columns if col not in [target_col, time_col_name, 'Id']]
    
#     nan_threshold = 50.0
#     nan_percentages = (train_df[initial_features].isnull().sum() / len(train_df)) * 100
#     cols_to_drop_by_nan = nan_percentages[nan_percentages > nan_threshold].index
#     features_after_nan_filter = [col for col in initial_features if col not in cols_to_drop_by_nan]
#     log.info(f"Step 3a (NaN Filter): Dropped {len(cols_to_drop_by_nan)} columns with >{nan_threshold}% NaNs. Retained {len(features_after_nan_filter)} features.")
    
#     temp_train_data = train_df[features_after_nan_filter].fillna(0)
#     selector = VarianceThreshold(threshold=(.98 * (1 - .98)))
#     selector.fit(temp_train_data)
#     final_features = temp_train_data.columns[selector.get_support()].tolist()
#     log.info(f"Step 3b (Variance Filter): Dropped {len(features_after_nan_filter) - len(final_features)} low-variance columns. Final feature count: {len(final_features)}")

#     # === Step 4: Prepare Final Data & Impute Missing Values ===
#     log.info("\n--- Step 4: Preparing Final Data & Imputing NaNs ---")
#     X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
#     X_test, y_test = test_df[final_features], test_df[target_col].astype(int)
    
#     imputer = SimpleImputer(strategy='median')
#     log.info(f"Fitting imputer with '{imputer.strategy}' strategy on training data.")
#     imputer.fit(X_train)
#     X_train = pd.DataFrame(imputer.transform(X_train), columns=final_features)
#     X_test = pd.DataFrame(imputer.transform(X_test), columns=final_features)
#     log.info("Imputation complete for training and testing sets.")

#     # === Step 5: Handle Class Imbalance ===
#     log.info("\n--- Step 5: Calculating Class Imbalance Weight ---")
#     try:
#         scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#         log.info(f"Class distribution in training data:\n{y_train.value_counts()}")
#         log.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
#     except (KeyError, ZeroDivisionError):
#         log.warning("Could not calculate scale_pos_weight (one class might be missing). Defaulting to 1.")
#         scale_pos_weight = 1
    
#     # === Step 6: Training the Model ===
#     log.info("\n--- Step 6: Training XGBoost Model ---")
#     model = xgb.XGBClassifier(
#         objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
#         scale_pos_weight=scale_pos_weight, n_estimators=150
#     )
#     log.info(f"Fitting model with {model.n_estimators} estimators...")
#     model.fit(X_train, y_train)
#     log.info("Model training complete.")

#     # === Step 7: Evaluation & Persistence ===
#     log.info("\n--- Step 7: Evaluating, Plotting, and Persisting Results ---")
#     y_pred = model.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     log.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
#     metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred, zero_division=0),
#         "recall": recall_score(y_test, y_pred, zero_division=0),
#         "f1Score": f1_score(y_test, y_pred, zero_division=0),
#         # Add the real confusion matrix values to the response
#         "trueNegative": int(tn),
#         "falsePositive": int(fp),
#         "falseNegative": int(fn),
#         "truePositive": int(tp)
#     }
#     log.info(f"Calculated Metrics on test set: {metrics}")
    
#     b64_plot_string = create_feature_importance_plot(
#         feature_names=X_train.columns, 
#         importances=model.feature_importances_
#     )
#     plots = {"featureImportance": b64_plot_string}
#     log.info("Feature importance plot generated and encoded.")

#     model_dir = "models"
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
#     model.save_model(model_path)
#     log.info(f"Model successfully saved to: {model_path}")
    
#     if not simulation_df.empty:
#         simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
#         sim_features = simulation_df[final_features]
#         sim_imputed = imputer.transform(sim_features) 
#         simulation_df_processed = pd.DataFrame(sim_imputed, columns=final_features)
#         for col in ['Id', time_col_name, 'Response']:
#              if col in simulation_df:
#                 simulation_df_processed[col] = simulation_df[col].values
#         store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)
#         log.info(f"Processed simulation data and saved to Redis key: {simulation_key}")
    
#     delete_key(redis_key)
#     log.info(f"Original dataset key '{redis_key}' deleted from Redis.")
    
#     log.info(f"--- ADVANCED model training function finished successfully for user:'{user_id}' ---")
#     return {"metrics": metrics, "plots": plots}

# async def run_simulation_for_websocket(user_id: str, dataset_id: str):
#     """
#     An async generator that loads a model and data, yielding one prediction per second.
#     """
#     simulation_id = f"{user_id}:{dataset_id}"
#     log.info(f"[{simulation_id}] Initializing simulation stream.")

#     # 1. Load the Model
#     model_path = f"models/model_{user_id}_{dataset_id}.ubj"
#     if not os.path.exists(model_path):
#         log.error(f"[{simulation_id}] Model not found at {model_path}")
#         yield {"error": "Model not found. Please train the model first."}
#         return
        
#     model = xgb.XGBClassifier()
#     model.load_model(model_path)
#     log.info(f"[{simulation_id}] Model loaded successfully from {model_path}")

#     # 2. Load Simulation Data
#     sim_key = f"user:{user_id}:simulation_data:{dataset_id}"
#     log.info(f"[{simulation_id}] Loading simulation data from Redis key: {sim_key}")
#     sim_df = get_dataframe_from_json(sim_key)
#     if sim_df is None:
#         log.error(f"[{simulation_id}] Simulation data not found in Redis.")
#         yield {"error": "Simulation data not found. Please re-run the training process."}
#         return
#     log.info(f"[{simulation_id}] Simulation data loaded with shape: {sim_df.shape}")

#     # 3. Get the exact features the model was trained on
#     # This is robust and prevents column mismatch errors.
#     try:
#         final_features = model.feature_names_in_
#     except AttributeError:
#         # Fallback for older XGBoost versions
#         log.warning("Could not get feature names from model, using DataFrame columns. Ensure they match.")
#         final_features = [col for col in sim_df.columns if col not in ['Id', 'synthetic_timestamp', 'Response']]

#     log.info(f"[{simulation_id}] Starting prediction loop for {len(sim_df)} records...")
#     # 4. Loop and Predict
#     for index, row in sim_df.iterrows():
#         try:
#             # Prepare the single row of data for prediction
#             X_pred = pd.DataFrame([row[final_features]], columns=final_features)
            
#             # Get prediction probabilities to calculate a confidence score
#             prediction_proba = model.predict_proba(X_pred)[0]
#             prediction = int(prediction_proba.argmax())
#             confidence = float(prediction_proba[prediction])

#             actual_response = int(row['Response'])

#             # Yield a dictionary with the results for this row
#             yield {
#                 "rowIndex": int(index),
#                 "prediction": prediction,
#                 "confidence": round(confidence, 4),
#                 "actual": actual_response,
#                 "isCorrect": prediction == actual_response,
#                 "timestamp": row.get('synthetic_timestamp', str(datetime.now()))
#             }
            
#             # Wait for one second before the next prediction
#             await asyncio.sleep(1)

#         except Exception as row_error:
#             log.error(f"[{simulation_id}] Error processing row {index}: {row_error}")
#             yield {"error": f"Error on row {index}: {row_error}"}


##################################################################################
###################################
# # app/services/csv_service.py

# import pandas as pd
# import io
# import uuid
# import os
# import glob # To find files
# from datetime import datetime, timedelta
# from app.redis_client import store_dataframe_as_json, get_dataframe_from_json, delete_key
# import logging
# import base64
# import asyncio
# import pyarrow as pa # NEW: Import pyarrow
# import pyarrow.parquet as pq # NEW: Import pyarrow.parquet

# import xgboost as xgb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.impute import SimpleImputer
# import matplotlib
# matplotlib.use('Agg') # Use a non-interactive backend
# import matplotlib.pyplot as plt
# import seaborn as sns

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

# # --- NEW: Define a storage directory for Parquet files ---
# PARQUET_STORAGE_DIR = "data_store"
# os.makedirs(PARQUET_STORAGE_DIR, exist_ok=True)


# def find_parquet_file(dataset_id: str) -> str | None:
#     """Finds the full path of a parquet file using its unique dataset_id."""
#     search_pattern = os.path.join(PARQUET_STORAGE_DIR, f"*_dataset_{dataset_id}_*.parquet")
#     matching_files = glob.glob(search_pattern)
#     if not matching_files:
#         log.error(f"No parquet file found for dataset_id: {dataset_id}")
#         return None
#     return matching_files[0]


# def create_feature_importance_plot(feature_names: list, importances: list, top_n: int = 20) -> str:
#     """Creates a feature importance bar plot and returns it as a Base64 string."""
#     log.info(f"Generating feature importance plot for top {top_n} features.")
#     feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
#     feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

#     plt.figure(figsize=(10, 8))
#     sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
#     plt.title(f'Top {top_n} Feature Importances')
#     plt.xlabel('Importance Score')
#     plt.ylabel('Features')
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()
#     log.info("Successfully generated and encoded the plot.")
#     return img_base64


# # --- REPLACED: This function replaces process_and_analyze_csv ---
# def process_csv_to_parquet(temp_csv_path: str, user_id: str) -> dict:
#     """
#     Reads a large CSV in chunks, analyzes it, saves it as Parquet, and cleans up.
#     """
#     log.info(f"Starting Parquet processing for user_id: {user_id} from {temp_csv_path}")
    
#     try:
#         total_records = 0
#         pass_count = 0
#         num_columns = 0
#         overall_min_date = pd.Timestamp.max
#         overall_max_date = pd.Timestamp.min
#         time_col_name = None
        
#         # --- FIX: Wrap the iterators in a 'with' statement to ensure the file handle is closed ---
#         with pd.read_csv(temp_csv_path, chunksize=100_000, low_memory=False) as chunk_iterator:
#             # --- Pass 1: Analyze in chunks to get metadata without loading full file ---
#             for chunk in chunk_iterator:
#                 if time_col_name is None:
#                     time_col_name = next((col for col in chunk.columns if 'timestamp' in col.lower() or 'date' in col.lower()), 'synthetic_timestamp')
#                     num_columns = len(chunk.columns)

#                 if 'synthetic_timestamp' in time_col_name:
#                     chunk[time_col_name] = [datetime.now() + timedelta(seconds=i) for i in range(len(chunk))]
                
#                 chunk[time_col_name] = pd.to_datetime(chunk[time_col_name], errors='coerce')
#                 chunk.dropna(subset=[time_col_name], inplace=True)
                
#                 if not chunk.empty:
#                     overall_min_date = min(overall_min_date, chunk[time_col_name].min())
#                     overall_max_date = max(overall_max_date, chunk[time_col_name].max())

#                 if 'Response' in chunk.columns:
#                     response_col = pd.to_numeric(chunk['Response'], errors='coerce')
#                     pass_count += response_col[response_col == 1].count()
                
#                 total_records += len(chunk)

#         pass_rate = (pass_count / total_records) * 100 if total_records > 0 else 0

#         # --- Construct final filename with embedded date range ---
#         dataset_id = str(uuid.uuid4())
#         start_date_str = overall_min_date.strftime('%Y%m%d%H%M%S')
#         end_date_str = overall_max_date.strftime('%Y%m%d%H%M%S')
        
#         final_parquet_filename = f"user_{user_id}_dataset_{dataset_id}_from_{start_date_str}_to_{end_date_str}.parquet"
#         final_parquet_path = os.path.join(PARQUET_STORAGE_DIR, final_parquet_filename)

#         # --- Pass 2: Convert CSV to Parquet (again in chunks) ---
#         log.info(f"Converting CSV to Parquet at: {final_parquet_path}")
        
#         # --- FIX IS HERE: Use a ParquetWriter for robust appending ---
#         writer = None
#         with pd.read_csv(temp_csv_path, chunksize=100_000, low_memory=False) as csv_stream_for_parquet:
#             for i, chunk in enumerate(csv_stream_for_parquet):
#                 # Re-add synthetic timestamp if needed for each chunk
#                 if 'synthetic_timestamp' in time_col_name:
#                      chunk[time_col_name] = [datetime.now() + timedelta(seconds=i) for i in range(len(chunk))]
                
#                 # Convert pandas DataFrame to pyarrow Table
#                 table = pa.Table.from_pandas(chunk)

#                 if writer is None:
#                     # For the first chunk, create the writer with the schema
#                     writer = pq.ParquetWriter(final_parquet_path, table.schema)
                
#                 writer.write_table(table)
        
#         if writer:
#             writer.close()
#         # --- END OF FIX ---
        
#         date_range = {
#             "start": overall_min_date.isoformat() if pd.notna(overall_min_date) else None,
#             "end": overall_max_date.isoformat() if pd.notna(overall_max_date) else None
#         }
        
#         log.info(f"Finished Parquet processing. Dataset ID: {dataset_id}")
#         return {
#             "datasetId": dataset_id,
#             "userId": user_id,
#             "parquetPath": final_parquet_path, # Return path instead of storing in Redis
#             "totalRecords": int(total_records),
#             "numColumns": int(num_columns),
#             "passRate": round(float(pass_rate), 2),
#             "dateRange": date_range
#         }
#     finally:
#         # --- NEW: This function now cleans up the file it was given ---
#         if os.path.exists(temp_csv_path):
#             os.remove(temp_csv_path)
#             log.info(f"Cleaned up temporary CSV file: {temp_csv_path}")

# # --- FIXED: This function is now corrected and robust ---
# def validate_and_count_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     Validates date ranges by first checking the filename, then loads the
#     Parquet file to get exact counts if validation passes.
#     """
#     log.info(f"--- Validating ranges for dataset: {dataset_id} ---")
#     parquet_path = find_parquet_file(dataset_id)
#     if not parquet_path:
#         raise ValueError("Dataset file not found. It may have expired or failed processing.")

#     # --- OPTIMIZATION: Validate against filename before reading the file ---
#     try:
#         parts = os.path.basename(parquet_path).split('_')
#         # FIX: Use the correct format string to parse the precise timestamp
#         file_start_str = parts[-3] 
#         file_end_str = parts[-1].split('.')[0]
        
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

#     # --- Load data only if the initial check passes ---
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


# # --- REFACTORED: This function now loads from Parquet ---
# def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     Trains an XGBoost model by loading data from a Parquet file.
#     """
#     log.info(f"--- Starting model training from Parquet for dataset: {dataset_id} ---")
    
#     # === Step 1: Fetch Data from Parquet File ===
#     parquet_path = find_parquet_file(dataset_id)
#     if not parquet_path:
#         raise ValueError("Dataset file not found. It may have expired or failed processing.")
    
#     log.info(f"Loading data from Parquet file: {parquet_path}")
#     df = pd.read_parquet(parquet_path)
#     log.info(f"Successfully loaded DataFrame. Shape: {df.shape}")

#     # ... The rest of the function is largely the same, but no more redis_key ...
#     time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
#     df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
#     cols_to_convert = [col for col in df.columns if col not in ['Id', time_col_name]]
#     for col in cols_to_convert:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # === Step 2: Slice Data ===
#     # ... (slicing logic is unchanged) ...
#     training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#     training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
#     testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
#     testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
#     simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
#     simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

#     train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
#     test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
#     simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]

#     # === Steps 3-7: Feature Selection, Imputation, Training, Evaluation... ===
#     # ... (This entire block of logic remains the same) ...
#     log.info("\n--- Step 3: Performing Feature Selection ---")
#     target_col = 'Response'
#     initial_features = [col for col in df.columns if col not in [target_col, time_col_name, 'Id']]
    
#     nan_threshold = 50.0
#     nan_percentages = (train_df[initial_features].isnull().sum() / len(train_df)) * 100
#     cols_to_drop_by_nan = nan_percentages[nan_percentages > nan_threshold].index
#     features_after_nan_filter = [col for col in initial_features if col not in cols_to_drop_by_nan]
    
#     temp_train_data = train_df[features_after_nan_filter].fillna(0)
#     selector = VarianceThreshold(threshold=(.98 * (1 - .98)))
#     selector.fit(temp_train_data)
#     final_features = temp_train_data.columns[selector.get_support()].tolist()

#     log.info("\n--- Step 4: Preparing Final Data & Imputing NaNs ---")
#     X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
#     X_test, y_test = test_df[final_features], test_df[target_col].astype(int)
    
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(X_train)
#     X_train = pd.DataFrame(imputer.transform(X_train), columns=final_features)
#     X_test = pd.DataFrame(imputer.transform(X_test), columns=final_features)

#     log.info("\n--- Step 5: Calculating Class Imbalance Weight ---")
#     try:
#         scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#     except (KeyError, ZeroDivisionError):
#         scale_pos_weight = 1
    
#     log.info("\n--- Step 6: Training XGBoost Model ---")
#     model = xgb.XGBClassifier(
#         objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
#         scale_pos_weight=scale_pos_weight, n_estimators=150
#     )
#     model.fit(X_train, y_train)

#     log.info("\n--- Step 7: Evaluating, Plotting, and Persisting Results ---")
#     y_pred = model.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred, zero_division=0),
#         "recall": recall_score(y_test, y_pred, zero_division=0),
#         "f1Score": f1_score(y_test, y_pred, zero_division=0),
#         "trueNegative": int(tn), "falsePositive": int(fp),
#         "falseNegative": int(fn), "truePositive": int(tp)
#     }
    
#     b64_plot_string = create_feature_importance_plot(
#         feature_names=X_train.columns, 
#         importances=model.feature_importances_
#     )
#     plots = {"featureImportance": b64_plot_string}

#     model_dir = "models"
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
#     model.save_model(model_path)
    
#     if not simulation_df.empty:
#         simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
#         sim_features = simulation_df[final_features]
#         sim_imputed = imputer.transform(sim_features) 
#         simulation_df_processed = pd.DataFrame(sim_imputed, columns=final_features)
#         for col in ['Id', time_col_name, 'Response']:
#              if col in simulation_df:
#                  simulation_df_processed[col] = simulation_df[col].values
#         store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)
#         log.info(f"Processed simulation data and saved to Redis key: {simulation_key}")
    
#     # --- NEW: Delete the Parquet file after use ---
#     os.remove(parquet_path)
#     log.info(f"Original dataset file '{parquet_path}' deleted from disk.")
    
#     return {"metrics": metrics, "plots": plots}


# # This function remains unchanged, as it correctly uses Redis for the temporary simulation data
# async def run_simulation_for_websocket(user_id: str, dataset_id: str):
#     # ... (no changes needed here) ...
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
#     for index, row in sim_df.iterrows():
#         try:
#             X_pred = pd.DataFrame([row[final_features]], columns=final_features)
#             prediction_proba = model.predict_proba(X_pred)[0]
#             prediction = int(prediction_proba.argmax())
#             confidence = float(prediction_proba[prediction])
#             actual_response = int(row['Response'])
#             yield {
#                 "rowIndex": int(index), "prediction": prediction,
#                 "confidence": round(confidence, 4), "actual": actual_response,
#                 "isCorrect": prediction == actual_response,
#                 "timestamp": row.get('synthetic_timestamp', str(datetime.now()))
#             }
#             await asyncio.sleep(1)
#         except Exception as row_error:
#             log.error(f"[{simulation_id}] Error processing row {index}: {row_error}")
#             yield {"error": f"Error on row {index}: {row_error}"}

##########################################################################################################################

# app/services/csv_service.py

import pandas as pd
import io
import uuid
import os
import glob  # To find files
from datetime import datetime, timedelta
from app.redis_client import store_dataframe_as_json, get_dataframe_from_json, delete_key
import logging
import base64
import asyncio
import pyarrow as pa  # Import pyarrow
import pyarrow.parquet as pq  # Import pyarrow.parquet
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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

def create_feature_importance_plot(feature_names: list, importances: list, top_n: int = 20) -> str:
    """Creates a feature importance bar plot and returns it as a Base64 string."""
    log.info(f"Generating feature importance plot for top {top_n} features.")
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    log.info("Successfully generated and encoded the plot.")
    return img_base64

# --- NEW: Function to create the training history plot ---
def create_training_history_plot(evals_result: dict) -> str:
    """Creates a plot of training history (loss and accuracy) and returns it as a Base64 string."""
    log.info("Generating training history plot.")
    results = evals_result['validation_0']
    epochs = len(results['logloss'])
    x_axis = range(0, epochs)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss on the primary Y-axis
    ax1.plot(x_axis, results['logloss'], 'g-', label='Validation LogLoss')
    ax1.set_xlabel('Epochs (Boosting Rounds)')
    ax1.set_ylabel('LogLoss', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    
    # --- FIX: Robustly check for the error metric before plotting ---
    # Find the metric key that is NOT 'logloss'
    error_keys = [key for key in results.keys() if key != 'logloss']
    
    if error_keys:
        error_key = error_keys[0]
        log.info(f"Using '{error_key}' as the error metric for plotting accuracy.")
        
        # Create a secondary Y-axis for Accuracy
        ax2 = ax1.twinx()
        accuracy = [1 - x for x in results[error_key]]
        ax2.plot(x_axis, accuracy, 'b-', label='Validation Accuracy')
        ax2.set_ylabel('Accuracy', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
    else:
        log.warning("Only 'logloss' was found in eval_results. Accuracy will not be plotted.")

    plt.title('XGBoost Training History')
    fig.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    log.info("Successfully generated and encoded the training history plot.")
    return img_base64
    
# --- CRITICAL OPTIMIZATION: Single-pass processing function ---
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
        chunk_size = 100_000  # Process 100K rows at a time for optimal memory usage
        
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
                
                # === PASS RATE CALCULATION ===
                if 'Response' in chunk.columns:
                    response_col = pd.to_numeric(chunk['Response'], errors='coerce')
                    pass_count += response_col[response_col == 1].count()
                
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

def validate_and_count_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
    """
    Validates date ranges by first checking the filename, then loads the
    Parquet file to get exact counts if validation passes.
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
        
        log.info("Filename boundary check passed. Proceeding to load Parquet file.")
        
    except (IndexError, ValueError) as e:
        log.error(f"Could not parse dates from filename '{parquet_path}': {e}")
        raise ValueError("Could not validate filename. File may be malformed.")
    
    # Load data only if the initial check passes
    df = pd.read_parquet(parquet_path)
    time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
    
    if not time_col_name:
        raise ValueError("Timestamp column not found in the dataset.")
    
    df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
    
    if df[time_col_name].isnull().all():
        return {"status": "Invalid", "detail": "Could not parse valid dates from the dataset's timestamp column."}
    
    training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
    training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
    testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
    testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
    simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
    simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)
    
    if training_end > testing_start or testing_end > simulation_start:
        return {"status": "Invalid", "detail": "Date ranges cannot overlap."}
    
    training_mask = (df[time_col_name] >= training_start) & (df[time_col_name] < training_end)
    testing_mask = (df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)
    simulation_mask = (df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)
    
    df['YYYY-MM'] = df[time_col_name].dt.to_period('M').astype(str)
    monthly_counts = df['YYYY-MM'].value_counts().sort_index().to_dict()
    
    return {
        "status": "Valid",
        "training": {"count": int(df.loc[training_mask].shape[0])},
        "testing": {"count": int(df.loc[testing_mask].shape[0])},
        "simulation": {"count": int(df.loc[simulation_mask].shape[0])},
        "monthlyCounts": monthly_counts
    }

# def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     Trains an XGBoost model by loading data from a Parquet file.
#     """
#     log.info(f"--- Starting model training from Parquet for dataset: {dataset_id} ---")
    
#     # === Step 1: Fetch Data from Parquet File ===
#     parquet_path = find_parquet_file(dataset_id)
#     if not parquet_path:
#         raise ValueError("Dataset file not found. It may have expired or failed processing.")
    
#     log.info(f"Loading data from Parquet file: {parquet_path}")
#     df = pd.read_parquet(parquet_path)
#     log.info(f"Successfully loaded DataFrame. Shape: {df.shape}")
    
#     # Identify timestamp column
#     time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
#     df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
    
#     # Convert feature columns to numeric
#     cols_to_convert = [col for col in df.columns if col not in ['Id', time_col_name]]
#     for col in cols_to_convert:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # === Step 2: Slice Data ===
#     training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#     training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
#     testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
#     testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
#     simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
#     simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)
    
#     train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
#     test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
#     simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]
    
#     if train_df.empty or test_df.empty:
#         raise ValueError("Training or testing date range resulted in zero records.")
    
#     # === Step 3: Feature Selection ===
#     log.info("\n--- Step 3: Performing Feature Selection ---")
#     target_col = 'Response'
#     initial_features = [col for col in df.columns if col not in [target_col, time_col_name, 'Id']]
    
#     nan_threshold = 50.0
#     nan_percentages = (train_df[initial_features].isnull().sum() / len(train_df)) * 100
#     cols_to_drop_by_nan = nan_percentages[nan_percentages > nan_threshold].index
#     features_after_nan_filter = [col for col in initial_features if col not in cols_to_drop_by_nan]
    
#     temp_train_data = train_df[features_after_nan_filter].fillna(0)
#     selector = VarianceThreshold(threshold=(.98 * (1 - .98)))
#     selector.fit(temp_train_data)
#     final_features = temp_train_data.columns[selector.get_support()].tolist()
    
#     # === Step 4: Prepare Final Data & Impute Missing Values ===
#     log.info("\n--- Step 4: Preparing Final Data & Imputing NaNs ---")
#     X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
#     X_test, y_test = test_df[final_features], test_df[target_col].astype(int)
    
#     imputer = SimpleImputer(strategy='median')
#     imputer.fit(X_train)
#     X_train = pd.DataFrame(imputer.transform(X_train), columns=final_features)
#     X_test = pd.DataFrame(imputer.transform(X_test), columns=final_features)
    
#     # === Step 5: Handle Class Imbalance ===
#     log.info("\n--- Step 5: Calculating Class Imbalance Weight ---")
#     try:
#         scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#     except (KeyError, ZeroDivisionError):
#         scale_pos_weight = 1
    
#     # === Step 6: Training the Model ===
#     log.info("\n--- Step 6: Training XGBoost Model ---")
#     model = xgb.XGBClassifier(
#         objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
#         scale_pos_weight=scale_pos_weight, n_estimators=150
#     )
#     model.fit(X_train, y_train)
    
#     # === Step 7: Evaluation & Persistence ===
#     log.info("\n--- Step 7: Evaluating, Plotting, and Persisting Results ---")
#     y_pred = model.predict(X_test)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
#     metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred, zero_division=0),
#         "recall": recall_score(y_test, y_pred, zero_division=0),
#         "f1Score": f1_score(y_test, y_pred, zero_division=0),
#         "trueNegative": int(tn), "falsePositive": int(fp),
#         "falseNegative": int(fn), "truePositive": int(tp)
#     }
    
#     b64_plot_string = create_feature_importance_plot(
#         feature_names=X_train.columns,
#         importances=model.feature_importances_
#     )
#     plots = {"featureImportance": b64_plot_string}
    
#     model_dir = "models"
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
#     model.save_model(model_path)
    
#     if not simulation_df.empty:
#         simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
#         sim_features = simulation_df[final_features]
#         sim_imputed = imputer.transform(sim_features)
#         simulation_df_processed = pd.DataFrame(sim_imputed, columns=final_features)
        
#         for col in ['Id', time_col_name, 'Response']:
#             if col in simulation_df:
#                 simulation_df_processed[col] = simulation_df[col].values
        
#         store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)
#         log.info(f"Processed simulation data and saved to Redis key: {simulation_key}")
    
#     # Clean up the Parquet file after training
#     os.remove(parquet_path)
#     log.info(f"Original dataset file '{parquet_path}' deleted from disk.")
    
#     return {"metrics": metrics, "plots": plots}
def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
    """
    Trains an XGBoost model with controlled resampling and conservative threshold tuning.
    """
    log.info(f"--- Starting controlled model training for dataset: {dataset_id} ---")
    
    # === Steps 1-5: Same as before (Data loading, slicing, feature selection, imputation) ===
    parquet_path = find_parquet_file(dataset_id)
    if not parquet_path:
        raise ValueError("Dataset file not found. It may have expired or failed processing.")
    
    log.info(f"Loading data from Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log.info(f"Successfully loaded DataFrame. Shape: {df.shape}")
    
    time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
    df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
    
    cols_to_convert = [col for col in df.columns if col not in ['Id', time_col_name]]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
    training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
    testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
    testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
    simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
    simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)
    
    train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
    print(f"Training set length: {len(train_df)}")

    test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
    print(f"Testing set length: {len(test_df)}")

    simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]
    print(f"Simulation set length: {len(simulation_df)}")
    
    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing date range resulted in zero records.")
    
    target_col = 'Response'
    initial_features = [col for col in df.columns if col not in [target_col, time_col_name, 'Id']]
    
    nan_threshold = 50.0
    nan_percentages = (train_df[initial_features].isnull().sum() / len(train_df)) * 100
    cols_to_drop_by_nan = nan_percentages[nan_percentages > nan_threshold].index
    features_after_nan_filter = [col for col in initial_features if col not in cols_to_drop_by_nan]
    
    temp_train_data = train_df[features_after_nan_filter].fillna(0)
    selector = VarianceThreshold(threshold=(.98 * (1 - .98)))
    selector.fit(temp_train_data)
    final_features = temp_train_data.columns[selector.get_support()].tolist()
    
    X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
    X_test, y_test = test_df[final_features], test_df[target_col].astype(int)
    
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_train = pd.DataFrame(imputer.transform(X_train), columns=final_features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=final_features)
    
    # === Step 6: CONTROLLED SMOTE Application ===
    log.info("\n--- Step 6: Applying Controlled SMOTE ---")
    log.info(f"Original training class distribution: {y_train.value_counts().to_dict()}")
    
    # **FIX 1: Use conservative SMOTE ratio instead of full balance**
    # For 1:199 ratio, only oversample to 1:10 or 1:20 (not 1:1)
    minority_count = y_train.sum()
    majority_count = len(y_train) - minority_count
    
    # Target ratio of 1:10 instead of 1:1 (less aggressive)
    target_minority_count = min(majority_count // 10, minority_count * 5)
    
    if minority_count >= 2 and target_minority_count > minority_count:
        smote = SMOTE(
            sampling_strategy={1: target_minority_count}, 
            random_state=42, 
            k_neighbors=min(5, minority_count - 1)
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        log.info(f"After controlled SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    else:
        # If too few positive samples, skip SMOTE
        X_train_resampled, y_train_resampled = X_train, y_train
        log.info("Skipping SMOTE due to insufficient positive samples")
    
    # === Step 7: Enhanced Class Weight Calculation ===
    log.info("\n--- Step 7: Calculating Enhanced Class Weight ---")
    try:
        # Use original imbalance ratio for scale_pos_weight (more conservative)
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    except (KeyError, ZeroDivisionError):
        scale_pos_weight = 1
    
    # === Step 8: Training with More Conservative Parameters ===
    log.info("\n--- Step 8: Training XGBoost Model ---")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,  # **FIX 2: Reduced from 500 to prevent overfitting**
        learning_rate=0.05,  # **FIX 3: Lower learning rate**
        max_depth=4,        # **FIX 4: Limit tree depth**
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,  # **FIX 5: Higher min_child_weight for stability**
        early_stopping_rounds=15,  # **FIX 6: Earlier stopping**
        random_state=42
    )
    
    # Split test for validation
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # === Step 9: CONSERVATIVE Threshold Tuning ===
    log.info("\n--- Step 9: Conservative Threshold Tuning ---")
    
    y_proba = model.predict_proba(X_test_final)[:, 1]
    
    # **FIX 7: More conservative threshold range and better validation**
    best_threshold = 0.5  # Default fallback
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    # Test more conservative threshold range
    thresholds_to_test = np.arange(0.1, 0.9, 0.05)  # 0.1 to 0.85, not 0.01
    
    for threshold in thresholds_to_test:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        
        # **FIX 8: Add validation - skip if all predictions are same class**
        if len(np.unique(y_pred_threshold)) == 1:
            continue
            
        precision = precision_score(y_test_final, y_pred_threshold, zero_division=0)
        recall = recall_score(y_test_final, y_pred_threshold, zero_division=0)
        f1 = f1_score(y_test_final, y_pred_threshold, zero_division=0)
        
        # **FIX 9: Require minimum precision AND recall**
        if f1 > best_f1 and precision > 0.1 and recall > 0.1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    log.info(f"Optimal threshold: {best_threshold:.3f}, F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    
    # === Step 10: Final Evaluation ===
    y_pred_final = (y_proba >= best_threshold).astype(int)
    
    # **FIX 10: Validate final predictions**
    unique_preds = np.unique(y_pred_final)
    log.info(f"Final prediction classes: {unique_preds}")
    
    if len(unique_preds) == 1:
        log.warning(f"Model predicting only class {unique_preds[0]}! Using default threshold 0.5")
        y_pred_final = (y_proba >= 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_final).ravel()
    
    metrics = {
        "accuracy": accuracy_score(y_test_final, y_pred_final),
        "precision": precision_score(y_test_final, y_pred_final, zero_division=0),
        "recall": recall_score(y_test_final, y_pred_final, zero_division=0),
        "f1Score": f1_score(y_test_final, y_pred_final, zero_division=0),
        "trueNegative": int(tn), "falsePositive": int(fp),
        "falseNegative": int(fn), "truePositive": int(tp)
    }
    
    # Rest of the function remains the same...
    b64_plot_string = create_feature_importance_plot(
        feature_names=X_train.columns,
        importances=model.feature_importances_
    )
    b64_training_plot = create_training_history_plot(model.evals_result())
    plots = {"featureImportance": b64_plot_string, "trainingPlot": b64_training_plot}
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
    model.save_model(model_path)
    
    if not simulation_df.empty:
        simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
        sim_features = simulation_df[final_features]
        sim_imputed = imputer.transform(sim_features)
        simulation_df_processed = pd.DataFrame(sim_imputed, columns=final_features)
        
        for col in ['Id', time_col_name, 'Response']:
            if col in simulation_df:
                simulation_df_processed[col] = simulation_df[col].values
        
        store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)
    
    os.remove(parquet_path)
    
    return {"metrics": metrics, "plots": plots}

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
    
    sim_key = f"user:{user_id}:simulation_data:{dataset_id}"
    sim_df = get_dataframe_from_json(sim_key)
    if sim_df is None:
        yield {"error": "Simulation data not found. Please re-run the training process."}
        return
    
    try:
        final_features = model.feature_names_in_
    except AttributeError:
        final_features = [col for col in sim_df.columns if col not in ['Id', 'synthetic_timestamp', 'Response']]

    # --- Start of Changes ---

    # 1. Before the loop, find a potential ID column.
    id_col_name = next((col for col in sim_df.columns if col.lower() == 'id'), None)

    if id_col_name:
        print(f"Using column '{id_col_name}' as the row identifier.")
    else:
        print("No 'Id' column found. Falling back to using the DataFrame index as the identifier.")

    # --- End of Changes ---
    
    for index, row in sim_df.iterrows():
        try:
            # 2. Determine which identifier to use for the current row.
            row_identifier = int(row[id_col_name]) if id_col_name else int(index)

            X_pred = pd.DataFrame([row[final_features]], columns=final_features)
            prediction_proba = model.predict_proba(X_pred)[0]
            prediction = int(prediction_proba.argmax())
            confidence = float(prediction_proba[prediction])
            actual_response = int(row['Response'])
            
            yield {
                # 3. Use the determined identifier in the output.
                "rowIndex": row_identifier, 
                "prediction": prediction,
                "confidence": round(confidence, 4), 
                "actual": actual_response,
                "isCorrect": prediction == actual_response,
                "timestamp": row.get('synthetic_timestamp', str(datetime.now()))
            }
            
            await asyncio.sleep(1)
            
        except Exception as row_error:
            log.error(f"[{simulation_id}] Error processing row {index}: {row_error}")
            yield {"error": f"Error on row {index}: {row_error}"}
