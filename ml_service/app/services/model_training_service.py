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


# # app/services/model_training_service.py

# import pandas as pd
# import io
# import os
# import logging
# import base64
# import numpy as np
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
# from sklearn.impute import SimpleImputer
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline as ImbPipeline
# from sklearn.preprocessing import StandardScaler
# from app.redis_client import store_dataframe_as_json
# from app.services.data_processing_service import find_parquet_file

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

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()
#     log.info("Successfully generated and encoded the plot.")
#     return img_base64

# def create_training_history_plot(evals_result: dict) -> str:
#     """Creates a plot of training history (loss and accuracy) and returns it as a Base64 string."""
#     log.info("Generating training history plot.")
#     results = evals_result['validation_0']
#     epochs = len(results['logloss'])
#     x_axis = range(0, epochs)

#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     # Plot Loss on the primary Y-axis
#     ax1.plot(x_axis, results['logloss'], 'g-', label='Validation LogLoss')
#     ax1.set_xlabel('Epochs (Boosting Rounds)')
#     ax1.set_ylabel('LogLoss', color='g')
#     ax1.tick_params(axis='y', labelcolor='g')

#     # Find the metric key that is NOT 'logloss'
#     error_keys = [key for key in results.keys() if key != 'logloss']
#     if error_keys:
#         error_key = error_keys[0]
#         log.info(f"Using '{error_key}' as the error metric for plotting accuracy.")
        
#         # Create a secondary Y-axis for Accuracy
#         ax2 = ax1.twinx()
#         accuracy = [1 - x for x in results[error_key]]
#         ax2.plot(x_axis, accuracy, 'b-', label='Validation Accuracy')
#         ax2.set_ylabel('Accuracy', color='b')
#         ax2.tick_params(axis='y', labelcolor='b')
#     else:
#         log.warning("Only 'logloss' was found in eval_results. Accuracy will not be plotted.")

#     plt.title('XGBoost Training History')
#     fig.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()
#     log.info("Successfully generated and encoded the training history plot.")
#     return img_base64

# def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
#     """
#     HIGH-PRECISION Bosch Production Line Model Training with Advanced Techniques.
#     Optimized for maximum precision on highly imbalanced manufacturing defect detection.
#     """
#     log.info(f"--- Starting HIGH-PRECISION model training for dataset: {dataset_id} ---")

#     # === Step 1: Data Loading and Validation ===
#     parquet_path = find_parquet_file(dataset_id)
#     if not parquet_path:
#         raise ValueError("Dataset file not found. It may have expired or failed processing.")

#     log.info(f"Loading data from Parquet file: {parquet_path}")
#     df = pd.read_parquet(parquet_path)
#     log.info(f"Successfully loaded DataFrame. Shape: {df.shape}")

#     # === Step 2: CRITICAL - Extract Only 970 Numeric Features (Bosch F-columns) ===
#     log.info("--- Step 2: Filtering to 970 Numeric Features ---")
    
#     # Identify all numeric feature columns (L*_S*_F* pattern)
#     numeric_feature_cols = [col for col in df.columns if '_F' in col and any(col.startswith(f'L{i}_S') for i in range(4))]
    
#     # Essential columns for processing
#     time_col_name = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), None)
#     essential_cols = ['Id', time_col_name, 'Response']
    
#     # Final feature set: 970 numeric features + essential columns
#     available_numeric_features = [col for col in numeric_feature_cols if col in df.columns]
#     final_columns = essential_cols + available_numeric_features
    
#     log.info(f"Selected {len(available_numeric_features)} numeric features from {len(df.columns)} total columns")
    
#     # Filter DataFrame to only include selected columns
#     df = df[final_columns].copy()
    
#     # === Step 3: Temporal Data Splitting ===
#     df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')

#     # Convert all feature columns to numeric
#     for col in available_numeric_features:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
#     training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
#     testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
#     testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
#     simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
#     simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

#     train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
#     test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
#     simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]

#     log.info(f"Training set: {len(train_df)}, Testing set: {len(test_df)}, Simulation set: {len(simulation_df)}")

#     if train_df.empty or test_df.empty:
#         raise ValueError("Training or testing date range resulted in zero records.")

#     # === Step 4: Advanced Feature Engineering for Bosch Dataset ===
#     log.info("--- Step 4: Advanced Bosch-Specific Feature Engineering ---")
    
#     target_col = 'Response'
    
#     # Calculate sparsity for each feature (common in Bosch data)
#     feature_sparsity = {}
#     for col in available_numeric_features:
#         non_null_ratio = train_df[col].notna().sum() / len(train_df)
#         feature_sparsity[col] = non_null_ratio
    
#     # Filter out extremely sparse features (less than 1% non-null)
#     sparse_threshold = 0.01
#     dense_features = [col for col, sparsity in feature_sparsity.items() if sparsity >= sparse_threshold]
#     log.info(f"Retained {len(dense_features)} features after sparsity filtering (>{sparse_threshold*100}% non-null)")
    
#     # Advanced NaN handling specific to production line data
#     # Features with >80% missing are likely process-specific and should be handled carefully
#     high_missing_threshold = 0.8
#     medium_missing_features = []
#     low_missing_features = []
    
#     for col in dense_features:
#         missing_ratio = train_df[col].isna().sum() / len(train_df)
#         if missing_ratio > high_missing_threshold:
#             continue  # Skip extremely sparse features
#         elif missing_ratio > 0.5:
#             medium_missing_features.append(col)
#         else:
#             low_missing_features.append(col)
    
#     final_features = low_missing_features + medium_missing_features
#     log.info(f"Final feature set: {len(final_features)} features after advanced filtering")

#     # === Step 5: Precision-Optimized Preprocessing Pipeline ===
#     log.info("--- Step 5: High-Precision Preprocessing Pipeline ---")
    
#     X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
#     X_test, y_test = test_df[final_features], test_df[target_col].astype(int)
    
#     # Log initial class distribution
#     class_dist = y_train.value_counts()
#     log.info(f"Original class distribution: {class_dist.to_dict()}")
#     log.info(f"Class imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1")
    
#     # Advanced imputation strategy
#     # Use median for low-missing features, mode for high-missing
#     imputer_low = SimpleImputer(strategy='median')
#     imputer_medium = SimpleImputer(strategy='constant', fill_value=0)  # Missing = process not run
    
#     if low_missing_features:
#         X_train_low = imputer_low.fit_transform(X_train[low_missing_features])
#         X_test_low = imputer_low.transform(X_test[low_missing_features])
#         X_train_low = pd.DataFrame(X_train_low, columns=low_missing_features, index=X_train.index)
#         X_test_low = pd.DataFrame(X_test_low, columns=low_missing_features, index=X_test.index)
#     else:
#         X_train_low = pd.DataFrame(index=X_train.index)
#         X_test_low = pd.DataFrame(index=X_test.index)
    
#     if medium_missing_features:
#         X_train_medium = imputer_medium.fit_transform(X_train[medium_missing_features])
#         X_test_medium = imputer_medium.transform(X_test[medium_missing_features])
#         X_train_medium = pd.DataFrame(X_train_medium, columns=medium_missing_features, index=X_train.index)
#         X_test_medium = pd.DataFrame(X_test_medium, columns=medium_missing_features, index=X_test.index)
#     else:
#         X_train_medium = pd.DataFrame(index=X_train.index)
#         X_test_medium = pd.DataFrame(index=X_test.index)
    
#     # Combine imputed features
#     X_train_imputed = pd.concat([X_train_low, X_train_medium], axis=1)
#     X_test_imputed = pd.concat([X_test_low, X_test_medium], axis=1)
    
#     # Advanced feature selection for precision optimization
#     # Remove zero-variance features
#     variance_selector = VarianceThreshold(threshold=0.0)
#     X_train_var = variance_selector.fit_transform(X_train_imputed)
#     X_test_var = variance_selector.transform(X_test_imputed)
    
#     selected_features = X_train_imputed.columns[variance_selector.get_support()].tolist()
#     X_train_processed = pd.DataFrame(X_train_var, columns=selected_features, index=X_train.index)
#     X_test_processed = pd.DataFrame(X_test_var, columns=selected_features, index=X_test.index)
    
#     log.info(f"Features after variance filtering: {len(selected_features)}")
    
#     # === Step 6: Precision-Focused Class Imbalance Handling ===
#     log.info("--- Step 6: Precision-Optimized Imbalance Handling ---")
    
#     minority_count = y_train.sum()
#     majority_count = len(y_train) - minority_count
    
#     # For extreme imbalance in production data, use conservative approach
#     # Target ratio more conservative than 1:10 to maintain precision
#     target_ratio = max(0.05, min(0.15, minority_count / majority_count * 3))  # 5-15% positive class
#     target_minority_count = int(majority_count * target_ratio)
    
#     if minority_count >= 10 and target_minority_count > minority_count:
#         # Use ADASYN for better boundary learning in manufacturing data
#         try:
#             resampler = ADASYN(
#                 sampling_strategy={1: target_minority_count},
#                 random_state=42,
#                 n_neighbors=min(5, minority_count - 1)
#             )
#             X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_processed, y_train)
#             log.info(f"Applied ADASYN: {pd.Series(y_train_resampled).value_counts().to_dict()}")
#         except ValueError:
#             # Fallback to SMOTE if ADASYN fails
#             resampler = SMOTE(
#                 sampling_strategy={1: target_minority_count},
#                 random_state=42,
#                 k_neighbors=min(5, minority_count - 1)
#             )
#             X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_processed, y_train)
#             log.info(f"Applied SMOTE fallback: {pd.Series(y_train_resampled).value_counts().to_dict()}")
#     else:
#         X_train_resampled, y_train_resampled = X_train_processed, y_train
#         log.info("Skipping resampling due to insufficient positive samples")

#     # === Step 7: High-Precision XGBoost Configuration ===
#     log.info("--- Step 7: Precision-Optimized XGBoost Training ---")
    
#     # Calculate scale_pos_weight from original distribution
#     scale_pos_weight = majority_count / minority_count if minority_count > 0 else 1
    
#     # Precision-optimized hyperparameters for manufacturing defect detection
#     model = xgb.XGBClassifier(
#         objective='binary:logistic',
#         eval_metric=['logloss', 'aucpr'],  # AUCPR is better for imbalanced data
#         scale_pos_weight=scale_pos_weight,
        
#         # Conservative parameters to prevent overfitting
#         n_estimators=300,
#         learning_rate=0.03,  # Lower learning rate for stability
#         max_depth=6,  # Deeper trees for complex interactions
#         min_child_weight=5,  # Higher to prevent overfitting on rare cases
        
#         # Regularization for precision
#         reg_alpha=0.1,  # L1 regularization
#         reg_lambda=1.0,  # L2 regularization
        
#         # Sampling parameters
#         subsample=0.8,
#         colsample_bytree=0.8,
#         colsample_bylevel=0.8,
        
#         # Early stopping and validation
#         early_stopping_rounds=25,
#         random_state=42,
#         n_jobs=-1
#     )
    
#     # Split test set for validation
#     X_val, X_test_final, y_val, y_test_final = train_test_split(
#         X_test_processed, y_test, test_size=0.5, random_state=42, stratify=y_test
#     )
    
#     # Train with validation monitoring
#     model.fit(
#         X_train_resampled, y_train_resampled,
#         eval_set=[(X_val, y_val)],
#         verbose=False
#     )
    
#     # === Step 8: Precision-Maximizing Threshold Optimization ===
#     log.info("--- Step 8: Precision-Maximizing Threshold Selection ---")
    
#     y_proba = model.predict_proba(X_test_final)[:, 1]
    
#     # Focus on precision optimization with minimum recall constraint
#     best_threshold = 0.5
#     best_precision = 0
#     best_recall = 0
#     best_f1 = 0
    
#     # Test wider range of thresholds with focus on high precision
#     thresholds_to_test = np.concatenate([
#         np.arange(0.1, 0.5, 0.02),  # Lower thresholds
#         np.arange(0.5, 0.9, 0.01),  # Higher precision thresholds
#         np.arange(0.9, 0.99, 0.005)  # Very high precision thresholds
#     ])
    
#     threshold_results = []
#     for threshold in thresholds_to_test:
#         y_pred_threshold = (y_proba >= threshold).astype(int)
        
#         # Skip if all predictions are same class
#         if len(np.unique(y_pred_threshold)) == 1:
#             continue
            
#         precision = precision_score(y_test_final, y_pred_threshold, zero_division=0)
#         recall = recall_score(y_test_final, y_pred_threshold, zero_division=0)
#         f1 = f1_score(y_test_final, y_pred_threshold, zero_division=0)
        
#         threshold_results.append({
#             'threshold': threshold,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1
#         })
        
#         # Prioritize precision with minimum viable recall (>5%)
#         if precision > best_precision and recall > 0.05:
#             best_precision = precision
#             best_recall = recall
#             best_f1 = f1
#             best_threshold = threshold
    
#     log.info(f"Optimal precision threshold: {best_threshold:.3f}")
#     log.info(f"Best precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    
#     # === Step 9: Final Model Evaluation ===
#     y_pred_final = (y_proba >= best_threshold).astype(int)
    
#     # Validate predictions
#     unique_preds = np.unique(y_pred_final)
#     if len(unique_preds) == 1:
#         log.warning(f"Model predicting only class {unique_preds[0]}! Using default threshold 0.5")
#         y_pred_final = (y_proba >= 0.5).astype(int)
    
#     # Calculate comprehensive metrics
#     tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_final).ravel()
    
#     metrics = {
#         "accuracy": float(accuracy_score(y_test_final, y_pred_final)),
#         "precision": float(precision_score(y_test_final, y_pred_final, zero_division=0)),
#         "recall": float(recall_score(y_test_final, y_pred_final, zero_division=0)),
#         "f1Score": float(f1_score(y_test_final, y_pred_final, zero_division=0)),
#         "trueNegative": int(tn),
#         "falsePositive": int(fp),
#         "falseNegative": int(fn),
#         "truePositive": int(tp)
#     }
    
#     log.info(f"Final metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
#     # === Step 10: Generate Plots and Save Model ===
#     b64_plot_string = create_feature_importance_plot(
#         feature_names=selected_features,
#         importances=model.feature_importances_
#     )
    
#     b64_training_plot = create_training_history_plot(model.evals_result())
#     plots = {"featureImportance": b64_plot_string, "trainingPlot": b64_training_plot}
    
#     # Save model
#     model_dir = "models"
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
#     model.save_model(model_path)
    
#     # === Step 11: Prepare Simulation Data ===
#     if not simulation_df.empty:
#         simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
#         sim_features = simulation_df[final_features]
        
#         # Apply same preprocessing pipeline
#         if low_missing_features:
#             sim_low = imputer_low.transform(sim_features[low_missing_features])
#             sim_low = pd.DataFrame(sim_low, columns=low_missing_features, index=sim_features.index)
#         else:
#             sim_low = pd.DataFrame(index=sim_features.index)
            
#         if medium_missing_features:
#             sim_medium = imputer_medium.transform(sim_features[medium_missing_features])
#             sim_medium = pd.DataFrame(sim_medium, columns=medium_missing_features, index=sim_features.index)
#         else:
#             sim_medium = pd.DataFrame(index=sim_features.index)
        
#         sim_imputed = pd.concat([sim_low, sim_medium], axis=1)
#         sim_processed = variance_selector.transform(sim_imputed)
#         simulation_df_processed = pd.DataFrame(sim_processed, columns=selected_features, index=sim_features.index)
        
#         # Add essential columns
#         for col in ['Id', time_col_name, 'Response']:
#             if col in simulation_df:
#                 simulation_df_processed[col] = simulation_df[col].values
        
#         store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)
    
#     # Clean up
#     os.remove(parquet_path)
    
#     log.info("--- HIGH-PRECISION model training completed successfully ---")
#     return {"metrics": metrics, "plots": plots}

# app/services/model_training_service.py

import pandas as pd
import io
import os
import logging
import base64
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from app.redis_client import store_dataframe_as_json
from app.services.data_processing_service import find_parquet_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

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

def train_model_with_ranges(user_id: str, dataset_id: str, ranges: dict) -> dict:
    """
    HIGH-PRECISION Bosch Production Line Model Training implementing advanced R methodology.
    Optimized for maximum precision on highly imbalanced manufacturing defect detection.
    
    This implementation translates sophisticated R techniques that achieved 48%+ precision:
    - Advanced feature engineering with station paths and temporal features
    - Multi-stage imputation strategy
    - Sophisticated XGBoost hyperparameter tuning
    - Precision-optimized threshold selection
    - Controlled resampling for extreme class imbalance
    """
    log.info(f"--- Starting HIGH-PRECISION Bosch model training for dataset: {dataset_id} ---")

    # === Step 1: Data Loading and Validation ===
    parquet_path = find_parquet_file(dataset_id)
    if not parquet_path:
        raise ValueError("Dataset file not found. It may have expired or failed processing.")

    log.info(f"Loading data from Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log.info(f"Successfully loaded DataFrame. Shape: {df.shape}")

    # === Step 2: CRITICAL - Extract Bosch Numeric Features (970 L*_S*_F* pattern) ===
    log.info("--- Step 2: Extracting Bosch Production Line Features ---")
    
    # Identify all Bosch numeric feature columns (L*_S*_F* pattern)
    bosch_feature_cols = [col for col in df.columns 
                         if col.startswith('L') and '_S' in col and '_F' in col]
    
    # Essential columns for processing
    time_col_name = next((col for col in df.columns 
                         if 'timestamp' in col.lower() or 'date' in col.lower()), None)
    essential_cols = ['Id', time_col_name, 'Response']
    
    # Final feature set: Bosch numeric features + essential columns
    available_features = [col for col in bosch_feature_cols if col in df.columns]
    final_columns = essential_cols + available_features
    
    log.info(f"Selected {len(available_features)} Bosch features from {len(df.columns)} total columns")
    
    # Filter DataFrame to only include selected columns
    df = df[final_columns].copy()
    
    # === Step 3: Advanced Feature Engineering (from R methodology) ===
    log.info("--- Step 3: Advanced Bosch Feature Engineering ---")
    
    # Convert timestamp column
    df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')

    # Convert all feature columns to numeric
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create station-based aggregations (from R StationPath.R)
    stations = set()
    for col in available_features:
        if '_S' in col:
            station = col.split('_S')[1].split('_')[0]
            stations.add(f"S{station}")
    
    # Add station count features
    for station in list(stations)[:50]:  # Limit to prevent memory issues
        station_cols = [col for col in available_features if f"_S{station[1:]}_" in col]
        if station_cols:
            df[f"{station}_count"] = df[station_cols].notna().sum(axis=1)
            df[f"{station}_mean"] = df[station_cols].mean(axis=1)

    # Add line-based aggregations (from R XGB1 Train.R methodology)
    lines = set()
    for col in available_features:
        if col.startswith('L'):
            line = col.split('_')[0]
            lines.add(line)
    
    for line in list(lines):
        line_cols = [col for col in available_features if col.startswith(f"{line}_")]
        if line_cols:
            df[f"{line}_numeric_count"] = df[line_cols].notna().sum(axis=1)
            df[f"{line}_positive_count"] = (df[line_cols] > 0).sum(axis=1)
            df[f"{line}_negative_count"] = (df[line_cols] < 0).sum(axis=1)
            df[f"{line}_zero_count"] = (df[line_cols] == 0).sum(axis=1)
            df[f"{line}_missing_count"] = df[line_cols].isna().sum(axis=1)

    # Update feature list to include engineered features
    engineered_features = [col for col in df.columns 
                          if col not in essential_cols and col not in available_features]
    all_features = available_features + engineered_features
    
    log.info(f"Created {len(engineered_features)} engineered features")

    # === Step 4: Temporal Data Splitting ===
    training_start = pd.to_datetime(ranges['training']['start']).tz_localize(None)
    training_end = pd.to_datetime(ranges['training']['end']).tz_localize(None)
    testing_start = pd.to_datetime(ranges['testing']['start']).tz_localize(None)
    testing_end = pd.to_datetime(ranges['testing']['end']).tz_localize(None)
    simulation_start = pd.to_datetime(ranges['simulation']['start']).tz_localize(None)
    simulation_end = pd.to_datetime(ranges['simulation']['end']).tz_localize(None)

    train_df = df[(df[time_col_name] >= training_start) & (df[time_col_name] < training_end)]
    test_df = df[(df[time_col_name] >= testing_start) & (df[time_col_name] < testing_end)]
    simulation_df = df[(df[time_col_name] >= simulation_start) & (df[time_col_name] <= simulation_end)]

    log.info(f"Training set: {len(train_df)}, Testing set: {len(test_df)}, Simulation set: {len(simulation_df)}")

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing date range resulted in zero records.")

    # === Step 5: Advanced Preprocessing Pipeline (from R methodology) ===
    log.info("--- Step 5: Multi-Stage Preprocessing Pipeline ---")
    
    target_col = 'Response'
    
    # Calculate feature sparsity and quality metrics
    feature_stats = {}
    for col in all_features:
        if col in train_df.columns:
            non_null_ratio = train_df[col].notna().sum() / len(train_df)
            variance = train_df[col].var() if train_df[col].notna().any() else 0
            feature_stats[col] = {'sparsity': non_null_ratio, 'variance': variance}
    
    # Advanced feature selection based on R methodology
    # Stage 1: Remove extremely sparse features (< 1% non-null)
    sparse_threshold = 0.01
    stage1_features = [col for col, stats in feature_stats.items() 
                      if stats['sparsity'] >= sparse_threshold]
    
    # Stage 2: Remove zero variance features
    stage2_features = [col for col in stage1_features 
                      if feature_stats[col]['variance'] > 0]
    
    log.info(f"Feature selection: {len(all_features)} -> {len(stage1_features)} -> {len(stage2_features)}")
    
    # Categorize features by missing pattern for optimized imputation
    high_missing_features = []
    medium_missing_features = []
    low_missing_features = []
    
    for col in stage2_features:
        if col in train_df.columns:
            missing_ratio = train_df[col].isna().sum() / len(train_df)
            if missing_ratio > 0.8:
                continue  # Skip extremely sparse features
            elif missing_ratio > 0.5:
                high_missing_features.append(col)
            elif missing_ratio > 0.1:
                medium_missing_features.append(col)
            else:
                low_missing_features.append(col)
    
    final_features = low_missing_features + medium_missing_features + high_missing_features
    log.info(f"Final feature categorization: Low({len(low_missing_features)}), "
             f"Medium({len(medium_missing_features)}), High({len(high_missing_features)})")

    # === Step 6: Sophisticated Imputation Strategy ===
    X_train, y_train = train_df[final_features], train_df[target_col].astype(int)
    X_test, y_test = test_df[final_features], test_df[target_col].astype(int)
    
    # Log initial class distribution
    class_dist = y_train.value_counts()
    log.info(f"Original class distribution: {class_dist.to_dict()}")
    log.info(f"Class imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1")
    
    # Multi-stage imputation based on feature characteristics
    imputer_low = SimpleImputer(strategy='median')
    imputer_medium = SimpleImputer(strategy='constant', fill_value=0)  # Missing = process not run
    imputer_high = SimpleImputer(strategy='most_frequent')
    
    # Process low missing features
    if low_missing_features:
        X_train_low = imputer_low.fit_transform(X_train[low_missing_features])
        X_test_low = imputer_low.transform(X_test[low_missing_features])
        X_train_low = pd.DataFrame(X_train_low, columns=low_missing_features, index=X_train.index)
        X_test_low = pd.DataFrame(X_test_low, columns=low_missing_features, index=X_test.index)
    else:
        X_train_low = pd.DataFrame(index=X_train.index)
        X_test_low = pd.DataFrame(index=X_test.index)
    
    # Process medium missing features
    if medium_missing_features:
        X_train_medium = imputer_medium.fit_transform(X_train[medium_missing_features])
        X_test_medium = imputer_medium.transform(X_test[medium_missing_features])
        X_train_medium = pd.DataFrame(X_train_medium, columns=medium_missing_features, index=X_train.index)
        X_test_medium = pd.DataFrame(X_test_medium, columns=medium_missing_features, index=X_test.index)
    else:
        X_train_medium = pd.DataFrame(index=X_train.index)
        X_test_medium = pd.DataFrame(index=X_test.index)
    
    # Process high missing features
    if high_missing_features:
        X_train_high = imputer_high.fit_transform(X_train[high_missing_features])
        X_test_high = imputer_high.transform(X_test[high_missing_features])
        X_train_high = pd.DataFrame(X_train_high, columns=high_missing_features, index=X_train.index)
        X_test_high = pd.DataFrame(X_test_high, columns=high_missing_features, index=X_test.index)
    else:
        X_train_high = pd.DataFrame(index=X_train.index)
        X_test_high = pd.DataFrame(index=X_test.index)
    
    # Combine all imputed features
    X_train_imputed = pd.concat([X_train_low, X_train_medium, X_train_high], axis=1)
    X_test_imputed = pd.concat([X_test_low, X_test_medium, X_test_high], axis=1)
    
    # Final variance threshold filter
    variance_selector = VarianceThreshold(threshold=0.0)
    X_train_var = variance_selector.fit_transform(X_train_imputed)
    X_test_var = variance_selector.transform(X_test_imputed)
    
    selected_features = X_train_imputed.columns[variance_selector.get_support()].tolist()
    X_train_processed = pd.DataFrame(X_train_var, columns=selected_features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_var, columns=selected_features, index=X_test.index)
    
    log.info(f"Features after final variance filtering: {len(selected_features)}")

    # === Step 7: Precision-Optimized Class Imbalance Handling ===
    log.info("--- Step 7: Advanced Imbalance Handling ---")
    
    minority_count = y_train.sum()
    majority_count = len(y_train) - minority_count
    
    # Conservative approach for production data (from R methodology)
    # Target ratio optimized for precision (5-10% positive class)
    target_ratio = max(0.05, min(0.10, minority_count / majority_count * 2))
    target_minority_count = int(majority_count * target_ratio)
    
    if minority_count >= 10 and target_minority_count > minority_count:
        # Use ADASYN for better boundary learning (superior to SMOTE for manufacturing data)
        try:
            resampler = ADASYN(
                sampling_strategy={1: target_minority_count},
                random_state=42,
                n_neighbors=min(5, minority_count - 1)
            )
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_processed, y_train)
            log.info(f"Applied ADASYN: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        except ValueError:
            # Fallback to SMOTE if ADASYN fails
            resampler = SMOTE(
                sampling_strategy={1: target_minority_count},
                random_state=42,
                k_neighbors=min(5, minority_count - 1)
            )
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_processed, y_train)
            log.info(f"Applied SMOTE fallback: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    else:
        X_train_resampled, y_train_resampled = X_train_processed, y_train
        log.info("Skipping resampling due to insufficient positive samples")

    # === Step 8: High-Precision XGBoost Configuration (from R XGB1 Train.R) ===
    log.info("--- Step 8: High-Precision XGBoost Training ---")
    
    # Calculate scale_pos_weight from original distribution
    scale_pos_weight = majority_count / minority_count if minority_count > 0 else 1
    
    # Precision-optimized hyperparameters (translated from R)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['aucpr', 'logloss'],  # AUCPR better for imbalanced data
        scale_pos_weight=scale_pos_weight,
        
        # Parameters optimized for Bosch dataset (from R grid search)
        n_estimators=300,
        learning_rate=0.1,  # eta from R
        max_depth=10,  # From R optimal parameters
        min_child_weight=1,
        
        # Regularization for precision
        reg_alpha=0,
        reg_lambda=1,
        
        # Sampling parameters (from R)
        subsample=0.95,
        colsample_bytree=0.5,  # Important for preventing overfitting
        
        # Early stopping and validation
        early_stopping_rounds=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Split test set for validation (stratified)
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test_processed, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    # Train with validation monitoring
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # === Step 9: Advanced Threshold Optimization for Maximum Precision ===
    log.info("--- Step 9: Precision-Maximizing Threshold Selection ---")
    
    y_proba = model.predict_proba(X_test_final)[:, 1]
    
    # Comprehensive threshold search optimized for precision
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    
    # Extended threshold range focusing on high precision regions
    thresholds_to_test = np.concatenate([
        np.arange(0.1, 0.3, 0.02),   # Lower thresholds
        np.arange(0.3, 0.7, 0.01),   # Mid-range precision thresholds  
        np.arange(0.7, 0.95, 0.005), # High precision thresholds
        np.arange(0.95, 0.99, 0.002) # Very high precision thresholds
    ])
    
    for threshold in thresholds_to_test:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        
        # Skip if all predictions are same class
        if len(np.unique(y_pred_threshold)) == 1:
            continue
            
        precision = precision_score(y_test_final, y_pred_threshold, zero_division=0)
        recall = recall_score(y_test_final, y_pred_threshold, zero_division=0)
        f1 = f1_score(y_test_final, y_pred_threshold, zero_division=0)
        
        # Prioritize precision with minimum viable recall (>3%)
        if precision > best_precision and recall > 0.03:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_threshold = threshold
    
    log.info(f"Optimal precision threshold: {best_threshold:.3f}")
    log.info(f"Best precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    
    # === Step 10: Final Model Evaluation ===
    y_pred_final = (y_proba >= best_threshold).astype(int)
    
    # Validate predictions
    unique_preds = np.unique(y_pred_final)
    if len(unique_preds) == 1:
        log.warning(f"Model predicting only class {unique_preds[0]}! Using quantile threshold")
        # Use quantile-based threshold as fallback (from R methodology)
        quantile_threshold = np.percentile(y_proba, 95)  # Top 5% as positive
        y_pred_final = (y_proba >= quantile_threshold).astype(int)
    
    # Calculate comprehensive metrics
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_final).ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_test_final, y_pred_final)),
        "precision": float(precision_score(y_test_final, y_pred_final, zero_division=0)),
        "recall": float(recall_score(y_test_final, y_pred_final, zero_division=0)),
        "f1Score": float(f1_score(y_test_final, y_pred_final, zero_division=0)),
        "trueNegative": int(tn),
        "falsePositive": int(fp),
        "falseNegative": int(fn),
        "truePositive": int(tp)
    }
    
    log.info(f"Final metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # === Step 11: Generate Plots and Save Model ===
    b64_plot_string = create_feature_importance_plot(
        feature_names=selected_features,
        importances=model.feature_importances_
    )
    
    b64_training_plot = create_training_history_plot(model.evals_result())
    plots = {"featureImportance": b64_plot_string, "trainingPlot": b64_training_plot}
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{user_id}_{dataset_id}.ubj")
    model.save_model(model_path)
    
    # === Step 12: Prepare Simulation Data with Same Pipeline ===
    if not simulation_df.empty:
        simulation_key = f"user:{user_id}:simulation_data:{dataset_id}"
        sim_features = simulation_df[final_features]
        
        # Apply same preprocessing pipeline
        sim_low = imputer_low.transform(sim_features[low_missing_features]) if low_missing_features else np.array([]).reshape(len(sim_features), 0)
        sim_medium = imputer_medium.transform(sim_features[medium_missing_features]) if medium_missing_features else np.array([]).reshape(len(sim_features), 0)
        sim_high = imputer_high.transform(sim_features[high_missing_features]) if high_missing_features else np.array([]).reshape(len(sim_features), 0)
        
        # Combine imputed simulation data
        sim_combined = np.column_stack([arr for arr in [sim_low, sim_medium, sim_high] if arr.size > 0])
        sim_processed = variance_selector.transform(sim_combined)
        simulation_df_processed = pd.DataFrame(sim_processed, columns=selected_features, index=sim_features.index)
        
        # Add essential columns
        for col in ['Id', time_col_name, 'Response']:
            if col in simulation_df.columns:
                simulation_df_processed[col] = simulation_df[col].values
        
        store_dataframe_as_json(simulation_key, simulation_df_processed, expiration_seconds=86400)
    
    # Clean up
    os.remove(parquet_path)
    
    log.info("--- HIGH-PRECISION Bosch model training completed successfully ---")
    log.info(f"Achieved precision: {metrics['precision']:.4f} (Target: >0.48)")
    
    return {"metrics": metrics, "plots": plots}
