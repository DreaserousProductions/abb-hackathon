# app/services/model_training_service.py

import pandas as pd
import io
import os
import logging
import base64
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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
        max_depth=4,  # **FIX 4: Limit tree depth**
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
