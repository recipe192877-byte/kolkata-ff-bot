import pandas as pd
import numpy as np
import joblib
import warnings
import os
import json
from datetime import datetime, timedelta, timezone
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from vector_memory import KolkataVectorMemory
warnings.filterwarnings('ignore')

# ── Self-Learning AI Brain ──
brain = KolkataVectorMemory(context_size=5, db_path='kolkata_brain.json')

MODEL_FILE = 'xgb_model.joblib'
FREQ_FILE = 'freq_model.joblib' # This file is no longer used, freq_model is stored in MODEL_FILE
DATA_FILE = 'kolkata_ff_history_advanced.csv'
STATS_FILE = 'backtest_stats.json'
CONFIG_FILE = 'ai_config.json'

# Default AI Configuration (can be updated by AI Council Evolution)
DEFAULT_AI_CONFIG = {
    "ml_weight": 0.50,
    "memory_weight": 0.15,
    "freq_weight": 0.35,
    "confidence_threshold": 20.0,
    "risk_tolerance": "MEDIUM"
}

def load_ai_config():
    """Load dynamic AI weights and settings."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Ensure all keys exist
                for k, v in DEFAULT_AI_CONFIG.items():
                    if k not in config:
                        config[k] = v
                return config
        except Exception:
            return DEFAULT_AI_CONFIG
    return DEFAULT_AI_CONFIG

def save_ai_config(config):
    """Save updated AI weights and settings."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# ============================================================
#                FEATURE ENGINEERING V3 (DEEP)
# ============================================================

def calculate_patti_sum(patti_val):
    if pd.isna(patti_val) or str(patti_val).strip() == '':
        return 0
    clean_patti = str(patti_val).strip()
    return sum(int(d) for d in clean_patti if d.isdigit())

def extract_patti_digits(patti_val):
    """Extract individual digits from Patti for pattern analysis."""
    if pd.isna(patti_val) or str(patti_val).strip() == '':
        return 0, 0, 0
    try:
        s = str(int(float(patti_val))).zfill(3)
        return int(s[0]), int(s[1]), int(s[2])
    except (ValueError, TypeError):
        return 0, 0, 0

def compute_gap_features(singles_series, current_idx, max_gap_history=50):
    """Compute how many draws since each digit (0-9) last appeared, looking backwards."""
    gaps = {}
    
    # Ensure singles_series is a Series and can be indexed
    if isinstance(singles_series, np.ndarray):
        singles_series = pd.Series(singles_series)

    for d in range(10):
        # Look backwards from current_idx, up to max_gap_history
        lookback_slice = singles_series.iloc[max(0, current_idx - max_gap_history):current_idx]
        
        # Check if the digit exists in the lookback slice
        if d in lookback_slice.values:
            # Find the most recent occurrence
            last_occurrence_idx = lookback_slice[lookback_slice == d].index[-1]
            gap = current_idx - last_occurrence_idx
            gaps[f'Gap_{d}'] = gap
        else:
            gaps[f'Gap_{d}'] = max_gap_history + 1  # If not found, set to max gap + 1
    return gaps

def compute_frequency_features(singles_series, current_idx, window=20):
    """Compute frequency of each digit in last N draws."""
    start = max(0, current_idx - window)
    recent = singles_series.iloc[start:current_idx]
    freq = {}
    if len(recent) == 0: # Handle case when there's no sufficient history
        return {f'Freq_{d}': 0.1 for d in range(10)} # Default to uniform if no data
    counts = recent.value_counts()
    for d in range(10):
        freq[f'Freq_{d}'] = counts.get(d, 0) / len(recent)
    return freq

def load_and_preprocess_data(filepath=DATA_FILE):
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['Single', 'Patti']) # Ensure both are present
        df['Single'] = df['Single'].astype(int)
        
        df['Date_Obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        # Drop rows where Date conversion failed
        df = df.dropna(subset=['Date_Obj']) 
        df = df.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).reset_index(drop=True)
        
        # === CORE FEATURES ===
        df['DayOfWeek'] = df['Date_Obj'].dt.dayofweek
        df['Month'] = df['Date_Obj'].dt.month
        df['Patti_Sum'] = df['Patti'].apply(calculate_patti_sum)
        df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # === CYCLICAL TIME ENCODING (captures periodic patterns better than raw) ===
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # === PATTI DIGIT FEATURES ===
        patti_digits = df['Patti'].apply(extract_patti_digits)
        df['Patti_D1'] = patti_digits.apply(lambda x: x[0])
        df['Patti_D2'] = patti_digits.apply(lambda x: x[1])
        df['Patti_D3'] = patti_digits.apply(lambda x: x[2])
        
        # === ROLLING / LAG FEATURES ===
        df['Is_Even'] = (df['Single'] % 2 == 0).astype(int)
        df['Rolling_Even_5'] = df['Is_Even'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0.5)
        
        # Ensure 'Prev_Day_Same_Bazi' uses integer type for a cleaner feature
        df['Prev_Day_Same_Bazi'] = df.groupby('Bazi')['Single'].shift(1).fillna(-1) # Use -1 or some indicator if no value
        
        # Moving Averages using previous values
        df['MA_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).mean().fillna(4.5)
        df['MA_30'] = df['Single'].shift(1).rolling(window=30, min_periods=1).mean().fillna(4.5)
        df['STD_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).std().fillna(3.0)
        df['Patti_MA_7'] = df['Patti_Sum'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
        
        # === GAP FEATURES: How many draws since digit X last appeared ===
        print("Computing gap & frequency features (this may take a moment)...")
        gap_data = []
        freq_data = []
        # Calculate these features using the current `df['Single']` values, not shifted
        # The shift for prediction happens when building the X_cols
        for i in range(len(df)):
            gaps = compute_gap_features(df['Single'], i)
            freqs = compute_frequency_features(df['Single'], i, window=20)
            gap_data.append(gaps)
            freq_data.append(freqs)
        
        gap_df = pd.DataFrame(gap_data, index=df.index)
        freq_df = pd.DataFrame(freq_data, index=df.index)
        df = pd.concat([df, gap_df, freq_df], axis=1)
        
        # === BAZI-SPECIFIC FREQUENCY (last 30 same-bazi draws) ===
        for d in range(10):
            df[f'Bazi_Freq_{d}'] = 0.0 # Initialize column
        
        for bazi_num in df['Bazi'].unique():
            bazi_mask = df['Bazi'] == bazi_num
            # Operate on a temporary copy to avoid SettingWithCopyWarning
            temp_bazi_df = df.loc[bazi_mask].copy() 
            for d in range(10):
                # Calculate frequency based on previous 30 values within this bazi group
                temp_bazi_df[f'Bazi_Freq_{d}'] = (temp_bazi_df['Single'] == d).astype(int).shift(1).rolling(window=30, min_periods=1).mean().fillna(0.1)
            df.loc[bazi_mask, [f'Bazi_Freq_{d}' for d in range(10)]] = temp_bazi_df[[f'Bazi_Freq_{d}' for d in range(10)]].values
        
        # === STREAK FEATURES ===
        df['Same_As_Prev'] = (df['Single'].shift(1) == df['Single'].shift(2)).astype(int).fillna(0) # Shifted for prediction
        
        # Repeat_In_3 (Previous 3 unique singles for target prediction)
        df['Unique_In_Last_3'] = 0
        df['Single_lag1'] = df['Single'].shift(1)
        df['Single_lag2'] = df['Single'].shift(2)
        df['Single_lag3'] = df['Single'].shift(3) 

        def count_unique_for_prediction(row):
            s = set()
            if pd.notna(row['Single_lag1']): s.add(row['Single_lag1'])
            if pd.notna(row['Single_lag2']): s.add(row['Single_lag2'])
            if pd.notna(row['Single_lag3']): s.add(row['Single_lag3'])
            return len(s)
        
        df['Unique_In_Last_3'] = df.apply(count_unique_for_prediction, axis=1)
        df['Repeat_In_3'] = (df['Unique_In_Last_3'] < 3).astype(int) # 1 if less than 3 unique values
        
        # We don't need 'Single_lagX' in final features
        df = df.drop(columns=['Single_lag1', 'Single_lag2', 'Single_lag3'])

        original_df = df.copy()
        
        # === BUILD FEATURE MATRIX ===
        # The actual target for machine learning is df['Single']
        # All features must be based on values PRIOR to the target
        
        # Define base features that don't need shifting or are already defined as "previous"
        base_features = (
            ['Date', 'Date_Obj', 'Bazi', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos', 'Is_Weekend',
             'MA_7', 'MA_30', 'STD_7', 'Patti_MA_7', 
             'Rolling_Even_5', 'Prev_Day_Same_Bazi', 'Same_As_Prev', 'Repeat_In_3']
            + [f'Gap_{d}' for d in range(10)]
            + [f'Freq_{d}' for d in range(10)]
            + [f'Bazi_Freq_{d}' for d in range(10)]
        )
        
        # These features relate to the 'Patti' and 'Single' of the PREVIOUS DRAW
        lag_features_for_X = [
            'Patti_D1', 'Patti_D2', 'Patti_D3', # These are current draw's patti digits, for prediction they should be prev
            'Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_4_Single', 'Prev_5_Single',
            'Patti_Sum' # This is current draw's patti sum, for prediction it should be prev
        ]
        
        # Apply shifts to create predictive features
        features_df = df[base_features].copy()
        
        features_df['Patti_D1'] = df['Patti_D1'].shift(1)
        features_df['Patti_D2'] = df['Patti_D2'].shift(1)
        features_df['Patti_D3'] = df['Patti_D3'].shift(1)
        
        features_df['Prev_1_Single'] = df['Single'].shift(1)
        features_df['Prev_2_Single'] = df['Single'].shift(2)
        features_df['Prev_3_Single'] = df['Single'].shift(3)
        features_df['Prev_4_Single'] = df['Single'].shift(4)
        features_df['Prev_5_Single'] = df['Single'].shift(5)
        features_df['Prev_Patti_Sum'] = df['Patti_Sum'].shift(1)
        
        features_df['Target_Single'] = df['Single'] # This is the target for ML
        
        features_df = features_df.dropna() # Drop rows that don't have all features due to shifting
        
        # The offset here needs to align with the maximum shift or window size used for features
        # e.g., if a feature uses a 30-day rolling window, the first 30 rows cannot be used for prediction.
        # Adjusted min_offset to account for all features including rolling/gap/freq calculation setup
        min_offset = max(
            30,  # MA_30, Bazi_Freq_D for 30 days
            50,  # Gap_d max history
            5    # Prev_5_Single
        )
        
        # Further filter `features_df` based on `min_offset`
        features_df = features_df.iloc[min_offset:].reset_index(drop=True)
        original_df = original_df.iloc[min_offset:].reset_index(drop=True) # Align original_df similarly
        
        return features_df, original_df
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None, None

# ============================================================
#              MODEL ARCHITECTURE V3 (ANTI-OVERFIT)
# ============================================================

def get_feature_columns():
    """Single source of truth for feature column names."""
    return (
        ['Bazi', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos', 'Is_Weekend',
         'MA_7', 'MA_30', 'STD_7', 'Patti_MA_7', 'Patti_D1', 'Patti_D2', 'Patti_D3',
         'Rolling_Even_5', 'Prev_Day_Same_Bazi', 'Same_As_Prev', 'Repeat_In_3']
        + [f'Gap_{d}' for d in range(10)]
        + [f'Freq_{d}' for d in range(10)]
        + [f'Bazi_Freq_{d}' for d in range(10)]
        + ['Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_4_Single', 'Prev_5_Single',
           'Prev_Patti_Sum']
    )

def create_ensemble_models():
    """Create 4 diverse models with STRONG regularization to prevent overfitting."""
    
    # XGBoost — low depth, high regularization
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,           # L1 regularization
        reg_lambda=5.0,          # L2 regularization
        min_child_weight=10,     # Minimum samples per leaf
        gamma=1.0,               # Minimum loss reduction
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False  # Suppress the deprecation warning
    )
    
    # LightGBM — complementary to XGB
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=5.0,
        min_child_samples=15,
        random_state=42,
        verbose=-1
    )
    
    # Random Forest — lower depth, balanced
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=8,
        min_samples_split=15,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    
    # Gradient Boosting — sklearn variant, different algorithm
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        min_samples_leaf=10,
        random_state=42
    )
    
    return {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'rf': rf_model,
        'gb': gb_model
    }

def build_frequency_model(df):
    """Build a frequency-based model that captures number distribution biases."""
    freq_model = {}
    
    # Overall frequency
    overall_counts = df['Single'].value_counts(normalize=True).reindex(range(10), fill_value=0.01)
    freq_model['overall'] = {d: overall_counts.get(d, 0.01) for d in range(10)}
    
    # Bazi-specific frequency
    for bazi in range(1, 9):
        bazi_data = df[df['Bazi'] == bazi]
        if len(bazi_data) > 10:
            bazi_counts = bazi_data['Single'].value_counts(normalize=True).reindex(range(10), fill_value=0.01)
            freq_model[f'bazi_{bazi}'] = {d: bazi_counts.get(d, 0.01) for d in range(10)}
        else:
            freq_model[f'bazi_{bazi}'] = freq_model['overall']
    
    # Day-specific frequency
    for day in range(7):
        day_data = df[df['Date_Obj'].dt.dayofweek == day]
        if len(day_data) > 20:
            day_counts = day_data['Single'].value_counts(normalize=True).reindex(range(10), fill_value=0.01)
            freq_model[f'day_{day}'] = {d: day_counts.get(d, 0.01) for d in range(10)}
        else:
            freq_model[f'day_{day}'] = freq_model['overall']
    
    return freq_model

# ============================================================
#               TRAINING & VALIDATION
# ============================================================

def generate_oos_stats(features):
    """True Out-of-Sample validation using TimeSeriesSplit."""
    features = features.sort_values(by=['Date_Obj', 'Bazi']).reset_index(drop=True)
    unique_dates = features['Date_Obj'].dt.date.unique()
    
    if len(unique_dates) < 21: # Need at least 3 weeks of data for meaningful 21-day OOS
        print("Warning: Not enough unique dates for a robust OOS validation (need >= 21 days).")
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0, "oos_accuracy_pct": 0.0}
    
    X_cols = get_feature_columns()
    
    # Use last 21 days as test, rest as train (proper temporal split)
    cutoff_date = unique_dates[-21] 
    
    train_df = features[features['Date_Obj'].dt.date < cutoff_date]
    test_df = features[features['Date_Obj'].dt.date >= cutoff_date]
    
    if train_df.empty or test_df.empty:
        print("Error: Train or test set is empty for OOS validation.")
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0, "oos_accuracy_pct": 0.0}

    X_train = train_df[X_cols]
    y_train = train_df['Target_Single']
    X_test = test_df[X_cols]
    y_test = test_df['Target_Single']
    
    # Ensure all possible target classes are present in y_train for consistent model behavior
    for model_name, model in create_ensemble_models().items():
        if hasattr(model, 'classes_'):
            # This is primarily for models like LGBM and XGBoost which fit with specific classes.
            # If a model was trained on [1,2,3] and then infers on [0..9], it might fail.
            # Here we just make sure y_train has enough diversity.
            pass

    # Train individual models (no stacking during validation — avoids data leakage)
    models_oos = create_ensemble_models()
    all_probs = []
    
    for name, model in models_oos.items():
        # Handle potential for unseen classes in test data for tree-based models if not all 0-9 in train
        # This is less of an issue for classification where all 0-9 are expected outputs.
        model.fit(X_train, y_train)
        
        # Get predictions for all 10 classes
        try:
            probs = model.predict_proba(X_test)
            # Ensure probabilities are for classes 0-9, padding if a class is missing from model.classes_
            full_probs = np.zeros((len(X_test), 10))
            for i, cls in enumerate(model.classes_):
                if cls in range(10): # Ensure class is valid digit
                    full_probs[:, int(cls)] = probs[:, i]
            all_probs.append(full_probs)
        except Exception as e:
            print(f"Warning: {name} failed predict_proba on OOS: {e}. Skipping this model for OOS accuracy.")
            # If a model fails, we cannot use its probabilities, treat it as a uniform distribution or skip
            all_probs.append(np.full((len(X_test), 10), 0.1)) # Fallback to uniform distribution

    if not all_probs: # If all models failed to predict probabilities
        print("Error: No models successfully predicted probabilities for OOS.")
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0, "oos_accuracy_pct": 0.0}
    
    # Average ensembling (simpler, less overfit than stacking)
    ensemble_probs = np.mean(all_probs, axis=0)
    
    is_match_list = []
    for i, true_val in enumerate(y_test.values): # Use .values for consistent indexing
        sorted_indices = ensemble_probs[i].argsort()[::-1][:3]
        is_match_list.append(int(true_val) in sorted_indices)
    
    matches = np.array(is_match_list)
    test_df_copy = test_df.copy()
    test_df_copy['Matched'] = matches
    
    # Getting today and week stats based on test_df, specifically the last day in test_df
    last_date_in_test_df = test_df_copy['Date'].iloc[-1]
    
    today_mask = test_df_copy['Date'] == last_date_in_test_df
    today_matches_count = int(test_df_copy[today_mask]['Matched'].sum())
    today_total = int(today_mask.sum())
    
    unique_dates_in_test = test_df_copy['Date_Obj'].dt.date.unique()
    last_7_dates = unique_dates_in_test[-7:] if len(unique_dates_in_test) >= 7 else unique_dates_in_test
    week_mask = test_df_copy['Date_Obj'].dt.date.isin(last_7_dates)
    week_matches_count = int(test_df_copy[week_mask]['Matched'].sum())
    week_total = int(week_mask.sum())
    
    prev_correct = bool(matches[-1]) if len(matches) > 0 else False
    
    win_streak = 0
    # Iterate backwards through 'matches' to determine streak
    for i in range(len(matches) -1, -1, -1):
        if matches[i]: 
            win_streak += 1
        else: 
            break
            
    lose_streak = 0
    for i in range(len(matches) -1, -1, -1):
        if not matches[i]: 
            lose_streak += 1
        else: 
            break
    
    overall_accuracy = float(matches.mean() * 100) if len(matches) > 0 else 0.0
    
    stats = {
        "today_matches": f"{today_matches_count}/{today_total}",
        "week_matches": f"{week_matches_count}/{week_total}",
        "prev_correct": prev_correct,
        "winning_streak": int(win_streak),
        "losing_streak": int(lose_streak),
        "oos_accuracy_pct": round(overall_accuracy, 1)
    }
    
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        print(f"Error saving OOS stats: {e}")
            
    print(f"OOS Validation: {overall_accuracy:.1f}% top-3 accuracy over {len(matches)} predictions (random baseline: 30%)")
    return stats

def train_and_save_model():
    features, original_df = load_and_preprocess_data()
    if features is None or len(features) < 100: # Increased minimum data for meaningful training
        print("Not enough data to train advanced ML model.")
        return False
    
    # Generate OOS stats first
    print("Generating pure OOS Validation Stats...")
    generate_oos_stats(features)
    
    X_cols = get_feature_columns()
    X = features[X_cols]
    y = features['Target_Single']
    
    # Reindex y to ensure it aligns with X after feature engineering
    y = y.loc[X.index] 

    # Train all 4 models on full data
    print(f"Training 4-Model Ensemble on {len(X)} records...")
    models = create_ensemble_models()
    trained_models = {}
    
    for name, model in models.items():
        if len(np.unique(y)) < 10:
            print(f"Warning: Not all 10 digits (0-9) present in training target 'y'. This might affect model {name}.")
        model.fit(X, y)
        trained_models[name] = model
        print(f"  [OK] {name.upper()} trained")
    
    # Build frequency model
    freq_model = build_frequency_model(original_df) # Use original_df for frequency to capture true distributions
    
    # Save everything
    save_package = {
        'models': trained_models,
        'freq_model': freq_model,
        'feature_columns': X_cols # Save feature columns for consistency during prediction
    }
    
    try:
        joblib.dump(save_package, MODEL_FILE)
        print(f"V3 Deep Ensemble (XGB+LGB+RF+GB + Freq) trained on {len(X)} records and saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

# ============================================================
#               LIVE PREDICTION & STATS
# ============================================================

def blend_predictions(ml_probs, freq_model, bazi, day_of_week, recent_singles=None):
    """Blend Machine Learning probabilities with Frequency model and Vector Memory."""
    
    # Load dynamic weights from config
    config = load_ai_config()
    ml_weight = config.get("ml_weight", DEFAULT_AI_CONFIG["ml_weight"])
    memory_weight = config.get("memory_weight", DEFAULT_AI_CONFIG["memory_weight"])
    freq_weight = config.get("freq_weight", DEFAULT_AI_CONFIG["freq_weight"])
    
    # Normalize weights just in case
    total = ml_weight + memory_weight + freq_weight
    if total > 0:
        ml_weight /= total
        memory_weight /= total
        freq_weight /= total
    else: # Fallback if all weights are zero or negative
        ml_weight, memory_weight, freq_weight = DEFAULT_AI_CONFIG["ml_weight"], DEFAULT_AI_CONFIG["memory_weight"], DEFAULT_AI_CONFIG["freq_weight"]
    
    # ML prediction (ensemble average)
    ml_dist = ml_probs.copy()
    
    # Frequency prediction: combine overall + bazi + day priors
    freq_dist = np.zeros(10)
    overall = freq_model.get('overall', {d: 0.1 for d in range(10)})
    bazi_freq = freq_model.get(f'bazi_{bazi}', overall)
    day_freq = freq_model.get(f'day_{day_of_week}', overall)
    
    for d in range(10):
        # Weighted average of frequencies, ensure non-zero fallback for missing keys
        freq_dist[d] = (overall.get(d, 0.01) * 0.3 + bazi_freq.get(d, 0.01) * 0.4 + day_freq.get(d, 0.01) * 0.3)
    
    # Normalize frequency distribution
    freq_dist = freq_dist / (freq_dist.sum() + 1e-9) # Add epsilon to avoid division by zero
    
    # Vector Memory prediction (AI Brain)
    memory_dist = np.zeros(10)
    # Check if brain has enough context and capacity
    if recent_singles is not None and brain.get_brain_capacity() >= brain.context_size:
        memory_boost_raw = brain.get_prediction_boost(recent_singles, bazi)
        if memory_boost_raw is not None and sum(memory_boost_raw) > 0:
            memory_dist = np.array(memory_boost_raw)
            memory_dist = memory_dist / (memory_dist.sum() + 1e-9)
        else:
            memory_weight = 0.0 # If brain signals no boost or all zeros, reduce its weight
    else:
        memory_weight = 0.0 # If not enough recent singles or brain capacity is low, disable memory
        
    # Re-normalize if memory