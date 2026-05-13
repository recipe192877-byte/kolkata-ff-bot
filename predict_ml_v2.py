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
FREQ_FILE = 'freq_model.joblib' # This is deprecated, frequency model is now part of MODEL_FILE
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
        except json.JSONDecodeError: # Handle empty or malformed JSON
            print(f"Warning: {CONFIG_FILE} is corrupted or empty. Using default config.")
            return DEFAULT_AI_CONFIG
        except Exception as e:
            print(f"Error loading AI config: {e}. Using default config.")
            return DEFAULT_AI_CONFIG
    return DEFAULT_AI_CONFIG

def save_ai_config(config):
    """Save updated AI weights and settings."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving AI config: {e}")

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
        # Ensure it's treated as a 3-digit string, padding with 0s if necessary
        s = str(int(float(patti_val))).zfill(3)
        return int(s[0]), int(s[1]), int(s[2])
    except (ValueError, TypeError):
        return 0, 0, 0

def compute_gap_features(singles_series, current_idx):
    """Compute how many draws since each digit (0-9) last appeared."""
    gaps = {}
    for d in range(10):
        # Look backwards from current_idx, up to 50 previous draws
        found = False
        # Use .iloc just in case index is not sequential
        for gap in range(1, min(current_idx + 1, 51)):
            if singles_series.iloc[current_idx - gap] == d:
                gaps[f'Gap_{d}'] = gap
                found = True
                break
        if not found:
            gaps[f'Gap_{d}'] = 50  # Max gap value if not found in last 50
    return gaps

def compute_frequency_features(singles_series, current_idx, window=20):
    """Compute frequency of each digit in last N draws."""
    start = max(0, current_idx - window)
    recent = singles_series.iloc[start:current_idx]
    freq = {}
    counts = recent.value_counts(normalize=True) # Normalize to get frequencies directly
    for d in range(10):
        freq[f'Freq_{d}'] = counts.get(d, 0) # Default to 0 if digit not found
    return freq

def load_and_preprocess_data(filepath=DATA_FILE):
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['Single', 'Patti']) # Ensure both are present
        df['Single'] = df['Single'].astype(int)
        
        # Use dd/mm/yyyy explicitly for consistency
        df['Date_Obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date_Obj']) # Drop rows where date parsing failed
        
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
        patti_digits_df = df['Patti'].apply(lambda x: pd.Series(extract_patti_digits(x)))
        patti_digits_df.columns = ['Patti_D1', 'Patti_D2', 'Patti_D3']
        df = pd.concat([df, patti_digits_df], axis=1)

        # === ROLLING / LAG FEATURES ===
        df['Is_Even'] = (df['Single'] % 2 == 0).astype(int)
        # Shift to prevent data leakage (use previous data)
        df['Rolling_Even_5'] = df['Is_Even'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0.5)
        
        # Corrected Prev_Day_Same_Bazi: ensure it looks for previous _day's_ result for the same Bazi
        df['Prev_Day_Same_Bazi'] = df.groupby('Bazi')['Single'].shift(8).fillna(df['Single'].shift(1)).fillna(0) # shift by 8 (max bazi-count) to get previous day same bazi
        
        # Moving Averages - shifted to prevent leakage
        df['MA_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).mean().fillna(4.5)
        df['MA_30'] = df['Single'].shift(1).rolling(window=30, min_periods=1).mean().fillna(4.5)
        df['STD_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).std().fillna(df['Single'].std()) # Use overall std if not enough data
        df['Patti_MA_7'] = df['Patti_Sum'].shift(1).rolling(window=7, min_periods=1).mean().fillna(df['Patti_Sum'].mean())
        
        # === GAP FEATURES: How many draws since digit X last appeared ===
        print("Computing gap & frequency features (this may take a moment)...")
        gap_data = []
        freq_data = []
        # Min index for reliable feature calculation
        min_reliable_idx = max(50, 20) # for gaps and freqs calculation
        
        for i in range(len(df)):
            if i < min_reliable_idx: # Not enough history for initial rows
                gaps = {f'Gap_{d}': 50 for d in range(10)}
                # Using 0.1 for initial frequencies or uniform distribution proxy
                freqs = {f'Freq_{d}': 0.1 for d in range(10)} 
            else:
                gaps = compute_gap_features(df['Single'], i)
                freqs = compute_frequency_features(df['Single'], i, window=20)
            gap_data.append(gaps)
            freq_data.append(freqs)
        
        gap_df = pd.DataFrame(gap_data, index=df.index)
        freq_df = pd.DataFrame(freq_data, index=df.index)
        df = pd.concat([df, gap_df, freq_df], axis=1)
        
        # === BAZI-SPECIFIC FREQUENCY (last 30 same-bazi draws) ===
        # Initialize columns
        for d in range(10):
            df[f'Bazi_Freq_{d}'] = np.nan # Use NaN to distinguish from calculated zeros
        
        for bazi_num in df['Bazi'].unique():
            bazi_mask = df['Bazi'] == bazi_num
            bazi_df_temp = df[bazi_mask].copy() # Work on a copy to avoid SettingWithCopyWarning
            for d in range(10):
                # Calculate frequency using shifted 'Single' to prevent leakage
                bazi_df_temp[f'Bazi_Freq_{d}'] = (bazi_df_temp['Single'].shift(1) == d).astype(int).rolling(window=30, min_periods=1).mean()
            # Update the original df with calculated frequencies
            df.loc[bazi_mask, [f'Bazi_Freq_{d}' for d in range(10)]] = bazi_df_temp[[f'Bazi_Freq_{d}' for d in range(10)]].values
        
        # Fill remaining NaNs (e.g., for early records in a Bazi) with a default (e.g., 0.1 for uniform distribution)
        for d in range(10):
            df[f'Bazi_Freq_{d}'] = df[f'Bazi_Freq_{d}'].fillna(0.1)

        # === STREAK FEATURES ===
        df['Same_As_Prev'] = (df['Single'] == df['Single'].shift(1)).astype(int)
        
        # A more efficient way to compute 'Repeat_In_3'
        df['Single_lag1'] = df['Single'].shift(1)
        df['Single_lag2'] = df['Single'].shift(2)
        df['Repeat_In_3'] = ((df['Single_lag1'] == df['Single_lag2']) | \
                             (df['Single_lag1'] == df['Single_lag3']) | \
                             (df['Single_lag2'] == df['Single_lag3'])).astype(int) # This was missing Single_lag3
        df['Single_lag3'] = df['Single'].shift(3) # define it here for repeat in 3
        df['Repeat_In_3'] = ((df['Single_lag1'] == df['Single_lag2']) |
                             (df['Single_lag1'] == df['Single_lag3']) |
                             (df['Single_lag2'] == df['Single_lag3'])).astype(int)
        df.drop(columns=['Single_lag1', 'Single_lag2', 'Single_lag3'], inplace=True) # clean up
        
        original_df = df.copy() # Keep a copy before dropping rows
        
        # === BUILD FEATURE MATRIX ===
        feature_cols_base = (
            ['Date', 'Date_Obj', 'Bazi', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos', 'Is_Weekend',
             'MA_7', 'MA_30', 'STD_7', 'Patti_MA_7', 'Patti_D1', 'Patti_D2', 'Patti_D3',
             'Rolling_Even_5', 'Prev_Day_Same_Bazi', 'Same_As_Prev', 'Repeat_In_3']
            + [f'Gap_{d}' for d in range(10)]
            + [f'Freq_{d}' for d in range(10)]
            + [f'Bazi_Freq_{d}' for d in range(10)]
        )
        
        features = df[feature_cols_base].copy()
        
        # Lag features should also be available at the point of prediction
        features['Prev_1_Single'] = df['Single'].shift(1)
        features['Prev_2_Single'] = df['Single'].shift(2)
        features['Prev_3_Single'] = df['Single'].shift(3)
        features['Prev_4_Single'] = df['Single'].shift(4)
        features['Prev_5_Single'] = df['Single'].shift(5)
        
        features['Prev_Patti_Sum'] = df['Patti_Sum'].shift(1)
        features['Prev_Patti_D1'] = df['Patti_D1'].shift(1)
        features['Prev_Patti_D2'] = df['Patti_D2'].shift(1)
        features['Prev_Patti_D3'] = df['Patti_D3'].shift(1)
        
        features['Target_Single'] = df['Single'] # The target variable
        
        # Drop rows with any NaN values that resulted from shifting or initial feature calculation
        # This is where we ensure all features are valid for ALL rows we use.
        features = features.dropna()
        
        # Align original_df with features after dropping NaNs
        original_df = original_df.loc[features.index].reset_index(drop=True)
        features = features.reset_index(drop=True)
        
        return features, original_df
        
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
           'Prev_Patti_Sum', 'Prev_Patti_D1', 'Prev_Patti_D2', 'Prev_Patti_D3']
    )

def create_ensemble_models():
    """Create 4 diverse models with STRONG regularization to prevent overfitting."""
    
    # XGBoost — low depth, high regularization
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, # Increased estimators
        max_depth=4,      # Slightly increased depth
        learning_rate=0.03, # Slightly increased learning rate
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,           # L1 regularization
        reg_lambda=5.0,          # L2 regularization
        min_child_weight=10,     # Minimum samples per leaf
        gamma=1.0,               # Minimum loss reduction
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False # Suppress warning
    )
    
    # LightGBM — complementary to XGB
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,       # Increased estimators
        max_depth=4,            # Slightly increased depth
        learning_rate=0.03,     # Slightly increased learning rate
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
        n_estimators=250, # Increased estimators
        max_depth=5,      # Slightly increased depth
        min_samples_leaf=8,
        min_samples_split=15,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced', # Important for imbalanced classes
        n_jobs=-1 # Use all available cores
    )
    
    # Gradient Boosting — sklearn variant, different algorithm
    gb_model = GradientBoostingClassifier(
        n_estimators=150, # Increased estimators
        max_depth=4,      # Slightly increased depth
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
    # Add a small smoothing factor (Laplace smoothing) to prevent zero probabilities
    alpha = 1
    overall_counts = df['Single'].value_counts()
    total_overall = overall_counts.sum() + alpha * 10 # Add alpha for each possible digit (0-9)
    freq_model['overall'] = {d: (overall_counts.get(d, 0) + alpha) / total_overall for d in range(10)}
    
    # Bazi-specific frequency
    for bazi in range(1, 9):
        bazi_data = df[df['Bazi'] == bazi]
        if len(bazi_data) > 10:
            bazi_counts = bazi_data['Single'].value_counts()
            total_bazi = bazi_counts.sum() + alpha * 10
            freq_model[f'bazi_{bazi}'] = {d: (bazi_counts.get(d, 0) + alpha) / total_bazi for d in range(10)}
        else:
            freq_model[f'bazi_{bazi}'] = freq_model['overall'] # Fallback to overall
    
    # Day-specific frequency
    for day in range(7):
        day_data = df[df['Date_Obj'].dt.dayofweek == day]
        if len(day_data) > 20: # More data required for reliable day-specific stats
            day_counts = day_data['Single'].value_counts()
            total_day = day_counts.sum() + alpha * 10
            freq_model[f'day_{day}'] = {d: (day_counts.get(d, 0) + alpha) / total_day for d in range(10)}
        else:
            freq_model[f'day_{day}'] = freq_model['overall'] # Fallback to overall
    
    return freq_model

# ============================================================
#               TRAINING & VALIDATION
# ============================================================

def generate_oos_stats(features):
    """True Out-of-Sample validation using TimeSeriesSplit."""
    features = features.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True])
    unique_dates = features['Date_Obj'].dt.date.unique()
    
    if len(unique_dates) < 21: # Need enough dates for a meaningful 21-day OOS test
        print("Warning: Not enough unique dates for deep OOS validation. Skipping.")
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0, "oos_accuracy_pct": 0.0}
    
    X_cols = get_feature_columns()
    
    # Use last 21 days as test, rest as train (proper temporal split)
    # The cutoff should be based on the start of the 21-day test period
    cutoff_date = unique_dates[-21]
    
    train_df = features[features['Date_Obj'].dt.date < cutoff_date]
    test_df = features[features['Date_Obj'].dt.date >= cutoff_date]
    
    if train_df.empty or test_df.empty:
        print("Warning: Train or test set is empty during OOS validation. Skipping.")
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0, "oos_accuracy_pct": 0.0}
    
    X_train = train_df[X_cols]
    y_train = train_df['Target_Single']
    X_test = test_df[X_cols]
    y_test = test_df['Target_Single']
    
    # Ensure X_train and X_test have the same columns in the same order
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0) # fill with 0 for new cols
    X_train = X_train[X_cols] # Re-filter to ensure only expected cols remain
    X_test = X_test[X_cols]
    
    # Train individual models (no stacking during validation — avoids data leakage)
    models = create_ensemble_models()
    all_ml_probs = []
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            
            # Ensure all classes 0-9 are represented, even if a model didn't predict them in training
            full_probs = np.zeros((len(X_test), 10))
            # Map model.classes_ to column indices correctly
            for i, cls in enumerate(model.classes_):
                if 0 <= int(cls) <= 9:
                    full_probs[:, int(cls)] = probs[:, i]
            all_ml_probs.append(full_probs)
        except Exception as e:
            print(f"Error training/predicting with {name} during OOS validation: {e}")
            continue # Skip this model if it fails
    
    if not all_ml_probs: # If all models failed to train
        print("Error: No models could be trained for OOS validation.")
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0, "oos_accuracy_pct": 0.0}
        
    # Average ensembling (simpler, less overfit than stacking)
    ensemble_ml_probs = np.mean(all_ml_probs, axis=0)
    
    # Re-build frequency model on the training data for fair application
    freq_model = build_frequency_model(train_df[['Single', 'Bazi', 'Date_Obj']])

    # Apply blending for OOS predictions
    blended_matches = []
    for i in range(len(test_df)):
        bazi = int(test_df.iloc[i]['Bazi'])
        day_of_week = test_df.iloc[i]['Date_Obj'].dayofweek
        
        # Get recent singles for vector memory from training data + already predicted test data
        recent_singles_for_brain = []
        if i > 0: # Get singles from previous test predictions and training data
            current_date_obj = test_df.iloc[i]['Date_Obj']
            # Look up to 5 previous results, prioritize from actuals before current test split
            temp_df_for_brain = pd.concat([train_df, test_df.iloc[:i]])
            recent_singles_for_brain = temp_df_for_brain['Single'].tail(5).values.tolist()
            
        blended_probs = blend_predictions(ensemble_ml_probs[i], freq_model, bazi, day_of_week, recent_singles=recent_singles_for_brain)
        
        sorted_indices = blended_probs.argsort()[::-1][:3]
        blended_matches.append(int(y_test.iloc[i]) in sorted_indices)
    
    matches = np.array(blended_matches)
    
    test_df_copy = test_df.copy()
    test_df_copy['Matched'] = matches
    
    # Use timezone-aware current time for 'today_str'
    kolkata_tz = timezone(timedelta(hours=5, minutes=30))
    today_kolkata = datetime.now(kolkata_tz).strftime('%d/%m/%Y')
    
    last_date_in_test_df = test_df_copy['Date'].iloc[-1]
    
    # Determine today's matches based on the last recorded date in the test set.
    # This might be yesterday or older if the data is not fully up-to-date.
    today_mask = (test_df_copy['Date'] == last_date_in_test_df)
    today_matches_count = int(test_df_copy[today_mask]['Matched'].sum())
    today_total = int(today_mask.sum())
    
    unique_test_dates = test_df_copy['Date'].unique()
    last_7_test_dates = unique_test_dates[-7:] if len(unique_test_dates) >= 7 else unique_test_dates
    week_mask = test_df_copy['Date'].isin(last_7_test_dates)
    week_matches_count = int(test_df_copy[week_mask]['Matched'].sum())
    week_total = int(week_mask.sum())
    
    prev_correct = bool(matches[-1]) if len(matches) > 0 else False
    
    win_streak = 0
    # Iterate backwards from the _end_ of the matches list
    for m in reversed(matches):
        if m: win_streak += 1
        else: break
    
    lose_streak = 0
    # Iterate backwards from the _end_ of the matches list
    for m in reversed(matches):
        if not m: lose_streak += 1
        else: break
    
    overall_accuracy = float(matches.mean() * 100) if len(matches) > 0 else 0.0
    
    stats = {
        "today_matches": f"{today_matches_count}/{today_total}",
        "week_matches": f"{week_matches_count}/{week_total}",
        "prev_correct": prev_correct,
        "winning_streak": int(win_streak),
        "losing_streak": int(lose_streak),
        "oos_accuracy_pct": round(overall_accuracy, 1)
    }
    
    # Store OOS stats for general info
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        print(f"Error saving OOS stats: {e}")
        
    print(f"OOS Validation: {overall_accuracy:.1f}% top-3 accuracy over {len(matches)} predictions (random baseline: 30%)")
    return stats

def train_and_save_model():
    features, original_df = load_and_preprocess_data()
    if features is None or len(features) < 100: # Increased minimum data requirement
        print("Not enough data to train advanced ML model. Need at least 100 processed records.")
        return False
    
    # Generate OOS stats first - this helps in evaluating the model without seeing the future data
    print("Generating pure OOS Validation Stats...")
    generate_oos_stats(features.copy()) # Pass a copy to avoid in-place modifications
    
    X_cols = get_feature_columns()
    X = features[X_cols]
    y = features['Target_Single']
    
    # Handle potential missing columns in X if get_feature_columns adds new ones not in the loaded data
    # (e.g. if new features are added but not in the historical CSV for older records)
    missing_cols = set(X_cols) - set(X.columns)
    for c in missing_cols:
        X[c] = 0 # Add missing columns with a default value (e.g. 0)
    X = X[X_cols] # Ensure column order
    
    # Train all 4 models on full data
    print(f"Training 4-Model Ensemble on {len(X)} records...")
    models = create_ensemble_models()
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"  [OK] {name.upper()} trained")
        except Exception as e:
            print(f"Error training model {name}: {e}. Skipping this model.")
            # If a model fails, it won't be included in trained_models
            
    if not trained_models:
        print("Error: No models were successfully trained.")
        return False
            
    # Build frequency model using the full original_df
    freq_model = build_frequency_model(original_df)
    
    # Save everything
    save_package = {
        'models': trained_models,
        'freq_model': freq_model
    }
    
    joblib.dump(save_package, MODEL_FILE)
    print(f"V3 Deep Ensemble (XGB+LGB+RF+GB + Freq) trained on {len(X)} records and saved successfully.")
    return True

# ============================================================
#               LIVE PREDICTION & STATS
# ============================================================

def blend_predictions(ml_probs, freq_model, bazi, day_of_week, recent_singles=None):
    """Blend Machine Learning probabilities with Frequency model and Vector Memory."""
    
    config = load_ai_config()
    ml_weight = config.get("ml_weight", DEFAULT_AI_CONFIG["ml_weight"])
    memory_weight = config.get("memory_weight", DEFAULT_AI_CONFIG["memory_weight"])
    freq_weight = config.get("freq_weight", DEFAULT_AI_CONFIG["freq_weight"])
    
    # Sanitize inputs to ml_probs, ensure it's a valid probability distribution
    if not isinstance(ml_probs, np.ndarray) or ml_probs.shape != (10,) or ml_probs.sum() == 0:
        ml_probs = np.full(10, 0.1) # Default to uniform if invalid
    ml_dist = ml_probs / ml_probs.sum() # Normalize to ensure it's a distribution
    
    # Frequency prediction: combine overall + bazi + day priors
    freq_dist_values = np.zeros(10)
    overall_freq_map = freq_model.get('overall', {d: 0.1 for d in range(10)})
    bazi_freq_map = freq_model.get(f'bazi_{bazi}', overall_freq_map)
    day_freq_map = freq_model.get(f'day_{day_of_week}', overall_freq_map)
    
    for d in range(10):
        # Use average of overall, bazi, day frequencies
        freq_dist_values[d] = (overall_freq_map.get(d, 0.1) + bazi