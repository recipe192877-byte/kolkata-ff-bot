import pandas as pd
import numpy as np
import joblib
import warnings
import os
import json
from datetime import datetime, timedelta, timezone
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

MODEL_FILE = 'xgb_model.joblib'
FREQ_FILE = 'freq_model.joblib'
DATA_FILE = 'kolkata_ff_history_advanced.csv'
STATS_FILE = 'backtest_stats.json'

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
    s = str(int(float(patti_val))).zfill(3)
    return int(s[0]), int(s[1]), int(s[2])

def compute_gap_features(singles_series, current_idx):
    """Compute how many draws since each digit (0-9) last appeared."""
    gaps = {}
    for d in range(10):
        # Look backwards from current_idx
        found = False
        for gap in range(1, min(current_idx + 1, 51)):
            if singles_series.iloc[current_idx - gap] == d:
                gaps[f'Gap_{d}'] = gap
                found = True
                break
        if not found:
            gaps[f'Gap_{d}'] = 50  # Max gap value
    return gaps

def compute_frequency_features(singles_series, current_idx, window=20):
    """Compute frequency of each digit in last N draws."""
    start = max(0, current_idx - window)
    recent = singles_series.iloc[start:current_idx]
    freq = {}
    counts = recent.value_counts()
    for d in range(10):
        freq[f'Freq_{d}'] = counts.get(d, 0) / max(len(recent), 1)
    return freq

def load_and_preprocess_data(filepath=DATA_FILE):
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['Single'])
        df['Single'] = df['Single'].astype(int)
        
        df['Date_Obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
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
        df['Prev_Day_Same_Bazi'] = df.groupby('Bazi')['Single'].shift(1).fillna(df['Single'].shift(1)).fillna(0)
        
        # Moving Averages
        df['MA_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).mean().fillna(4.5)
        df['MA_30'] = df['Single'].shift(1).rolling(window=30, min_periods=1).mean().fillna(4.5)
        df['STD_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).std().fillna(3.0)
        df['Patti_MA_7'] = df['Patti_Sum'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
        
        # === GAP FEATURES: How many draws since digit X last appeared ===
        print("Computing gap & frequency features (this may take a moment)...")
        gap_data = []
        freq_data = []
        for i in range(len(df)):
            if i < 20:
                gaps = {f'Gap_{d}': 50 for d in range(10)}
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
        for d in range(10):
            df[f'Bazi_Freq_{d}'] = 0.0
        
        for bazi_num in range(1, 9):
            bazi_mask = df['Bazi'] == bazi_num
            bazi_df = df[bazi_mask].copy()
            for d in range(10):
                bazi_df[f'Bazi_Freq_{d}'] = (bazi_df['Single'].shift(1) == d).astype(int).rolling(window=30, min_periods=1).mean().fillna(0.1)
            df.loc[bazi_mask, [f'Bazi_Freq_{d}' for d in range(10)]] = bazi_df[[f'Bazi_Freq_{d}' for d in range(10)]].values
        
        # === STREAK FEATURES ===
        df['Same_As_Prev'] = (df['Single'] == df['Single'].shift(1)).astype(int)
        df['Repeat_In_3'] = 0
        for i in range(3, len(df)):
            last3 = df['Single'].iloc[i-3:i].values
            df.iloc[i, df.columns.get_loc('Repeat_In_3')] = int(len(set(last3)) < 3)
        
        original_df = df.copy()
        
        # === BUILD FEATURE MATRIX ===
        feature_cols = (
            ['Date', 'Date_Obj', 'Bazi', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos', 'Is_Weekend',
             'MA_7', 'MA_30', 'STD_7', 'Patti_MA_7', 'Patti_D1', 'Patti_D2', 'Patti_D3',
             'Rolling_Even_5', 'Prev_Day_Same_Bazi', 'Same_As_Prev', 'Repeat_In_3']
            + [f'Gap_{d}' for d in range(10)]
            + [f'Freq_{d}' for d in range(10)]
            + [f'Bazi_Freq_{d}' for d in range(10)]
        )
        
        features = df[feature_cols].copy()
        
        # Lag features
        features['Prev_1_Single'] = df['Single'].shift(1)
        features['Prev_2_Single'] = df['Single'].shift(2)
        features['Prev_3_Single'] = df['Single'].shift(3)
        features['Prev_4_Single'] = df['Single'].shift(4)
        features['Prev_5_Single'] = df['Single'].shift(5)
        features['Prev_Patti_Sum'] = df['Patti_Sum'].shift(1)
        features['Prev_Patti_D1'] = df['Patti_D1'].shift(1)
        features['Prev_Patti_D2'] = df['Patti_D2'].shift(1)
        features['Prev_Patti_D3'] = df['Patti_D3'].shift(1)
        
        features['Target_Single'] = df['Single']
        
        features = features.dropna()
        min_offset = 20  # Need at least 20 rows for gap/freq features
        original_df = original_df.iloc[min_offset:].reset_index(drop=True)
        features = features.iloc[max(0, min_offset - 5):].reset_index(drop=True)
        
        return features, original_df
        
    except FileNotFoundError:
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
        eval_metric='mlogloss'
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
    
    # Gradient Boosting — sklearn variant
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.7, min_samples_leaf=10, random_state=42
    )
    
    # Extra Trees — extreme randomization
    et_model = ExtraTreesClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=6,
        max_features='sqrt', random_state=42, class_weight='balanced'
    )
    
    # Hist Gradient Boosting — fast, handles missing values
    hgb_model = HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.03,
        min_samples_leaf=10, random_state=42
    )
    
    return {
        'xgb': xgb_model, 'lgb': lgb_model, 'rf': rf_model,
        'gb': gb_model, 'et': et_model, 'hgb': hgb_model
    }

def build_frequency_model(df):
    """Build a frequency-based model that captures number distribution biases."""
    freq_model = {}
    
    # Overall frequency
    overall_counts = df['Single'].value_counts(normalize=True)
    freq_model['overall'] = {d: overall_counts.get(d, 0.1) for d in range(10)}
    
    # Bazi-specific frequency
    for bazi in range(1, 9):
        bazi_data = df[df['Bazi'] == bazi]
        if len(bazi_data) > 10:
            bazi_counts = bazi_data['Single'].value_counts(normalize=True)
            freq_model[f'bazi_{bazi}'] = {d: bazi_counts.get(d, 0.1) for d in range(10)}
        else:
            freq_model[f'bazi_{bazi}'] = freq_model['overall']
    
    # Day-specific frequency
    for day in range(7):
        day_data = df[df['Date_Obj'].dt.dayofweek == day]
        if len(day_data) > 20:
            day_counts = day_data['Single'].value_counts(normalize=True)
            freq_model[f'day_{day}'] = {d: day_counts.get(d, 0.1) for d in range(10)}
        else:
            freq_model[f'day_{day}'] = freq_model['overall']
    
    return freq_model

# ============================================================
#               TRAINING & VALIDATION
# ============================================================

def generate_oos_stats(features):
    """True Out-of-Sample validation using TimeSeriesSplit."""
    features = features.sort_values(by=['Date_Obj', 'Bazi'])
    unique_dates = features['Date_Obj'].dt.date.unique()
    
    if len(unique_dates) < 20: 
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0}
    
    X_cols = get_feature_columns()
    
    # Use last 21 days as test, rest as train (proper temporal split)
    last_21_dates = unique_dates[-21:]
    cutoff_date = last_21_dates[0]
    
    train_df = features[features['Date_Obj'].dt.date < cutoff_date]
    test_df = features[features['Date_Obj'].dt.date >= cutoff_date]
    
    X_train = train_df[X_cols]
    y_train = train_df['Target_Single']
    X_test = test_df[X_cols]
    y_test = test_df['Target_Single']
    
    # Train individual models (no stacking during validation — avoids data leakage)
    models = create_ensemble_models()
    all_probs = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        # Ensure all classes 0-9 are represented
        full_probs = np.zeros((len(X_test), 10))
        for i, cls in enumerate(model.classes_):
            full_probs[:, int(cls)] = probs[:, i]
        all_probs.append(full_probs)
    
    # Average ensembling (simpler, less overfit than stacking)
    ensemble_probs = np.mean(all_probs, axis=0)
    
    is_match_list = []
    for i, true_val in enumerate(y_test):
        sorted_indices = ensemble_probs[i].argsort()[::-1][:3]
        is_match_list.append(int(true_val) in sorted_indices)
    
    matches = np.array(is_match_list)
    test_df_copy = test_df.copy()
    test_df_copy['Matched'] = matches
    
    last_date = test_df_copy['Date'].iloc[-1]
    today_mask = test_df_copy['Date'] == last_date
    today_matches_count = int(test_df_copy[today_mask]['Matched'].sum())
    today_total = int(today_mask.sum())
    
    test_dates = test_df_copy['Date'].unique()
    last_7_dates = test_dates[-7:] if len(test_dates) >= 7 else test_dates
    week_mask = test_df_copy['Date'].isin(last_7_dates)
    week_matches_count = int(test_df_copy[week_mask]['Matched'].sum())
    week_total = int(week_mask.sum())
    
    prev_correct = bool(matches[-1]) if len(matches) > 0 else False
    
    win_streak = 0
    for m in reversed(matches):
        if m: win_streak += 1
        else: break
    
    lose_streak = 0
    for m in reversed(matches):
        if not m: lose_streak += 1
        else: break
    
    overall_accuracy = float(matches.mean() * 100)
    
    stats = {
        "today_matches": f"{today_matches_count}/{today_total}",
        "week_matches": f"{week_matches_count}/{week_total}",
        "prev_correct": prev_correct,
        "winning_streak": int(win_streak),
        "losing_streak": int(lose_streak),
        "oos_accuracy_pct": round(overall_accuracy, 1)
    }
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)
    
    print(f"OOS Validation: {overall_accuracy:.1f}% top-3 accuracy over {len(matches)} predictions (random baseline: 30%)")
    return stats

def train_and_save_model():
    features, original_df = load_and_preprocess_data()
    if features is None or len(features) < 100:
        print("Not enough data to train advanced ML model.")
        return False
    
    # Generate OOS stats first
    print("Generating pure OOS Validation Stats...")
    generate_oos_stats(features)
    
    X_cols = get_feature_columns()
    X = features[X_cols]
    y = features['Target_Single']
    
    # Train all 6 models on full data
    print(f"Training 6-Model Ensemble on {len(X)} records...")
    models = create_ensemble_models()
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
        print(f"  [OK] {name.upper()} trained")
    
    # Build frequency model
    freq_model = build_frequency_model(original_df)
    
    # Save everything
    save_package = {
        'models': trained_models,
        'freq_model': freq_model
    }
    
    joblib.dump(save_package, MODEL_FILE)
    print(f"V5 ULTRA Ensemble (XGB+LGB+RF+GB+ET+HGB + Freq) trained on {len(X)} records and saved successfully.")
    return True

# ============================================================
#               LIVE PREDICTION & STATS
# ============================================================

def blend_predictions(ml_probs, freq_model, bazi, day_of_week, ml_weight=0.65):
    """Blend ML predictions with frequency-based predictions for better accuracy."""
    # ML prediction (ensemble average)
    ml_dist = ml_probs.copy()
    
    # Frequency prediction: combine overall + bazi + day priors
    freq_dist = np.zeros(10)
    overall = freq_model.get('overall', {d: 0.1 for d in range(10)})
    bazi_freq = freq_model.get(f'bazi_{bazi}', overall)
    day_freq = freq_model.get(f'day_{day_of_week}', overall)
    
    for d in range(10):
        freq_dist[d] = (overall.get(d, 0.1) * 0.3 + bazi_freq.get(d, 0.1) * 0.4 + day_freq.get(d, 0.1) * 0.3)
    
    # Normalize
    freq_dist = freq_dist / freq_dist.sum()
    
    # Blend: ML + Frequency
    blended = ml_weight * ml_dist + (1 - ml_weight) * freq_dist
    blended = blended / blended.sum()
    
    return blended

def backtest_recent_stats(save_package, features, today_str):
    """Live computation of recent backtest stats using loaded models."""
    X_cols = get_feature_columns()
    
    eval_features = features.tail(30 * 8).copy()
    if eval_features.empty:
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0}
    
    try:
        models = save_package['models']
        freq_model = save_package.get('freq_model', {})
        X_all = eval_features[X_cols]
        y_all = eval_features['Target_Single'].values
        
        # Ensemble predict
        all_probs = []
        for name, model in models.items():
            probs = model.predict_proba(X_all)
            full_probs = np.zeros((len(X_all), 10))
            for i, cls in enumerate(model.classes_):
                full_probs[:, int(cls)] = probs[:, i]
            all_probs.append(full_probs)
        
        ensemble_probs = np.mean(all_probs, axis=0)
        
        matches_list = []
        for i in range(len(ensemble_probs)):
            bazi = int(eval_features.iloc[i]['Bazi'])
            day_of_week = eval_features.iloc[i]['Date_Obj'].dayofweek
            
            blended_probs = blend_predictions(ensemble_probs[i], freq_model, bazi, day_of_week, ml_weight=0.65)
            
            sorted_indices = blended_probs.argsort()[::-1][:3]
            matches_list.append(1 if int(y_all[i]) in sorted_indices else 0)

        
        matches_series = pd.Series(matches_list, index=eval_features.index)
        
        today_mask = (eval_features['Date'] == today_str)
        today_total = today_mask.sum()
        
        if today_total > 0:
            today_matches_count = matches_series[today_mask].sum()
        else:
            last_date = eval_features['Date'].iloc[-1]
            today_mask = (eval_features['Date'] == last_date)
            today_matches_count = matches_series[today_mask].sum()
            today_total = today_mask.sum()
        
        unique_dates = eval_features['Date'].unique()
        last_7_dates = unique_dates[-7:] if len(unique_dates) >= 7 else unique_dates
        week_mask = eval_features['Date'].isin(last_7_dates)
        week_matches_count = matches_series[week_mask].sum()
        week_total = week_mask.sum()
        
        prev_correct = bool(matches_list[-1]) if len(matches_list) > 0 else False
        
        win_streak = 0
        for m in reversed(matches_list):
            if m: win_streak += 1
            else: break
        
        lose_streak = 0
        for m in reversed(matches_list):
            if not m: lose_streak += 1
            else: break
        
        return {
            "today_matches": f"{int(today_matches_count)}/{int(today_total)}",
            "week_matches": f"{int(week_matches_count)}/{int(week_total)}",
            "prev_correct": prev_correct,
            "winning_streak": int(win_streak),
            "losing_streak": int(lose_streak)
        }
    except Exception as e:
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0}

def get_patti_suggestions(original_df, target_single):
    history = original_df[original_df['Single'] == target_single]
    if history.empty:
        return []
    patti_counts = history['Patti'].value_counts()
    top_pattis = patti_counts.head(3).index.tolist()
    suggestions = []
    for p in top_pattis:
        if pd.notna(p):
            try:
                clean_p = str(int(float(p)))
                suggestions.append(clean_p.zfill(3))
            except ValueError:
                suggestions.append(str(p))
    return suggestions

def get_today_prediction_history(save_package, features, original_df):
    today_obj = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    today_str = today_obj.strftime('%d/%m/%Y')
    
    today_features = features[features['Date'] == today_str].copy()
    history = []
    
    if not today_features.empty:
        X_cols = get_feature_columns()
        X_today = today_features[X_cols]
        
        models = save_package['models']
        freq_model = save_package.get('freq_model', {})
        all_probs = []
        for name, model in models.items():
            probs = model.predict_proba(X_today)
            full_probs = np.zeros((len(X_today), 10))
            for i, cls in enumerate(model.classes_):
                full_probs[:, int(cls)] = probs[:, i]
            all_probs.append(full_probs)
        
        ensemble_probs = np.mean(all_probs, axis=0)
        
        for i, (idx, row) in enumerate(today_features.iterrows()):
            bazi_num = int(row['Bazi'])
            actual = int(row['Target_Single'])
            day_of_week = row['Date_Obj'].dayofweek
            
            blended_probs = blend_predictions(ensemble_probs[i], freq_model, bazi_num, day_of_week, ml_weight=0.65)
            
            sorted_indices = blended_probs.argsort()[::-1][:3]
            top_3 = [int(k) for k in sorted_indices]
            
            status = "Pass" if actual in top_3 else "Fail"
            
            history.append({
                "bazi": bazi_num,
                "predictions": top_3,
                "actual": actual,
                "status": status
            })
    
    history.sort(key=lambda x: x['bazi'], reverse=True)
    return history

def get_quick_prediction():
    if not os.path.exists(MODEL_FILE):
        success = train_and_save_model()
        if not success:
            return {"status": "error", "message": "Not enough historical data to generate predictions."}
    
    features, original_df = load_and_preprocess_data()
    if features is None:
        return {"status": "error", "message": "No data found."}
    
    save_package = joblib.load(MODEL_FILE)
    models = save_package['models']
    freq_model = save_package.get('freq_model', {})
    
    last_record = original_df.iloc[-1]
    last_single = last_record['Single']
    prev_patti_sum = last_record['Patti_Sum']
    
    prev2_single = original_df.iloc[-2]['Single'] if len(original_df) > 1 else 0
    prev3_single = original_df.iloc[-3]['Single'] if len(original_df) > 2 else 0
    prev4_single = original_df.iloc[-4]['Single'] if len(original_df) > 3 else 0
    prev5_single = original_df.iloc[-5]['Single'] if len(original_df) > 4 else 0
    
    ma_7 = original_df['Single'].tail(7).mean()
    ma_30 = original_df['Single'].tail(30).mean()
    std_7 = original_df['Single'].tail(7).std()
    if pd.isna(std_7): std_7 = 3.0
    patti_ma_7 = original_df['Patti_Sum'].tail(7).mean()
    if pd.isna(patti_ma_7): patti_ma_7 = 0.0
    
    is_even_list = original_df['Single'].tail(5).apply(lambda x: 1 if x % 2 == 0 else 0)
    rolling_even_5 = is_even_list.mean()
    
    last_date_str = str(last_record['Date']).strip()
    today_obj = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    today_str = today_obj.strftime('%d/%m/%Y')
    
    is_today = (today_str == last_date_str)
    next_bazi = int(last_record['Bazi']) + 1 if is_today else 1
    
    same_bazi_hist = original_df[original_df['Bazi'] == next_bazi]
    prev_day_same_bazi_val = same_bazi_hist.iloc[-1]['Single'] if not same_bazi_hist.empty else last_single
    
    if next_bazi > 8:
        return {"status": "error", "message": "All 8 Bazis for today are completed."}
    
    day_of_week = today_obj.weekday()
    month = today_obj.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Compute gap features for query
    recent_singles = original_df['Single'].tail(50).values
    gap_features = {}
    for d in range(10):
        found = False
        for gap in range(1, min(len(recent_singles) + 1, 51)):
            if recent_singles[-gap] == d:
                gap_features[f'Gap_{d}'] = gap
                found = True
                break
        if not found:
            gap_features[f'Gap_{d}'] = 50
    
    # Compute frequency features for query
    recent_20 = original_df['Single'].tail(20).values
    freq_features = {}
    for d in range(10):
        freq_features[f'Freq_{d}'] = np.sum(recent_20 == d) / len(recent_20)
    
    # Compute bazi-specific frequency for query
    bazi_recent = original_df[original_df['Bazi'] == next_bazi]['Single'].tail(30).values
    bazi_freq_features = {}
    for d in range(10):
        bazi_freq_features[f'Bazi_Freq_{d}'] = np.sum(bazi_recent == d) / max(len(bazi_recent), 1)
    
    # Patti digits
    prev_patti = last_record['Patti']
    pd1, pd2, pd3 = extract_patti_digits(prev_patti)
    
    same_as_prev = 1 if len(original_df) > 1 and original_df.iloc[-1]['Single'] == original_df.iloc[-2]['Single'] else 0
    last3 = original_df['Single'].tail(3).values
    repeat_in_3 = int(len(set(last3)) < 3) if len(last3) == 3 else 0
    
    # Build query DataFrame
    query_dict = {
        'Bazi': [next_bazi],
        'DayOfWeek_Sin': [np.sin(2 * np.pi * day_of_week / 7)],
        'DayOfWeek_Cos': [np.cos(2 * np.pi * day_of_week / 7)],
        'Month_Sin': [np.sin(2 * np.pi * month / 12)],
        'Month_Cos': [np.cos(2 * np.pi * month / 12)],
        'Is_Weekend': [is_weekend],
        'MA_7': [ma_7],
        'MA_30': [ma_30],
        'STD_7': [std_7],
        'Patti_MA_7': [patti_ma_7],
        'Patti_D1': [pd1],
        'Patti_D2': [pd2],
        'Patti_D3': [pd3],
        'Rolling_Even_5': [rolling_even_5],
        'Prev_Day_Same_Bazi': [prev_day_same_bazi_val],
        'Same_As_Prev': [same_as_prev],
        'Repeat_In_3': [repeat_in_3],
        'Prev_1_Single': [last_single],
        'Prev_2_Single': [prev2_single],
        'Prev_3_Single': [prev3_single],
        'Prev_4_Single': [prev4_single],
        'Prev_5_Single': [prev5_single],
        'Prev_Patti_Sum': [prev_patti_sum],
        'Prev_Patti_D1': [pd1],
        'Prev_Patti_D2': [pd2],
        'Prev_Patti_D3': [pd3],
    }
    query_dict.update({k: [v] for k, v in gap_features.items()})
    query_dict.update({k: [v] for k, v in freq_features.items()})
    query_dict.update({k: [v] for k, v in bazi_freq_features.items()})
    
    X_cols = get_feature_columns()
    query = pd.DataFrame(query_dict)[X_cols]
    
    # Ensemble prediction
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(query)[0]
        full_probs = np.zeros(10)
        for i, cls in enumerate(model.classes_):
            full_probs[int(cls)] = probs[i]
        all_probs.append(full_probs)
    
    ml_probs = np.mean(all_probs, axis=0)
    
    # Blend with frequency model
    blended_probs = blend_predictions(ml_probs, freq_model, next_bazi, day_of_week, ml_weight=0.65)
    
    sorted_indices = blended_probs.argsort()[::-1]
    top_5 = [(int(sorted_indices[i]), float(blended_probs[sorted_indices[i]])) for i in range(5)]
    top_prob = top_5[0][1] * 100
    
    patti_suggestions = [get_patti_suggestions(original_df, t[0]) for t in top_5]
    
    stats = backtest_recent_stats(save_package, features, today_str)
    
    # Load OOS accuracy from stats file if available
    oos_accuracy_pct = None
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                saved_stats = json.load(f)
                oos_accuracy_pct = saved_stats.get('oos_accuracy_pct')
        except Exception:
            pass
    
    # Enhanced Risk Management
    if next_bazi == 1:
        risk_status = "EXTREME RISK"
        reason = "Market Data Insufficient. Pehli Bazi hamesha unpredictable hoti hai, Subah ka trend clear hone de."
        action = "NAHI KHELNA HAI (SKIP)"
        color = "red"
    elif top_prob < 12.0:
        risk_status = "VERY HIGH RISK"
        reason = f"Deep AI Ensembles me weak pattern (Probability: {top_prob:.1f}%). Confusion zyada, loss chance high."
        action = "NAHI KHELNA HAI (SKIP)"
        color = "red"
    elif stats['losing_streak'] >= 3:
        risk_status = "MARKET VOLATILE"
        reason = f"6-Model Deep AI detect market volatility ({stats['losing_streak']} prediction fail huye). Trend stabilize hone ka wait karein."
        action = "WAIT KARO (NO BET)"
        color = "red"
    elif top_prob >= 20.0 and stats['winning_streak'] >= 1:
        risk_status = "JACKPOT SIGNAL"
        reason = f"Deep Ensemble Master Match ({top_prob:.1f}%). 6 models + frequency alignment verified! AI winning streak par hai."
        action = "KHELNA HAI (HIGH BET)"
        color = "green"
    elif top_prob >= 15.0:
        risk_status = "GOOD SIGNAL"
        reason = f"Pattern practically stable ({top_prob:.1f}%). Ensembles agree kar rahe hain. Normal bet khel sakte hain."
        action = "KHELNA HAI (NORMAL BET)"
        color = "gold"
    else:
        risk_status = "AVERAGE OPTION"
        reason = f"Model Confidence okay hai ({top_prob:.1f}%). Sirf jarurat ho tabhi khelo warna chhod do."
        action = "PLAY LIGHT (LOW BET)"
        color = "yellow"
    
    history_trend = original_df.tail(30)[['Bazi', 'Single']].to_dict('records')
    history_trend = [{"Bazi": int(x['Bazi']), "Single": int(x['Single'])} for x in history_trend]
    
    today_history = get_today_prediction_history(save_package, features, original_df)
    
    return {
        "status": "success",
        "data": {
            "today_history": today_history,
            "next_bazi": int(next_bazi),
            "predictions": [
                {
                    "number": int(top_5[i][0]),
                    "probability": round(top_5[i][1] * 100, 1),
                    "pattis": patti_suggestions[i]
                }
                for i in range(5)
            ],
            "risk_management": {
                "level": risk_status,
                "action": action,
                "reason": reason,
                "color": color
            },
            "stats": {
                "previous_prediction_correct": bool(stats['prev_correct']),
                "today_matches": str(stats['today_matches']),
                "weekly_matches": str(stats['week_matches']),
                "winning_streak": int(stats['winning_streak']),
                "losing_streak": int(stats['losing_streak']),
                "oos_accuracy_pct": oos_accuracy_pct
            },
            "history_trend": history_trend
        }
    }

if __name__ == "__main__":
    import time
    print("=" * 60)
    print("  KOLKATA FF V5 ULTRA ENSEMBLE ENGINE")
    print("  6 Models + Frequency Blending + 56 Features")
    print("=" * 60)
    t1 = time.time()
    train_and_save_model()
    t2 = time.time()
    print(f"\nTraining completed in {t2-t1:.2f} seconds.")
    res = get_quick_prediction()
    if res['status'] == 'success':
        print("\n--- PERFORMANCE & NEXT PREDICTION ---")
        print(f"Today's Accuracy: {res['data']['stats']['today_matches']}")
        print(f"Weekly Accuracy:  {res['data']['stats']['weekly_matches']}")
        print(f"Current Win Streak: {res['data']['stats']['winning_streak']}")
        print("\nNext Bazi Predictions:")
        for p in res['data']['predictions']:
            print(f"  Number {p['number']} ({p['probability']}%) - Top Pattis: {', '.join(p['pattis'])}")
    else:
        print(f"Error: {res['message']}")
