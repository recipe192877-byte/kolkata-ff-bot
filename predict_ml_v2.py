import pandas as pd
import numpy as np
import joblib
import warnings
import os
import json
from datetime import datetime, timedelta
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

MODEL_FILE = 'xgb_model.joblib'
DATA_FILE = 'kolkata_ff_history_advanced.csv'
STATS_FILE = 'backtest_stats.json'

def calculate_patti_sum(patti_val):
    if pd.isna(patti_val) or str(patti_val).strip() == '':
        return 0
    clean_patti = str(patti_val).strip()
    return sum(int(d) for d in clean_patti if d.isdigit())

def load_and_preprocess_data(filepath=DATA_FILE):
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['Single'])
        df['Single'] = df['Single'].astype(int)
        
        df['Date_Obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).reset_index(drop=True)
        
        # Deep Feature Engineering
        df['DayOfWeek'] = df['Date_Obj'].dt.dayofweek
        df['Month'] = df['Date_Obj'].dt.month
        df['Patti_Sum'] = df['Patti'].apply(calculate_patti_sum)
        
        df['Is_Even'] = (df['Single'] % 2 == 0).astype(int)
        df['Rolling_Even_5'] = df['Is_Even'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
        df['Prev_Day_Same_Bazi'] = df.groupby('Bazi')['Single'].shift(1).fillna(df['Single'].shift(1)).fillna(0)
        
        # Calculate Moving Averages for long-term trends
        df['MA_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
        df['MA_30'] = df['Single'].shift(1).rolling(window=30, min_periods=1).mean().fillna(0)
        df['STD_7'] = df['Single'].shift(1).rolling(window=7, min_periods=1).std().fillna(0.0)
        
        original_df = df.copy()
        
        features = df[['Date', 'Date_Obj', 'Bazi', 'DayOfWeek', 'Month', 'MA_7', 'MA_30', 'STD_7', 'Rolling_Even_5', 'Prev_Day_Same_Bazi']].copy()
        
        # New Lag Features for Better Series Prediction
        features['Prev_1_Single'] = df['Single'].shift(1)
        features['Prev_2_Single'] = df['Single'].shift(2)
        features['Prev_3_Single'] = df['Single'].shift(3)
        features['Prev_4_Single'] = df['Single'].shift(4)
        features['Prev_5_Single'] = df['Single'].shift(5)
        features['Prev_Patti_Sum'] = df['Patti_Sum'].shift(1)
        
        features['Target_Single'] = df['Single']
        
        features = features.dropna()
        original_df = original_df.iloc[5:].reset_index(drop=True)
        
        return features, original_df
        
    except FileNotFoundError:
        return None, None

def create_stacking_model():
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=4, 
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    meta_learner = LogisticRegression(max_iter=500)
    
    # Stacking ensures models weight themselves properly according to OOS error
    model = StackingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('lgb', lgb_model)],
        final_estimator=meta_learner,
        passthrough=False,
        cv=3
    )
    return model

def generate_oos_stats(features):
    features = features.sort_values(by=['Date_Obj', 'Bazi'])
    unique_dates = features['Date_Obj'].dt.date.unique()
    
    if len(unique_dates) < 20: 
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0}
        
    last_14_dates = unique_dates[-14:]
    cutoff_date = last_14_dates[0]
    
    train_df = features[features['Date_Obj'].dt.date < cutoff_date]
    test_df = features[features['Date_Obj'].dt.date >= cutoff_date]
    
    X_cols = ['Bazi', 'DayOfWeek', 'Month', 'MA_7', 'MA_30', 'STD_7', 'Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_4_Single', 'Prev_5_Single', 'Prev_Patti_Sum', 'Rolling_Even_5', 'Prev_Day_Same_Bazi']
    
    X_train = train_df[X_cols]
    y_train = train_df['Target_Single']
    X_test = test_df[X_cols]
    y_test = test_df['Target_Single']
    
    # Train validation model to get True Out-of-Sample Stats
    val_model = create_stacking_model()
    val_model.fit(X_train, y_train)
    
    probabilities = val_model.predict_proba(X_test)
    classes = val_model.classes_
    
    is_match_list = []
    for prob_row, true_val in zip(probabilities, y_test):
        sorted_indices = prob_row.argsort()[::-1][:3]
        top_3_preds = [classes[i] for i in sorted_indices]
        is_match_list.append(true_val in top_3_preds)
        
    matches = pd.Series(is_match_list).values
    test_df['Matched'] = matches
    
    last_date = test_df['Date'].iloc[-1]
    today_mask = test_df['Date'] == last_date
    today_matches_count = test_df[today_mask]['Matched'].sum()
    today_total = today_mask.sum()
    
    test_dates = test_df['Date'].unique()
    last_7_dates = test_dates[-7:] if len(test_dates)>=7 else test_dates
    week_mask = test_df['Date'].isin(last_7_dates)
    week_matches_count = test_df[week_mask]['Matched'].sum()
    week_total = week_mask.sum()
    
    prev_correct = bool(matches[-1]) if len(matches) > 0 else False
    
    win_streak = 0
    for m in reversed(matches):
        if m: win_streak += 1
        else: break
        
    lose_streak = 0
    for m in reversed(matches):
        if not m: lose_streak += 1
        else: break
        
    stats = {
        "today_matches": f"{today_matches_count}/{today_total}",
        "week_matches": f"{week_matches_count}/{week_total}",
        "prev_correct": prev_correct,
        "winning_streak": int(win_streak),
        "losing_streak": int(lose_streak)
    }
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)
        
    return stats

def train_and_save_model():
    features, _ = load_and_preprocess_data()
    if features is None or len(features) < 100:
        print("Not enough data to train advanced ML model.")
        return False
        
    # Before training final model, generate strict Out-Of-Sample validation stats natively!
    print("Generating pure OOS Validation Stats...")
    generate_oos_stats(features)
        
    X_cols = ['Bazi', 'DayOfWeek', 'Month', 'MA_7', 'MA_30', 'STD_7', 'Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_4_Single', 'Prev_5_Single', 'Prev_Patti_Sum', 'Rolling_Even_5', 'Prev_Day_Same_Bazi']
    X = features[X_cols]
    y = features['Target_Single']
    
    model = create_stacking_model()
    model.fit(X, y)
    
    joblib.dump(model, MODEL_FILE)
    print(f"Deep Stacking AI Model trained on {len(X)} historical records and saved successfully.")
    
    return True

def backtest_recent_stats(original_df, features):
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
    return [str(p) for p in top_pattis if pd.notna(p)]

def get_today_prediction_history(model, features, original_df):
    today_obj = datetime.utcnow() + timedelta(hours=5, minutes=30)
    today_str = today_obj.strftime('%d/%m/%Y')
    
    today_features = features[features['Date'] == today_str].copy()
    history = []
    
    if not today_features.empty:
        X_cols = ['Bazi', 'DayOfWeek', 'Month', 'MA_7', 'MA_30', 'STD_7', 'Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_4_Single', 'Prev_5_Single', 'Prev_Patti_Sum', 'Rolling_Even_5', 'Prev_Day_Same_Bazi']
        X_today = today_features[X_cols]
        probs = model.predict_proba(X_today)
        classes = model.classes_
        
        for i, (idx, row) in enumerate(today_features.iterrows()):
            bazi_num = int(row['Bazi'])
            actual = int(row['Target_Single'])
            
            sorted_indices = probs[i].argsort()[::-1][:3]
            top_3 = [int(classes[k]) for k in sorted_indices]
            
            status = "Pass" if actual in top_3 else "Fail"
            
            history.append({
                "bazi": bazi_num,
                "predictions": top_3,
                "actual": actual,
                "status": status
            })
            
    # Sort history newest on top
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
         
    model = joblib.load(MODEL_FILE)
    
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
    if pd.isna(std_7): std_7 = 0.0
    
    is_even_list = original_df['Single'].tail(5).apply(lambda x: 1 if x%2==0 else 0)
    rolling_even_5 = is_even_list.mean()
    
    last_date_str = str(last_record['Date']).strip()
    
    today_obj = datetime.utcnow() + timedelta(hours=5, minutes=30)
    today_str = today_obj.strftime('%d/%m/%Y')
    
    is_today = (today_str == last_date_str)
    next_bazi = int(last_record['Bazi']) + 1 if is_today else 1
    
    same_bazi_hist = original_df[original_df['Bazi'] == next_bazi]
    prev_day_same_bazi_val = same_bazi_hist.iloc[-1]['Single'] if not same_bazi_hist.empty else last_single
    
    if next_bazi > 8:
        return {"status": "error", "message": "All 8 Bazis for today are completed."}
        
    day_of_week = today_obj.weekday()
    month = today_obj.month
    
    query = pd.DataFrame({
        'Bazi': [next_bazi], 
        'DayOfWeek': [day_of_week],
        'Month': [month],
        'MA_7': [ma_7],
        'MA_30': [ma_30],
        'STD_7': [std_7],
        'Prev_1_Single': [last_single], 
        'Prev_2_Single': [prev2_single],
        'Prev_3_Single': [prev3_single],
        'Prev_4_Single': [prev4_single],
        'Prev_5_Single': [prev5_single],
        'Prev_Patti_Sum': [prev_patti_sum],
        'Rolling_Even_5': [rolling_even_5],
        'Prev_Day_Same_Bazi': [prev_day_same_bazi_val]
    })
    
    probabilities = model.predict_proba(query)[0]
    prob_dict = {num: float(prob) for num, prob in zip(model.classes_, probabilities)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    top_3 = sorted_probs[:3]
    top_prob = top_3[0][1] * 100
    
    patti_suggestions_1 = get_patti_suggestions(original_df, int(top_3[0][0]))
    patti_suggestions_2 = get_patti_suggestions(original_df, int(top_3[1][0]))
    patti_suggestions_3 = get_patti_suggestions(original_df, int(top_3[2][0]))
    
    stats = backtest_recent_stats(original_df, features)
    
    # Enhanced ML Risk Logic
    if next_bazi == 1:
        risk_status = "EXTREME RISK"
        reason = "Pehli Bazi sabse unpredictable hoti hai. Market ka trend clear nahi hai."
        action = "NAHI KHELNA HAI (SKIP)"
        color = "red"
    elif top_prob < 15.0:
        risk_status = "VERY HIGH RISK"
        reason = f"AI ko naya pattern samajh nahi aa raha (Probability: {top_prob:.1f}%). Loss ka chance hai."
        action = "NAHI KHELNA HAI (SKIP)"
        color = "red"
    elif stats['losing_streak'] >= 2:
        risk_status = "MARKET VOLATILE"
        reason = f"Abhi market unstable chal raha hai ({stats['losing_streak']} prediction fail huye). Trend badalne do."
        action = "WAIT KARO (NO BET)"
        color = "red"
    elif top_prob >= 28.0 and stats['winning_streak'] >= 1:
        risk_status = "JACKPOT CHANCE"
        reason = f"Bahut strong pattern match hua hai ({top_prob:.1f}%). AI winning streak par hai."
        action = "KHELNA HAI (HIGH BET)"
        color = "green"
    elif top_prob >= 20.0:
        risk_status = "GOOD SIGNAL"
        reason = f"Pattern stable hai ({top_prob:.1f}%). Safely khel sakte hain."
        action = "KHELNA HAI (NORMAL BET)"
        color = "gold"
    else:
        risk_status = "MEDIUM RISK"
        reason = f"Average chance ({top_prob:.1f}%). Agar zaruri ho tabhi khelo warna wait karo."
        action = "PLAY LIGHT (LOW BET)"
        color = "yellow"
        
    history_trend = original_df.tail(30)[['Bazi', 'Single']].to_dict('records')
    history_trend = [{"Bazi": int(x['Bazi']), "Single": int(x['Single'])} for x in history_trend]
    
    today_history = get_today_prediction_history(model, features, original_df)
    
    return {
        "status": "success",
        "data": {
            "today_history": today_history,
            "next_bazi": int(next_bazi),
            "predictions": [
                {
                    "number": int(top_3[0][0]), 
                    "probability": round(top_3[0][1] * 100, 1),
                    "pattis": patti_suggestions_1
                },
                {
                    "number": int(top_3[1][0]), 
                    "probability": round(top_3[1][1] * 100, 1),
                    "pattis": patti_suggestions_2
                },
                {
                    "number": int(top_3[2][0]), 
                    "probability": round(top_3[2][1] * 100, 1),
                    "pattis": patti_suggestions_3
                }
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
                "losing_streak": int(stats['losing_streak'])
            },
            "history_trend": history_trend
        }
    }

if __name__ == "__main__":
    import time
    print("Initializing Deep Learning Engine and Backtesting...")
    t1 = time.time()
    train_and_save_model()
    t2 = time.time()
    print(f"Training completed in {t2-t1:.2f} seconds.")
    res = get_quick_prediction()
    if res['status'] == 'success':
        print("\n--- PERFORMANCE & NEXT PREDICTION ---")
        print(f"Today's Accuracy: {res['data']['stats']['today_matches']}")
        print(f"Weekly Accuracy:  {res['data']['stats']['weekly_matches']}")
        print(f"Current Win Streak: {res['data']['stats']['winning_streak']}")
        print("\nNext Bazi Predictions:")
        for p in res['data']['predictions']:
            print(f"Number {p['number']} ({p['probability']}%) - Top Pattis: {', '.join(p['pattis'])}")
