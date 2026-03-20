import pandas as pd
import numpy as np
import joblib
import warnings
import os
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
warnings.filterwarnings('ignore')

MODEL_FILE = 'xgb_model.joblib'
DATA_FILE = 'kolkata_ff_history_advanced.csv'

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
        
        original_df = df.copy()
        
        features = df[['Bazi', 'DayOfWeek', 'Month']].copy()
        features['Prev_1_Single'] = df['Single'].shift(1)
        features['Prev_2_Single'] = df['Single'].shift(2)
        features['Prev_3_Single'] = df['Single'].shift(3)
        features['Prev_Patti_Sum'] = df['Patti_Sum'].shift(1)
        
        features['Target_Single'] = df['Single']
        
        features = features.dropna()
        original_df = original_df.iloc[3:].reset_index(drop=True)
        
        return features, original_df
        
    except FileNotFoundError:
        return None, None

def train_and_save_model():
    features, _ = load_and_preprocess_data()
    if features is None or len(features) < 100:
        print("Not enough data to train advanced Hybrid model.")
        return False
        
    X = features[['Bazi', 'DayOfWeek', 'Month', 'Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_Patti_Sum']]
    y = features['Target_Single']
    
    # Advanced XGBoost Engine
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Random Forest Engine
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )

    # Hybrid Ensemble (Soft Voting for highly accurate probability distribution)
    model = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        voting='soft',
        weights=[1.5, 1.0] # Prioritize XGBoost slightly more
    )
    
    model.fit(X, y)
    
    joblib.dump(model, MODEL_FILE)
    print(f"Hybrid AI Model (XGB+RF) trained on {len(X)} historical records and saved successfully.")
    return True

def backtest_recent_stats(original_df, features):
    try:
        model = joblib.load(MODEL_FILE)
    except:
        return {"today_matches": "0/0", "week_matches": "0/0", "prev_correct": False, "winning_streak": 0, "losing_streak": 0}
        
    X_all = features[['Bazi', 'DayOfWeek', 'Month', 'Prev_1_Single', 'Prev_2_Single', 'Prev_3_Single', 'Prev_Patti_Sum']]
    y_all = features['Target_Single']
    
    predictions = model.predict(X_all)
    matches = (predictions == y_all).values
    
    # Today's matches
    last_date = original_df.iloc[-1]['Date']
    today_mask = (original_df['Date'] == last_date).values
    today_matches_count = matches[today_mask].sum()
    today_total = today_mask.sum()
    
    # Weekly matches
    unique_dates = original_df['Date'].unique()
    last_7_dates = unique_dates[-7:] if len(unique_dates) >= 7 else unique_dates
    week_mask = original_df['Date'].isin(last_7_dates).values
    week_matches_count = matches[week_mask].sum()
    week_total = week_mask.sum()
    
    prev_correct = bool(matches[-1]) if len(matches) > 0 else False
    
    streak = 0
    for m in reversed(matches):
        if m: streak += 1
        else: break
        
    losing_streak = 0
    for m in reversed(matches):
        if not m: losing_streak += 1
        else: break
        
    return {
        "today_matches": f"{today_matches_count}/{today_total}",
        "week_matches": f"{week_matches_count}/{week_total}",
        "prev_correct": prev_correct,
        "winning_streak": streak,
        "losing_streak": losing_streak
    }

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
    
    last_date_str = str(last_record['Date']).strip()
    
    today_obj = datetime.utcnow() + timedelta(hours=5, minutes=30)
    today_str = today_obj.strftime('%d/%m/%Y')
    
    is_today = (today_str == last_date_str)
    next_bazi = int(last_record['Bazi']) + 1 if is_today else 1
    
    if next_bazi > 8:
        return {"status": "error", "message": "All 8 Bazis for today are completed."}
        
    day_of_week = today_obj.weekday()
    month = today_obj.month
    
    query = pd.DataFrame({
        'Bazi': [next_bazi], 
        'DayOfWeek': [day_of_week],
        'Month': [month],
        'Prev_1_Single': [last_single], 
        'Prev_2_Single': [prev2_single],
        'Prev_3_Single': [prev3_single],
        'Prev_Patti_Sum': [prev_patti_sum]
    })
    
    probabilities = model.predict_proba(query)[0]
    prob_dict = {num: float(prob) for num, prob in zip(model.classes_, probabilities)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    top_2 = sorted_probs[:2]
    top_prob = top_2[0][1] * 100
    
    stats = backtest_recent_stats(original_df, features)
    
    # Hardcore Risk Management Engine (Hinglish integration as requested)
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
        
    return {
        "status": "success",
        "data": {
            "next_bazi": int(next_bazi),
            "predictions": [
                {"number": int(top_2[0][0]), "probability": round(top_2[0][1] * 100, 1)},
                {"number": int(top_2[1][0]), "probability": round(top_2[1][1] * 100, 1)}
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
            }
        }
    }

if __name__ == "__main__":
    train_and_save_model()
    import json
    print(json.dumps(get_quick_prediction(), indent=2))
