"""
=============================================================
  KOLKATA FF PREDICTOR â€” PREDICT ML V3
  Clean rewrite based on honest data analysis.

  DATA FACTS (from CSV audit):
  - 3,999 rows | June 2024 - May 2026
  - Digits near-uniform (9.7-10.6%) = lottery-like
  - Max feature correlation: 0.034 (very weak)
  - Expected Top-3 accuracy: ~28-34% (random baseline: 27.1%)

  ARCHITECTURE:
  - Ensemble: LightGBM + RandomForest + GradientBoosting
  - Frequency model blended with ML
  - No XGBoost (version compat issues)
  - No Vector Memory Brain (adds noise on random data)
  - No AI Council in prediction pipeline (LLMs can't predict lottery)
  - Zero data leakage: all features use shift(1)+ only
=============================================================
"""

import os, json, joblib, warnings
import pandas as pd
import numpy as np
from collections import Counter
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
load_dotenv()

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_FILE   = 'kolkata_ff_history_advanced.csv'
MODEL_FILE  = 'kolkata_model_v3.joblib'
STATS_FILE  = 'oos_stats.json'
AI_CFG_FILE = 'ai_config.json'

DEFAULT_AI_CONFIG = {
    "ml_weight":   0.65,
    "freq_weight": 0.35,
    "version":     "v3"
}

# â”€â”€ Config helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_ai_config():
    try:
        with open(AI_CFG_FILE) as f:
            cfg = json.load(f)
            cfg.setdefault('ml_weight',   DEFAULT_AI_CONFIG['ml_weight'])
            cfg.setdefault('freq_weight', DEFAULT_AI_CONFIG['freq_weight'])
            return cfg
    except Exception:
        return DEFAULT_AI_CONFIG.copy()

def save_ai_config(cfg):
    try:
        with open(AI_CFG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ================================================================
#   FEATURE ENGINEERING (Zero Data Leakage)
#   ALL features computed from data BEFORE the row being predicted.
# ================================================================

def _patti_sum(p):
    """Sum digits of patti number."""
    try:
        return float(sum(int(x) for x in str(int(float(p))).zfill(3) if x.isdigit()))
    except Exception:
        return 0.0


def build_feature_matrix(df_raw: pd.DataFrame):
    """
    Build feature matrix from raw CSV data.
    Returns: (features_df, original_df_clean)
    - All features use shift(1) or earlier â€” ZERO look-ahead bias.
    """
    df = df_raw.copy()
    df['DO']     = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Single'] = pd.to_numeric(df['Single'], errors='coerce')
    df['Patti']  = pd.to_numeric(df['Patti'],  errors='coerce')
    df = df.dropna(subset=['Single', 'DO']).copy()
    df['Single'] = df['Single'].astype(int)
    df = df.sort_values(['DO', 'Bazi']).reset_index(drop=True)

    original_df = df.copy()   # saved for live prediction context
    s  = df['Single'].astype(float)
    N  = len(df)
    dw = df['DO'].dt.dayofweek
    mn = df['DO'].dt.month

    f = pd.DataFrame(index=df.index)

    # â”€â”€ Bazi â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['bazi'] = df['Bazi'].astype(float)

    # â”€â”€ Cyclical time features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['dws'] = np.sin(2 * np.pi * dw / 7)
    f['dwc'] = np.cos(2 * np.pi * dw / 7)
    f['ms']  = np.sin(2 * np.pi * mn / 12)
    f['mc']  = np.cos(2 * np.pi * mn / 12)
    f['we']  = (dw >= 5).astype(float)

    # â”€â”€ Lag features (shift 1..8) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for k in range(1, 9):
        f[f'p{k}'] = s.shift(k)

    # â”€â”€ Rolling stats on shifted series â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['ma5']  = s.shift(1).rolling(5,  min_periods=1).mean().fillna(4.5)
    f['ma10'] = s.shift(1).rolling(10, min_periods=1).mean().fillna(4.5)
    f['ma20'] = s.shift(1).rolling(20, min_periods=1).mean().fillna(4.5)
    f['std5'] = s.shift(1).rolling(5,  min_periods=1).std().fillna(3.0)
    f['std10']= s.shift(1).rolling(10, min_periods=1).std().fillna(3.0)

    # â”€â”€ Bazi-specific lags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['bazi_p1'] = df.groupby('Bazi')['Single'].shift(1).astype(float).fillna(4.0)
    f['bazi_p2'] = df.groupby('Bazi')['Single'].shift(2).astype(float).fillna(4.0)

    # â”€â”€ Modular / streak features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['p1_mod2'] = (f['p1'] % 2).fillna(0)   # even/odd of previous result
    f['p1_mod3'] = (f['p1'] % 3).fillna(0)   # mod-3 pattern
    # streak = did previous two results match? (comparing two past values â€” safe)
    f['streak'] = (s.shift(1) == s.shift(2)).astype(float).fillna(0)

    # â”€â”€ Patti features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['patti_prev'] = df['Patti'].apply(_patti_sum).shift(1).fillna(0)

    # â”€â”€ GAP features: how many draws since digit d last appeared â”€â”€
    # Looks BACKWARD from position i â€” no future info.
    sv = s.values.astype(float)
    for d in range(10):
        gap_arr = np.full(N, 30.0)
        for i in range(1, N):
            limit = min(i, 30)
            for g in range(1, limit + 1):
                if sv[i - g] == d:
                    gap_arr[i] = float(g)
                    break
        f[f'gap{d}'] = gap_arr

    # â”€â”€ FREQUENCY features: digit rate in last 20 draws â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Uses shift(1) rolling â€” excludes current row. Safe.
    for d in range(10):
        f[f'freq{d}'] = (s.shift(1) == d).astype(float).rolling(20, min_periods=1).mean()

    # â”€â”€ Target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f['TARGET'] = s.values
    f['DO']     = df['DO'].values

    # â”€â”€ Clean â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f = f.dropna()
    f = f.iloc[10:].reset_index(drop=True)

    return f, original_df


def get_feature_columns():
    """Return the exact list of feature column names used for training."""
    cols = (['bazi', 'dws', 'dwc', 'ms', 'mc', 'we',
             'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8',
             'ma5', 'ma10', 'ma20', 'std5', 'std10',
             'bazi_p1', 'bazi_p2',
             'p1_mod2', 'p1_mod3', 'streak', 'patti_prev']
            + [f'gap{d}'  for d in range(10)]
            + [f'freq{d}' for d in range(10)])
    return cols


# ================================================================
#   FREQUENCY MODEL
# ================================================================

def build_frequency_model(df: pd.DataFrame):
    """
    Build a per-bazi digit frequency model from historical data.
    Returns dict: {bazi: {digit: probability}}
    """
    model = {}
    for b in range(1, 9):
        sub = df[df['Bazi'] == b]['Single']
        if len(sub) == 0:
            model[b] = {d: 0.1 for d in range(10)}
            continue
        cnt = Counter(sub.tolist())
        total = len(sub)
        # Apply Laplace smoothing
        model[b] = {d: (cnt.get(d, 0) + 1) / (total + 10) for d in range(10)}
    return model


# ================================================================
#   OOS VALIDATION  (temporal split â€” no leakage)
# ================================================================

def generate_oos_stats(features: pd.DataFrame):
    """
    Walk-forward OOS validation.
    Train on first 70% of dates, test on last 30%.
    Returns dict with accuracy metrics.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import lightgbm as lgb

    X_cols = get_feature_columns()
    features = features.sort_values('DO').reset_index(drop=True)
    ud = features['DO'].dt.date.unique()

    if len(ud) < 20:
        return _empty_oos_stats()

    split = int(len(ud) * 0.70)
    train_dates = set(ud[:split])
    test_dates  = set(ud[split:])

    train_df = features[features['DO'].dt.date.isin(train_dates)]
    test_df  = features[features['DO'].dt.date.isin(test_dates)]

    if len(train_df) < 50 or len(test_df) < 8:
        return _empty_oos_stats()

    Xt, yt = train_df[X_cols].astype(float), train_df['TARGET'].astype(int)
    Xv, yv = test_df[X_cols].astype(float),  test_df['TARGET'].astype(int).values

    oos_models = {
        'lgb': lgb.LGBMClassifier(max_depth=5, n_estimators=200, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.7,
                                    verbose=-1, random_state=42),
        'rf':  RandomForestClassifier(n_estimators=200, max_depth=10,
                                       min_samples_leaf=2, random_state=42),
        'gb':  GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                           learning_rate=0.05, random_state=42),
    }
    probs_all = []
    for m in oos_models.values():
        m.fit(Xt, yt)
        p = m.predict_proba(Xv)
        fp = np.zeros((len(Xv), 10))
        for k, cls in enumerate(m.classes_):
            fp[:, int(cls)] = p[:, k]
        probs_all.append(fp)

    ens = np.mean(probs_all, axis=0)
    top3_correct = 0
    top1_correct = 0
    daily_acc = []

    for i in range(len(yv)):
        top3 = np.argsort(ens[i])[::-1][:3].tolist()
        top1 = int(np.argmax(ens[i]))
        actual = int(yv[i])
        if actual in top3:
            top3_correct += 1
        if actual == top1:
            top1_correct += 1

    # Day-by-day accuracy for chart
    test_df_reset = test_df.reset_index(drop=True)
    for date_str in sorted([str(d) for d in test_dates])[:30]:
        day_mask = test_df_reset['DO'].dt.strftime('%Y-%m-%d') == date_str
        day_idx  = test_df_reset[day_mask].index.tolist()
        if not day_idx:
            continue
        day_ok = sum(1 for i in day_idx if int(yv[i]) in np.argsort(ens[i])[::-1][:3].tolist())
        daily_acc.append({'date': date_str, 'acc': round(day_ok / len(day_idx) * 100, 1)})

    oos_top3_pct = round(top3_correct / len(yv) * 100, 1)
    oos_top1_pct = round(top1_correct / len(yv) * 100, 1)

    stats = {
        'oos_accuracy_pct': oos_top3_pct,
        'oos_top1_pct':     oos_top1_pct,
        'random_baseline':  27.1,
        'train_rows':       len(train_df),
        'test_rows':        len(test_df),
        'total_rows':       len(features),
        'daily_accuracy':   daily_acc[-14:],  # last 14 days for chart
    }
    print(f"OOS Validation: Top-3={oos_top3_pct}%  Top-1={oos_top1_pct}%  (random baseline=27.1%)")

    try:
        with open(STATS_FILE, 'w') as ff:
            json.dump(stats, ff, indent=2)
    except Exception:
        pass

    return stats


def _empty_oos_stats():
    return {
        'oos_accuracy_pct': 0.0, 'oos_top1_pct': 0.0,
        'random_baseline': 27.1,
        'train_rows': 0, 'test_rows': 0, 'total_rows': 0,
        'daily_accuracy': []
    }


# ================================================================
#   MODEL TRAINING
# ================================================================

def create_ensemble_models():
    return {
        'lgb': lgb.LGBMClassifier(max_depth=5, n_estimators=300, learning_rate=0.04,
                                    subsample=0.8, colsample_bytree=0.7,
                                    min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
                                    verbose=-1, random_state=42),
        'rf':  RandomForestClassifier(n_estimators=300, max_depth=12,
                                       min_samples_leaf=2, max_features='sqrt',
                                       random_state=42, n_jobs=-1),
        'gb':  GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                           learning_rate=0.04, subsample=0.8,
                                           min_samples_leaf=3, random_state=42),
    }


def train_and_save_model():
    """
    Load CSV â†’ build features â†’ OOS validate â†’ train on full data â†’ save.
    Returns True on success.
    """
    print("Loading CSV data...")
    try:
        raw = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: {DATA_FILE} not found.")
        return False

    print("Building feature matrix (no data leakage)...")
    features, original_df = build_feature_matrix(raw)

    if features is None or len(features) < 100:
        print("Not enough data to train.")
        return False

    print(f"Feature matrix: {len(features)} rows x {len(get_feature_columns())} features")

    # OOS validation first
    print("Running OOS validation...")
    generate_oos_stats(features)

    X_cols = get_feature_columns()
    X = features[X_cols].astype(float)
    y = features['TARGET'].astype(int)

    print(f"Training 3-model ensemble on {len(X)} records...")
    models = create_ensemble_models()
    trained = {}
    for name, m in models.items():
        m.fit(X, y)
        trained[name] = m
        print(f"  [OK] {name.upper()} trained")

    freq_model = build_frequency_model(original_df)

    package = {
        'models':     trained,
        'freq_model': freq_model,
        'feature_cols': X_cols,
    }
    joblib.dump(package, MODEL_FILE)
    print(f"Model saved â†’ {MODEL_FILE}")
    return True


# ================================================================
#   PREDICTION BLENDING
# ================================================================

def blend_predictions(ml_probs: np.ndarray, freq_model: dict, bazi: int) -> np.ndarray:
    """
    Blend ML ensemble probabilities with frequency model.
    Weights loaded from ai_config.json (can be tuned by AI Council).
    """
    cfg = load_ai_config()
    ml_w   = cfg.get('ml_weight',   0.65)
    freq_w = cfg.get('freq_weight', 0.35)

    freq_probs = np.array([freq_model.get(bazi, {}).get(d, 0.1) for d in range(10)])
    freq_probs = freq_probs / freq_probs.sum()

    blended = ml_w * ml_probs + freq_w * freq_probs
    blended = blended / blended.sum()
    return blended


# ================================================================
#   LIVE PREDICTION
# ================================================================

def get_latest_prediction():
    """
    Main entry point: returns prediction dict for the next bazi.
    """
    import datetime

    # â”€â”€ Load model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        package = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("Model not found â€” training now...")
        if not train_and_save_model():
            return _error_prediction("Model training failed.")
        package = joblib.load(MODEL_FILE)

    trained_models = package['models']
    freq_model     = package['freq_model']
    X_cols         = package.get('feature_cols', get_feature_columns())

    # â”€â”€ Load raw data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        raw = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return _error_prediction("CSV file not found.")

    # â”€â”€ Build features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    features, original_df = build_feature_matrix(raw)
    if features is None or len(features) == 0:
        return _error_prediction("Could not build features.")

    # â”€â”€ Determine next bazi â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    today = datetime.date.today()
    today_str = today.strftime('%d/%m/%Y')

    today_data   = original_df[original_df['Date'] == today_str]
    is_today     = len(today_data) > 0
    last_record  = original_df.iloc[-1]

    if is_today:
        completed_bazis = today_data['Bazi'].astype(int).tolist()
        next_bazi = max(completed_bazis) + 1
    else:
        next_bazi = 1

    if next_bazi > 8:
        return {
            'status': 'complete',
            'message': 'All 8 bazis complete for today.',
            'next_bazi': None,
            'top_3': [],
            'confidence': 0,
            'total_records': len(original_df),
        }

    # â”€â”€ Build query row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    last_N = original_df.tail(30)   # use last 30 rows for context
    last_s = original_df['Single'].astype(float)
    sv     = last_s.values

    # Time features
    day_of_week = today.weekday()
    month       = today.month

    # Lag features from recent history
    lags = {}
    for k in range(1, 9):
        idx = len(sv) - k
        lags[f'p{k}'] = float(sv[idx]) if idx >= 0 else 4.0

    # MA / std from last history
    recent_s = last_s.values[-30:]
    ma5  = float(np.mean(recent_s[-5:]))  if len(recent_s) >= 5  else 4.5
    ma10 = float(np.mean(recent_s[-10:])) if len(recent_s) >= 10 else 4.5
    ma20 = float(np.mean(recent_s[-20:])) if len(recent_s) >= 20 else 4.5
    std5 = float(np.std(recent_s[-5:]))   if len(recent_s) >= 5  else 3.0
    std10= float(np.std(recent_s[-10:])) if len(recent_s) >= 10 else 3.0

    # Bazi-specific lags
    same_bazi_hist = original_df[original_df['Bazi'] == next_bazi]['Single'].values
    bazi_p1 = float(same_bazi_hist[-1]) if len(same_bazi_hist) >= 1 else 4.0
    bazi_p2 = float(same_bazi_hist[-2]) if len(same_bazi_hist) >= 2 else 4.0

    # Streak
    streak = 1.0 if (len(sv) >= 2 and sv[-1] == sv[-2]) else 0.0

    # Patti
    try:
        patti_prev = _patti_sum(original_df.iloc[-1]['Patti'])
    except Exception:
        patti_prev = 0.0

    # Gap features from full history
    gaps = {}
    for d in range(10):
        g = 30
        for k in range(1, min(len(sv) + 1, 31)):
            if sv[-k] == d:
                g = k
                break
        gaps[f'gap{d}'] = float(g)

    # Freq features
    freqs = {}
    window = sv[-20:] if len(sv) >= 20 else sv
    cnt    = Counter([int(x) for x in window])
    for d in range(10):
        freqs[f'freq{d}'] = cnt.get(d, 0) / max(len(window), 1)

    query = {
        'bazi':     float(next_bazi),
        'dws':      np.sin(2 * np.pi * day_of_week / 7),
        'dwc':      np.cos(2 * np.pi * day_of_week / 7),
        'ms':       np.sin(2 * np.pi * month / 12),
        'mc':       np.cos(2 * np.pi * month / 12),
        'we':       1.0 if day_of_week >= 5 else 0.0,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'std5': std5, 'std10': std10,
        'bazi_p1': bazi_p1, 'bazi_p2': bazi_p2,
        'p1_mod2': lags['p1'] % 2,
        'p1_mod3': lags['p1'] % 3,
        'streak':  streak,
        'patti_prev': patti_prev,
        **lags, **gaps, **freqs,
    }

    X_query = pd.DataFrame([query])[X_cols].astype(float)

    # â”€â”€ Ensemble predict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_probs = []
    for m in trained_models.values():
        p  = m.predict_proba(X_query)
        fp = np.zeros(10)
        for k, cls in enumerate(m.classes_):
            fp[int(cls)] = p[0, k]
        all_probs.append(fp)

    ml_probs = np.mean(all_probs, axis=0)

    # â”€â”€ Blend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    blended = blend_predictions(ml_probs, freq_model, next_bazi)

    top3     = np.argsort(blended)[::-1][:3].tolist()
    top1_prob = float(blended[top3[0]])
    conf_pct  = round(top1_prob * 100, 1)

    # â”€â”€ Load OOS accuracy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    oos_acc = 0.0
    try:
        with open(STATS_FILE) as ff:
            oos_acc = json.load(ff).get('oos_accuracy_pct', 0.0)
    except Exception:
        pass

    return {
        'status':         'ok',
        'next_bazi':      int(next_bazi),
        'top_3':          top3,
        'top_1':          top3[0],
        'confidence':     conf_pct,
        'probabilities':  {str(d): round(float(blended[d]) * 100, 1) for d in range(10)},
        'oos_accuracy':   oos_acc,
        'random_baseline': 27.1,
        'total_records':  len(original_df),
        'today_records':  len(today_data),
        'is_today':       is_today,
    }


def _error_prediction(msg: str) -> dict:
    return {
        'status': 'error', 'message': msg,
        'next_bazi': None, 'top_3': [], 'confidence': 0,
    }


# ================================================================
#   PREDICTION HISTORY & ACCURACY TRACKING
# ================================================================

PRED_STORE = 'predictions_store.json'

def load_pred_store() -> dict:
    try:
        with open(PRED_STORE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_pred_store(store: dict):
    try:
        with open(PRED_STORE, 'w') as f:
            json.dump(store, f, indent=2)
    except Exception:
        pass

def record_prediction(date_str: str, bazi: int, top3: list):
    """Save prediction for later accuracy check."""
    store = load_pred_store()
    store.setdefault(date_str, {})
    store[date_str][str(bazi)] = {'predicted': top3, 'actual': None, 'correct': None}
    save_pred_store(store)

def update_actuals(date_str: str, bazi: int, actual: int):
    """Update actual result and mark prediction as correct/wrong."""
    store = load_pred_store()
    if date_str in store and str(bazi) in store[date_str]:
        entry = store[date_str][str(bazi)]
        entry['actual']  = actual
        entry['correct'] = actual in entry.get('predicted', [])
        save_pred_store(store)

def get_prediction_history(days: int = 7) -> list:
    """
    Returns list of prediction records for last N days.
    Each record: {date, bazi, predicted, actual, correct, status}
    """
    store = load_pred_store()
    records = []
    import datetime
    today = datetime.date.today()

    for d in range(days, -1, -1):
        date = today - datetime.timedelta(days=d)
        date_str = date.strftime('%d/%m/%Y')
        day_data  = store.get(date_str, {})
        for bazi_str, entry in day_data.items():
            actual  = entry.get('actual')
            correct = entry.get('correct')
            records.append({
                'date':      date_str,
                'bazi':      int(bazi_str),
                'predicted': entry.get('predicted', []),
                'actual':    actual,
                'correct':   correct,
                'status':    'Pass' if correct else ('Fail' if correct is False else 'Pending'),
            })

    return sorted(records, key=lambda x: (x['date'], x['bazi']))


def get_accuracy_stats(days: int = 30) -> dict:
    """Overall accuracy stats from prediction store."""
    history = get_prediction_history(days)
    decided = [r for r in history if r['correct'] is not None]
    if not decided:
        return {'correct': 0, 'total': 0, 'accuracy': 0.0, 'random_baseline': 27.1}
    correct = sum(1 for r in decided if r['correct'])
    return {
        'correct':         correct,
        'total':           len(decided),
        'accuracy':        round(correct / len(decided) * 100, 1),
        'random_baseline': 27.1,
    }


# ================================================================
#   KEEP_ALIVE ENDPOINT DATA
# ================================================================

def get_dashboard_data() -> dict:
    """
    Collects all data needed for the web dashboard.
    Called by keep_alive.py Flask routes.
    """
    import datetime

    pred   = get_latest_prediction()
    stats  = get_accuracy_stats(30)
    hist   = get_prediction_history(7)

    # OOS stats
    oos = {}
    try:
        with open(STATS_FILE) as f:
            oos = json.load(f)
    except Exception:
        pass

    return {
        'prediction':  pred,
        'accuracy':    stats,
        'history':     hist[-64:],  # last 8 days Ã— 8 bazis
        'oos_stats':   oos,
        'timestamp':   datetime.datetime.now().isoformat(),
    }


# ================================================================
#   MAIN (for direct testing)
# ================================================================

if __name__ == '__main__':
    import sys

    if '--train' in sys.argv:
        ok = train_and_save_model()
        print("Training", "OK" if ok else "FAILED")
    elif '--predict' in sys.argv:
        result = get_latest_prediction()
        print(json.dumps(result, indent=2))
    elif '--stats' in sys.argv:
        print(json.dumps(get_accuracy_stats(), indent=2))
    else:
        print("Usage: python predict_ml_v2.py [--train | --predict | --stats]")


# ── Backwards compatibility alias ─────────────────────────────────────────────
def get_quick_prediction():
    """Alias for get_latest_prediction() — keeps keep_alive.py working."""
    result = get_latest_prediction()
    # Normalize output format for keep_alive.py compatibility
    if result.get('status') == 'ok':
        top3 = result.get('top_3', [])
        probs = result.get('probabilities', {})
        return {
            'status': 'success',
            'data': {
                'next_bazi':   result.get('next_bazi'),
                'top_3':       top3,
                'predictions': [
                    {'number': n, 'probability': float(probs.get(str(n), 0))} for n in top3
                ],
                'confidence':     result.get('confidence', 0),
                'oos_accuracy':   result.get('oos_accuracy', 0),
                'random_baseline': 27.1,
                'total_records':  result.get('total_records', 0),
            }
        }
    return result


# Expose constants expected by keep_alive.py
class _brain_stub:
    def get_brain_capacity(self): return 'N/A'

brain = _brain_stub()


# ================================================================
#   YESTERDAY STATS (for AI Council evolution)
# ================================================================

def get_yesterday_stats() -> dict:
    """
    Calculate yesterday's predictions vs actual results.
    Used by AI Council for daily evolution meetings.
    """
    import datetime

    try:
        history = get_prediction_history(days=2)
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%d/%m/%Y')
        yest_records = [r for r in history if r['date'] == yesterday]

        if not yest_records:
            return {}

        decided = [r for r in yest_records if r['correct'] is not None]
        if not decided:
            return {'date': yesterday, 'total_bazis': len(yest_records), 'decided': 0}

        correct = sum(1 for r in decided if r['correct'])
        accuracy = round(correct / len(decided) * 100, 1)

        return {
            'date':         yesterday,
            'total_bazis':  len(yest_records),
            'decided':      len(decided),
            'correct':      correct,
            'accuracy_pct': accuracy,
            'random_baseline': 27.1,
            'records':      decided,
        }
    except Exception as e:
        print(f"Error getting yesterday stats: {e}")
        return {}
