from flask import Flask, render_template, jsonify, request
from threading import Thread
import os
import time
import json
import scraper
import predict_ml_v2 as predict_ml

app = Flask(__name__)
last_scrape_time = 0

# ── Response Cache ──
_cache = {}
CACHE_TTL = 60  # seconds


def _get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict')
def api_predict():
    global last_scrape_time
    try:
        # Check cache first
        cached = _get_cached('predict')
        if cached:
            return jsonify(cached)

        # Auto-fetch latest result every 5 minutes max
        current_time = time.time()
        if (current_time - last_scrape_time) > 300:
            print("Auto-syncing kolkataff.tv latest result before prediction...")
            scraper.scrape_kolkata_ff()
            last_scrape_time = current_time
            
        result = predict_ml.get_quick_prediction()
        _set_cache('predict', result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"})


@app.route('/api/retrain')
def api_retrain():
    # FIX #5: Basic auth for retrain endpoint
    retrain_key = os.environ.get("RETRAIN_KEY")
    if retrain_key:
        provided_key = request.args.get('key', '')
        if provided_key != retrain_key:
            return jsonify({"status": "error", "message": "Unauthorized. Provide ?key=YOUR_RETRAIN_KEY"})
    
    try:
        success = predict_ml.train_and_save_model()
        _cache.clear()  # Invalidate cache after retrain
        if success:
            return jsonify({"status": "success", "message": "Model retrained successfully on latest data."})
        else:
            return jsonify({"status": "error", "message": "Not enough data to retrain."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Retrain error: {str(e)}"})


@app.route('/api/stats')
def api_stats():
    try:
        if os.path.exists(predict_ml.STATS_FILE):
            with open(predict_ml.STATS_FILE, 'r') as f:
                stats = json.load(f)
            return jsonify({"status": "success", "data": stats})
        return jsonify({"status": "error", "message": "No stats available. Train the model first."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Stats error: {str(e)}"})


@app.route('/api/health')
def api_health():
    """Health check endpoint for monitoring."""
    has_model = os.path.exists(predict_ml.MODEL_FILE)
    has_data = os.path.exists(predict_ml.DATA_FILE)
    
    # Get OOS accuracy if available
    oos_accuracy = None
    if os.path.exists(predict_ml.STATS_FILE):
        try:
            with open(predict_ml.STATS_FILE, 'r') as f:
                stats = json.load(f)
                oos_accuracy = stats.get('oos_accuracy_pct')
        except Exception:
            pass
    
    return jsonify({
        "status": "ok",
        "model_loaded": has_model,
        "data_available": has_data,
        "oos_accuracy_pct": oos_accuracy,
        "uptime": int(time.time()),
        "cache_size": len(_cache)
    })


@app.route('/api/heatmap')
def api_heatmap():
    """Return digit probability distribution for heatmap display."""
    try:
        cached = _get_cached('heatmap')
        if cached:
            return jsonify(cached)

        result = predict_ml.get_quick_prediction()
        if result['status'] != 'success':
            return jsonify(result)

        # Build heatmap from predictions
        preds = result['data']['predictions']
        heatmap = {}
        for p in preds:
            heatmap[str(p['number'])] = p['probability']

        resp = {"status": "success", "heatmap": heatmap}
        _set_cache('heatmap', resp)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Heatmap error: {str(e)}"})


def run():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    # FIX #1: Set thread as daemon so it dies when main process exits
    t = Thread(target=run, daemon=True)
    t.start()
