from flask import Flask, render_template, jsonify, request
from threading import Thread, Lock
import os
import time
import json
import scraper
import predict_ml_v2 as predict_ml
from ruflo_healer import RuFloHealer
from ai_council import council

ruflo_upgrader = RuFloHealer()

app = Flask(__name__)
last_scrape_time = 0
_scrape_lock = Lock()  # Thread-safe lock to prevent concurrent scrapes

# ── Response Cache (bounded) ──
_cache = {}
CACHE_TTL = 60  # seconds
CACHE_MAX_SIZE = 10  # max entries


def _get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
        else:
            del _cache[key]  # Evict expired entry
    return None


def _set_cache(key, data):
    # Evict oldest entries if cache is full
    while len(_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest_key]
    _cache[key] = (data, time.time())


def _background_scrape():
    """Run scraper in background thread to avoid Render 30-sec timeout."""
    global last_scrape_time
    if not _scrape_lock.acquire(blocking=False):
        return  # Another scrape is already running
    try:
        print("Background scrape started...")
        scraper.scrape_kolkata_ff()
        last_scrape_time = time.time()
    except Exception as e:
        print(f"Background scrape error: {e}")
    finally:
        _scrape_lock.release()


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

        # FIX: Auto-fetch in background thread (non-blocking) to avoid timeout
        current_time = time.time()
        if (current_time - last_scrape_time) > 300:
            Thread(target=_background_scrape, daemon=True).start()
            
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
        "brain_capacity": predict_ml.brain.get_brain_capacity(),
        "healer": "RuFlo Autonomous Code Upgrader is ACTIVE",
        "uptime": int(time.time()),
        "cache_size": len(_cache)
    })

def _daily_ruflo_scan():
    """Background loop that runs RuFlo Healer once a day."""
    while True:
        try:
            print("[BACKGROUND] Triggering daily RuFlo Autonomous Code Upgrade...")
            ruflo_upgrader.scan_and_upgrade()
            time.sleep(86400) # Sleep for 24 hours
        except Exception as e:
            print(f"[BACKGROUND] Daily RuFlo Scan failed: {e}")
            time.sleep(3600) # Retry in 1 hour on failure


@app.route('/api/heatmap')
def api_heatmap():
    """Return full 10-digit probability distribution for heatmap display."""
    try:
        cached = _get_cached('heatmap')
        if cached:
            return jsonify(cached)

        result = predict_ml.get_quick_prediction()
        if result['status'] != 'success':
            return jsonify(result)

        # Build heatmap from predictions — fill all 10 digits
        preds = result['data']['predictions']
        heatmap = {str(d): 0.0 for d in range(10)}  # Initialize all digits to 0
        for p in preds:
            heatmap[str(p['number'])] = p['probability']

        resp = {"status": "success", "heatmap": heatmap}
        _set_cache('heatmap', resp)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Heatmap error: {str(e)}"})


@app.route('/api/heal')
def api_heal():
    """Manual trigger for RuFlo Autonomous Code Scan."""
    try:
        # Start a background thread so we don't block the API response
        Thread(target=ruflo_upgrader.scan_and_upgrade, daemon=True).start()
        return jsonify({"status": "success", "message": "RuFlo Autonomous Code Upgrade scan has been manually triggered in the background."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Heal trigger error: {str(e)}"})


@app.route('/api/heal/status')
def api_heal_status():
    """Check Auto-Healer health."""
    try:
        return jsonify({"status": "success", "data": "RuFlo Autonomous Upgrader is Online and Active."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Interactive RuFlo Chat for coding and modifications."""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"status": "error", "message": "No message provided."})
            
        user_message = data['message']
        # This might take a while if files are modified, so we do it synchronously for now 
        # because the frontend will show a typing indicator
        result = ruflo_upgrader.chat_and_execute(user_message)
        
        return jsonify({
            "status": "success",
            "reply": result.get("reply", "Done."),
            "files_modified": [mod.get("filename") for mod in result.get("modifications", [])]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/council')
def api_council():
    """Hold an AI Council meeting — multiple AI models discuss and give consensus prediction."""
    try:
        # First get the ML prediction data
        cached_pred = _get_cached('predict')
        if cached_pred:
            prediction_data = cached_pred
        else:
            prediction_data = predict_ml.get_quick_prediction()
            _set_cache('predict', prediction_data)

        if prediction_data.get('status') != 'success':
            return jsonify({"status": "error", "message": "ML prediction failed. Cannot hold council meeting."})

        # Check cache for council (reuse if recent)
        cached_council = _get_cached('council')
        if cached_council:
            return jsonify(cached_council)

        # Hold the meeting
        result = council.hold_meeting(prediction_data)
        if result.get('status') == 'success':
            _set_cache('council', result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Council error: {str(e)}"})


@app.route('/api/council/last')
def api_council_last():
    """Get the last council meeting result."""
    last = council.get_last_meeting()
    if last:
        return jsonify(last)
    return jsonify({"status": "error", "message": "No council meeting held yet. Call /api/council first."})


@app.route('/api/council/log')
def api_council_log():
    """Get all council meeting history."""
    return jsonify({"status": "success", "meetings": council.get_meeting_log()})
@app.route('/api/evolution')
def api_evolution():
    """Get the daily AI Evolution report (yesterday's stats + AI config updates)."""
    try:
        if os.path.exists('daily_report.json'):
            with open('daily_report.json', 'r') as f:
                return jsonify({"status": "success", "data": json.load(f)})
        else:
            return jsonify({"status": "pending", "message": "No daily evolution report generated yet. It will run automatically at midnight."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Evolution report error: {str(e)}"})


def run():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    # Set web server thread as daemon so it dies when main process exits
    t = Thread(target=run, daemon=True)
    t.start()
    
    # Start the Daily RuFlo Code Upgrade Scanner
    t_ruflo = Thread(target=_daily_ruflo_scan, daemon=True)
    t_ruflo.start()
