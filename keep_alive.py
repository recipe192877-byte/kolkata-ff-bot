from flask import Flask, render_template, jsonify
from threading import Thread
import os
import time
import scraper
import predict_ml_v2 as predict_ml

app = Flask(__name__)
last_scrape_time = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict')
def api_predict():
    global last_scrape_time
    try:
        # Auto-fetch latest result from Kolkata FF every 5 minutes maximum
        current_time = time.time()
        if (current_time - last_scrape_time) > 300:
            print("Auto-syncing kolkataff.tv latest result before prediction...")
            scraper.scrape_kolkata_ff()
            last_scrape_time = current_time
            
        result = predict_ml.get_quick_prediction()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"})

@app.route('/api/retrain')
def api_retrain():
    try:
        success = predict_ml.train_and_save_model()
        if success:
            return jsonify({"status": "success", "message": "Model retrained successfully on latest data."})
        else:
            return jsonify({"status": "error", "message": "Not enough data to retrain."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Retrain error: {str(e)}"})

@app.route('/api/stats')
def api_stats():
    try:
        import json
        if os.path.exists(predict_ml.STATS_FILE):
            with open(predict_ml.STATS_FILE, 'r') as f:
                stats = json.load(f)
            return jsonify({"status": "success", "data": stats})
        return jsonify({"status": "error", "message": "No stats available. Train the model first."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Stats error: {str(e)}"})

def run():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.start()
