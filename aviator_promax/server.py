import threading, csv, os
from datetime import datetime
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__, static_folder="dashboard", template_folder="dashboard")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aviator_log.csv")

# Thread-safe lock to prevent _state race conditions
_state_lock = threading.Lock()

_state = {
    "prediction": {
        "action": "WATCH", "cashout": "-", "confidence": 0,
        "conf_label": "Starting...", "reason": "Bot initializing...",
        "signal": "neutral", "votes": {}, "stats": {}, "total_score": 0
    },
    "history": [], "last_mult": None, "round_num": 0, "status": "Initializing..."
}

# Global predictor reference for manual entry
_predictor_ref = None

def set_predictor(p):
    global _predictor_ref
    _predictor_ref = p

def ensure_csv():
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "Timestamp", "Round", "Multiplier",
                "Prediction", "Cashout", "Signal", "Source"
            ])

def log_round(rn, mult, pred, source="auto"):
    ensure_csv()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ts, rn, mult,
                pred.get("action", "?"),
                pred.get("cashout", "?"),
                pred.get("signal", "?"),
                source
            ])
    except Exception as e:
        print(f"[SERVER] CSV write error: {e}")

def push_update(prediction, history, last_mult, round_num, status="Live"):
    """Thread-safe state update + WebSocket emit."""
    global _state
    new_state = {
        "prediction": prediction,
        "history": list(history)[-50:],
        "last_mult": last_mult,
        "round_num": round_num,
        "status": status,
    }
    with _state_lock:
        _state = new_state
    socketio.emit("update", new_state)
    if last_mult is not None:
        log_round(round_num, last_mult, prediction)

def push_status(msg):
    """Thread-safe status update."""
    with _state_lock:
        _state["status"] = msg
    socketio.emit("status", {"msg": msg})
    print(f"[BOT] {msg}")

@app.route("/")
def index():
    path = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard not found. Please check dashboard/index.html</h1>", 404

@app.route("/api/state")
def state():
    with _state_lock:
        return jsonify(dict(_state))

@app.route("/api/history")
def history_export():
    rows = []
    if os.path.exists(HISTORY_CSV):
        try:
            with open(HISTORY_CSV, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(rows)

@app.route("/api/manual_result", methods=["POST"])
def manual_result():
    """Manually add a round result for the predictor."""
    global _predictor_ref
    if _predictor_ref is None:
        return jsonify({"error": "Predictor not initialized yet"}), 503

    data = request.get_json(force=True, silent=True) or {}
    try:
        value = float(data.get("value", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value. Send a number like 2.35"}), 400

    if value < 1.0 or value > 200.0:
        return jsonify({"error": "Value must be between 1.0 and 200.0"}), 400

    _predictor_ref.add(value)
    _predictor_ref.save_history()        # <-- FIX: persist history on manual entry too
    pred = _predictor_ref.predict()
    hist = _predictor_ref.get_history_list()
    rn   = _predictor_ref.round_num

    # Push update to dashboard via WebSocket
    push_update(pred, hist, value, rn, "Manual")
    # NOTE: log_round is called inside push_update already, so NOT called again here
    # (was a duplicate logging bug in the old code)

    return jsonify({
        "ok": True,
        "value": value,
        "round_num": rn,
        "prediction": pred,
    })

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Clear predictor history (useful for fresh session)."""
    global _predictor_ref
    if _predictor_ref is None:
        return jsonify({"error": "Predictor not initialized yet"}), 503
    _predictor_ref.history.clear()
    _predictor_ref.round_num = 0
    _predictor_ref.save_history()
    push_update(_predictor_ref.predict(), [], None, 0, "History Cleared")
    return jsonify({"ok": True, "msg": "History cleared"})

def run_server(host="0.0.0.0", port=5000):
    ensure_csv()
    print(f"[SERVER] Dashboard running at http://localhost:{port}")
    socketio.run(
        app, host=host, port=port,
        debug=False, use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
