import threading, csv, os
from datetime import datetime
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__, static_folder="dashboard", template_folder="dashboard")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aviator_log.csv")

# ─── Thread-safe state ────────────────────────────────────────────────────────
_state_lock = threading.Lock()

_state = {
    "prediction": {
        "action": "WATCH", "cashout": "-", "confidence": 0,
        "conf_label": "Starting...", "reason": "Bot initializing...",
        "signal": "neutral", "votes": {}, "stats": {}, "total_score": 0
    },
    "history": [], "last_mult": None, "round_num": 0,
    "status": "Initializing...",
    "accuracy": {"total": 0, "correct": 0, "pct": 0},
    "session": {"rounds": 0, "start_time": None, "max_mult": 0.0, "min_mult": 999.0},
}

# ─── Globals ──────────────────────────────────────────────────────────────────
_predictor_ref    = None
_last_logged_round = 0          # FIX: avoid duplicate CSV rows
_last_push_time   = None        # heartbeat tracking
_scraper_reset_event = threading.Event()  # signals scraper to clear last_history


# ─── Public accessors ─────────────────────────────────────────────────────────
def set_predictor(p):
    global _predictor_ref
    _predictor_ref = p

def get_reset_event():
    return _scraper_reset_event


# ─── CSV helpers ──────────────────────────────────────────────────────────────
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


# ─── Accuracy updater ─────────────────────────────────────────────────────────
def _update_accuracy(prev_signal, actual_mult):
    """Compare last prediction signal vs the round result."""
    with _state_lock:
        acc = _state["accuracy"]
        acc["total"] += 1
        correct = False
        if prev_signal == "bet"  and actual_mult >= 2.0:  correct = True
        elif prev_signal == "safe" and actual_mult >= 1.5: correct = True
        elif prev_signal in ("skip", "wait") and actual_mult < 2.0: correct = True
        elif prev_signal == "neutral":                      correct = True
        if correct:
            acc["correct"] += 1
        acc["pct"] = round(acc["correct"] / acc["total"] * 100, 1) if acc["total"] else 0


# ─── Push helpers ─────────────────────────────────────────────────────────────
def push_update(prediction, history, last_mult, round_num, status="Live"):
    """Thread-safe state update + WebSocket emit. Only logs once per new round."""
    global _state, _last_logged_round, _last_push_time

    # --- Accuracy: compare previous prediction vs incoming actual result ---
    is_new_round = (last_mult is not None) and (round_num > _last_logged_round)
    if is_new_round:
        prev_signal = _state.get("prediction", {}).get("signal", "neutral")
        _update_accuracy(prev_signal, last_mult)

    # --- Session stats ---
    with _state_lock:
        sess = _state["session"]
        if last_mult is not None:
            if sess["start_time"] is None:
                sess["start_time"] = datetime.now().strftime("%H:%M:%S")
            sess["rounds"]   = round_num
            sess["max_mult"] = max(sess.get("max_mult", 0.0),   last_mult)
            sess["min_mult"] = min(sess.get("min_mult", 999.0), last_mult)
        acc  = _state["accuracy"].copy()
        sess = sess.copy()

    new_state = {
        "prediction": prediction,
        "history":    list(history)[-50:],
        "last_mult":  last_mult,
        "round_num":  round_num,
        "status":     status,
        "accuracy":   acc,
        "session":    sess,
    }

    with _state_lock:
        _state.update(new_state)
        _last_push_time = datetime.now()

    socketio.emit("update", new_state)

    # --- FIX: Log CSV only once per new round ---
    if is_new_round:
        log_round(round_num, last_mult, prediction, "auto")
        _last_logged_round = round_num

def push_status(msg):
    """Thread-safe status-only update."""
    global _last_push_time
    with _state_lock:
        _state["status"] = msg
        _last_push_time  = datetime.now()
    socketio.emit("status", {"msg": msg})
    print(f"[BOT] {msg}")


# ─── Flask routes ─────────────────────────────────────────────────────────────
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
    _predictor_ref.save_history()
    pred = _predictor_ref.predict()
    hist = _predictor_ref.get_history_list()
    rn   = _predictor_ref.round_num

    push_update(pred, hist, value, rn, "Manual")
    return jsonify({"ok": True, "value": value, "round_num": rn, "prediction": pred})

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    global _predictor_ref, _last_logged_round
    if _predictor_ref is None:
        return jsonify({"error": "Predictor not initialized yet"}), 503

    _predictor_ref.history.clear()
    _predictor_ref.round_num = 0
    _predictor_ref.save_history()
    _last_logged_round = 0

    # FIX: tell the scraper to clear its last_history list on next tick
    _scraper_reset_event.set()

    # Reset accuracy + session in state
    with _state_lock:
        _state["accuracy"] = {"total": 0, "correct": 0, "pct": 0}
        _state["session"]  = {"rounds": 0, "start_time": None,
                               "max_mult": 0.0, "min_mult": 999.0}

    push_update(_predictor_ref.predict(), [], None, 0, "History Cleared")
    return jsonify({"ok": True, "msg": "History cleared"})

@app.route("/api/heartbeat")
def heartbeat():
    """Returns seconds since last scraper push (for health monitoring)."""
    global _last_push_time
    if _last_push_time is None:
        return jsonify({"seconds_ago": None, "status": "not_started"})
    secs = (datetime.now() - _last_push_time).total_seconds()
    return jsonify({"seconds_ago": round(secs, 1),
                    "status": "ok" if secs < 60 else "stale"})


# ─── Server entry ─────────────────────────────────────────────────────────────
def run_server(host="0.0.0.0", port=5000):
    ensure_csv()
    print(f"[SERVER] Dashboard running at http://localhost:{port}")
    socketio.run(
        app, host=host, port=port,
        debug=False, use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
