import threading, time, webbrowser, os, sys, socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import ProMaxPredictor
from server   import run_server, push_update, push_status, set_predictor, get_reset_event
from scraper  import scrape_loop_with_retry

SERVER_PORT = 5000

def _wait_for_flask(port=SERVER_PORT, timeout=15):
    """FIX: wait until Flask is actually listening before opening browser."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False

def start_scraper(predictor, reset_event):
    """Run the passive Chrome tracker with auto-retry in its own thread."""
    try:
        scrape_loop_with_retry(predictor, push_update, push_status, reset_event)
    except Exception as e:
        push_status(f"Scraper thread crashed unexpectedly: {e}")

def main():
    predictor    = ProMaxPredictor()
    reset_event  = get_reset_event()

    predictor.load_history()
    set_predictor(predictor)

    print("=" * 55)
    print("  AVIATOR PRO MAX PREDICTION BOT  v2.0")
    print("  Passive Chrome Tracker + AI Predictor")
    print("=" * 55)
    print()
    print("  INSTRUCTIONS:")
    print("  1. Run OPEN_CHROME.bat to open Chrome in debug mode.")
    print("  2. Log in and open the Aviator game page in Chrome.")
    print("  3. The bot will auto-detect and start tracking!")
    print()
    print("=" * 55)

    # Start scraper in background thread
    scraper_thread = threading.Thread(
        target=start_scraper,
        args=(predictor, reset_event),
        daemon=True,
        name="ScraperThread",
    )
    scraper_thread.start()
    print("[OK] Scraper started - waiting for Chrome...")

    # Start Flask in a non-blocking thread so we can wait for readiness
    flask_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "0.0.0.0", "port": SERVER_PORT},
        daemon=True,
        name="FlaskThread",
    )
    flask_thread.start()
    print("[OK] Starting dashboard server...")

    # FIX: wait until Flask is actually ready before opening browser
    if _wait_for_flask(SERVER_PORT):
        try:
            webbrowser.open(f"http://localhost:{SERVER_PORT}")
            print(f"[OK] Dashboard opened -> http://localhost:{SERVER_PORT}")
        except Exception:
            print(f"[!] Open http://localhost:{SERVER_PORT} manually in your browser")
    else:
        print(f"[!] Server not ready in time. Open http://localhost:{SERVER_PORT} manually.")

    # Keep main thread alive (Flask runs in daemon thread, would die otherwise)
    try:
        while flask_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user.")

if __name__ == "__main__":
    main()
