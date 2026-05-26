import threading, time, webbrowser, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import ProMaxPredictor
from server   import run_server, push_update, push_status, set_predictor
from scraper  import scrape_loop_with_retry

def start_scraper(predictor):
    """Run the passive Chrome tracker with auto-retry in its own thread."""
    try:
        scrape_loop_with_retry(predictor, push_update, push_status)
    except Exception as e:
        push_status(f"Scraper thread crashed unexpectedly: {e}")

def main():
    predictor = ProMaxPredictor()

    # Restore saved history from disk (if available)
    predictor.load_history()

    # Wire predictor into Flask server for /api/manual_result
    set_predictor(predictor)

    print("=" * 55)
    print("  AVIATOR PRO MAX PREDICTION BOT")
    print("  Passive Chrome Tracker + AI Predictor")
    print("=" * 55)
    print()
    print("  INSTRUCTIONS:")
    print("  1. Run OPEN_CHROME.bat to open Chrome in debug mode.")
    print("  2. Log in and open the Aviator game page in Chrome.")
    print("  3. The bot will auto-detect and start tracking!")
    print()
    print("=" * 55)

    # Start scraper in BACKGROUND thread (daemon: dies with main process)
    scraper_thread = threading.Thread(
        target=start_scraper,
        args=(predictor,),
        daemon=True,
        name="ScraperThread",
    )
    scraper_thread.start()
    print("[OK] Scraper started - waiting for Chrome...")

    # Open dashboard in browser after a short delay
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:5000")
        print("[OK] Dashboard opened -> http://localhost:5000")
    except Exception:
        print("[!] Open http://localhost:5000 manually in your browser")

    # Run Flask server in MAIN thread (blocks until Ctrl+C)
    print("[OK] Starting dashboard server...")
    try:
        run_server(host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user.")
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print("\n[ERROR] Port 5000 is already in use!")
            print("  -> Close any other running instance of the bot first.")
            print("  -> Or change the port in server.py / main.py")
        else:
            raise

if __name__ == "__main__":
    main()
