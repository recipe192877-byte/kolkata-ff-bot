import threading, time, webbrowser, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import ProMaxPredictor
from server import run_server, push_update, push_status, set_predictor
from scraper import scrape_loop_with_retry

def start_scraper(predictor):
    """Run scraper with auto-retry in its own thread."""
    scrape_loop_with_retry(predictor, push_update, push_status)

def main():
    predictor = ProMaxPredictor()

    # Load saved history from disk if available
    predictor.load_history()

    # Wire predictor into server so /api/manual_result can use it
    set_predictor(predictor)

    print("=" * 55)
    print("  AVIATOR PRO MAX PREDICTION BOT")
    print("  Starting all systems...")
    print("=" * 55)

    # Start scraper in BACKGROUND thread
    scraper_thread = threading.Thread(
        target=start_scraper,
        args=(predictor,),
        daemon=True,
        name="ScraperThread"
    )
    scraper_thread.start()
    print("[OK] Chrome bot started in background")

    # Wait a moment then open browser
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:5000")
        print("[OK] Dashboard opened in browser -> http://localhost:5000")
    except:
        print("[!] Open http://localhost:5000 manually in browser")

    # Run Flask server in MAIN thread (blocking)
    print("[OK] Starting dashboard server...")
    run_server(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
