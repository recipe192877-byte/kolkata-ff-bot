import os, time, subprocess, socket
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

CHROME_EXE  = os.getenv("CHROME_EXE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PROFILE_DIR = os.getenv("PROFILE_DIR", r"C:\chrome_aviator_manual")
AVIATOR_URL = os.getenv("AVIATOR_URL", "https://pari-betting.com/en/casino/instant-games/game/spribe-in-aviator-insta")
PORT = int(os.getenv("DEBUG_PORT", "9222"))

GAME_SELECTORS = [
    ".stats-list .bubble-multiplier",
    ".payouts .payout-item",
    "div[class*='bubble']",
    "div[class*='multiplier']",
    ".stats-list div",
    ".payouts-block div",
    ".bet-results__coeff",
    "app-game-history div",
    "[class*='history'] div",
]

def _port_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", PORT))
            return True
        except:
            return False

def scrape_loop_with_retry(predictor, push_fn, status_fn):
    """Outer retry loop — reconnects automatically on any crash."""
    while True:
        try:
            scrape_loop(predictor, push_fn, status_fn)
        except KeyboardInterrupt:
            status_fn("Bot stopped by user.")
            break
        except Exception as e:
            status_fn(f"Scraper crashed: {str(e)[:80]}. Retrying in 10s...")
            time.sleep(10)

def scrape_loop(predictor, push_fn, status_fn):
    from playwright.sync_api import sync_playwright

    status_fn("Checking Chrome remote debugging port...")

    chrome_proc = None
    if not _port_open():
        status_fn("Launching Google Chrome in debug mode...")
        cmd = f'"{CHROME_EXE}" --remote-debugging-port={PORT} --user-data-dir="{PROFILE_DIR}"'
        chrome_proc = subprocess.Popen(cmd, shell=True)
        # Wait for Chrome to start (up to 15s)
        for _ in range(15):
            time.sleep(1)
            if _port_open():
                break
        else:
            status_fn("Chrome did not start in time. Retrying...")
            if chrome_proc:
                chrome_proc.terminate()
            return
    else:
        status_fn("Chrome debug session already active. Connecting...")

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(f"http://localhost:{PORT}")
            ctx = browser.contexts[0]
        except Exception as e:
            status_fn(f"CDP connection failed: {e}")
            if chrome_proc:
                chrome_proc.terminate()
            return

        # Find existing Aviator page or open new
        page = None
        for p in ctx.pages:
            try:
                if "spribe-in-aviator-insta" in p.url or "aviator-next.spribegaming" in p.url:
                    page = p
                    status_fn("Existing Aviator game tab detected!")
                    break
            except:
                pass

        if not page:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            status_fn("Navigating to Aviator game...")
            try:
                page.goto(AVIATOR_URL, timeout=30000, wait_until="domcontentloaded")
                time.sleep(5)
                _dismiss_popups(page)
            except Exception as e:
                status_fn(f"Nav error: {e}")
                time.sleep(5)

        # Auto-click Play button if game iframe not yet loaded
        frame = _find_frame(page)
        if not frame:
            status_fn("Looking for Play button...")
            try:
                for btn in page.locator("button").all()[:20]:
                    if btn.inner_text(timeout=500).strip().lower() == "play":
                        status_fn("Play button found! Clicking...")
                        btn.click()
                        time.sleep(10)
                        break
            except Exception as e:
                status_fn(f"Play button error: {e}")

        # Find iframe (wait up to 60s)
        frame = None
        for i in range(30):
            frame = _find_frame(page)
            if frame:
                status_fn("Aviator game iframe found!")
                break
            time.sleep(2)
            if i % 5 == 4:
                status_fn(f"Waiting for Aviator to load... ({(i+1)*2}s)")

        if not frame:
            frame = page.main_frame
            status_fn("Using main frame (iframe not found).")

        # Find data selector
        time.sleep(3)
        selector = _find_selector(frame)
        if selector:
            status_fn("LIVE data found! Predictions starting!")
        else:
            selector = ".stats-list div"
            status_fn("Tracking active.")

        # Live prediction loop
        last_history = []
        last_mult    = None

        while True:
            try:
                elements = frame.locator(selector).all()
                current = [v for el in elements
                           if (v := _parse(el.inner_text())) is not None]

                if current:
                    if not last_history:
                        last_history = current
                        for v in reversed(current[:30]):
                            predictor.add(v)
                        last_mult = current[0]
                        predictor.save_history()
                    else:
                        new_rounds = []
                        for v in current:
                            if v == last_history[0]: break
                            new_rounds.append(v)
                        if new_rounds:
                            for v in reversed(new_rounds):
                                predictor.add(v)
                            last_history = current
                            last_mult    = new_rounds[0]
                            predictor.save_history()

                    pred = predictor.predict()
                    push_fn(pred, predictor.get_history_list(),
                            last_mult, predictor.round_num, "Live")

                time.sleep(1.5)

            except KeyboardInterrupt:
                status_fn("Bot stopped.")
                raise
            except Exception as e:
                status_fn(f"Error: {str(e)[:60]}")
                time.sleep(3)
                # Check if page is still alive
                try:
                    _ = page.url
                except:
                    status_fn("Page closed. Reconnecting...")
                    raise RuntimeError("Page closed, need reconnect")

        browser.close()


def _dismiss_popups(page):
    for sel in ["button:has-text('No, thanks')", "button:has-text('Accept')",
                "button:has-text('Close')", "[aria-label='Close']",
                "[class*='close-btn']", "[class*='modal__close']"]:
        try:
            for el in page.locator(sel).all()[:2]:
                if el.is_visible(timeout=400):
                    el.click(timeout=400); time.sleep(0.2)
        except: pass

def _find_frame(page):
    for f in page.frames:
        u = f.url.lower()
        if any(k in u for k in ["aviator","spribe","aviatorgame"]):
            return f
    return None

def _find_selector(frame):
    for sel in GAME_SELECTORS:
        try:
            valid = sum(1 for el in frame.locator(sel).all()[:5]
                       if _parse(el.inner_text()) is not None)
            if valid >= 2: return sel
        except: continue
    return None

def _parse(txt):
    try:
        return float(txt.strip().replace("x","").replace(",",".").replace(" ",""))
    except: return None
