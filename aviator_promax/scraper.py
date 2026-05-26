import os, time, socket
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

AVIATOR_URL = os.getenv("AVIATOR_URL", "https://pari-betting.com/en/casino/instant-games/game/spribe-in-aviator-insta")
PORT = int(os.getenv("DEBUG_PORT", "9222"))

GAME_SELECTORS = [
    ".payouts-block .payout",
    ".payouts-block div",
    ".stats-list .bubble-multiplier",
    ".payouts .payout-item",
    "div[class*='bubble']",
    "div[class*='multiplier']",
    ".stats-list div",
    ".bet-results__coeff",
    "app-game-history div",
    "[class*='history'] div",
]

def _port_open():
    """Check if Chrome remote debug port is active."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", PORT))
            return True
        except:
            return False

def scrape_loop_with_retry(predictor, push_fn, status_fn):
    """Outer retry loop - passively waits for Chrome, reconnects on crash."""
    while True:
        try:
            if not _port_open():
                status_fn(
                    "Waiting for Chrome... "
                    "Open Chrome with remote debugging on port 9222 "
                    "(use OPEN_CHROME.bat)."
                )
                time.sleep(5)
                continue
            scrape_loop(predictor, push_fn, status_fn)
        except KeyboardInterrupt:
            status_fn("Bot stopped by user.")
            break
        except Exception as e:
            err = str(e)[:80]
            status_fn(f"Disconnected: {err}. Reconnecting in 5s...")
            time.sleep(5)

def scrape_loop(predictor, push_fn, status_fn):
    from playwright.sync_api import sync_playwright

    status_fn("Chrome debug session detected! Connecting...")

    with sync_playwright() as pw:
        # --- Connect to running Chrome via CDP ---
        try:
            browser = pw.chromium.connect_over_cdp(f"http://localhost:{PORT}")
        except Exception as e:
            status_fn(f"CDP connection failed: {e}")
            time.sleep(5)
            return

        try:
            ctx = browser.contexts[0]
        except IndexError:
            status_fn("No browser context found. Is Aviator open in Chrome?")
            time.sleep(5)
            return

        # --- Wait for Aviator tab to be opened by user ---
        page = None
        while True:
            try:
                pages = ctx.pages
            except Exception:
                status_fn("Chrome connection lost. Waiting to reconnect...")
                return

            for p in pages:
                try:
                    url = p.url
                    if (
                        "spribe-in-aviator-insta" in url
                        or "aviator-next.spribegaming" in url
                        or "spribe" in url.lower()
                    ):
                        page = p
                        status_fn("Aviator game tab detected!")
                        break
                except Exception:
                    pass

            if page:
                break

            status_fn(
                "Chrome connected. "
                "Please open the Aviator game page in Chrome to start tracking..."
            )
            time.sleep(5)

        # --- Auto-click Play button if iframe not yet loaded ---
        if not _find_frame(page):
            status_fn("Looking for Play button...")
            try:
                for btn in page.locator("button").all()[:20]:
                    try:
                        txt = btn.text_content(timeout=1000).strip().lower()
                        if txt == "play":
                            status_fn("Play button found! Clicking...")
                            btn.click()
                            time.sleep(10)
                            break
                    except Exception:
                        pass
            except Exception as e:
                status_fn(f"Play button scan error: {str(e)[:50]}")

        # --- Wait for game iframe (up to 60s) ---
        frame = None
        for i in range(30):
            try:
                frame = _find_frame(page)
                if frame:
                    status_fn("Aviator game iframe found!")
                    break
            except Exception as e:
                status_fn(f"Iframe error: {str(e)[:50]}")
                return
            time.sleep(2)
            if i > 0 and i % 5 == 4:
                status_fn(f"Waiting for Aviator to load... ({(i+1)*2}s elapsed)")

        if not frame:
            status_fn("Aviator game elements not detected in any frame. Retrying...")
            time.sleep(5)
            return

        # --- Detect working data selector ---
        time.sleep(3)
        selector = _find_selector(frame)
        if selector:
            status_fn("LIVE data found! Predictions starting!")
        else:
            selector = ".payouts-block .payout"
            status_fn("Tracking active (fallback selector).")

        # --- Live prediction loop ---
        last_history = []
        last_mult = None
        consecutive_empty = 0

        while True:
            try:
                elements = frame.locator(selector).all()
                current = []
                for el in elements:
                    try:
                        txt = el.text_content(timeout=1000)
                        v = _parse(txt)
                        if v is not None:
                            current.append(v)
                    except Exception:
                        continue

                if current:
                    consecutive_empty = 0
                    if not last_history:
                        # First batch of data
                        last_history = current
                        for v in reversed(current[:30]):
                            predictor.add(v)
                        last_mult = current[0]
                        predictor.save_history()
                    else:
                        # Detect new rounds since last read
                        new_rounds = []
                        for v in current:
                            if v == last_history[0]:
                                break
                            new_rounds.append(v)

                        if new_rounds:
                            for v in reversed(new_rounds):
                                predictor.add(v)
                            last_history = current
                            last_mult = new_rounds[0]
                            predictor.save_history()

                    pred = predictor.predict()
                    push_fn(
                        pred,
                        predictor.get_history_list(),
                        last_mult,
                        predictor.round_num,
                        "Live",
                    )
                else:
                    consecutive_empty += 1
                    # If we see 30+ consecutive empty reads, selector may have broken
                    if consecutive_empty == 30:
                        status_fn("No data from selector. Trying to re-detect...")
                        new_sel = _find_selector(frame)
                        if new_sel and new_sel != selector:
                            selector = new_sel
                            status_fn(f"Switched to selector: {selector}")
                        consecutive_empty = 0

                time.sleep(1.5)

            except KeyboardInterrupt:
                status_fn("Bot stopped by user.")
                raise
            except Exception as e:
                err = str(e)[:60]
                status_fn(f"Tracking error: {err}")
                time.sleep(3)
                # Check if page connection is still alive
                try:
                    _ = page.url
                except Exception:
                    status_fn("Page closed or tab navigated away. Reconnecting...")
                    raise RuntimeError("Page closed - need reconnect")


# ===== HELPERS =====

def _find_frame(page):
    """Find the Aviator game iframe inside the page."""
    try:
        # Look for a frame containing payouts-block or stats-list or bubble-multiplier
        for f in page.frames:
            try:
                if (
                    f.locator(".payouts-block").count() > 0
                    or f.locator(".stats-list").count() > 0
                    or f.locator(".bubble-multiplier").count() > 0
                ):
                    return f
            except Exception:
                pass
    except Exception:
        pass
    return None

def _find_selector(frame):
    """Try each known selector and return the first one that yields numeric data."""
    for sel in GAME_SELECTORS:
        try:
            els = frame.locator(sel).all()
            valid = 0
            for el in els[:5]:
                try:
                    txt = el.text_content(timeout=1000).strip()
                    if _parse(txt) is not None:
                        valid += 1
                except Exception:
                    continue
            if valid >= 2:
                return sel
        except Exception:
            continue
    return None

def _parse(txt):
    """Parse a multiplier string like '2.35x' -> 2.35. Returns None on failure."""
    try:
        cleaned = txt.strip().replace("x", "").replace(",", ".").replace(" ", "")
        v = float(cleaned)
        # Sanity check: Aviator multipliers are always >= 1.00
        if v >= 1.0:
            return v
    except Exception:
        pass
    return None
