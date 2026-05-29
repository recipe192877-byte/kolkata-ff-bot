import os, time, socket
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

AVIATOR_URL = os.getenv("AVIATOR_URL",
    "https://pari-betting.com/en/casino/instant-games/game/spribe-in-aviator-insta")
PORT = int(os.getenv("DEBUG_PORT", "9222"))

# ─── Selector priority list (confirmed working ones first) ───────────────────
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

# ─── CDP port check ───────────────────────────────────────────────────────────
def _port_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", PORT))
            return True
        except:
            return False


# ─── Outer retry loop ─────────────────────────────────────────────────────────
def scrape_loop_with_retry(predictor, push_fn, status_fn, reset_event=None):
    """Passively waits for Chrome on port 9222, reconnects on any crash."""
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
            scrape_loop(predictor, push_fn, status_fn, reset_event)
        except KeyboardInterrupt:
            status_fn("Bot stopped by user.")
            break
        except Exception as e:
            err = str(e)[:80]
            status_fn(f"Disconnected: {err}. Reconnecting in 5s...")
            time.sleep(5)


# ─── Main scrape loop ─────────────────────────────────────────────────────────
def scrape_loop(predictor, push_fn, status_fn, reset_event=None):
    from playwright.sync_api import sync_playwright

    status_fn("Chrome debug session detected! Connecting...")

    with sync_playwright() as pw:
        # Connect to running Chrome via CDP
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

        # ── Wait for Aviator tab ─────────────────────────────────────────────
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
                    # FIX: more specific URL check to avoid wrong Spribe games
                    if (
                        "spribe-in-aviator" in url.lower()
                        or "aviator-next.spribegaming" in url.lower()
                        or ("spribe" in url.lower() and "aviator" in url.lower())
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

        # ── Auto-click Play button if needed ─────────────────────────────────
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

        # ── Wait for game iframe ──────────────────────────────────────────────
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

        # ── Detect primary working selector ──────────────────────────────────
        time.sleep(3)
        selector = _find_selector(frame)
        using_fallback = selector is None
        if selector:
            status_fn("LIVE data found! Predictions starting!")
        else:
            selector = ".payouts-block .payout"
            status_fn("Tracking active (fallback selector).")

        # ── Live prediction loop ──────────────────────────────────────────────
        last_history     = []
        last_mult        = None
        consecutive_empty = 0
        frame_miss_count  = 0          # FIX: track frame health

        while True:
            try:
                # FIX: honour reset signal from clear_history API
                if reset_event is not None and reset_event.is_set():
                    last_history = []
                    last_mult    = None
                    reset_event.clear()
                    status_fn("History cleared. Resuming tracking...")

                elements = frame.locator(selector).all()
                current  = []
                for el in elements:
                    try:
                        txt = el.text_content(timeout=1000)
                        v   = _parse(txt)
                        if v is not None:
                            current.append(v)
                    except Exception:
                        continue

                if current:
                    consecutive_empty = 0
                    frame_miss_count  = 0
                    if not last_history:
                        # First batch of data — seed history
                        last_history = current
                        for v in reversed(current[:30]):
                            predictor.add(v)
                        last_mult = current[0]
                        predictor.save_history()
                    else:
                        # FIX: sequence-overlap detection (handles identical values)
                        new_count = _detect_new_rounds(current, last_history)
                        new_rounds = current[:new_count]

                        if new_rounds:
                            for v in reversed(new_rounds):
                                predictor.add(v)
                            last_history = current
                            last_mult    = new_rounds[0]
                            predictor.save_history()

                    pred = predictor.predict()
                    push_fn(
                        pred,
                        predictor.get_history_list(),
                        last_mult,
                        predictor.round_num,
                        "Live" if not using_fallback else "Live (fallback)",
                    )

                else:
                    consecutive_empty += 1
                    frame_miss_count  += 1

                    # After 30 empty reads — try re-detecting selector AND frame
                    if consecutive_empty >= 30:
                        status_fn("No data. Re-detecting frame & selector...")
                        # FIX: re-run _find_frame in case iframe reloaded
                        new_frame = _find_frame(page)
                        if new_frame:
                            frame = new_frame
                        new_sel = _find_selector(frame)
                        if new_sel:
                            selector      = new_sel
                            using_fallback = False
                            status_fn(f"Recovered! Using selector: {selector}")
                        consecutive_empty = 0

                time.sleep(1.5)

            except KeyboardInterrupt:
                status_fn("Bot stopped by user.")
                raise
            except Exception as e:
                err = str(e)[:60]
                status_fn(f"Tracking error: {err}")
                time.sleep(3)
                try:
                    _ = page.url
                except Exception:
                    status_fn("Page closed. Reconnecting...")
                    raise RuntimeError("Page closed - need reconnect")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _detect_new_rounds(current, last_history):
    """
    Sequence-overlap detection.
    Returns how many items at the TOP of 'current' are new since 'last_history'.
    Handles identical consecutive round values correctly.
    """
    if not last_history:
        return len(current)
    # For each position i, compare current[i:i+k] vs last_history[0:k]
    # k adapts to available elements so the comparison always works
    for i in range(len(current)):
        remaining = len(current) - i
        k = min(3, remaining, len(last_history))
        if k == 0:
            break
        if current[i : i + k] == last_history[:k]:
            return i
    # No overlap found - treat up to 10 as new to avoid spam
    return min(len(current), 10)


def _find_frame(page):
    """Find the Aviator game iframe by checking for known game elements."""
    try:
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
    """Try each known selector; return first one yielding >= 2 valid values."""
    for sel in GAME_SELECTORS:
        try:
            els   = frame.locator(sel).all()
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
    """Parse '2.35x' -> 2.35. Returns None on failure."""
    try:
        cleaned = txt.strip().replace("x", "").replace(",", ".").replace(" ", "")
        v = float(cleaned)
        if v >= 1.0:
            return v
    except Exception:
        pass
    return None
