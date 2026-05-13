"""
Auto-Healer 2.0 — AI-Powered Code Consultant for Kolkata FF Bot
Uses OpenRouter API to diagnose and suggest fixes for runtime errors.
Logs all healing events to heal_log.json for transparency.
"""
import os
import json
import time
import traceback
import inspect
import functools
import requests
from datetime import datetime, timezone, timedelta

HEAL_LOG_FILE = 'heal_log.json'
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
MAX_HEALS_PER_HOUR = 5
IST = timezone(timedelta(hours=5, minutes=30))


class AutoHealer:
    """AI-powered self-healing system that diagnoses runtime errors."""

    def __init__(self):
        self.api_key = os.environ.get('OPENROUTER_API_KEY', '')
        self.heal_log = []
        self.heal_timestamps = []  # For rate limiting
        self._load_log()

    def _load_log(self):
        """Load healing history from disk."""
        if os.path.exists(HEAL_LOG_FILE):
            try:
                with open(HEAL_LOG_FILE, 'r') as f:
                    self.heal_log = json.load(f)
            except Exception:
                self.heal_log = []

    def _save_log(self):
        """Save healing history to disk (keep last 100 entries)."""
        try:
            with open(HEAL_LOG_FILE, 'w') as f:
                json.dump(self.heal_log[-100:], f, indent=2)
        except Exception as e:
            print(f"[HEALER] Error saving log: {e}")

    def _is_rate_limited(self):
        """Check if we've exceeded the heal rate limit."""
        now = time.time()
        # Remove timestamps older than 1 hour
        self.heal_timestamps = [t for t in self.heal_timestamps if now - t < 3600]
        return len(self.heal_timestamps) >= MAX_HEALS_PER_HOUR

    def get_log(self):
        """Return the heal log for API display."""
        return self.heal_log[-20:]  # Last 20 entries

    def get_status(self):
        """Return healer status."""
        now = time.time()
        active_heals = len([t for t in self.heal_timestamps if now - t < 3600])
        return {
            "api_key_set": bool(self.api_key),
            "total_heals": len(self.heal_log),
            "heals_this_hour": active_heals,
            "rate_limit": f"{active_heals}/{MAX_HEALS_PER_HOUR}",
            "last_heal": self.heal_log[-1]['timestamp'] if self.heal_log else None
        }

    def diagnose(self, error_type, error_message, traceback_str, function_name, code_context=""):
        """Send error to AI for diagnosis and fix suggestion."""
        if not self.api_key:
            diagnosis = f"[HEALER] No API key set. Set OPENROUTER_API_KEY environment variable. Error: {error_message}"
            self._log_event(function_name, error_type, error_message, diagnosis, ai_used=False)
            return diagnosis

        if self._is_rate_limited():
            diagnosis = f"[HEALER] Rate limited ({MAX_HEALS_PER_HOUR}/hour). Error logged but not sent to AI: {error_message}"
            self._log_event(function_name, error_type, error_message, diagnosis, ai_used=False)
            return diagnosis

        # Build the AI prompt
        prompt = f"""You are an expert Python developer debugging a Kolkata FF prediction bot.
The bot scrapes lottery results, trains ML models (XGBoost, LightGBM, RF, GB ensemble), and serves predictions via Flask API.

A runtime error occurred:

**Function:** `{function_name}`
**Error Type:** `{error_type}`
**Error Message:** `{error_message}`

**Full Traceback:**
```
{traceback_str[-1500:]}
```

{f'**Code Context:**{chr(10)}```python{chr(10)}{code_context[-1000:]}{chr(10)}```' if code_context else ''}

Please provide:
1. **Root Cause** (1-2 sentences)
2. **Fix** (exact code change needed)
3. **Prevention** (how to avoid this in future)

Be concise and specific."""

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'HTTP-Referer': 'https://kolkata-ff-bot.onrender.com',
                    'X-Title': 'Kolkata FF Auto-Healer'
                },
                json={
                    'model': 'openrouter/free',
                    'messages': [
                        {'role': 'system', 'content': 'You are a senior Python developer. Give concise, actionable fixes.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'max_tokens': 512
                },
                timeout=30
            )

            self.heal_timestamps.append(time.time())

            if response.status_code == 200:
                data = response.json()
                diagnosis = data.get('choices', [{}])[0].get('message', {}).get('content', 'No diagnosis available.')
            else:
                diagnosis = f"API Error {response.status_code}: {response.text[:200]}"

        except requests.exceptions.Timeout:
            diagnosis = f"[HEALER] AI request timed out. Error was: {error_message}"
        except Exception as e:
            diagnosis = f"[HEALER] AI call failed: {str(e)}. Original error: {error_message}"

        self._log_event(function_name, error_type, error_message, diagnosis, ai_used=True)
        return diagnosis

    def _log_event(self, function_name, error_type, error_message, diagnosis, ai_used=False):
        """Log a healing event."""
        event = {
            "timestamp": datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST'),
            "function": function_name,
            "error_type": error_type,
            "error_message": str(error_message)[:300],
            "diagnosis": str(diagnosis)[:1000],
            "ai_used": ai_used
        }
        self.heal_log.append(event)
        self._save_log()
        ai_label = "[AI]" if ai_used else "[LOCAL]"
        print(f"\n[HEALER] {ai_label} diagnosis for {function_name}:")
        print(f"  Error: {error_message[:150]}")
        print(f"  Fix: {diagnosis[:200]}")


# Global healer instance
healer = AutoHealer()


def self_healing(max_retries=2, fallback_value=None):
    """
    Decorator that wraps functions with auto-healing capability.
    On crash: captures error, asks AI for diagnosis, retries, or returns fallback.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    error_msg = str(e)
                    tb_str = traceback.format_exc()

                    # Try to get source code context
                    try:
                        source = inspect.getsource(func)
                    except Exception:
                        source = ""

                    diagnosis = healer.diagnose(
                        error_type=error_type,
                        error_message=error_msg,
                        traceback_str=tb_str,
                        function_name=func.__name__,
                        code_context=source
                    )

                    if attempt < max_retries:
                        wait = (attempt + 1) * 5  # 5s, 10s backoff
                        print(f"[HEALER] Retrying {func.__name__} in {wait}s (attempt {attempt + 2}/{max_retries + 1})...")
                        time.sleep(wait)
                    else:
                        print(f"[HEALER] {func.__name__} failed after {max_retries + 1} attempts. Using fallback.")

            # All retries exhausted — return fallback
            if fallback_value is not None:
                return fallback_value
            raise last_error

        return wrapper
    return decorator
