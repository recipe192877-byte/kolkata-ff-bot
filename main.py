from keep_alive import keep_alive
from bot import start_bot

if __name__ == "__main__":
    print("=" * 55)
    print("  KOLKATA FF v3.0 — AI BRAIN + AUTO-HEALER EDITION")
    print("=" * 55)
    # Start the web server in a background thread
    keep_alive()
    # Start the main bot process (with auto-healing)
    start_bot()
