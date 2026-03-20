from keep_alive import keep_alive
from bot import start_bot

if __name__ == "__main__":
    # Start the web server in a background thread
    keep_alive()
    # Start the main bot process
    start_bot()
