import time
import datetime
import traceback
import scraper
import predict_ml_v2 as predict_ml

def start_bot():
    print("==================================================")
    print(" KOLKATA FF BACKGROUND WORKER STARTED (WEB MODE)  ")
    print(" (Scrapes new data & retrains AI every 10 mins)   ")
    print("==================================================")
    
    while True:
        try:
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Automatic background scrape initiated...")
            # Fetch latest data
            scraper.scrape_kolkata_ff()
            
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Refreshing AI Model with latest data...")
            predict_ml.train_and_save_model()
            
            # Wait 10 minutes (600 seconds)
            time.sleep(600)
            
        except KeyboardInterrupt:
            print("\nBackground worker stopped manually.")
            break
        except Exception as e:
            print(f"Background worker error: {e}")
            traceback.print_exc()
            time.sleep(60) # Try again in 1 min if error

if __name__ == "__main__":
    start_bot()
