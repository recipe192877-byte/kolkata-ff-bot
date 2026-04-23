import time
import datetime
import traceback
import os
import pandas as pd
import scraper
import predict_ml_v2 as predict_ml

def start_bot():
    print("==================================================")
    print(" KOLKATA FF BACKGROUND WORKER v2.0 (WEB MODE)     ")
    print(" 6-Model Ensemble | Auto-Scrape every 60 mins     ")
    print("==================================================")
    
    last_record_count = 0
    
    while True:
        try:
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Automatic background scrape initiated...")
            # Fetch latest data
            scraper.scrape_kolkata_ff()
            
            # Check if data actually changed before expensive retrain
            try:
                df = pd.read_csv(predict_ml.DATA_FILE)
                current_count = len(df)
            except Exception:
                current_count = 0
                
            if current_count != last_record_count:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New data detected ({current_count} vs {last_record_count}). Retraining AI Model...")
                predict_ml.train_and_save_model()
                last_record_count = current_count
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No new data. Skipping retrain. ({current_count} records)")
            
            # Wait 60 minutes (3600 seconds) — Stacking model training is expensive
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("\nBackground worker stopped manually.")
            break
        except Exception as e:
            print(f"Background worker error: {e}")
            traceback.print_exc()
            time.sleep(120) # Try again in 2 min if error

if __name__ == "__main__":
    start_bot()
