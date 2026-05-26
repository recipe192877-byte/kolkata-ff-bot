import time
import datetime
import traceback
import os
import pandas as pd
import scraper
import predict_ml_v2 as predict_ml
from dotenv import load_dotenv

load_dotenv()

def safe_scrape():
    """Scrape directly, AI diagnosis is now handled by RuFlo Upgrader daily."""
    return scraper.scrape_kolkata_ff()

def safe_retrain():
    """Retrain directly, AI diagnosis is now handled by RuFlo Upgrader daily."""
    return predict_ml.train_and_save_model()


def start_bot():
    print("==================================================")
    print(" KOLKATA FF BACKGROUND WORKER v3.0 (AI POWERED)   ")
    print(" 4-Model Ensemble | Vector Memory | Auto-Healer   ")
    print("==================================================")
    print(f" Brain Capacity: {predict_ml.brain.get_brain_capacity()} patterns")
    print(f" Healer Status:  {'ONLINE' if os.environ.get('GEMINI_API_KEY') else 'NO API KEY (set GEMINI_API_KEY)'}")
    print("==================================================")
    
    last_record_count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    last_evolution_date = None
    
    # Initialize last_record_count from existing data if available
    try:
        df = pd.read_csv(predict_ml.DATA_FILE)
        last_record_count = len(df)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Initial record count: {last_record_count}")
    except FileNotFoundError:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Data file not found at {predict_ml.DATA_FILE}. Starting with 0 records.")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error initializing record count: {e}. Starting with 0 records.")


    while True:
        try:
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Automatic background scrape initiated...")
            
            # Fetch latest data (auto-healed)
            safe_scrape()
            
            # Check if data actually changed before expensive retrain
            current_count = 0 
            try:
                df = pd.read_csv(predict_ml.DATA_FILE)
                current_count = len(df)
            except FileNotFoundError:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Data file not found after scrape. Possible scraping issue or first run.")
                current_count = 0 # If file not found, treat as 0 records
            except Exception as e:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error reading data file to get current count: {e}")
                current_count = 0 # Fallback to 0 if an error occurs during read
                
            if current_count > last_record_count:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New data detected ({current_count} vs {last_record_count}). Retraining AI Model...")
                safe_retrain()
                last_record_count = current_count
                # Brain is already rebuilt inside train_and_save_model() â€” no need to save again
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Retrain complete. Brain capacity: {predict_ml.brain.get_brain_capacity()} patterns")

            elif current_count < last_record_count:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Data count decreased ({current_count} vs {last_record_count}). This shouldn't happen. Re-initializing last_record_count.")
                last_record_count = current_count
                # Even if data count decreased, we should still retrain IF current_count > 0.
                # A decrease in data might mean some data was cleaned or removed, which could also warrant a retrain.
                # But for the purpose of "only retrain on new data" as per comment, we won't retrain here.
                # If the underlying cause is data corruption, a later scrape will hopefully fix it.
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%M')}] No new data. Skipping retrain. ({current_count} records)")
            
            # Daily AI Evolution - run once per day (with retry on failure)
            current_date = datetime.datetime.now().date()
            if last_evolution_date is None or current_date != last_evolution_date:
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Running Daily AI Evolution...")
                try:
                    yesterday_stats = predict_ml.get_yesterday_stats()
                    if yesterday_stats and yesterday_stats.get('total_bazis', 0) > 0:
                        from ai_council import council
                        import json
                        
                        evo_result = council.hold_evolution_meeting(yesterday_stats)
                        
                        report = {
                            "date_run": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                            "target_date": yesterday_stats.get('date'),
                            "yesterday_stats": yesterday_stats,
                            "evolution_result": evo_result
                        }
                        
                        if evo_result.get('status') == 'success' and 'config' in evo_result:
                            predict_ml.save_ai_config(evo_result['config'])
                            print(f"[EVOLUTION] Successfully updated AI config: {evo_result.get('reason')}")
                            last_evolution_date = current_date  # Mark as done ONLY on success
                        else:
                            print(f"[EVOLUTION] Meeting failed: {evo_result.get('message', 'Unknown error')}. Will retry on next cycle.")
                            # DON'T set last_evolution_date â€” allow retry on next loop cycle
                            
                        # Ensure the 'reports' directory exists
                        os.makedirs('reports', exist_ok=True)
                        report_filename = f"reports/daily_report_{current_date.strftime('%Y%m%d')}.json"
                        with open(report_filename, 'w') as f:
                            json.dump(report, f, indent=4)
                        # Keep the root daily_report.json updated for the API
                        with open('daily_report.json', 'w') as f:
                            json.dump(report, f, indent=4)
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Daily evolution report saved to {report_filename} and daily_report.json")

                    else:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Not enough data for yesterday to run AI Evolution or stats not available.")
                        last_evolution_date = current_date  # No data to evolve on, skip for today
                except ImportError:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] AI Council module not found. Skipping daily evolution.")
                    last_evolution_date = current_date  # Module missing, won't help to retry
                except Exception as evo_err:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error during daily evolution: {evo_err}")
                    traceback.print_exc()
                    # DON'T set last_evolution_date â€” allow retry on next loop cycle

            # Reset failure counter on success
            consecutive_failures = 0
            
            # Wait 60 minutes
            time.sleep(300)  # 5 min cycle for fresh bazi
            
        except KeyboardInterrupt:
            print("\nBackground worker stopped manually.")
            predict_ml.brain.save_brain()
            break
        except Exception as e:
            consecutive_failures += 1
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Background worker error (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
            traceback.print_exc()
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[CRITICAL] {MAX_CONSECUTIVE_FAILURES} consecutive failures. Resetting state and waiting 10 minutes before retrying...")
                consecutive_failures = 0 # Reset to allow future attempts
                time.sleep(600)
            else:
                # Exponential backoff: 2min, 4min, 8min, etc. up to 10 minutes (600s)
                # Modified to start from 120s if consecutive_failures is 1
                wait_time = min(120 * (2 ** (consecutive_failures - 1)), 600)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Retrying in {wait_time}s...")
                time.sleep(wait_time)

if __name__ == "__main__":
    start_bot()
