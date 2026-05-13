import time
import datetime
import traceback
import os
import pandas as pd
import scraper
import predict_ml_v2 as predict_ml
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
    print(f" Healer Status:  {'ONLINE' if os.environ.get('OPENROUTER_API_KEY') else 'NO API KEY (set OPENROUTER_API_KEY)'}")
    print("==================================================")
    
    last_record_count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    last_evolution_date = None
    
    while True:
        try:
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Automatic background scrape initiated...")
            
            # Fetch latest data (auto-healed)
            safe_scrape()
            
            # Check if data actually changed before expensive retrain
            try:
                df = pd.read_csv(predict_ml.DATA_FILE)
                current_count = len(df)
            except Exception:
                current_count = 0
                
            if current_count != last_record_count:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New data detected ({current_count} vs {last_record_count}). Retraining AI Model...")
                safe_retrain()
                last_record_count = current_count
                
                # Save brain after successful retrain cycle
                predict_ml.brain.save_brain()
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Brain saved. Capacity: {predict_ml.brain.get_brain_capacity()} patterns")
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No new data. Skipping retrain. ({current_count} records)")
            
            # Daily AI Evolution - run once per day
            current_date = datetime.datetime.now().date()
            if current_date != last_evolution_date:
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
                        else:
                            print(f"[EVOLUTION] Meeting failed or skipped: {evo_result.get('message', 'Unknown error')}")
                            
                        with open('daily_report.json', 'w') as f:
                            json.dump(report, f, indent=4)
                            
                        last_evolution_date = current_date
                except Exception as evo_err:
                    print(f"Error during daily evolution: {evo_err}")
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            # Wait 60 minutes
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("\nBackground worker stopped manually.")
            predict_ml.brain.save_brain()
            break
        except Exception as e:
            consecutive_failures += 1
            print(f"Background worker error (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
            traceback.print_exc()
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[CRITICAL] {MAX_CONSECUTIVE_FAILURES} consecutive failures. Resetting state and waiting 10 minutes...")
                consecutive_failures = 0
                time.sleep(600)
            else:
                # Exponential backoff: 2min, 4min, 8min, etc.
                wait = min(120 * (2 ** (consecutive_failures - 1)), 600)
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

if __name__ == "__main__":
    start_bot()
