import time
import datetime
import traceback
import os
import pandas as pd
import pytz
import scraper
import predict_ml_v2 as predict_ml
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  SCRAPE SCHEDULE (IST)
#  Kolkata FF bazi results come at approx:
#    10:03  11:31  13:00  14:28  15:57  17:25  18:54  20:22
#  We scrape 37 min after each bazi (starting 10:40):
#    10:40  12:20  14:00  15:40  17:20  19:00  20:40
#  Interval = 100 minutes (1 hour 40 minutes)
# ─────────────────────────────────────────────────────────────────────────────

SCRAPE_START_HOUR   = 10
SCRAPE_START_MINUTE = 40
SCRAPE_INTERVAL_MIN = 100   # 1h 40m


def get_ist_now():
    """Return current datetime in IST."""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.datetime.now(ist)


def wait_until_next_scrape():
    """
    Sleep until the next scheduled scrape time.
    Schedule: 10:40 IST, then every 100 min: 12:20, 14:00, 15:40, 17:20, 19:00, 20:40
    After 20:40 (last scrape), wait until NEXT MORNING 10:40.
    """
    now = get_ist_now()
    today_first = now.replace(
        hour=SCRAPE_START_HOUR,
        minute=SCRAPE_START_MINUTE,
        second=0, microsecond=0
    )

    # Step forward from 10:40 in 100-min increments
    next_scrape = today_first
    while next_scrape <= now:
        next_scrape += datetime.timedelta(minutes=SCRAPE_INTERVAL_MIN)

    # If next_scrape crossed midnight (past 20:40 today), jump to tomorrow 10:40
    tomorrow_first = (now + datetime.timedelta(days=1)).replace(
        hour=SCRAPE_START_HOUR,
        minute=SCRAPE_START_MINUTE,
        second=0, microsecond=0
    )
    if next_scrape.date() > now.date() or next_scrape.hour < SCRAPE_START_HOUR:
        next_scrape = tomorrow_first

    wait_sec = (next_scrape - now).total_seconds()
    wait_min = int(wait_sec // 60)
    wait_s   = int(wait_sec % 60)

    print(f"[SCHEDULER] IST now    : {now.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"[SCHEDULER] Next scrape: {next_scrape.strftime('%d/%m/%Y %H:%M IST')}")
    print(f"[SCHEDULER] Waiting    : {wait_min}m {wait_s}s ...")
    time.sleep(max(wait_sec, 5))


def safe_scrape():
    """Scrape latest data from Kolkata FF website."""
    return scraper.scrape_kolkata_ff()


def safe_retrain():
    """Retrain ML model on latest data."""
    return predict_ml.train_and_save_model()


def start_bot():
    print("=" * 52)
    print("  KOLKATA FF WORKER v3.0 — IST SMART SCHEDULER  ")
    print("=" * 52)
    print(f"  Schedule : 10:40 IST then every {SCRAPE_INTERVAL_MIN} min (1h 40m)")
    print(f"  Times    : 10:40  12:20  14:00  15:40  17:20  19:00  20:40")
    print(f"  API Key  : {'SET ✓' if os.environ.get('GEMINI_API_KEY') else 'NOT SET ✗'}")
    print("=" * 52)

    last_record_count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    last_evolution_date = None

    # Initialize last_record_count from existing data
    try:
        df = pd.read_csv(predict_ml.DATA_FILE)
        last_record_count = len(df)
        print(f"[INIT] Loaded {last_record_count} existing records from CSV.")
    except FileNotFoundError:
        print(f"[INIT] No CSV found at {predict_ml.DATA_FILE}. Starting fresh.")
    except Exception as e:
        print(f"[INIT] Error reading CSV: {e}")

    while True:
        try:
            now_ist = get_ist_now()
            print(f"\n[{now_ist.strftime('%H:%M:%S IST')}] ── Scrape cycle starting ──")

            # ── Scrape latest data ─────────────────────────────────────
            safe_scrape()

            # ── Check if new data arrived ──────────────────────────────
            current_count = 0
            try:
                df = pd.read_csv(predict_ml.DATA_FILE)
                current_count = len(df)
            except FileNotFoundError:
                print(f"[{now_ist.strftime('%H:%M:%S')}] CSV not found after scrape.")
            except Exception as e:
                print(f"[{now_ist.strftime('%H:%M:%S')}] Error reading CSV: {e}")

            if current_count > last_record_count:
                added = current_count - last_record_count
                print(f"[{now_ist.strftime('%H:%M:%S')}] NEW DATA: +{added} records ({current_count} total). Retraining...")
                safe_retrain()
                last_record_count = current_count
                print(f"[{now_ist.strftime('%H:%M:%S')}] Retrain complete.")

            elif current_count < last_record_count:
                print(f"[{now_ist.strftime('%H:%M:%S')}] Record count dropped ({current_count} < {last_record_count}). Re-syncing.")
                last_record_count = current_count

            else:
                print(f"[{now_ist.strftime('%H:%M:%S')}] No new data. ({current_count} records)")

            # ── Daily AI Evolution (once per day) ──────────────────────
            current_date = now_ist.date()
            if last_evolution_date is None or current_date != last_evolution_date:
                print(f"\n[{now_ist.strftime('%H:%M:%S')}] Running Daily AI Evolution...")
                try:
                    yesterday_stats = predict_ml.get_yesterday_stats()
                    if yesterday_stats and yesterday_stats.get('total_bazis', 0) > 0:
                        from ai_council import council
                        import json

                        evo_result = council.hold_evolution_meeting(yesterday_stats)

                        report = {
                            "date_run": now_ist.strftime('%d/%m/%Y %H:%M:%S'),
                            "target_date": yesterday_stats.get('date'),
                            "yesterday_stats": yesterday_stats,
                            "evolution_result": evo_result
                        }

                        if evo_result.get('status') == 'success' and 'config' in evo_result:
                            predict_ml.save_ai_config(evo_result['config'])
                            print(f"[EVOLUTION] Config updated: {evo_result.get('reason')}")
                            last_evolution_date = current_date
                        else:
                            print(f"[EVOLUTION] Failed: {evo_result.get('message', 'Unknown error')}. Will retry.")

                        os.makedirs('reports', exist_ok=True)
                        report_file = f"reports/daily_report_{current_date.strftime('%Y%m%d')}.json"
                        with open(report_file, 'w') as f:
                            json.dump(report, f, indent=4)
                        with open('daily_report.json', 'w') as f:
                            json.dump(report, f, indent=4)
                        print(f"[{now_ist.strftime('%H:%M:%S')}] Evolution report saved.")

                    else:
                        print(f"[{now_ist.strftime('%H:%M:%S')}] Not enough data for evolution. Skipping.")
                        last_evolution_date = current_date

                except ImportError:
                    print(f"[{now_ist.strftime('%H:%M:%S')}] AI Council not found. Skipping evolution.")
                    last_evolution_date = current_date
                except Exception as evo_err:
                    print(f"[{now_ist.strftime('%H:%M:%S')}] Evolution error: {evo_err}")
                    traceback.print_exc()

            # Reset failure counter
            consecutive_failures = 0

            # ── Wait until next scheduled scrape time ─────────────────
            wait_until_next_scrape()

        except KeyboardInterrupt:
            print("\n[BOT] Stopped manually.")
            break
        except Exception as e:
            consecutive_failures += 1
            print(f"[{get_ist_now().strftime('%H:%M:%S')}] Error (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
            traceback.print_exc()

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[CRITICAL] {MAX_CONSECUTIVE_FAILURES} consecutive failures. Waiting 10 min...")
                consecutive_failures = 0
                time.sleep(600)
            else:
                wait_time = min(120 * (2 ** (consecutive_failures - 1)), 600)
                print(f"[BOT] Retrying in {wait_time}s...")
                time.sleep(wait_time)


if __name__ == "__main__":
    start_bot()

