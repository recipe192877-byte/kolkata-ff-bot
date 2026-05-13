from bot import start_bot
import datetime
import predict_ml_v2 as predict_ml
from ai_council import council
import json
import time

print("TESTING DAILY EVOLUTION...")

yesterday_stats = predict_ml.get_yesterday_stats()
if yesterday_stats:
    print(f"Stats loaded for {yesterday_stats.get('date')} - Acc: {yesterday_stats.get('accuracy_pct')}%")
    
    evo_result = council.hold_evolution_meeting(yesterday_stats)
    print("Evolution Result:", json.dumps(evo_result, indent=2))
    
    if evo_result.get('status') == 'success':
        predict_ml.save_ai_config(evo_result['config'])
        print("Config saved!")
        print("Loaded config:", predict_ml.load_ai_config())
        
        report = {
            "date_run": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "target_date": yesterday_stats.get('date'),
            "yesterday_stats": yesterday_stats,
            "evolution_result": evo_result
        }
        with open('daily_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        print("daily_report.json created.")
else:
    print("Could not load stats. Missing data or model files.")
