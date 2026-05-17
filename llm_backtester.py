import pandas as pd
import requests
import json
import time
import os
import argparse

MODELS = [
    'nousresearch/hermes-3-llama-3.1-405b:free',
    'meta-llama/llama-3.3-70b-instruct:free',
    'nvidia/nemotron-3-super-120b-a12b:free',
    'openai/gpt-oss-120b:free',
    'arcee-ai/trinity-large-thinking:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    'deepseek/deepseek-r1:free',
    'deepseek/deepseek-chat:free',
    'qwen/qwen3-next-80b-a3b-instruct:free',
    'google/gemma-4-31b-it:free',
    'deepseek/deepseek-v4-flash:free',
    'openai/gpt-oss-20b:free',
    'google/gemma-4-26b-a4b-it:free',
    'nvidia/nemotron-3-nano-30b-a3b:free',
]

def load_api_key():
    key = os.environ.get('OPENROUTER_API_KEY', '')
    if not key:
        print("WARNING: OPENROUTER_API_KEY environment variable not set. Please set it before running.")
    return key

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure Single is integer
    df = df.dropna(subset=['Single'])
    df['Single'] = df['Single'].astype(int)
    return df

def get_prediction(model, history_list, api_key):
    history_str = " -> ".join(map(str, history_list))
    prompt = f"""You are an expert Kolkata FF pattern analyst.
Here are the recent 20 results in chronological order: {history_str}
Predict the top 3 numbers (0-9) for the next bazi.
You MUST reply ONLY with a valid JSON array containing exactly 3 integers, e.g., [1, 5, 8]. Do not include any other text or markdown formatting. Just the array."""

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.1,
                'max_tokens': 50
            },
            timeout=15
        )
        if response.status_code == 200:
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            content = content.replace("```json", "").replace("```", "").strip()
            idx_start = content.find('[')
            idx_end = content.find(']') + 1
            if idx_start != -1 and idx_end > idx_start:
                arr = json.loads(content[idx_start:idx_end])
                return arr[:3]
        else:
            print(f"  [!] API Error {response.status_code} for {model}")
    except Exception as e:
        print(f"  [!] Exception for {model}: {str(e)[:50]}")
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5, help='Number of recent bazis to test')
    parser.add_argument('--delay', type=int, default=2, help='Delay between API calls in seconds')
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        return

    csv_path = 'kolkata_ff_history_advanced.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = load_data(csv_path)
    
    if len(df) < args.samples + 20:
        print("Not enough data in CSV.")
        return

    start_idx = len(df) - args.samples
    test_indices = list(range(start_idx, len(df)))

    stats = {model: {'direct_hits': 0, 'top3_hits': 0, 'misses': 0, 'errors': 0} for model in MODELS}

    print(f"Starting backtest on last {args.samples} bazis across {len(MODELS)} AI models...")
    
    count = 1
    for idx in test_indices:
        actual_number = int(df.iloc[idx]['Single'])
        date = df.iloc[idx]['Date']
        bazi = df.iloc[idx]['Bazi']
        
        print(f"\n[{count}/{args.samples}] Testing Date: {date}, Bazi: {bazi} | Actual Winner: {actual_number}")
        
        history_df = df.iloc[idx-20:idx]
        history_list = history_df['Single'].tolist()
        
        for model in MODELS:
            predicted_numbers = get_prediction(model, history_list, api_key)
            
            if not predicted_numbers or len(predicted_numbers) == 0:
                stats[model]['errors'] += 1
            else:
                if actual_number == predicted_numbers[0]:
                    stats[model]['direct_hits'] += 1
                    stats[model]['top3_hits'] += 1
                elif actual_number in predicted_numbers:
                    stats[model]['top3_hits'] += 1
                else:
                    stats[model]['misses'] += 1
            
            time.sleep(args.delay)
            
        count += 1

    leaderboard = []
    for model, s in stats.items():
        total_valid = s['direct_hits'] + s['top3_hits'] + s['misses']
        if total_valid > 0:
            acc_top1 = (s['direct_hits'] / total_valid) * 100
            acc_top3 = (s['top3_hits'] / total_valid) * 100
        else:
            acc_top1 = 0
            acc_top3 = 0
            
        score = (s['top3_hits'] * 1) + (s['direct_hits'] * 2)
        leaderboard.append({
            'model': model,
            'top1_acc': round(acc_top1, 1),
            'top3_acc': round(acc_top3, 1),
            'direct_hits': s['direct_hits'],
            'top3_hits': s['top3_hits'],
            'errors': s['errors'],
            'score': score
        })

    leaderboard.sort(key=lambda x: x['score'], reverse=True)

    print("\n\n🏆 TOP 11 AI MODELS LEADERBOARD 🏆")
    print("-" * 85)
    print(f"{'Rank':<5} | {'Model':<40} | {'Top3 Acc':<10} | {'Direct Hits':<12}")
    print("-" * 85)
    
    for i, entry in enumerate(leaderboard[:11], 1):
        print(f"{i:<5} | {entry['model']:<40} | {entry['top3_acc']:<8}% | {entry['direct_hits']}")
        
    print("-" * 85)
    print("\nFull stats saved to backtest_results.json")
    
    with open("backtest_results.json", "w") as f:
        json.dump(leaderboard, f, indent=4)

if __name__ == "__main__":
    main()
