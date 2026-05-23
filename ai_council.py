"""
AI Council — Multi-Model Swarm Intelligence for Kolkata FF Predictions
Multiple AI models hold a "meeting", each analyzes the data independently,
then a Chairman AI summarizes the consensus prediction.

Uses OpenRouter API to call multiple free AI models simultaneously.
"""
import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
IST = timezone(timedelta(hours=5, minutes=30))

# Council Members — each AI model has a unique "personality" and analysis style
COUNCIL_MEMBERS = [
    {
        'name': 'StatBot',
        'role': 'Statistical Analyst',
        'model': 'openrouter/free',
        'style': 'You are StatBot, a pure statistics expert. Analyze ONLY mathematical patterns like frequency, gaps, and probability distributions. Give your top 3 number predictions with statistical reasoning.'
    },
    {
        'name': 'PatternAI',
        'role': 'Pattern Recognition Expert',
        'model': 'openrouter/free',
        'style': 'You are PatternAI, a pattern recognition specialist. Focus ONLY on sequential patterns, streaks, repetitions, and cyclical trends in the data. Give your top 3 number predictions based on pattern analysis.'
    },
    {
        'name': 'RiskGuard',
        'role': 'Risk Assessment Officer',
        'model': 'openrouter/free',
        'style': 'You are RiskGuard, a risk management expert. Analyze the volatility, streaks, and confidence levels. Rate the overall risk (HIGH/MEDIUM/LOW) and suggest whether to play or skip. Also give your top 3 number predictions.'
    },
]

# Chairman prompt — synthesizes all council opinions
CHAIRMAN_PROMPT = """You are the Chairman of the Kolkata FF AI Council. Three AI experts just analyzed the prediction data and gave their opinions.

Here are their analyses:

{council_opinions}

Now synthesize their opinions into a FINAL CONSENSUS:
1. **Final Top 3 Numbers** (pick the numbers that most experts agree on)
2. **Confidence Level** (HIGH/MEDIUM/LOW based on agreement level)
3. **Risk Assessment** (should the user play or skip?)
4. **One-line reasoning** in Hindi

Format your response EXACTLY as JSON:
{{
  "numbers": [X, Y, Z],
  "confidence": "HIGH/MEDIUM/LOW",
  "risk": "PLAY/SKIP/WAIT",
  "reason_hindi": "...",
  "agreement": "X/3 experts agree"
}}

IMPORTANT: Return ONLY the JSON, no other text."""


class AICouncil:
    """Multi-AI Swarm Intelligence for consensus-based predictions."""

    def __init__(self):
        # Re-load .env as safety net (no-op if already loaded or file missing)
        load_dotenv(override=False)
        self.api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        self.gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()
        self.last_meeting = None
        self.meeting_log = []
        # Diagnostic logging for deployment debugging
        if self.gemini_key:
            masked = self.gemini_key[:8] + '...' + self.gemini_key[-4:]
            print(f"[COUNCIL] Gemini API Key loaded: {masked}")
        else:
            print("[COUNCIL] ⚠️ WARNING: No GEMINI_API_KEY found! Set it in environment variables.")

    def _call_ai(self, model, system_prompt, user_prompt, max_tokens=8000):
        """Call direct Gemini API with fallback to OpenRouter on error or quota limit."""
        # 1. Try Direct Gemini API first (if key exists)
        if self.gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {
                            "text": system_prompt
                        }
                    ]
                },
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": max_tokens
                }
            }
            headers = {
                "Content-Type": "application/json"
            }
            try:
                print(f"[COUNCIL] Trying direct Gemini API...")
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    print(f"[COUNCIL] Direct Gemini API failed with status code {response.status_code}: {response.text[:200]}")
            except Exception as e:
                print(f"[COUNCIL] Direct Gemini API exception: {e}")

        # 2. Fallback to OpenRouter (if key exists)
        if self.api_key:
            print(f"[COUNCIL] Falling back to OpenRouter...")
            url = OPENROUTER_URL
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/kolkata-ff-bot",
                "X-Title": "Kolkata FF Bot"
            }
            
            # Try specific robust free models
            models_to_try = []
            if model and model != 'openrouter/free':
                models_to_try.append(model)
            
            models_to_try.extend([
                "google/gemini-2.5-flash-preview:free",
                "google/gemini-2.0-flash-exp:free",
                "qwen/qwen-2.5-coder-32b-instruct:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "openrouter/free"
            ])
            
            # Remove duplicates while preserving order
            seen = set()
            models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]
            
            for model_name in models_to_try:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                }
                
                for attempt in range(2):
                    try:
                        print(f"[COUNCIL] Trying OpenRouter model: {model_name} (attempt {attempt+1})...")
                        response = requests.post(url, headers=headers, json=payload, timeout=45)
                        if response.status_code == 200:
                            data = response.json()
                            if 'choices' in data and len(data['choices']) > 0:
                                msg = data['choices'][0]['message']
                                content = msg.get('content')
                                if not content and msg.get('reasoning'):
                                    content = msg.get('reasoning')
                                if content:
                                    print(f"[COUNCIL] OpenRouter Success with {model_name}!")
                                    return content
                            print(f"[COUNCIL] OpenRouter response missing content for {model_name}: {data}")
                        else:
                            print(f"[COUNCIL] OpenRouter attempt failed for {model_name}: {response.status_code} - {response.text[:200]}")
                            time.sleep(attempt * 2 + 1)
                    except Exception as e:
                        print(f"[COUNCIL] OpenRouter exception for {model_name} on attempt {attempt+1}: {e}")
                        time.sleep(attempt * 2 + 1)
        
        return None

    def _build_data_prompt(self, prediction_data):
        """Build the data context prompt from current prediction state."""
        preds = prediction_data.get('data', {})
        predictions = preds.get('predictions', [])
        stats = preds.get('stats', {})
        risk = preds.get('risk_management', {})
        history = preds.get('history_trend', [])
        today_history = preds.get('today_history', [])

        # Last 20 results
        recent_numbers = [str(h['Single']) for h in history[-20:]] if history else []

        # Today's results
        today_results = ""
        if today_history:
            for t in today_history:
                status_icon = "PASS" if t['status'] == 'Pass' else "FAIL"
                today_results += f"  Bazi {t['bazi']}: Predicted {t['predictions']} | Actual: {t['actual']} | {status_icon}\n"

        prompt = f"""=== KOLKATA FF PREDICTION DATA ===
Next Bazi: {preds.get('next_bazi', '?')}
Date: {datetime.now(IST).strftime('%d/%m/%Y %A')}

ML Engine Predictions (4-Model Ensemble: XGB+LGB+RF+GB):
"""
        for p in predictions:
            prompt += f"  Number {p['number']} — Probability: {p['probability']}% — Top Pattis: {', '.join(p['pattis'])}\n"

        prompt += f"""
ML Risk Assessment: {risk.get('level', 'N/A')} — {risk.get('reason', '')}
ML Action: {risk.get('action', 'N/A')}

Recent 20 Results: {' -> '.join(recent_numbers)}

Stats:
  Today's Accuracy: {stats.get('today_matches', 'N/A')}
  Weekly Accuracy: {stats.get('weekly_matches', 'N/A')}
  Winning Streak: {stats.get('winning_streak', 0)}
  Losing Streak: {stats.get('losing_streak', 0)}
"""
        if today_results:
            prompt += f"\nToday's Prediction History:\n{today_results}"

        prompt += "\nBased on this data, what are your top 3 number predictions (0-9) and why?"

        return prompt

    def hold_meeting(self, prediction_data):
        """
        Hold an AI Council meeting:
        1. Each council member analyzes the data independently
        2. Chairman synthesizes the consensus
        Returns the full meeting result.
        """
        if not self.gemini_key:
            return {
                'status': 'error',
                'message': 'GEMINI_API_KEY not set. Set it in environment variables to enable AI Council.'
            }

        data_prompt = self._build_data_prompt(prediction_data)
        meeting_start = time.time()

        # Step 1: Each council member gives their analysis
        opinions = []
        for member in COUNCIL_MEMBERS:
            print(f"[COUNCIL] {member['name']} ({member['role']}) is analyzing...")
            response = self._call_ai(
                model=member['model'],
                system_prompt=member['style'],
                user_prompt=data_prompt,
                max_tokens=8000
            )

            if response:
                opinions.append({
                    'name': member['name'],
                    'role': member['role'],
                    'analysis': response
                })
                print(f"[COUNCIL] {member['name']} responded.")
            else:
                opinions.append({
                    'name': member['name'],
                    'role': member['role'],
                    'analysis': f"[{member['name']} was unable to respond]"
                })

            # Small delay between calls to respect rate limits
            time.sleep(1)

        # Step 2: Chairman synthesizes
        print("[COUNCIL] Chairman is synthesizing consensus...")
        council_text = ""
        for op in opinions:
            council_text += f"\n--- {op['name']} ({op['role']}) ---\n{op['analysis']}\n"

        chairman_prompt = CHAIRMAN_PROMPT.format(council_opinions=council_text)
        chairman_response = self._call_ai(
            model='openrouter/free',
            system_prompt='You are a senior decision maker. Always respond with valid JSON only.',
            user_prompt=chairman_prompt,
            max_tokens=8000
        )

        # Parse chairman's JSON response
        consensus = None
        if chairman_response:
            try:
                # Try to extract JSON from the response
                json_start = chairman_response.find('{')
                json_end = chairman_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    consensus = json.loads(chairman_response[json_start:json_end])
            except json.JSONDecodeError:
                pass

        if not consensus:
            consensus = {
                'numbers': [],
                'confidence': 'LOW',
                'risk': 'SKIP',
                'reason_hindi': 'AI Council consensus unclear. ML prediction follow karein.',
                'agreement': '0/3 experts agree'
            }

        meeting_duration = round(time.time() - meeting_start, 1)

        result = {
            'status': 'success',
            'meeting_id': int(time.time()),
            'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST'),
            'duration_seconds': meeting_duration,
            'council_members': len(opinions),
            'opinions': opinions,
            'consensus': consensus,
        }

        # Save to meeting log
        self.last_meeting = result
        self.meeting_log.append({
            'timestamp': result['timestamp'],
            'consensus_numbers': consensus.get('numbers', []),
            'confidence': consensus.get('confidence', 'LOW'),
            'risk': consensus.get('risk', 'SKIP')
        })
        # Keep last 50 meetings
        if len(self.meeting_log) > 50:
            self.meeting_log = self.meeting_log[-50:]

        print(f"[COUNCIL] Meeting complete in {meeting_duration}s. Consensus: {consensus.get('numbers', [])} ({consensus.get('confidence', 'N/A')})")
        return result

    def get_last_meeting(self):
        """Return the last meeting result."""
        return self.last_meeting

    def get_meeting_log(self):
        """Return meeting history."""
        return self.meeting_log

    def hold_evolution_meeting(self, yesterday_stats):
        """
        Hold a daily evolution meeting to tweak system weights based on performance.
        Returns updated config and reason.
        """
        if not self.gemini_key:
            return {"status": "error", "message": "No API Key for Evolution. Set GEMINI_API_KEY in environment."}

        evolution_prompt = f"""You are the Chief AI Architect for the Kolkata FF bot.
Your job is to optimize the system's prediction logic daily by tweaking algorithm weights.
Here is the performance report for yesterday:

Total Bazis Played: {yesterday_stats.get('total_bazis', 0)}
Correct Predictions: {yesterday_stats.get('correct_predictions', 0)}
Accuracy: {yesterday_stats.get('accuracy_pct', 0)}%
Pass/Fail Details:
{json.dumps(yesterday_stats.get('details', []), indent=2)}

Currently, the system blends three models:
1. ML Ensemble (xgboost, lightgbm, etc.)
2. Frequency Model (statistical)
3. Vector Memory (AI Brain pattern matching)

Based on yesterday's performance, provide the NEW optimal weights (they must sum to 1.0).
Also provide a 1-2 sentence reason for your change.

Format your response EXACTLY as JSON:
{{
  "ml_weight": 0.50,
  "memory_weight": 0.20,
  "freq_weight": 0.30,
  "reason": "Increased memory weight because patterns were highly repetitive yesterday."
}}

IMPORTANT: Return ONLY the JSON, no markdown formatting."""

        response = self._call_ai(
            model='openrouter/free',
            system_prompt='You are an elite machine learning engineer. Respond only with valid JSON.',
            user_prompt=evolution_prompt,
            max_tokens=8000
        )

        if not response:
            print("[EVOLUTION] AI returned no response. All models may be down.")
            return {"status": "error", "message": "AI returned no response. All free models may be temporarily unavailable."}

        try:
            # Strip markdown code fences that some models add
            import re
            cleaned = response.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

            # Extract JSON
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                config = json.loads(cleaned[json_start:json_end])
                
                # Validation
                ml = float(config.get('ml_weight', 0.5))
                mem = float(config.get('memory_weight', 0.15))
                freq = float(config.get('freq_weight', 0.35))
                total = ml + mem + freq
                
                if total > 0:
                    return {
                        "status": "success",
                        "config": {
                            "ml_weight": round(ml / total, 4),
                            "memory_weight": round(mem / total, 4),
                            "freq_weight": round(freq / total, 4)
                        },
                        "reason": config.get('reason', 'Automatic system tuning applied.')
                    }
            
            print(f"[EVOLUTION] Could not find valid JSON in response: {cleaned[:200]}")
        except json.JSONDecodeError as e:
            print(f"[EVOLUTION] JSON parse error: {e}. Raw response: {response[:200]}")
        except Exception as e:
            print(f"[EVOLUTION] Failed to parse AI response: {e}. Raw: {response[:200]}")

        return {"status": "error", "message": "Failed to generate evolution config"}


# Global council instance
council = AICouncil()
