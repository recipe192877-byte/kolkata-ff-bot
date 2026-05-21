import os
import requests
import json
import time
from github_sync import upload_to_github
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
# Target files to autonomously rewrite
TARGET_FILES = ['predict_ml_v2.py', 'scraper.py', 'bot.py', 'keep_alive.py']

# Multi-model fallback list (ALL known Free models on OpenRouter)
FREE_MODELS = [
    # Google Free Models
    'google/gemma-3-27b-it:free',
    'google/gemma-4-27b-it:free',
    'google/gemma-4-12b-it:free',
    'google/gemini-2.0-flash-exp:free',
    'google/gemini-2.5-flash-preview:free',
    # Meta Llama Free Models
    'meta-llama/llama-4-maverick:free',
    'meta-llama/llama-4-scout:free',
    'meta-llama/llama-3.3-70b-instruct:free',
    'meta-llama/llama-3.1-8b-instruct:free',
    # Mistral Free Models
    'mistralai/mistral-small-3.1-24b-instruct:free',
    'mistralai/devstral-small:free',
    'mistralai/pixtral-12b:free',
    # Qwen Free Models
    'qwen/qwen-2.5-72b-instruct:free',
    'qwen/qwen-2.5-coder-32b-instruct:free',
    'qwen/qwen3-235b-a22b:free',
    'qwen/qwen3-32b:free',
    'qwen/qwen3-30b-a3b:free',
    'qwen/qwen3-14b:free',
    'qwen/qwen3-8b:free',
    'qwen/qwen-2-7b-instruct:free',
    # Microsoft Free Models
    'microsoft/phi-4-reasoning-plus:free',
    'microsoft/phi-4-reasoning:free',
    'microsoft/phi-4-multimodal-instruct:free',
    'microsoft/phi-3-mini-128k-instruct:free',
    'microsoft/mai-ds-r1:free',
    # DeepSeek Free Models
    'deepseek/deepseek-r1-0528:free',
    'deepseek/deepseek-r1:free',
    'deepseek/deepseek-chat-v3-0324:free',
    'deepseek/deepseek-chat:free',
    # NVIDIA Free Models
    'nvidia/llama-3.1-nemotron-70b-instruct:free',
    'nvidia/llama-3.3-nemotron-super-49b-v1:free',
    # Other Free Models
    'moonshotai/kimi-vl-a3b-thinking:free',
    'nousresearch/hermes-3-llama-3.1-405b:free',
    'open-r1/olympiccoder-32b:free',
    'rekaai/reka-flash-3:free',
    # OpenRouter Auto-Free Router
    'openrouter/free',
]

class RuFloHealer:
    """
    Autonomous code upgrader. Reads source files, asks RuFlo AI to improve them,
    overwrites the local files, and pushes changes to GitHub.
    """
    def __init__(self):
        self.api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        self.gemini_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyCwb5cLbFWnaZqh4Uzkow2L6ZCO4RDzFtg').strip()
        
    def scan_and_upgrade(self):
        if not self.gemini_key:
            print("[RUFLO_HEALER] GEMINI_API_KEY not found. Skipping autonomous upgrade.")
            return

        print("\n" + "="*50)
        print("🤖 [RUFLO_HEALER] Starting Autonomous Code Upgrade Scan...")
        print("="*50)

        for filename in TARGET_FILES:
            if not os.path.exists(filename):
                print(f"[RUFLO_HEALER] Skipping {filename}: File not found.")
                continue
                
            print(f"[RUFLO_HEALER] Scanning {filename} for potential upgrades/fixes...")
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    original_code = f.read()
                    
                upgraded_code = self._get_ai_upgrade(filename, original_code)
                
                if upgraded_code and upgraded_code.strip() != original_code.strip():
                    if "def " in upgraded_code or "import " in upgraded_code: # Basic sanity check
                        print(f"[RUFLO_HEALER] RuFlo suggested valid upgrades for {filename}. Overwriting local file...")
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(upgraded_code)
                            
                        print(f"[RUFLO_HEALER] Pushing upgraded {filename} to GitHub...")
                        upload_to_github(filename)
                        time.sleep(2) # Prevent rate limiting
                    else:
                        print(f"[RUFLO_HEALER] Safety check failed. AI returned invalid code for {filename}. Aborting overwrite.")
                else:
                    print(f"[RUFLO_HEALER] No upgrades needed for {filename}. Code is optimal.")
                    
            except Exception as e:
                print(f"[RUFLO_HEALER] Critical Error processing {filename}: {e}")

        print("="*50)
        print("🤖 [RUFLO_HEALER] Scan Complete.")
        print("="*50)

    def _get_ai_upgrade(self, filename, original_code):
        prompt = f"""You are RuFlo, the Autonomous Master AI Developer for the Kolkata FF bot project.
I am providing you with the full source code for '{filename}'.

Your task:
1. Scan the code for bugs, inefficiencies, or missing edge-case handling.
2. Upgrade ML logic or web scraping logic if you see a clear improvement.
3. If the code is perfect, just return the exact original code.
4. If you make changes, return the ENTIRE updated Python file.
5. YOU MUST RETURN ONLY RAW PYTHON CODE. Do not use markdown blocks (```python). Do not include conversational text or explanations. Just the code.

Here is the code:
{original_code}
"""
        if not self.gemini_key:
            return None
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": "You are RuFlo, a master autonomous coding agent. Return ONLY raw, perfectly valid Python code. No markdown formatting."
                    }
                ]
            },
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8000
            }
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        for attempt in range(3):
            try:
                print(f"[RUFLO_HEALER] Attempting upgrade with direct Gemini API (attempt {attempt+1})...")
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    new_code = data['candidates'][0]['content']['parts'][0]['text']
                    new_code = new_code.replace("```python", "").replace("```", "").strip()
                    if len(new_code) > 100: 
                        return new_code
                elif response.status_code in [429, 503]:
                    print(f"[RUFLO_HEALER] Gemini API returned {response.status_code}. Retrying in {attempt*2 + 2}s...")
                    time.sleep(attempt * 2 + 2)
                else:
                    print(f"Gemini API failed. Status code: {response.status_code}")
                    break
            except Exception as e:
                print(f"[RUFLO_HEALER] AI call failed: {str(e)}")
                time.sleep(attempt * 2 + 2)
        return None

    def chat_and_execute(self, user_message):
        """Interactive chat with the user. Can answer questions or execute code changes."""
        if not self.gemini_key:
            return {"reply": "Error: GEMINI_API_KEY is not set. I cannot function.", "modifications": []}

        print(f"\n[RUFLO_CHAT] Received command: {user_message}")
        
        # Gather context
        context_files = TARGET_FILES + ['templates/index.html']
        context_str = ""
        for filename in context_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        context_str += f"\n--- {filename} ---\n{f.read()}\n"
                except Exception as e:
                    pass

        prompt = f"""You are RuFlo, the J.A.R.V.I.S.-like autonomous AI developer for the Kolkata FF bot.
The user has sent you a message/command.
Here is the current source code of the project:
{context_str}

User Command: "{user_message}"

If the user is asking a general question, answer it concisely.
If the user is asking you to ADD A FEATURE, FIX A BUG, or MODIFY THE CODE, you must generate the FULL UPDATED SOURCE CODE for any file that needs changing.

You MUST respond EXACTLY in this JSON format (no markdown blocks, just raw JSON):
{{
  "reply": "Your response to the user here...",
  "modifications": [
    {{
      "filename": "predict_ml_v2.py",
      "code": "FULL REWRITTEN PYTHON CODE HERE..."
    }}
  ]
}}

If no modifications are needed, leave the "modifications" array empty.
IMPORTANT: If you modify a file, you MUST provide the ENTIRE file's code, not just snippets.
"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": "You are RuFlo, a master autonomous coding agent. Return ONLY raw, valid JSON."
                    }
                ]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 8000
            }
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        for attempt in range(3):
            try:
                print(f"[RUFLO_CHAT] Attempting chat with direct Gemini API (attempt {attempt+1})...")
                response = requests.post(url, headers=headers, json=payload, timeout=120)

                if response.status_code == 200:
                    data = response.json()
                    content = data['candidates'][0]['content']['parts'][0]['text']
                    content = content.replace("```json", "").replace("```", "").strip()
                    try:
                        result = json.loads(content)
                        
                        # Apply modifications if any
                        modifications = result.get("modifications", [])
                        for mod in modifications:
                            filename = mod.get("filename")
                            code = mod.get("code")
                            if filename and code and ("def " in code or "import " in code or "html" in code.lower()):
                                try:
                                    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
                                    with open(filename, 'w', encoding='utf-8') as f:
                                        f.write(code)
                                    upload_to_github(filename)
                                    time.sleep(1)
                                except Exception as e:
                                    print(f"Error saving {filename}: {e}")
                        
                        return result
                    except json.JSONDecodeError:
                        print("[RUFLO_CHAT] Error parsing JSON from response.")
                elif response.status_code in [429, 503]:
                    print(f"[RUFLO_CHAT] Gemini API returned {response.status_code}. Retrying in {attempt*2 + 2}s...")
                    time.sleep(attempt * 2 + 2)
                else:
                    print(f"Gemini API failed. Status code: {response.status_code}")
                    break
            except Exception as e:
                print(f"Error during API call: {e}")
                time.sleep(attempt * 2 + 2)

        return {"reply": "Error: Gemini AI model failed to respond.", "modifications": []}

if __name__ == "__main__":
    healer = RuFloHealer()
    healer.scan_and_upgrade()
