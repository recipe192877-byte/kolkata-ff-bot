import os
import requests
import json
import time
from github_sync import upload_to_github

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
# Target files to autonomously rewrite
TARGET_FILES = ['predict_ml_v2.py', 'scraper.py', 'bot.py', 'keep_alive.py']

class RuFloHealer:
    """
    Autonomous code upgrader. Reads source files, asks RuFlo AI to improve them,
    overwrites the local files, and pushes changes to GitHub.
    """
    def __init__(self):
        self.api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        
    def scan_and_upgrade(self):
        if not self.api_key:
            print("[RUFLO_HEALER] OPENROUTER_API_KEY not found. Skipping autonomous upgrade.")
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
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'HTTP-Referer': 'https://kolkata-ff-bot.onrender.com',
                    'X-Title': 'Kolkata FF Autonomous Upgrader'
                },
                json={
                    'model': 'google/gemini-2.5-flash', # Or another capable coder model
                    'messages': [
                        {'role': 'system', 'content': 'You are RuFlo, a master autonomous coding agent. Return ONLY raw, perfectly valid Python code. No markdown formatting.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'max_tokens': 8000 # Need high token limit to return full files
                },
                timeout=120 # AI needs time to read and rewrite entire files
            )

            if response.status_code == 200:
                data = response.json()
                new_code = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Strip potential markdown artifacts just in case AI disobeys
                new_code = new_code.replace("```python", "").replace("```", "").strip()
                return new_code
            else:
                print(f"API Error {response.status_code}: {response.text[:200]}")
                return None

        except Exception as e:
            print(f"[RUFLO_HEALER] AI call failed: {str(e)}")
            return None

    def chat_and_execute(self, user_message):
        """Interactive chat with the user. Can answer questions or execute code changes."""
        if not self.api_key:
            return {"reply": "Error: OPENROUTER_API_KEY is not set. I cannot function.", "modifications": []}

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
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'HTTP-Referer': 'https://kolkata-ff-bot.onrender.com',
                    'X-Title': 'Kolkata FF Autonomous Upgrader'
                },
                json={
                    'model': 'google/gemini-2.5-flash',
                    'messages': [
                        {'role': 'system', 'content': 'You are RuFlo, a master autonomous coding agent. Return ONLY raw, valid JSON.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'max_tokens': 15000 
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Parse JSON
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    return {"reply": "Error: RuFlo returned an invalid JSON response. Please try your command again.", "modifications": []}

                modifications = result.get("modifications", [])
                
                # Apply modifications
                for mod in modifications:
                    filename = mod.get("filename")
                    code = mod.get("code")
                    if filename and code and ("def " in code or "import " in code or "html" in code.lower()):
                        print(f"[RUFLO_CHAT] Applying changes to {filename}...")
                        try:
                            os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(code)
                            print(f"[RUFLO_CHAT] Pushing {filename} to GitHub...")
                            upload_to_github(filename)
                            time.sleep(1)
                        except Exception as e:
                            print(f"[RUFLO_CHAT] Error saving {filename}: {e}")

                return result
            else:
                return {"reply": f"API Error {response.status_code}: {response.text[:100]}", "modifications": []}

        except Exception as e:
            return {"reply": f"Error contacting RuFlo API: {str(e)}", "modifications": []}

if __name__ == "__main__":
    healer = RuFloHealer()
    healer.scan_and_upgrade()
