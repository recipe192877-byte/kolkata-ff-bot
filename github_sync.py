import os
import base64
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "recipe192877-byte"
REPO_NAME = "kolkata-ff-bot"

def upload_to_github(file_path="kolkata_ff_history_advanced.csv"):
    """
    Uploads a file to the GitHub repository.
    Used by the AI Council to push rewritten .py code or data CSVs.
    """
    if not GITHUB_TOKEN:
        print(f"GitHub token not found. Skipping auto-upload for {file_path}.")
        return

    print(f"Attempting to backup {file_path} to GitHub...")
    # Fix backslashes for github path if running on windows
    github_path = file_path.replace("\\", "/")
    # If the path has a leading path, we just want the basename for the repo root if it's meant to go there.
    # But usually, we want to respect the relative structure. We'll use basename for simplicity in root.
    github_path = os.path.basename(github_path)
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{github_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get the SHA of the existing file to update it
    sha = None
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        sha = response.json().get('sha')

    # Read the local file and encode it to base64
    try:
        with open(file_path, "rb") as file:
            content = file.read()
            encoded_content = base64.b64encode(content).decode('utf-8')
    except Exception as e:
        print(f"Error reading local file {file_path}: {e}")
        return

    commit_msg = f"Auto-sync latest data for {github_path}"
    if file_path.endswith('.py'):
        commit_msg = f"🤖 RuFlo Autonomous Code Upgrade: {github_path}"

    data = {
        "message": commit_msg,
        "content": encoded_content
    }
    if sha:
        data["sha"] = sha

    # Upload to github
    try:
        put_response = requests.put(url, headers=headers, json=data)
        if put_response.status_code in [200, 201]:
            print(f"Successfully synced {file_path} to GitHub!")
        else:
            print(f"Failed to sync {file_path}. Status: {put_response.status_code}")
            print(put_response.text)
    except Exception as e:
        print(f"Failed to upload to github: {e}")
