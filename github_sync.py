import os
import base64
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "recipe192877-byte"
REPO_NAME = "kolkata-ff-bot"
FILE_PATH = "kolkata_ff_history.csv"

def upload_to_github():
    if not GITHUB_TOKEN:
        print("GitHub token not found. Skipping auto-upload.")
        return

    print("Attempting to backup CSV to GitHub...")
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
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
        with open(FILE_PATH, "rb") as file:
            content = file.read()
            encoded_content = base64.b64encode(content).decode('utf-8')
    except Exception as e:
        print(f"Error reading local CSV: {e}")
        return

    data = {
        "message": "Auto-sync latest Koltata FF results",
        "content": encoded_content
    }
    if sha:
        data["sha"] = sha

    # Upload to github
    try:
        put_response = requests.put(url, headers=headers, json=data)
        if put_response.status_code in [200, 201]:
            print("Successfully synced CSV to GitHub!")
        else:
            print(f"Failed to sync. Status: {put_response.status_code}")
            print(put_response.text)
    except Exception as e:
        print(f"Failed to upload to github: {e}")
