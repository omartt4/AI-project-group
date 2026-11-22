#!/usr/bin/env python3
"""Check OpenRouter free model daily request quota."""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found in .env")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(f"{base_url}/key", headers=headers)
data = response.json()

if response.status_code != 200:
    print(f"ERROR: API returned {response.status_code}")
    print(data)
    exit(1)

free_daily = data.get('data', {}).get('free_model_daily_requests', {})
print(f"Used: {free_daily.get('used', 0)}")
print(f"Limit: {free_daily.get('limit', 0)}")
print(f"Remaining: {free_daily.get('remaining', 0)}")
