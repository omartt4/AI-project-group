"""Check available free embedding models on OpenRouter."""
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("OPENROUTER_API_KEY", "")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
    "X-Title": "Graph RAG Lab"
}

r = requests.get(f"{base_url}/models", headers=headers)
data = r.json()

print("Free embedding models:")
for m in data['data']:
    if 'embed' in m['id'].lower():
        pricing = m.get('pricing', {})
        if pricing.get('prompt', '0') == '0':
            print(f"  {m['id']}")
            if m.get('context_length'):
                print(f"    Context: {m['context_length']}")

print("\nAll free models (first 20):")
free_count = 0
for m in data['data']:
    pricing = m.get('pricing', {})
    if pricing.get('prompt', '0') == '0':
        print(f"  {m['id']}")
        free_count += 1
        if free_count >= 20:
            break

print("\nChecking for any model with 'embedding' capabilities:")
for m in data['data']:
    if 'embed' in m['id'].lower() or 'embedding' in m['id'].lower():
        print(f"  {m['id']}")
        print(f"    Pricing: {m.get('pricing', {})}")
