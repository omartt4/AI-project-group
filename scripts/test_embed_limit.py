import requests
import os
from dotenv import load_dotenv

load_dotenv('.env')
api_key = os.getenv('OPENROUTER_API_KEY')
base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

# Check limit before
headers = {'Authorization': f'Bearer {api_key}'}
r = requests.get(f'{base_url}/key', headers=headers)
data = r.json()
before = data['data']['free_model_daily_requests']['used']
print(f'Before embedding: {before}/50')

# Make a small embedding request
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/omartt4/AI-project-group',
    'X-Title': 'Graph RAG Lab'
}
data = {
    'model': 'nvidia/nemotron-3.5-embed-1b:free',
    'input': ['test']
}
r = requests.post(f'{base_url}/embeddings', headers=headers, json=data)
print(f'Embedding status: {r.status_code}')

# Check limit after
r = requests.get(f'{base_url}/key', headers=headers)
data = r.json()
after = data['data']['free_model_daily_requests']['used']
print(f'After embedding: {after}/50')
print(f'Embedding requests count: {after - before}')
