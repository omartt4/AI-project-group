import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv('.env')
api_key = os.getenv('OPENROUTER_API_KEY')
base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
headers = {'Authorization': f'Bearer {api_key}'}
r = requests.get(f'{base_url}/key', headers=headers)
data = r.json()
print(f'Free tier: {data["data"]["is_free_tier"]}')
print(f'Daily requests: {data["data"]["free_model_daily_requests"]["used"]}/{data["data"]["free_model_daily_requests"]["limit"]}')
print(f'Remaining: {data["data"]["free_model_daily_requests"]["remaining"]}')
if 'limit_reset' in data['data'] and data['data']['limit_reset']:
    reset_time = datetime.fromtimestamp(data['data']['limit_reset'], tz=timezone.utc)
    cairo_offset = timezone(timedelta(hours=2))
    cairo_time = reset_time.astimezone(cairo_offset)
    print(f'Limit reset: {reset_time} UTC / {cairo_time} UTC+2')
else:
    print('No reset time available')
