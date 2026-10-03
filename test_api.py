from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env file explicitly
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found")
    exit(1)

print(f"✅ API Key found (starts with: {api_key[:8]}...)")

client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Test'}]
)
print(f"✅ Connection successful!")
