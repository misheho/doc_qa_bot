# ──────────────────────────────────────────────────────────────────────
# FULL SETUP SEQUENCE (Run in order)
# ──────────────────────────────────────────────────────────────────────

# 1. Create project folder
mkdir -p doc_qa_bot && cd doc_qa_bot

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate

# 4. Upgrade pip (optional but recommended)
pip install --upgrade pip

# 5. Install all required packages at once
pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv

# 6. Create documents folder
mkdir documents

# 7. Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# 8. Test API connection (Option A - export method)
export $(cat .env | xargs)
python3 -c "from openai import OpenAI; import os; c=OpenAI(api_key=os.getenv('OPENAI_API_KEY')); r=c.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'user','content':'Hi'}]); print('✅ WORKS' if r.choices else '❌ FAIL')"

# 9. If that succeeds, create the full chatbot script
# (I'll provide this next)
