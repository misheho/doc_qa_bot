# 📁 Document Q&A Chatbot

An intelligent chatbot that answers questions about your uploaded documents using **LangChain**, **OpenAI**, and **Retrieval-Augmented Generation (RAG)**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-purple.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Document Ingestion** | Automatically loads PDFs and text files from a local folder |
| **Smart Retrieval** | Finds relevant document chunks based on your question |
| **Natural Answers** | Uses GPT-4o-mini (or any OpenAI model) to generate human-like responses |
| **Persistent Memory** | Vector database caches embeddings for faster subsequent queries |
| **CLI Interface** | Simple terminal-based interaction (ready for web UI expansion) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
- (Optional) PDF documents to test with

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/doc_qa_bot.git
cd doc_qa_bot

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
nano .env  # Replace with your actual OpenAI API key