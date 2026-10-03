#!/usr/bin/env python3
"""
Document Q&A Chatbot using LangChain
Ask questions about your uploaded PDF/text documents using retrieval-augmented generation (RAG).

Setup:
  1. Ensure .env file exists with OPENAI_API_KEY
  2. Place PDFs/TXT files in ./documents/ folder
  3. Run: python3 chat.py
"""

from dotenv import load_dotenv
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Load Environment Variables
# ──────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found!")
    print("   Fix: Check .env file contains: OPENAI_API_KEY=sk-...")
    exit(1)

print(f"✅ API key loaded (starts with: {API_KEY[:8]}...)")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Import LangChain Components
# ──────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Configuration
# ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
VECTOR_STORE_PATH = "./chroma_db"  # Where to save embeddings
NUM_RETRIEVED_DOCS = 3   # How many chunks to search

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Load Documents from ./documents/ folder
# ──────────────────────────────────────────────────────────────────────
def load_documents():
    """Load all PDF and TXT files from documents folder."""
    documents_folder = Path("documents")
    
    if not documents_folder.exists():
        print("❌ ERROR: 'documents' folder not found!")
        print("   Create it and add PDF/TXT files: mkdir documents")
        exit(1)
    
    docs = []
    file_count = 0
    
    for file_path in documents_folder.iterdir():
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            docs.extend(pages)
            file_count += 1
            print(f"   Loaded PDF: {file_path.name} ({len(pages)} pages)")
            
        elif file_path.suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path))
            pages = loader.load()
            docs.extend(pages)
            file_count += 1
            print(f"   Loaded text: {file_path.name} ({len(pages)} characters)")
    
    if file_count == 0:
        print("⚠️  WARNING: No documents found in ./documents/")
        print("   Add PDF or TXT files to the documents folder.")
        return None
    
    print(f"✅ Loaded {file_count} document(s), {len(docs)} total page(s)/chunk(s)")
    return docs

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Split Documents into Chunks
# ──────────────────────────────────────────────────────────────────────
def split_into_chunks(documents):
    """Break large documents into smaller searchable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# ──────────────────────────────────────────────────────────────────────
# STEP 6: Create Embeddings and Vector Store
# ──────────────────────────────────────────────────────────────────────
def create_vector_store(chunks):
    """Convert text chunks to vectors and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Either load existing DB or create new one
    if os.path.exists(VECTOR_STORE_PATH):
        print("ℹ️  Loading existing vector database...")
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
    else:
        print("🔄 Creating new vector database...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        print(f"✅ Vector store created with {vector_store.count()} vectors")
    
    return vector_store

# ──────────────────────────────────────────────────────────────────────
# STEP 7: Build the RAG Question-Answering Chain
# ──────────────────────────────────────────────────────────────────────
def build_qa_chain(vector_store):
    """Create the retrieval + LLM pipeline."""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": NUM_RETRIEVED_DOCS}
    )
    
    # Prompt template tells the model how to use the context
    template = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{{context}}
</context>

Question: {{question}}

Answer (be concise and cite which document section if possible):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Build the chain: retrieve → format → answer
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

# ──────────────────────────────────────────────────────────────────────
# STEP 8: Interactive Chat Loop
# ──────────────────────────────────────────────────────────────────────
def chat_loop(qa_chain):
    """Allow continuous questioning until user quits."""
    print("\n" + "="*60)
    print("🤖 DOCUMENT Q&A CHATBOT READY!")
    print("Type your question, then press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            question = input("📝 You: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Show thinking indicator
            print("🔍 Thinking...", end="\r")
            
            # Get answer from LLM
            answer = qa_chain.invoke(question)
            
            # Clear thinking indicator and show answer
            print("\r" + " " * 30 + "\r")  # Clear line
            print(f"💡 Answer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

# ──────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("📁 Document Q&A Chatbot - Initializing")
    print("="*60 + "\n")
    
    # Build the pipeline (takes 1-2 minutes for first run)
    print("STEP 1/4: Loading documents...")
    docs = load_documents()
    if docs is None:
        print("   Cannot proceed without documents.")
        exit(1)
    
    print("\nSTEP 2/4: Splitting into chunks...")
    chunks = split_into_chunks(docs)
    
    print("\nSTEP 3/4: Creating vector store...")
    vector_store = create_vector_store(chunks)
    
    print("\nSTEP 4/4: Building QA chain...")
    qa_chain = build_qa_chain(vector_store)
    
    # Start interactive chat
    print("\n")
    chat_loop(qa_chain)