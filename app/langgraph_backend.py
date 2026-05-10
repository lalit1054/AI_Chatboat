import os
import requests
import hashlib
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

# Postgres specific imports
import psycopg
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# RAG specific imports
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------
# Pinecone Init
# -------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def process_uploaded_file(file_bytes: bytes, filename: str, user_id: int, thread_id: str) -> dict:
    try:
        text = ""
        if filename.lower().endswith(".pdf"):
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.lower().endswith(".txt"):
            text = file_bytes.decode("utf-8")
        else:
            return {"success": False, "error": "Unsupported file format. Please upload PDF or TXT."}
            
        if not text.strip():
            return {"success": False, "error": "Could not extract any text from the file."}
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
        
        from langchain_core.documents import Document
        docs = [Document(page_content=chunk, metadata={"user_id": user_id, "thread_id": thread_id, "source": filename}) for chunk in chunks]
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
        
        return {"success": True, "message": f"Successfully processed and indexed {len(chunks)} paragraphs!"}
    except Exception as e:
        return {"success": False, "error": f"Error processing file: {str(e)}"}

# -------------------
# Database Init
# -------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")

pool = ConnectionPool(conninfo=DATABASE_URL, kwargs={"autocommit": True})

with pool.connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_metadata (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                user_id INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    conn.commit()

# -------------------
# Auth Helpers
# -------------------
def hash_password(password: str) -> str:
    salt = "ai_chatbot_secure_salt_2026"
    return hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def register_user(username: str, email: str, password: str) -> dict:
    password_hash = hash_password(password)
    try:
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                    (username, email, password_hash)
                )
                user_id = cursor.fetchone()[0]
                conn.commit()
                return {"success": True, "user_id": user_id, "username": username}
    except psycopg.errors.UniqueViolation:
        return {"success": False, "error": "Email already exists. Please sign in."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def authenticate_user(email: str, password: str) -> dict:
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, password_hash FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            
            if user and verify_password(password, user[2]):
                return {"success": True, "user_id": user[0], "username": user[1]}
            return {"success": False, "error": "Invalid email or password."}

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def search_my_documents(query: str, config: RunnableConfig) -> str:
    """Search through your uploaded personal documents for information. Use this when the user asks about their own uploaded files or specific data they provided."""
    try:
        user_id = config.get("configurable", {}).get("user_id")
        thread_id = config.get("configurable", {}).get("thread_id")
        if not user_id or not thread_id:
            return "Error: user_id or thread_id missing. Cannot perform secure search."
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        
        docs = vectorstore.similarity_search(query, k=4, filter={"user_id": user_id, "thread_id": thread_id})
        
        if not docs:
            return "No matching information found in your uploaded documents. Perhaps you haven't uploaded anything relevant yet."
            
        result = "Here is the relevant information from your documents:\n\n"
        for i, d in enumerate(docs):
            result += f"--- Excerpt {i+1} (Source: {d.metadata.get('source', 'Unknown')}) ---\n"
            result += d.page_content + "\n\n"
            
        return result
    except Exception as e:
        return f"Error searching documents: {str(e)}"

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers. Supported: add, sub, mul, div"""
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else: return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key: return {"error": "Alpha Vantage API key is missing."}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            return {"symbol": quote.get("01. symbol"), "price": quote.get("05. price"), "change": quote.get("09. change"), "change_percent": quote.get("10. change percent")}
        return {"error": "Invalid symbol or API rate limit exceeded.", "raw": data}
    except Exception as e:
        return {"error": f"API Request failed: {str(e)}"}

from datetime import datetime
@tool
def get_current_datetime() -> str:
    """Get the current date and time. Useful for answering questions about 'today', 'now', or calculating relative dates."""
    return datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

@tool
def get_weather(latitude: float, longitude: float) -> dict:
    """Get current weather given latitude and longitude."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    try:
        r = requests.get(url, timeout=5)
        return r.json().get("current_weather", {"error": "Weather data not found."})
    except Exception as e:
        return {"error": str(e)}

@tool
def get_wikipedia_summary(query: str) -> str:
    """Fetch a factual summary of a topic from Wikipedia. Use this for historical figures, places, events, and factual knowledge."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("extract", "No summary available.")
        return "Topic not found on Wikipedia. Try another search term."
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"

tools = [search_tool, get_stock_price, calculator, get_current_datetime, get_weather, get_wikipedia_summary, search_my_documents]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str

# -------------------
# 4. Nodes & Graph
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    summary = state.get("summary", "")
    
    sys_prompt_content = """You are a world-class, industry-grade AI assistant. 
Your core directive is to provide highly detailed, comprehensive, and logically structured answers.
When explaining concepts, solving problems, or providing guides, always break down your response into clear, step-by-step instructions or logical sections.
Use markdown formatting (bullet points, bold text, numbered lists, code blocks) to make complex information easy to read.
Never provide short or overly simplified answers unless explicitly asked. Always strive for maximum depth, accuracy, and professional clarity."""

    if summary:
        sys_prompt_content += f"\n\nHere is a summary of the earlier conversation: {summary}"
        
    sys_prompt = SystemMessage(content=sys_prompt_content)
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    response = llm_with_tools.invoke([sys_prompt] + filtered_messages)
    return {"messages": [response]}

def summarize_node(state: ChatState):
    summary = state.get("summary", "")
    messages = state["messages"]
    messages_to_summarize = messages[:-2]
    
    if not messages_to_summarize:
        return {}
        
    summary_prompt = f"Summarize the following conversation. If there is an existing summary, combine it with the new information into a single cohesive summary. Existing summary: {summary}\n\nNew conversation to summarize:\n"
    for m in messages_to_summarize:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        summary_prompt += f"{role}: {m.content}\n"
        
    summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = summary_llm.invoke(summary_prompt)
    
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id is not None]
    return {"summary": response.content, "messages": delete_messages}

def route_from_chat_node(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    if len(messages) > 6:
        return "summarize_node"
    return END

tool_node = ToolNode(tools)

# Set up Postgres checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize_node", summarize_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", route_from_chat_node)
graph.add_edge("tools", "chat_node")
graph.add_edge("summarize_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helpers
# -------------------
def retrieve_all_threads(user_id: int):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT thread_id, title FROM thread_metadata WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append({"id": row[0], "title": row[1] or "New Chat"})
            return result

def save_thread_title(thread_id: str, first_message: str, user_id: int):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT title FROM thread_metadata WHERE thread_id=%s", (str(thread_id),))
            if cursor.fetchone():
                return
                
    try:
        title_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=15, temperature=0)
        prompt = f"Generate a very short 2-4 word summary title for this message. No quotes, no punctuation: '{first_message}'"
        res = title_llm.invoke(prompt)
        title = res.content.replace('"', '').strip()
    except Exception:
        title = "New Chat"
        
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO thread_metadata (thread_id, title, user_id) VALUES (%s, %s, %s)", 
                (str(thread_id), title, user_id)
            )
            conn.commit()
