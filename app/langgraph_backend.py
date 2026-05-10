import os
import sqlite3
import requests
import hashlib
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# -------------------
# Database Init
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

with conn:
    # 1. Create Users Table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    
    # 2. Create Thread Metadata Table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    
    # Safely try to add user_id if migrating from older version
    try:
        conn.execute("ALTER TABLE thread_metadata ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists or other safe error

# -------------------
# Auth Helpers
# -------------------
def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = "ai_chatbot_secure_salt_2026"  # In production, use os.urandom(32) and store it with the hash
    return hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a stored password against one provided by user"""
    return hash_password(password) == hashed

def register_user(username: str, email: str, password: str) -> dict:
    password_hash = hash_password(password)
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            return {"success": True, "user_id": cursor.lastrowid, "username": username}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Email already exists. Please sign in."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def authenticate_user(email: str, password: str) -> dict:
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE email = ?", (email,))
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
    if not api_key: return {"error": "Alpha Vantage API key is missing. Please add ALPHA_VANTAGE_API_KEY to your .env file."}
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

tools = [search_tool, get_stock_price, calculator, get_current_datetime, get_weather, get_wikipedia_summary]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes & Graph
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helpers
# -------------------
def retrieve_all_threads(user_id: int):
    """Retrieve all threads for a specific user, ordered by creation time."""
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, title FROM thread_metadata WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = cursor.fetchall()
    
    result = []
    metadata_ids = set()
    for row in rows:
        result.append({"id": row[0], "title": row[1] or "New Chat"})
        metadata_ids.add(row[0])
        
    # We could also filter checkpointer list if needed, but the metadata table is the authoritative source for user-bound threads.
    # To prevent leaking threads from other users, we only return what's in thread_metadata for this user.
    return result

def save_thread_title(thread_id: str, first_message: str, user_id: int):
    """Generate a title for the thread using LLM and save it, bound to user_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM thread_metadata WHERE thread_id=?", (str(thread_id),))
    if cursor.fetchone():
        return # Title exists
        
    try:
        title_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=15, temperature=0)
        prompt = f"Generate a very short 2-4 word summary title for this message. No quotes, no punctuation: '{first_message}'"
        res = title_llm.invoke(prompt)
        title = res.content.replace('"', '').strip()
    except Exception:
        title = "New Chat"
        
    with conn:
        conn.execute(
            "INSERT INTO thread_metadata (thread_id, title, user_id) VALUES (?, ?, ?)", 
            (str(thread_id), title, user_id)
        )
