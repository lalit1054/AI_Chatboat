import streamlit as st
from app.langgraph_backend import chatbot, retrieve_all_threads, save_thread_title, register_user, authenticate_user
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import time

st.set_page_config(page_title="AI Assistant", page_icon="✨", layout="centered")

# =========================== Custom CSS ===========================
st.markdown("""
<style>
    [data-testid="stHeader"] { visibility: hidden; }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .main-title { font-weight: 600; background: -webkit-linear-gradient(45deg, #4f46e5, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; }
    .stChatMessage { border-radius: 12px; padding: 0.5rem; margin-bottom: 1rem; }
    [data-testid="stSidebar"] { background-color: #1a1c23; border-right: 1px solid #2d3748; }
    [data-testid="stChatInput"] { border-radius: 25px !important; border: 1px solid #4a5568 !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    /* Form styling for Auth */
    div[data-testid="stForm"] { border-radius: 15px; border: 1px solid #374151; padding: 2rem; background-color: #111827; }
    div[data-testid="stForm"] label p { color: #ec4899 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ======================= Session Initialization ===================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

# =========================== Auth Logic ===========================
def handle_logout():
    st.session_state["authenticated"] = False
    st.session_state["user_id"] = None
    st.session_state["username"] = None
    st.session_state["thread_id"] = None
    st.session_state["message_history"] = []

if not st.session_state["authenticated"]:
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Welcome to AI Assistant ✨</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.session_state["auth_mode"] == "login":
            st.markdown("### Sign In")
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In", use_container_width=True)
                
                if submitted:
                    if not email or not password:
                        st.error("Please fill in all fields.")
                    else:
                        res = authenticate_user(email, password)
                        if res["success"]:
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = res["user_id"]
                            st.session_state["username"] = res["username"]
                            st.rerun()
                        else:
                            st.error(res["error"])
            
            st.markdown("<br>Don't have an account?", unsafe_allow_html=True)
            if st.button("Create an account"):
                st.session_state["auth_mode"] = "signup"
                st.rerun()
                
        else:
            st.markdown("### Sign Up")
            with st.form("signup_form"):
                username = st.text_input("Username", placeholder="Your Name")
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submitted:
                    if not username or not email or not password:
                        st.error("Please fill in all fields.")
                    else:
                        res = register_user(username, email, password)
                        if res["success"]:
                            st.success("Account created! Logging you in...")
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = res["user_id"]
                            st.session_state["username"] = res["username"]
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(res["error"])
            
            st.markdown("<br>Already have an account?", unsafe_allow_html=True)
            if st.button("Back to Login"):
                st.session_state["auth_mode"] = "login"
                st.rerun()
    
    st.stop() # Stop execution of the rest of the app if not authenticated

# =========================== Main App ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.rerun()

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state or st.session_state["thread_id"] is None:
    st.session_state["thread_id"] = generate_thread_id()

# Fetch only the threads for the currently logged in user
chat_threads = retrieve_all_threads(st.session_state["user_id"])

# ============================ Sidebar ============================
with st.sidebar:
    st.markdown("<h1 style='color: #3b82f6; font-size: 2.2rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.5rem;'>✨ Lalit's AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f97316;'>Recent Conversations</h3>", unsafe_allow_html=True)
    
    for thread in chat_threads:
        is_active = thread['id'] == st.session_state.get("thread_id")
        button_type = "primary" if is_active else "secondary"
        
        if st.button(f"💬 {thread['title']}", key=f"btn_{thread['id']}", use_container_width=True, type=button_type):
            st.session_state["thread_id"] = thread['id']
            messages = load_conversation(thread['id'])
            
            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage): temp_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content: temp_messages.append({"role": "assistant", "content": msg.content})
            
            st.session_state["message_history"] = temp_messages
            st.rerun()

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True):
        handle_logout()
        st.rerun()

# ============================ Main UI ============================
username_display = st.session_state.get('username', '')
st.markdown(f"<h1 class='main-title'>Hi {username_display}! how can I help you today? 🚀</h1>", unsafe_allow_html=True)

if not st.session_state["message_history"]:
    st.caption("✨ I am powered by advanced LLMs. Try asking me a question!")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Message AI Assistant...")

if user_input:
    # Save title to DB with the correct user_id
    if not st.session_state["message_history"]:
        save_thread_title(st.session_state["thread_id"], user_input, st.session_state["user_id"])
    
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream({"messages": [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode="messages"):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None: status_holder["box"] = st.status(f"🧠 **Thinking... running `{tool_name}` tool**", expanded=False)
                    else: status_holder["box"].update(label=f"🧠 **Thinking... running `{tool_name}` tool**", state="running")
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        time.sleep(0.1)
        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None: status_holder["box"].update(label="✅ Tools executed successfully", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
