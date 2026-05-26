import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Page Config ---
st.set_page_config(page_title="Groq Chatbot", page_icon="⚡")
st.title("⚡ Groq + LangChain Chatbot")
st.caption("Powered by Groq's ultra-fast inference")

# --- Sidebar: API Key Input ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )
    model_choice = st.selectbox(
        "Choose Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
    )
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        height=100
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# --- Chat Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Build LangChain + Groq
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_choice,
        temperature=0.7
    )

    # Prepare messages with system prompt
    all_messages = [SystemMessage(content=system_prompt)] + st.session_state.messages

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(all_messages)
            st.write(response.content)

    # Save assistant response
    st.session_state.messages.append(AIMessage(content=response.content))