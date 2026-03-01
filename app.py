import streamlit as st
import requests
import pandas as pd
import os
from configs.config import config

API_URL = "http://localhost:8001"

st.set_page_config(page_title="RAG Analytics Chatbot", layout="wide")

def query_api(question: str, top_k: int = 5) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": question, "top_k": top_k},
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "latency_ms": 0, "num_sources": 0}

def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json().get("status") == "healthy"
    except:
        return False

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Context documents", min_value=1, max_value=11, value=5)
    st.divider()
    st.header("Sample Questions")
    sample_questions = [
        "What is the total revenue?",
        "Which region performs best?",
        "What is the utilization rate?",
        "How has revenue been trending?",
        "How are enterprise customers performing?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}"):
            st.session_state.pending_query = q
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()
    st.markdown("[GitHub](https://github.com/Haani76/rag-analytics-chatbot)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Header
st.title("RAG Analytics Chatbot")
st.markdown("**Natural language queries powered by RAG + Flan-T5**")
st.divider()

# API health check
if check_api_health():
    st.success("API connected and ready.")
else:
    st.error("API not reachable. Make sure the API server is running on port 8001.")

# Tabs
tab1, tab2 = st.tabs(["Data Overview", "About"])

with tab1:
    sales_path = os.path.join(config.RAW_DATA_DIR, "sales_data.csv")
    if os.path.exists(sales_path):
        sales_df = pd.read_csv(sales_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(sales_df):,}")
        col2.metric("Total Revenue", f"${sales_df['revenue'].sum():,.0f}")
        col3.metric("Customers", sales_df["customer_id"].nunique())
        col4.metric("Date Range", f"{sales_df['date'].min()[:7]} to {sales_df['date'].max()[:7]}")
        st.markdown("**Sales Data Preview:**")
        st.dataframe(sales_df.head(10), use_container_width=True)
        st.markdown("**Revenue by Region:**")
        region_rev = sales_df.groupby("region")["revenue"].sum().reset_index()
        st.bar_chart(region_rev.set_index("region"))

with tab2:
    st.markdown("""
    ### What is this?
    A RAG chatbot that answers natural language questions about business KPIs.

    ### How it works
    1. Business data is embedded using sentence-transformers
    2. User query is matched against stored documents using FAISS
    3. Most relevant documents are retrieved
    4. Flan-T5 generates a natural language answer with source citations

    ### Model Performance
    | Metric | Value |
    |---|---|
    | Embedding Model | all-MiniLM-L6-v2 |
    | LLM | google/flan-t5-small |
    | Vector Store | FAISS |
    | Documents Indexed | 11 KPI summaries |

    ### Links
    - [GitHub Repository](https://github.com/Haani/rag-analytics-chatbot)
    """)

st.divider()
st.subheader("Chat")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
        if message.get("latency_ms"):
            st.caption(f"Latency: {message['latency_ms']}ms")

# Handle sidebar button queries
if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_api(prompt, top_k=top_k)
        st.markdown(response["answer"])
        with st.expander("Sources"):
            for source in response.get("sources", []):
                st.markdown(f"- {source}")
        st.caption(f"Latency: {response.get('latency_ms', 0)}ms | Sources: {response.get('num_sources', 0)}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "latency_ms": response.get("latency_ms", 0),
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about your business KPIs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_api(prompt, top_k=top_k)
        st.markdown(response["answer"])
        with st.expander("Sources"):
            for source in response.get("sources", []):
                st.markdown(f"- {source}")
        st.caption(f"Latency: {response.get('latency_ms', 0)}ms | Sources: {response.get('num_sources', 0)}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "latency_ms": response.get("latency_ms", 0),
    })