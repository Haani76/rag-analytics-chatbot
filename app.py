import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import config

st.set_page_config(page_title="RAG Analytics Chatbot", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

@st.cache_resource(show_spinner=False)
def load_vector_store():
    from src.retrieval.vector_store import build_vector_store
    return build_vector_store()

@st.cache_resource(show_spinner=False)
def load_llm():
    from src.generation.llm_engine import LLMEngine
    return LLMEngine()

def get_answer(query: str, top_k: int = 5) -> dict:
    import time
    start = time.time()
    store = load_vector_store()
    llm = load_llm()
    context_docs = store.search(query, top_k=top_k)
    if context_docs:
        result = llm.generate(query, context_docs)
        retrieval_status = "success"
    else:
        result = llm.generate_without_context(query)
        retrieval_status = "no_context"
    latency_ms = round((time.time() - start) * 1000, 2)
    return {
        "answer": result["answer"],
        "sources": result["context_used"],
        "num_sources": result["num_context_docs"],
        "latency_ms": latency_ms,
        "retrieval_status": retrieval_status,
    }

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
        "Which product generates most revenue?",
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

# Header
st.title("RAG Analytics Chatbot")
st.markdown("**Natural language queries powered by RAG + Flan-T5**")
st.divider()

# Load models upfront with clear status
col1, col2 = st.columns(2)
with col1:
    with st.spinner("Loading embedding model..."):
        store = load_vector_store()
    st.success(f"Vector store ready — {len(store.documents)} documents")
with col2:
    with st.spinner("Loading Flan-T5..."):
        llm = load_llm()
    st.success("LLM ready")

# Tabs
tab1, tab2 = st.tabs(["Data Overview", "About"])

with tab1:
    sales_path = os.path.join(config.RAW_DATA_DIR, "sales_data.csv")
    if os.path.exists(sales_path):
        sales_df = pd.read_csv(sales_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(sales_df):,}")
        c2.metric("Total Revenue", f"${sales_df['revenue'].sum():,.0f}")
        c3.metric("Customers", sales_df["customer_id"].nunique())
        c4.metric("Date Range", f"{sales_df['date'].min()[:7]} to {sales_df['date'].max()[:7]}")
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
    1. Business data is embedded using sentence-transformers (all-MiniLM-L6-v2)
    2. User query is semantically matched against stored documents using FAISS
    3. Most relevant documents are retrieved
    4. Flan-T5 LLM generates a natural language answer with source citations

    ### Model Performance
    | Metric | Value |
    |---|---|
    | Embedding Model | all-MiniLM-L6-v2 |
    | LLM | google/flan-t5-small |
    | Vector Store | FAISS |
    | Documents Indexed | 11 KPI summaries |
    | Avg Latency | ~1-3 seconds (CPU) |

    ### Links
    - [GitHub Repository](https://github.com/Haani76/rag-analytics-chatbot)
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
            response = get_answer(prompt, top_k=top_k)
        st.markdown(response["answer"])
        with st.expander("Sources"):
            for source in response.get("sources", []):
                st.markdown(f"- {source}")
        st.caption(f"Latency: {response['latency_ms']}ms | Sources: {response['num_sources']}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "latency_ms": response["latency_ms"],
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about your business KPIs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(prompt, top_k=top_k)
        st.markdown(response["answer"])
        with st.expander("Sources"):
            for source in response.get("sources", []):
                st.markdown(f"- {source}")
        st.caption(f"Latency: {response['latency_ms']}ms | Sources: {response['num_sources']}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "latency_ms": response["latency_ms"],
    })