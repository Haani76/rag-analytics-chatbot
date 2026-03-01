import streamlit as st

st.title("Test")

@st.cache_resource
def load_store():
    from src.retrieval.vector_store import build_vector_store
    return build_vector_store()

@st.cache_resource
def load_llm():
    from src.generation.llm_engine import LLMEngine
    return LLMEngine()

st.write("Loading vector store...")
store = load_store()
st.write("Vector store loaded!")

st.write("Loading LLM...")
llm = load_llm()
st.write("LLM loaded!")

st.write("All done!")