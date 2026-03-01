import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Project
    PROJECT_NAME = os.getenv("PROJECT_NAME", "rag-analytics-chatbot")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # LLM
    LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # Retrieval
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

    # Vector Store
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/embeddings/chroma")

    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-analytics-chatbot")

    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8001))

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")


config = Config()