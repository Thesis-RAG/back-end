"""
Application settings loaded from environment variables and the .env file.
All fields can be overridden at runtime via environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


# Pydantic-settings model; values are read from .env or the process environment.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rag-role-enterprise-api"
    env: str = "dev"
    cors_origins: str = (
        "http://localhost:5173,http://localhost:8083,"
        "https://main.d37e45ukb2hsrw.amplifyapp.com,"
        "https://main.roles-aware-rag.amplifyapp.com"
    )
    gmail_redirect_uri: str = "http://localhost:8083/gmail/callback"

    database_url: str = "mysql+pymysql://rag:rag@mysql:3306/ragdb?charset=utf8mb4"
    redis_url: str = "redis://redis:6379/0"

    jwt_secret_key: str = "secret"
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = 1440

    openfga_url: str = "http://openfga:8080"
    openfga_store_name: str = "rag-enterprise"
    openfga_store_id: str = ""
    openfga_model_id: str = ""
    openfga_config_file: str = "/run/openfga/config.json"

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_public_endpoint: str = "localhost:9000"
    minio_bucket_raw: str = "raw"
    minio_bucket_processed: str = "processed"

    chroma_path: str = "/data/chroma"
    chroma_collection: str = "document_chunks"
    chroma_host: str = "chroma"
    chroma_port: int = 8000

    max_upload_size_mb: int = 50
    default_policy_version: str = "v1"
    gliner_model_name: str = "urchade/gliner_multi-v2.1"
    gliner_threshold: float = 0.3

    # LLM configuration
    llm_provider: str | None = None

    # OpenAI settings
    openai_api_key: str | None = None
    openai_api_base: str | None = None
    openai_model: str | None = "gpt-4o-mini"
    openai_embedding_model: str | None = "text-embedding-3-small"
    openai_embedding_dims: int = 1536

    # Olama local server settings (e.g. http://localhost:11434)
    olama_url: str | None = None
    olama_model: str | None = None

    llm_timeout_seconds: int = 30

    # Ontology graph (Neo4j) — the enterprise-ontology visualization, merged
    # in-process into this backend (app/services/ontology_sync_service.py,
    # app/services/ontology_graph_service.py). No separate service, no
    # shared-secret auth between services — it's all one process now.
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ontology12345"

    # Document content-signing private key (Ed25519, PEM). Kept as a file
    # outside the database on purpose — see document_signing_service.py.
    # Auto-generated on first use if the file doesn't exist yet.
    document_signing_key_path: str = "/app/.secrets/document_signing_ed25519.pem"


# Module-level singleton; import this instance throughout the application.
settings = Settings()
