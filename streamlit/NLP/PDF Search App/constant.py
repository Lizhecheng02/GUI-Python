import os
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    chroma_db_imp1='duckdb+parquet',
    persist_directory="db",
    anonymized_telemetry=False
)
