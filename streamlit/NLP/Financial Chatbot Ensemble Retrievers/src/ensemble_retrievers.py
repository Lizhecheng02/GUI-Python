from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *
from dotenv import find_dotenv, load_dotenv
import os
import weaviate

load_dotenv(find_dotenv())


def retriever_creation():

    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    auth_config = weaviate.AuthApiKey(
        api_key=weaviate_api_key
    )
    weaviate_client = weaviate.Client(
        url="",
        auth_client_secret=auth_config
    )

    dir_loader = DirectoryLoader(
        DATA_DIR_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = dir_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    input_text = text_splitter.split_documents(documents=docs)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDER,
        model_kwargs={"device": "cpu"}
    )

    weaviate_vectorstore = Weaviate.from_documents(
        input_text,
        embedding=embedding,
        client=weaviate_client,
        by_text=False
    )

    weaviate_retriever = weaviate_vectorstore.as_retriever(
        search_kwargs=SEARCH_KWARGS
    )
    bm25_retriever = BM25Retriever.from_documents(input_text)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[weaviate_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever
