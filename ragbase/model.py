from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from ragbase.config import Config

# def create_llm():
#     return ChatOllama(
#         model=Config.Model.LOCAL_LLM,  # Change to use the available model
#         temperature=Config.Model.TEMPERATURE,
#         max_tokens=Config.Model.MAX_TOKENS,
#         keep_alive=True,
#         timeout=120
#     )

def create_llm() -> BaseLanguageModel:
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            max_tokens=Config.Model.MAX_TOKENS,
            keep_alive=True,
            timeout=120
        )
    else:
        return ChatGroq(
            temperature=Config.Model.TEMPERATURE,
            model_name=Config.Model.REMOTE_LLM,
            max_tokens=Config.Model.MAX_TOKENS,
        )


def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)
