from pathlib import Path
from typing import List
import logging

from langchain_community.document_loaders import PyPDFium2Loader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbase.config import Config


class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
            add_start_index=True,
        )

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        documents = []
        for doc_path in doc_paths:
            try:
                # Choose the appropriate loader based on file extension
                if doc_path.suffix.lower() == '.pdf':
                    loader = PyPDFium2Loader(str(doc_path))
                else:
                    # Use TextLoader for other file types
                    loader = TextLoader(str(doc_path))
                
                loaded_documents = loader.load()
                if not loaded_documents:
                    logging.warning(f"No content loaded from {doc_path}")
                    continue
                    
                # Process loaded documents directly using split_documents
                split_docs = self.recursive_splitter.split_documents(loaded_documents)
                if not split_docs:
                    logging.warning(f"No documents created from {doc_path}")
                    continue
                    
                documents.extend(split_docs)
            except Exception as e:
                logging.exception(f"Error processing {doc_path}: {str(e)}")
        
        if not documents:
            raise ValueError("No documents were loaded. Please check your files.")
        
        # Create sample embeddings to verify before proceeding
        test_embedding = self.embeddings.embed_query("Test query")
        if not test_embedding or len(test_embedding) == 0:
            raise ValueError("Embedding model failed to generate embeddings")
            
        print(f"Successfully created {len(documents)} document chunks")
        
        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=Config.Path.DATABASE_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )
