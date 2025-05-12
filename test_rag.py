from ragbase.ingestor import Ingestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever
from ragbase.chain import create_chain
import pathlib
import asyncio

async def main():
    print("Initializing components...")
    llm = create_llm()
    print("LLM created")
    
    ingestor = Ingestor()
    print("Ingestor created")
    
    print("Processing PDF...")
    loop = asyncio.get_running_loop()
    vector_store = await loop.run_in_executor(None, ingestor.ingest, [pathlib.Path('tmp/test.pdf')])
    print("Vector store created")
    
    retriever = create_retriever(llm, vector_store=vector_store)
    print("Retriever created")
    
    chain = create_chain(llm, retriever)
    print("Chain created")
    
    print("\nTesting question answering...")
    print("Question: What is RAG?")
    print("Answer: ", end="", flush=True)
    
    async for chunk in chain.astream_events(
        {"question": "What is RAG?"}, 
        config={
            "configurable": {"session_id": "test-session-123"}
        },
        version="v2",
        include_names=["chain_answer"]
    ):
        if 'event' in chunk and chunk['event'] == 'on_chain_stream':
            print(chunk['data']['chunk'].content, end="", flush=True)
    
    print("\n\nAll components successfully initialized and working together!")

if __name__ == "__main__":
    asyncio.run(main()) 