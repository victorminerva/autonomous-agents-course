# This agent read all the .csv and .txt files and create a vectorial semantic index. 
# Also do some questions with natu
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from langchain.chat_models import ChatOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

def build_agent(data_path="data"):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("Please, define the environment variables OPENAI_API_KEY e OPENAI_BASE_URL")
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.3,
        max_tokens=512
    )

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader(input_dir=data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index.as_query_engine()
