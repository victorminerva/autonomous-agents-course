# This agent read all the .csv and .txt files and create a vectorial semantic index. 
# Also do some questions with natu
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

def build_agent(df):
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
        max_tokens=256  # Avoid credit/token errors
    )

    embed_model = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002",
        chunk_size=1000
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True
    )

    return agent, embed_model
