import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def get_llm(temperature=0):
    """Get a configured LLM instance"""
    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY", "fk222719-4TlnHx5wbaXtUm4CcneT1oLogM3TKGDB")
    api_base = os.getenv("OPENAI_API_BASE", "https://oa.api2d.net")
    model_name = os.getenv("OPENAI_MODEL_NAME", "o3-mini")
    
    # Create LLM instance
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    
    return llm 