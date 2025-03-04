import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from api_config import get_llm

# Load environment variables
load_dotenv()

# Define a simple calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {str(e)}"

# Define a simple weather tool
@tool
def get_weather(location: str) -> str:
    """Get weather information for a specified location"""
    weather_data = {
        "Beijing": "Sunny, 25°C",
        "Shanghai": "Cloudy, 22°C",
        "Guangzhou": "Rainy, 28°C",
        "Shenzhen": "Overcast, 27°C"
    }
    return weather_data.get(location, f"No weather information for {location}")

# Initialize tools and model
tools = [calculator, get_weather]
model = get_llm(temperature=0)

# Initialize memory saver
checkpointer = MemorySaver()

# Create agent using prebuilt function
agent = create_react_agent(model, tools, checkpointer=checkpointer)

if __name__ == "__main__":
    print("Simplified LangGraph Agent Example")
    print("Enter 'exit' to quit")
    
    # Create a session ID
    session_id = "demo-session"
    
    # Save conversation history
    messages = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Call the agent
        result = agent.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": session_id}}
        )
        
        # Update message history
        messages = result["messages"]
        
        # Print AI reply
        ai_message = messages[-1]
        print(f"\nAI: {ai_message['content']}") 