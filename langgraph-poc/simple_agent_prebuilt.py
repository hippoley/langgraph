import os
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
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

# Define the state schema
class AgentState(TypedDict):
    messages: List[Annotated[HumanMessage | AIMessage | ToolMessage | SystemMessage, "Messages"]]

# Initialize tools and model
tools = [calculator, get_weather]
model = get_llm(temperature=0)

# Define the agent function
def agent_node(state: AgentState):
    """Process messages and generate agent response"""
    messages = state["messages"]
    
    # Get the most recent message
    last_message = messages[-1]
    
    # If this is the first message, add a system message
    if len(messages) == 1 and isinstance(last_message, HumanMessage):
        system_message = SystemMessage(content="You are a helpful AI assistant that can use tools.")
        messages = [system_message] + messages
    
    # Generate AI response
    response = model.invoke(messages=messages, tools=tools)
    messages.append(response)
    
    # Check if the AI wants to use a tool
    if response.tool_calls:
        # Process each tool call
        for tool_call in response.tool_calls:
            # Get the tool and arguments
            tool_name = tool_call.name
            tool_args = tool_call.args
            
            # Find the matching tool
            for tool in tools:
                if tool.name == tool_name:
                    # Call the tool
                    tool_result = tool(**tool_args)
                    
                    # Create a tool message
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call.id
                    )
                    messages.append(tool_message)
    
    return {"messages": messages}

# Define the should_continue function
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if the agent should continue processing or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is from the AI and has no tool calls, we're done
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "end"
    
    # Otherwise, continue
    return "continue"

# Create the graph
workflow = StateGraph(AgentState)

# Add the agent node
workflow.add_node("agent", agent_node)

# Define the edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

# Compile the graph
agent = workflow.compile()

# Initialize memory saver
checkpointer = MemorySaver()

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
        
        # Create a human message
        human_message = HumanMessage(content=user_input)
        
        # Call the agent
        result = agent.invoke(
            {"messages": [human_message]},
            config={"configurable": {"thread_id": session_id}}
        )
        
        # Print AI reply
        ai_message = result["messages"][-1]
        if isinstance(ai_message, AIMessage):
            print(f"\nAI: {ai_message.content}") 