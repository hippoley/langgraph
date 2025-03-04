import os
import sys
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from api_config import get_llm

# Load environment variables
load_dotenv()

# Define a simple calculator tool with better error handling
@tool
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression.
    Examples: '2 + 2', '3 * 4', '10 / 2', '2 ** 3'
    """
    # 安全的数学表达式计算，避免使用 eval
    try:
        # 替换常见的数学运算符
        allowed_chars = set("0123456789+-*/().^ ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and +-*/().^ are allowed."
        
        # 替换 ^ 为 **
        expression = expression.replace("^", "**")
        
        # 使用更安全的方式计算
        import ast
        import math
        
        def safe_eval(expr):
            return ast.literal_eval(expr)
        
        # 尝试计算
        result = safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}. Please provide a valid mathematical expression."

# Define an expanded weather tool
@tool
def get_weather(location: str) -> str:
    """Get weather information for a specified location.
    Currently supports: Beijing, Shanghai, Guangzhou, Shenzhen, New York, London, Tokyo, Paris, and Sydney.
    """
    weather_data = {
        "Beijing": "Sunny, 25°C",
        "Shanghai": "Cloudy, 22°C",
        "Guangzhou": "Rainy, 28°C",
        "Shenzhen": "Overcast, 27°C",
        "New York": "Partly Cloudy, 18°C",
        "London": "Foggy, 15°C",
        "Tokyo": "Clear, 20°C",
        "Paris": "Sunny, 19°C",
        "Sydney": "Windy, 22°C"
    }
    
    # 尝试匹配不区分大小写的城市名
    for city in weather_data:
        if location.lower() == city.lower():
            return weather_data[city]
    
    return f"No weather information for {location}. Available cities: {', '.join(weather_data.keys())}"

# Add a new translation tool
@tool
def translate(text: str, target_language: str) -> str:
    """Translate text to the target language.
    Supported languages: Chinese, French, Spanish, Japanese, German.
    Examples: translate('hello', 'Chinese'), translate('goodbye', 'French')
    """
    translations = {
        "Chinese": {
            "hello": "你好", 
            "goodbye": "再见", 
            "thank you": "谢谢", 
            "yes": "是", 
            "no": "否",
            "how are you": "你好吗"
        },
        "French": {
            "hello": "bonjour", 
            "goodbye": "au revoir", 
            "thank you": "merci", 
            "yes": "oui", 
            "no": "non",
            "how are you": "comment allez-vous"
        },
        "Spanish": {
            "hello": "hola", 
            "goodbye": "adiós", 
            "thank you": "gracias", 
            "yes": "sí", 
            "no": "no",
            "how are you": "cómo estás"
        },
        "Japanese": {
            "hello": "こんにちは", 
            "goodbye": "さようなら", 
            "thank you": "ありがとう", 
            "yes": "はい", 
            "no": "いいえ",
            "how are you": "お元気ですか"
        },
        "German": {
            "hello": "hallo", 
            "goodbye": "auf wiedersehen", 
            "thank you": "danke", 
            "yes": "ja", 
            "no": "nein",
            "how are you": "wie geht es dir"
        }
    }
    
    # 规范化语言名称
    target_language = target_language.strip().capitalize()
    text_lower = text.lower().strip()
    
    if target_language in translations:
        if text_lower in translations[target_language]:
            return translations[target_language][text_lower]
        else:
            available_words = ", ".join(f"'{word}'" for word in translations[target_language].keys())
            return f"Translation not available for '{text}' to {target_language}. Available words: {available_words}"
    else:
        available_languages = ", ".join(translations.keys())
        return f"Language '{target_language}' not supported. Available languages: {available_languages}"

# Define the state schema
class AgentState(TypedDict):
    messages: List[Annotated[HumanMessage | AIMessage | ToolMessage | SystemMessage, "Messages"]]

# Initialize tools and model
tools = [calculator, get_weather, translate]
model = get_llm(temperature=0)

# Define the agent function
def agent_node(state: AgentState):
    """Process messages and generate agent response"""
    messages = state["messages"]
    
    # Get the most recent message
    last_message = messages[-1]
    
    # If this is the first message, add a system message
    if len(messages) == 1 and isinstance(last_message, HumanMessage):
        system_message = SystemMessage(content="""You are a helpful assistant with access to the following tools:
1. calculator: Calculate mathematical expressions
2. get_weather: Get weather information for major cities
3. translate: Translate simple phrases to different languages

Please use these tools when appropriate to answer user questions accurately.
""")
        messages = [system_message] + messages
    
    # If the last message is a tool message, we need to generate a new AI response
    if isinstance(last_message, ToolMessage):
        response = model.invoke(messages)
        return {"messages": messages + [response]}
    
    # If the last message is a human message, we need to generate an AI response
    if isinstance(last_message, HumanMessage):
        response = model.invoke(messages, tools=tools)
        return {"messages": messages + [response]}
    
    # If the last message is an AI message, we need to check if it wants to use a tool
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            # Process each tool call
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find the tool
                for tool in tools:
                    if tool.name == tool_name:
                        try:
                            # Call the tool
                            tool_result = tool(**tool_args)
                            
                            # Create a tool message
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                name=tool_name,
                                tool_call_id=tool_call["id"]
                            )
                            
                            return {"messages": messages + [tool_message]}
                        except Exception as e:
                            # Handle tool execution errors
                            error_message = f"Error executing {tool_name}: {str(e)}"
                            tool_message = ToolMessage(
                                content=error_message,
                                name=tool_name,
                                tool_call_id=tool_call["id"]
                            )
                            return {"messages": messages + [tool_message]}
        
        # If no tool calls, we're done
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
app = workflow.compile()

# Initialize memory saver
checkpointer = MemorySaver()

def print_colored(text, color="default"):
    """打印彩色文本"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "default": "\033[0m"
    }
    
    # Windows 命令行可能不支持 ANSI 颜色代码
    if sys.platform == "win32":
        print(text)
    else:
        print(f"{colors.get(color, colors['default'])}{text}{colors['default']}")

if __name__ == "__main__":
    print_colored("=== Simple LangGraph Agent Example ===", "cyan")
    print_colored("This agent can help with calculations, weather information, and translations.", "green")
    print_colored("Available tools:", "yellow")
    print_colored("1. calculator - Calculate mathematical expressions (e.g., '2 + 2', '3 * 4')", "yellow")
    print_colored("2. get_weather - Get weather for major cities (Beijing, Shanghai, New York, etc.)", "yellow")
    print_colored("3. translate - Translate simple phrases (hello, goodbye, etc.) to various languages", "yellow")
    print_colored("\nEnter 'exit' to quit", "magenta")
    
    # Create a session ID
    session_id = "demo-session"
    
    # Save conversation history
    messages = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print_colored("\nThank you for using the LangGraph Agent. Goodbye!", "cyan")
            break
        
        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        try:
            # Call the agent
            print_colored("Agent is thinking...", "blue")
            result = app.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Update message history
            messages = result["messages"]
            
            # Print AI reply
            ai_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
            if ai_message:
                print_colored(f"\nAI: {ai_message.content}", "green")
        except Exception as e:
            print_colored(f"\nError: {str(e)}", "red")
            print_colored("Please try again or type 'exit' to quit.", "red") 