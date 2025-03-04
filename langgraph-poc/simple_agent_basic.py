#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Agent Basic - 简化版本，避免使用 StateGraph
直接使用 OpenAI API 实现基本功能
"""

import os
import sys
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# 尝试导入 OpenAI 库
try:
    from openai import OpenAI
except ImportError:
    print("Installing OpenAI library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

# 定义工具函数
def calculator(expression):
    """计算数学表达式"""
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

def get_weather(location):
    """获取天气信息"""
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

def translate(text, target_language):
    """翻译文本"""
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
    
    text_lower = text.lower()
    
    if target_language in translations:
        if text_lower in translations[target_language]:
            return translations[target_language][text_lower]
        else:
            return f"Sorry, I don't know how to translate '{text}' to {target_language}."
    else:
        return f"Sorry, {target_language} is not supported. Available languages: {', '.join(translations.keys())}"

# 打印彩色文本
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

# 主函数
def main():
    print_colored("=== Simple Agent Basic Example ===", "cyan")
    print_colored("This agent can help with calculations, weather information, and translations.", "green")
    print_colored("Available tools:", "yellow")
    print_colored("1. calculator - Calculate mathematical expressions (e.g., '2 + 2', '3 * 4')", "yellow")
    print_colored("2. get_weather - Get weather for major cities (Beijing, Shanghai, New York, etc.)", "yellow")
    print_colored("3. translate - Translate simple phrases (hello, goodbye, etc.) to various languages", "yellow")
    print_colored("\nEnter 'exit' to quit", "magenta")
    
    # 获取 API 密钥
    api_key = os.getenv("OPENAI_API_KEY", "fk222719-4TlnHx5wbaXtUm4CcneT1oLogM3TKGDB")
    api_base = os.getenv("OPENAI_API_BASE", "https://oa.api2d.net")
    
    # 创建 OpenAI 客户端
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    # 保存对话历史
    messages = [
        {"role": "system", "content": """You are a helpful assistant that can use tools to help users.
You have access to the following tools:
1. calculator - Calculate mathematical expressions
2. get_weather - Get weather information for a location
3. translate - Translate text to another language

When you need to use a tool, respond in the following format:
```tool
{
  "tool": "tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

For example, to use the calculator:
```tool
{
  "tool": "calculator",
  "parameters": {
    "expression": "2 + 2"
  }
}
```

For weather:
```tool
{
  "tool": "get_weather",
  "parameters": {
    "location": "Beijing"
  }
}
```

For translation:
```tool
{
  "tool": "translate",
  "parameters": {
    "text": "hello",
    "target_language": "Chinese"
  }
}
```

Only use these tools when necessary. If you can answer directly, do so.
"""}
    ]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print_colored("\nThank you for using the Simple Agent. Goodbye!", "cyan")
            break
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 调用 OpenAI API
            print_colored("Agent is thinking...", "blue")
            response = client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0
            )
            
            # 获取回复
            ai_message = response.choices[0].message.content
            
            # 检查是否包含工具调用
            if "```tool" in ai_message:
                # 提取工具调用
                tool_start = ai_message.find("```tool") + 7
                tool_end = ai_message.find("```", tool_start)
                tool_json = ai_message[tool_start:tool_end].strip()
                
                try:
                    # 解析工具调用
                    tool_call = json.loads(tool_json)
                    tool_name = tool_call.get("tool")
                    parameters = tool_call.get("parameters", {})
                    
                    # 执行工具调用
                    if tool_name == "calculator":
                        result = calculator(parameters.get("expression", ""))
                    elif tool_name == "get_weather":
                        result = get_weather(parameters.get("location", ""))
                    elif tool_name == "translate":
                        result = translate(
                            parameters.get("text", ""),
                            parameters.get("target_language", "")
                        )
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    # 添加工具结果到消息历史
                    messages.append({"role": "assistant", "content": ai_message})
                    messages.append({"role": "system", "content": f"Tool result: {result}"})
                    
                    # 让 AI 解释结果
                    response = client.chat.completions.create(
                        model="o3-mini",
                        messages=messages,
                        temperature=0
                    )
                    
                    # 获取最终回复
                    final_message = response.choices[0].message.content
                    messages.append({"role": "assistant", "content": final_message})
                    
                    # 打印回复
                    print_colored(f"\nAI: {final_message}", "green")
                    
                except json.JSONDecodeError:
                    print_colored(f"\nAI: {ai_message}", "green")
                    messages.append({"role": "assistant", "content": ai_message})
            else:
                # 直接回复
                print_colored(f"\nAI: {ai_message}", "green")
                messages.append({"role": "assistant", "content": ai_message})
                
        except Exception as e:
            print_colored(f"\nError: {str(e)}", "red")
            print_colored("Please try again or type 'exit' to quit.", "red")

if __name__ == "__main__":
    main() 