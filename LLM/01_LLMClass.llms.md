# LLM Class

\- skip_showdoc: true

Author

Benedict Thekkel

## With design pattern

``` python
from abc import ABC, abstractmethod
from datetime import UTC, datetime
import uuid
import json
import os
from dotenv import load_dotenv
import ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------------- Adapter Pattern (LangChain LLM Wrapper) ----------------
class LLMAdapter(ABC):
    @abstractmethod
    def generate_response(self, messages: list) -> str:
        pass

class LangChainOpenAIAdapter(LLMAdapter):
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=os.getenv("openai_api_key")
        )

    def generate_response(self, messages: list) -> str:
        return self.llm.invoke(messages).content.strip()

class OllamaAdapter(LLMAdapter):
    def __init__(self, model="deepseek-r1:7b"):
        self.model = model

    def generate_response(self, messages: list) -> str:
        def get_role(msg):
            if isinstance(msg, HumanMessage):
                return "User"
            elif isinstance(msg, SystemMessage):
                return "System"
            elif isinstance(msg, AIMessage):
                return "Assistant"
            return "Unknown"

        prompt = "\n".join([f"{get_role(msg)}: {msg.content}" for msg in messages])
        response = ollama.generate(model=self.model, prompt=prompt)
        return response["response"].strip()
        
# ---------------- Prompt Strategy ----------------
class PromptStrategy(ABC):
    @abstractmethod
    def build_messages(self, user_input: str) -> list:
        pass

class QAStrategy(PromptStrategy):
    def build_messages(self, user_input: str) -> list:
        return [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Q: {user_input}\nA:")
        ]

class SummarizeStrategy(PromptStrategy):
    def build_messages(self, user_input: str) -> list:
        return [
            SystemMessage(content="You summarize input text."),
            HumanMessage(content=f"Summarize:\n{user_input}")
        ]

# ---------------- Chat History ----------------
class ChatHistory:
    def __init__(self):
        self.data = {}

    def log(self, session_id, role, message):
        self.data.setdefault(session_id, []).append({
            "timestamp": datetime.now(UTC).isoformat(),
            "role": role,
            "message": message
        })

    def show(self, session_id):
        for entry in self.data.get(session_id, []):
            print(f"{entry['timestamp']} [{entry['role']}]: {entry['message']}")

    def export(self, session_id):
        return json.dumps(self.data.get(session_id, []), indent=2)

# ---------------- LLM Client ----------------
class LLMClient:
    def __init__(self, adapter: LLMAdapter, strategy: PromptStrategy):
        self.adapter = adapter
        self.strategy = strategy
        self.history = ChatHistory()
        self.session_id = str(uuid.uuid4())

    def chat(self, user_input):
        messages = self.strategy.build_messages(user_input)
        self.history.log(self.session_id, "user", user_input)
        response = self.adapter.generate_response(messages)
        self.history.log(self.session_id, "assistant", response)
        return response

    def view_history(self):
        self.history.show(self.session_id)

    def export_history(self):
        return self.history.export(self.session_id)
```

## ChatGPT

``` python
# ---------------- Entry Point ----------------
adapter = LangChainOpenAIAdapter()
strategy = QAStrategy()  # or SummarizeStrategy()
bot = LLMClient(adapter, strategy)
```

``` python
print("\nChat session started. Type 'exit' to quit.\n")

while True:
    msg = input("You: ")
    if msg.strip().lower() in {"exit", "quit"}:
        break
    response = bot.chat(msg)
    print(f"Bot: {response}")
```

``` python
# View history in Jupyter
bot.view_history()

# Export as JSON
json_output = bot.export_history()
print(json_output)
```

    2025-04-21T14:38:45.428523 [user]: how is the weather today
    2025-04-21T14:38:46.342267 [assistant]: I'm sorry, but I don't have real-time data access to provide current weather updates. You can check the latest weather information through a weather app or website for your location.
    2025-04-21T14:38:52.994441 [user]: what day is it?
    2025-04-21T14:38:54.119662 [assistant]: I'm sorry, I can't provide real-time information or current dates. Please check your device's calendar or clock for the current day.
    2025-04-21T14:39:10.178454 [user]: whats the difference between django and fastapi
    2025-04-21T14:39:17.439090 [assistant]: Django and FastAPI are both web frameworks for Python, but they have different design philosophies and use cases. Here's a comparison of the two:

    1. **Speed and Performance**:
       - **FastAPI**: As the name suggests, FastAPI is designed for high performance. It is built on top of Starlette and is asynchronous by nature, making it well-suited for handling lots of concurrent requests efficiently. It's particularly beneficial for applications that require high throughput and low latency, such as APIs and microservices.
       - **Django**: Django is a synchronous framework, which can be less efficient for handling a large number of concurrent requests. However, with Django Channels, you can add asynchronous capabilities if needed. Django is more focused on rapid development and follows a "batteries-included" philosophy.

    2. **Features and Ecosystem**:
       - **Django**: It is a full-fledged web framework that comes with a wide range of features out of the box, such as an ORM (Object-Relational Mapping), an admin panel, authentication, forms, and more. This makes it ideal for building traditional monolithic web applications quickly.
       - **FastAPI**: It is minimalistic and focused on building APIs. It provides automatic generation of OpenAPI documentation and JSON Schema, which is very useful for API development. It doesn’t come with built-in components like Django’s ORM or admin interface, but it integrates well with third-party libraries.

    3. **Ease of Use and Learning Curve**:
       - **Django**: It is known for its "Django way" of doing things and has a steeper learning curve for beginners. However, once you are familiar with its conventions, it can significantly speed up development.
       - **FastAPI**: It is relatively easy to get started with, especially if you are familiar with modern Python features like type hints. The framework leverages Python type hints to provide input validation and serialization automatically.

    4. **Use Cases**:
       - **Django**: Ideal for building complex web applications that require a lot of built-in functionality and rapid prototyping. It's popular for building content management systems, social networks, and other web applications.
       - **FastAPI**: Best suited for building high-performance APIs and microservices. It is often used in projects that require asynchronous capabilities and high throughput, like real-time data processing.

    5. **Community and Support**:
       - **Django**: Has a large and mature community with extensive documentation and a wide range of third-party packages.
       - **FastAPI**: A newer framework with a rapidly growing community. It has excellent documentation and is becoming increasingly popular for API development.

    In summary, if you're building a traditional web application with a need for a lot of built-in tools and a structured framework, Django might be the better choice. If you're focused on building fast, efficient RESTful APIs with modern Python practices, FastAPI could be more suitable.
    2025-04-21T14:39:24.569336 [user]: exit/"
    2025-04-21T14:39:26.610750 [assistant]: It looks like your query might be incomplete or contain a typo. Could you please provide more context or clarify what you're asking about "exit/"? Are you looking for information on how to exit a program or command in a specific context, or is there something else you need help with? Let me know so I can assist you better!
    [
      {
        "timestamp": "2025-04-21T14:38:45.428523",
        "role": "user",
        "message": "how is the weather today"
      },
      {
        "timestamp": "2025-04-21T14:38:46.342267",
        "role": "assistant",
        "message": "I'm sorry, but I don't have real-time data access to provide current weather updates. You can check the latest weather information through a weather app or website for your location."
      },
      {
        "timestamp": "2025-04-21T14:38:52.994441",
        "role": "user",
        "message": "what day is it?"
      },
      {
        "timestamp": "2025-04-21T14:38:54.119662",
        "role": "assistant",
        "message": "I'm sorry, I can't provide real-time information or current dates. Please check your device's calendar or clock for the current day."
      },
      {
        "timestamp": "2025-04-21T14:39:10.178454",
        "role": "user",
        "message": "whats the difference between django and fastapi"
      },
      {
        "timestamp": "2025-04-21T14:39:17.439090",
        "role": "assistant",
        "message": "Django and FastAPI are both web frameworks for Python, but they have different design philosophies and use cases. Here's a comparison of the two:\n\n1. **Speed and Performance**:\n   - **FastAPI**: As the name suggests, FastAPI is designed for high performance. It is built on top of Starlette and is asynchronous by nature, making it well-suited for handling lots of concurrent requests efficiently. It's particularly beneficial for applications that require high throughput and low latency, such as APIs and microservices.\n   - **Django**: Django is a synchronous framework, which can be less efficient for handling a large number of concurrent requests. However, with Django Channels, you can add asynchronous capabilities if needed. Django is more focused on rapid development and follows a \"batteries-included\" philosophy.\n\n2. **Features and Ecosystem**:\n   - **Django**: It is a full-fledged web framework that comes with a wide range of features out of the box, such as an ORM (Object-Relational Mapping), an admin panel, authentication, forms, and more. This makes it ideal for building traditional monolithic web applications quickly.\n   - **FastAPI**: It is minimalistic and focused on building APIs. It provides automatic generation of OpenAPI documentation and JSON Schema, which is very useful for API development. It doesn\u2019t come with built-in components like Django\u2019s ORM or admin interface, but it integrates well with third-party libraries.\n\n3. **Ease of Use and Learning Curve**:\n   - **Django**: It is known for its \"Django way\" of doing things and has a steeper learning curve for beginners. However, once you are familiar with its conventions, it can significantly speed up development.\n   - **FastAPI**: It is relatively easy to get started with, especially if you are familiar with modern Python features like type hints. The framework leverages Python type hints to provide input validation and serialization automatically.\n\n4. **Use Cases**:\n   - **Django**: Ideal for building complex web applications that require a lot of built-in functionality and rapid prototyping. It's popular for building content management systems, social networks, and other web applications.\n   - **FastAPI**: Best suited for building high-performance APIs and microservices. It is often used in projects that require asynchronous capabilities and high throughput, like real-time data processing.\n\n5. **Community and Support**:\n   - **Django**: Has a large and mature community with extensive documentation and a wide range of third-party packages.\n   - **FastAPI**: A newer framework with a rapidly growing community. It has excellent documentation and is becoming increasingly popular for API development.\n\nIn summary, if you're building a traditional web application with a need for a lot of built-in tools and a structured framework, Django might be the better choice. If you're focused on building fast, efficient RESTful APIs with modern Python practices, FastAPI could be more suitable."
      },
      {
        "timestamp": "2025-04-21T14:39:24.569336",
        "role": "user",
        "message": "exit/\""
      },
      {
        "timestamp": "2025-04-21T14:39:26.610750",
        "role": "assistant",
        "message": "It looks like your query might be incomplete or contain a typo. Could you please provide more context or clarify what you're asking about \"exit/\"? Are you looking for information on how to exit a program or command in a specific context, or is there something else you need help with? Let me know so I can assist you better!"
      }
    ]

## Ollama Deepseek

``` python
# ---------------- Entry Point ----------------
adapter = OllamaAdapter()
strategy = QAStrategy()  # or SummarizeStrategy()
ollama_bot = LLMClient(adapter, strategy)
```

``` python
print("\nChat session started. Type 'exit' to quit.\n")

while True:
    msg = input("You: ")
    if msg.strip().lower() in {"exit", "quit"}:
        break
    response = ollama_bot.chat(msg)
    print(f"Bot: {response}")
```

    🧠 Chat Session Started — Type 'exit' to quit.

    👤 You:  exit

``` python
# View history in Jupyter
ollama_bot.view_history()

# Export as JSON
json_output = ollama_bot.export_history()
print(json_output)
```

    2025-04-21T16:39:53.306768 [user]: what model is this?
    2025-04-21T16:39:57.890272 [assistant]: <think>
    Okay, so the user asked "what model is this?" and I responded with "I'm unable to answer that question." Hmm, they might be referring to a specific product or item. Maybe it's related to cars or electronics? Since I can't see images or descriptions, I can only go by what's provided.

    Wait, maybe the user was asking about a car model but didn't specify any details like make or year. Without more info, it's hard to pinpoint. I should let them know that without additional context, I can't provide an accurate answer.
    </think>

    I'm sorry, but I need more information to answer your question. Could you please provide details about the item in question?
    2025-04-21T16:40:25.576062 [user]: whats the difference between fastapi and django
    2025-04-21T16:40:46.009118 [assistant]: <think>
    Okay, so I need to figure out the differences between FastAPI and Django. Both seem like modern web frameworks for Python, but they're used differently. Let me try to break this down step by step.

    First off, I know that Django is built with a specific set of principles in mind. It's part of the Django ecosystem, which includes Channels, DRF, etc., so it has a lot more built-in features out of the box compared to something like FastAPI. Django is server-side focused—so it handles all the backend stuff for you, including routing, templating, databases, and more.

    On the other hand, FastAPI seems to be more streamlined. It's meant specifically for building APIs without handling as much else on the server side. So maybe if I just need an API, FastAPI is easier to set up because it doesn't require so many other Django features. But does that mean it's less feature-rich? Probably.

    Performance-wise, Django might be better because it has built-in optimizations for things like database operations and request handling since it's handling all the server stuff itself. FastAPI, being lighter, could potentially serve requests faster because there's less overhead from handling more tasks on the server.

    Deployment also seems different. Django probably requires a full setup with all its components—like an ORM, templating engines, etc.—which can take longer to set up and configure properly. FastAPI might have a quicker deployment process since it's focused only on the API part.

    I've heard that Django has strong community support for certain use cases, like full-fledged web apps with rich features, but maybe not as much for just APIs. FastAPI is probably growing more in popularity because of its simplicity and scalability when dealing solely with APIs.

    In terms of security, both handle some aspects, but I think Django's built-in security features might be more comprehensive because it's part of a larger framework that takes care of a lot else on the server side. FastAPI would need to integrate additional security measures if needed, which could add complexity.

    For development and testing, Django probably has better tools since it's a full-stack framework. It might have integrated IDE support and debugging features right out of the box for web applications. FastAPI, being more minimalistic, might require using other frameworks like pytest or separate testing tools for different parts of the API.

    I also remember that Django uses Python 2 by default now, which is an older version, but maybe they're moving to Python 3. FastAPI supports both versions though, so if I need Python 2 functionality, Django still works, whereas with FastAPI, I'd have to use either.

    Thinking about scalability, Django might be a bit more resource-heavy because it's handling all the server-side stuff. FastAPI could scale better in environments where only APIs are needed without the extra overhead of Django's features.

    I'm also considering which one is easier to learn for someone new. Django has a steeper learning curve because of its complexity and the number of features, while FastAPI might be more approachable since it's focused on what it does best—building APIs quickly.

    Putting this all together, if I need a full-featured web application with everything server-side handled by Django, that's the way to go. But for an API-first project where simplicity and ease of setting up are key, FastAPI would be more suitable because it's lighter and easier to deploy.
    </think>

    **Comparative Analysis of FastAPI vs Django**

    1. **Framework Scope:**
       - **Django:** A full-stack web framework offering comprehensive features like routing, templating, databases, ORM, etc., ideal for complex web applications.
       - **FastAPI:** Primarily designed for building APIs with minimal server-side handling, making it highly efficient for API-centric projects.

    2. **Server-Side Features:**
       - **Django:** Handles a wide range of server-side tasks including database operations, templating, and routing, providing robust built-in optimizations.
       - **FastAPI:** Focuses solely on the backend APIs, reducing server-side overhead and improving performance for API requests.

    3. **Deployment Complexity:**
       - **Django:** Requires setting up multiple components (ORMs, templates), which can be time-consuming but ensures a fully functional web application.
       - **FastAPI:** Easier to deploy due to its streamlined approach, ideal for projects focused solely on APIs.

    4. **Performance:**
       - **Django:** May offer better performance due to optimized server-side tasks and built-in optimizations.
       - **FastAPI:** Typically faster as it's lightweight, especially in API-heavy environments with reduced overhead.

    5. **Security Features:**
       - **Django:** Built-in security features integrated into a robust framework, beneficial for full web applications.
       - **FastAPI:** May require additional security measures beyond the framework, adding complexity.

    6. **Development & Testing:**
       - **Django:** Integrated IDE support and comprehensive debugging tools suitable for complex web projects.
       - **FastAPI:** May necessitate separate testing frameworks like pytest for effective development and testing.

    7. **Python Version Support:**
       - **Django:** Currently on Python 2, transitioning to Python 3; may require additional steps for Python 2 features.
       - **FastAPI:** Supports both Python 2 and 3, offering flexibility.

    8. **Scalability:**
       - **Django:** May be less scalable due to server-side resource utilization.
       - **FastAPI:** Better suited for environments where only APIs are needed, potentially scaling more efficiently.

    9. **Learning Curve:**
       - **Django:** Steeper learning curve with its extensive features and complexity.
       - **FastAPI:** More approachable for new developers focusing on API development.

    **Conclusion:**
    - **Choose Django** for comprehensive web applications requiring a full-stack framework with built-in optimizations and security.
    - **Choose FastAPI** for projects centered around APIs, offering simplicity, ease of deployment, and scalability without the overhead of server-side tasks.
    [
      {
        "timestamp": "2025-04-21T16:39:53.306768",
        "role": "user",
        "message": "what model is this?"
      },
      {
        "timestamp": "2025-04-21T16:39:57.890272",
        "role": "assistant",
        "message": "<think>\nOkay, so the user asked \"what model is this?\" and I responded with \"I'm unable to answer that question.\" Hmm, they might be referring to a specific product or item. Maybe it's related to cars or electronics? Since I can't see images or descriptions, I can only go by what's provided.\n\nWait, maybe the user was asking about a car model but didn't specify any details like make or year. Without more info, it's hard to pinpoint. I should let them know that without additional context, I can't provide an accurate answer.\n</think>\n\nI'm sorry, but I need more information to answer your question. Could you please provide details about the item in question?"
      },
      {
        "timestamp": "2025-04-21T16:40:25.576062",
        "role": "user",
        "message": "whats the difference between fastapi and django"
      },
      {
        "timestamp": "2025-04-21T16:40:46.009118",
        "role": "assistant",
        "message": "<think>\nOkay, so I need to figure out the differences between FastAPI and Django. Both seem like modern web frameworks for Python, but they're used differently. Let me try to break this down step by step.\n\nFirst off, I know that Django is built with a specific set of principles in mind. It's part of the Django ecosystem, which includes Channels, DRF, etc., so it has a lot more built-in features out of the box compared to something like FastAPI. Django is server-side focused\u2014so it handles all the backend stuff for you, including routing, templating, databases, and more.\n\nOn the other hand, FastAPI seems to be more streamlined. It's meant specifically for building APIs without handling as much else on the server side. So maybe if I just need an API, FastAPI is easier to set up because it doesn't require so many other Django features. But does that mean it's less feature-rich? Probably.\n\nPerformance-wise, Django might be better because it has built-in optimizations for things like database operations and request handling since it's handling all the server stuff itself. FastAPI, being lighter, could potentially serve requests faster because there's less overhead from handling more tasks on the server.\n\nDeployment also seems different. Django probably requires a full setup with all its components\u2014like an ORM, templating engines, etc.\u2014which can take longer to set up and configure properly. FastAPI might have a quicker deployment process since it's focused only on the API part.\n\nI've heard that Django has strong community support for certain use cases, like full-fledged web apps with rich features, but maybe not as much for just APIs. FastAPI is probably growing more in popularity because of its simplicity and scalability when dealing solely with APIs.\n\nIn terms of security, both handle some aspects, but I think Django's built-in security features might be more comprehensive because it's part of a larger framework that takes care of a lot else on the server side. FastAPI would need to integrate additional security measures if needed, which could add complexity.\n\nFor development and testing, Django probably has better tools since it's a full-stack framework. It might have integrated IDE support and debugging features right out of the box for web applications. FastAPI, being more minimalistic, might require using other frameworks like pytest or separate testing tools for different parts of the API.\n\nI also remember that Django uses Python 2 by default now, which is an older version, but maybe they're moving to Python 3. FastAPI supports both versions though, so if I need Python 2 functionality, Django still works, whereas with FastAPI, I'd have to use either.\n\nThinking about scalability, Django might be a bit more resource-heavy because it's handling all the server-side stuff. FastAPI could scale better in environments where only APIs are needed without the extra overhead of Django's features.\n\nI'm also considering which one is easier to learn for someone new. Django has a steeper learning curve because of its complexity and the number of features, while FastAPI might be more approachable since it's focused on what it does best\u2014building APIs quickly.\n\nPutting this all together, if I need a full-featured web application with everything server-side handled by Django, that's the way to go. But for an API-first project where simplicity and ease of setting up are key, FastAPI would be more suitable because it's lighter and easier to deploy.\n</think>\n\n**Comparative Analysis of FastAPI vs Django**\n\n1. **Framework Scope:**\n   - **Django:** A full-stack web framework offering comprehensive features like routing, templating, databases, ORM, etc., ideal for complex web applications.\n   - **FastAPI:** Primarily designed for building APIs with minimal server-side handling, making it highly efficient for API-centric projects.\n\n2. **Server-Side Features:**\n   - **Django:** Handles a wide range of server-side tasks including database operations, templating, and routing, providing robust built-in optimizations.\n   - **FastAPI:** Focuses solely on the backend APIs, reducing server-side overhead and improving performance for API requests.\n\n3. **Deployment Complexity:**\n   - **Django:** Requires setting up multiple components (ORMs, templates), which can be time-consuming but ensures a fully functional web application.\n   - **FastAPI:** Easier to deploy due to its streamlined approach, ideal for projects focused solely on APIs.\n\n4. **Performance:**\n   - **Django:** May offer better performance due to optimized server-side tasks and built-in optimizations.\n   - **FastAPI:** Typically faster as it's lightweight, especially in API-heavy environments with reduced overhead.\n\n5. **Security Features:**\n   - **Django:** Built-in security features integrated into a robust framework, beneficial for full web applications.\n   - **FastAPI:** May require additional security measures beyond the framework, adding complexity.\n\n6. **Development & Testing:**\n   - **Django:** Integrated IDE support and comprehensive debugging tools suitable for complex web projects.\n   - **FastAPI:** May necessitate separate testing frameworks like pytest for effective development and testing.\n\n7. **Python Version Support:**\n   - **Django:** Currently on Python 2, transitioning to Python 3; may require additional steps for Python 2 features.\n   - **FastAPI:** Supports both Python 2 and 3, offering flexibility.\n\n8. **Scalability:**\n   - **Django:** May be less scalable due to server-side resource utilization.\n   - **FastAPI:** Better suited for environments where only APIs are needed, potentially scaling more efficiently.\n\n9. **Learning Curve:**\n   - **Django:** Steeper learning curve with its extensive features and complexity.\n   - **FastAPI:** More approachable for new developers focusing on API development.\n\n**Conclusion:**\n- **Choose Django** for comprehensive web applications requiring a full-stack framework with built-in optimizations and security.\n- **Choose FastAPI** for projects centered around APIs, offering simplicity, ease of deployment, and scalability without the overhead of server-side tasks."
      }
    ]

Back to top
