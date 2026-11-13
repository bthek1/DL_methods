# OpenAI using LangChain

Using the **ChatGPT API with LangChain** is simple and powerful. LangChain provides high-level abstractions for interacting with ChatGPT (and other LLMs), adding things like memory, tools, and prompt templates — making it great for chatbots, agents, and RAG systems.

Author

Benedict Thekkel

## ✅ 1. Install Dependencies

Install LangChain and OpenAI SDK:

``` bash
pip install langchain openai
```

Or with Poetry:

``` bash
poetry add langchain openai
```

------------------------------------------------------------------------

``` sh
pip install -U langchain-openai langchain-core langchain-community
```

## ✅ 2. Setup Environment

Set your OpenAI API key:

``` bash
export OPENAI_API_KEY=sk-...
```

Or load via `.env` and use `load_dotenv()`.

------------------------------------------------------------------------

## ✅ 3. Basic Usage with ChatGPT (gpt-3.5 / gpt-4 / gpt-4o)

``` python
from dotenv import load_dotenv
import os
load_dotenv()
```

    True

``` python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("openai_api_key"),
)

response = chat([
    HumanMessage(content="What's the capital of France?")
])

print(response.content)
```

## ✅ 4. Using Memory (ChatBot Style)

``` python
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Chat model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("openai_api_key")
)

# 2. Store for chat history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. Wrap with memory
chat = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=get_session_history,  # updated arg name
)

# 4. Simulate chat
session_id = "ben-session"

res1 = chat.invoke(
    [HumanMessage(content="Hi, I'm Ben")],
    config={"configurable": {"session_id": session_id}}
)
print(res1.content)

res2 = chat.invoke(
    [HumanMessage(content="What did I just tell you?")],
    config={"configurable": {"session_id": session_id}}
)
print(res2.content)
```

    Hello, Ben! How can I assist you today?
    You just told me your name, Ben. If there's anything else you'd like to share or ask, feel free to let me know!

## ✅ 5. Prompt Templates

``` python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# Define the prompt using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior software developer"),
    ("human", "{question}")
])

# Chat model
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("openai_api_key")
)

# Output parser (converts from message to string)
parser = StrOutputParser()

# Chain composition: Prompt -> LLM -> Parser
chain = prompt | llm | parser

# Run it
res = chain.invoke({"question": "how to use chatGPT api?"})
print(res)
```

    To use the ChatGPT API, you'll need to access the API provided by OpenAI. Here's a general guide on how to use it:

    1. **Set Up an OpenAI Account:**
       - Go to the [OpenAI website](https://www.openai.com/) and sign up or log in if you already have an account.

    2. **Access API Keys:**
       - Once logged in, navigate to the API section to obtain your API key. This key will be used to authenticate your requests.

    3. **Install Required Libraries:**
       - Make sure you have the necessary libraries to make HTTP requests. If you're using Python, the `requests` library is commonly used.

       ```bash
       pip install requests
       ```

    4. **Make API Requests:**
       - Use the API key to make requests to the ChatGPT API endpoint. Here’s a basic example using Python:

       ```python
       import requests

       # Define your API key and the endpoint URL
       api_key = 'your-api-key-here'
       url = 'https://api.openai.com/v1/chat/completions'

       # Set up the headers for the request
       headers = {
           'Authorization': f'Bearer {api_key}',
           'Content-Type': 'application/json'
       }

       # Define the parameters for the request
       data = {
           "model": "gpt-3.5-turbo",
           "messages": [
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": "Tell me a joke."}
           ],
           "max_tokens": 150,
           "temperature": 0.5
       }

       # Make the POST request to the API
       response = requests.post(url, headers=headers, json=data)

       # Parse the response
       if response.status_code == 200:
           response_data = response.json()
           print(response_data['choices'][0]['message']['content'])
       else:
           print(f"Request failed with status code {response.status_code}: {response.text}")
       ```

    5. **Handle the Response:**
       - If the request is successful, the response will include the generated text along with additional information like the token usage.

    6. **Configure Parameters:**
       - Adjust parameters such as `temperature`, `max_tokens`, and the conversation context in the `messages` array to achieve the desired behavior or output.

    7. **Security Considerations:**
       - Keep your API key secure. Do not hardcode it in publicly accessible places. Consider using environment variables or secure vault services.

    8. **Read the Documentation:**
       - OpenAI provides detailed [API documentation](https://platform.openai.com/docs/api-reference/introduction) that covers more advanced features and concepts you'll want to explore as you expand your implementation.

    With these steps, you should be able to start using the ChatGPT API to interact with the language model programmatically.

## ✅ 6. Streaming Output from ChatGPT

``` python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks.base import BaseCallbackHandler
import os

# Define a proper callback handler
class PrintChunkHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="")

# Set up the streaming model with callback
chat = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[PrintChunkHandler()],
    api_key=os.getenv("openai_api_key")
)

# Invoke the chat model with a message
res = chat.invoke([
    HumanMessage(content="Explain quantum computing simply.")
])
```

    Quantum computing is a new type of computing that uses the principles of quantum mechanics, which is the science of very small things like atoms and particles, to process information. Traditional computers use bits, which can be either a 0 or a 1, to perform tasks and calculations. Quantum computers, on the other hand, use quantum bits or "qubits."

    The unique features of qubits are:

    1. **Superposition**: Unlike bits, qubits can be both 0 and 1 at the same time. This allows quantum computers to process a vast amount of possibilities simultaneously.

    its can be linked together in such a way that the state of one qubit can depend on the state of another, no matter how far apart they are. This can lead to much faster processing speeds for certain tasks.

     the wrong ones.e**: Quantum algorithms can use interference to amplify the right answers and cancel out

    , they hold promise for complex problems in cryptography, optimization, and simulations of molecular structures. However, they are still in the experimental stage and are not yet practical for most applications.

## ✅ 7. Tool-Using Agent (with ChatGPT)

``` python
import os
from dotenv import load_dotenv
import numexpr
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool

load_dotenv()

# Define a safe calculator tool
def safe_calculator(expression: str) -> str:
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Useful for math operations like addition, subtraction, multiplication, and division."
    )
]

# Initialize the LLM
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("openai_api_key")
)

# Create the LangGraph ReAct agent executor
agent_executor = create_react_agent(
    model=chat_model,
    tools=tools,
)

# Create a message input for the agent
messages = [
    {"role": "user", "content": "What is 7 * (4 + 3)?"}
]

# Run the agent with a prompt
response = agent_executor.invoke({"messages": messages})

# Extract the final AI message from the returned state
ai_messages = response.get("messages", [])

if ai_messages:
    final_message = ai_messages[-1].content  # use .content for LangChain message objects
    print("\nFinal Answer:", final_message)
else:
    print("No response from agent.")
```

    Final Answer: The result of \(7 \times (4 + 3)\) is 49.

## ✅ 8. RAG with ChatGPT (Retrieval-Augmented Generation)

``` python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
import os

# Sample documents
docs = [
    Document(page_content="Climate change refers to long-term shifts in temperatures and weather patterns."),
    Document(page_content="Greenhouse gases are a major contributor to global warming."),
]

# Create and save FAISS index
embeddings = OpenAIEmbeddings(api_key=os.getenv("openai_api_key"))
db = FAISS.from_documents(docs, embeddings)
db.save_local("my_index")  # This creates my_index/index.faiss and index.pkl
```

``` python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

# 1. Load vector store
embeddings = OpenAIEmbeddings(api_key=os.getenv("openai_api_key"))
vectorstore = FAISS.load_local("my_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# 2. Correct prompt with {context}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who answers questions based on the context."),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

# 3. LLM setup
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("openai_api_key"),
    temperature=0,
)

# 4. Chain setup
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

# 5. Query
response = rag_chain.invoke({"input": "What did the document say about climate change?"})
print(response["answer"])
```

    The document mentioned that climate change refers to long-term shifts in temperatures and weather patterns.

## ✅ Summary: Updated LangChain Patterns (2024+)

| Feature | Modern Usage with LangChain |
|----|----|
| **Basic Chat** | `ChatOpenAI(model="gpt-4o")` with `invoke()` |
| **Streaming** | `ChatOpenAI(streaming=True)` + `callbacks` |
| **Memory** | `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` |
| **Prompt Template** | `ChatPromptTemplate` + `LLMChain` |
| **Tools / Agents** | `create_react_agent()` + `Tool[]` (LangGraph) |
| **RAG** | `create_retrieval_chain()` + `FAISS` / `Chroma` |

------------------------------------------------------------------------

Let me know if you’d like this in Markdown table format, visual diagram, or as a quick reference card!

------------------------------------------------------------------------

Back to top
