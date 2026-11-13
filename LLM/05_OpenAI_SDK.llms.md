# OpenAI SDK

Here’s **everything you need to know** about the official **OpenAI API Python SDK** (`openai`) — a comprehensive guide covering setup, authentication, endpoints, examples, best practices, and advanced features including streaming, function calling, tool use, and billing queries.

Author

Benedict Thekkel

## Example

``` python
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from IPython.display import Markdown, display


load_dotenv()

client = OpenAI(
  api_key=os.getenv('openai_api_key')
)
```

``` python
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "what language model is this?"}
  ],
  max_tokens=150,
)

content = completion.choices[0].message.content
display(Markdown(content))
```

I am based on OpenAI’s GPT-3 model, which is a language model designed for natural language understanding and generation. If you have any questions or need assistance, feel free to ask!

## 📦 1. Installation

``` bash
pip install openai
```

> ✅ This installs the [official SDK](https://github.com/openai/openai-python)

------------------------------------------------------------------------

## 🔐 2. Authentication

### Options:

``` python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Other methods: - Environment variable: `OPENAI_API_KEY` - `.openai` config file - Azure or custom base URL: `base_url="https://..."`

------------------------------------------------------------------------

## ⚙️ 3. SDK Client Overview

``` python
from openai import OpenAI
client = OpenAI()
```

Client supports: - `client.chat.completions.create(...)` - `client.images.generate(...)` - `client.embeddings.create(...)` - `client.audio.transcriptions.create(...)` - `client.files.create(...)` - `client.fine_tuning.jobs.create(...)` - and more…

------------------------------------------------------------------------

## 💬 4. Chat Completions

### Basic usage:

``` python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Options:

- `max_tokens`, `temperature`, `top_p`, `presence_penalty`
- `tools`, `tool_choice`
- `function_call` (deprecated in favor of `tools`)

------------------------------------------------------------------------

## 🔁 5. Streaming Responses

``` python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Stream this response"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

> ✅ Useful for real-time apps, CLI tools, or dashboards.

------------------------------------------------------------------------

## 🧠 6. Tool Use / Function Calling

``` python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    tool_choice="auto"
)
```

Check for `tool_calls` in response and call your function accordingly.

------------------------------------------------------------------------

## 🖼️ 7. Image Generation (`DALL·E`)

``` python
image = client.images.generate(
    prompt="A futuristic city skyline at dusk",
    model="dall-e-3",
    n=1,
    size="1024x1024"
)
print(image.data[0].url)
```

------------------------------------------------------------------------

## 🧬 8. Embeddings

``` python
response = client.embeddings.create(
    input="Deep learning is powerful.",
    model="text-embedding-3-small"
)
vector = response.data[0].embedding
```

Use for semantic search, vector databases, etc.

------------------------------------------------------------------------

## 🔊 9. Audio Transcription (Whisper)

``` python
with open("audio.mp3", "rb") as file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=file
    )
print(transcript.text)
```

------------------------------------------------------------------------

## 📁 10. File Upload + Fine-tuning

### Upload a file:

``` python
client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")
```

### Fine-tuning:

``` python
client.fine_tuning.jobs.create(training_file="file-abc123")
```

------------------------------------------------------------------------

## 📊 11. Billing / Usage (manual requests)

The SDK does **not** expose billing endpoints, but you can query them:

``` python
import requests

headers = {"Authorization": f"Bearer {api_key}"}
r = requests.get("https://api.openai.com/v1/dashboard/billing/usage?start_date=2024-04-01&end_date=2024-04-30", headers=headers)
print(r.json())
```

------------------------------------------------------------------------

## 📎 12. Error Handling

``` python
from openai import OpenAI, APIError, RateLimitError

try:
    ...
except RateLimitError:
    print("Rate limited.")
except APIError as e:
    print(f"OpenAI API error: {e}")
```

------------------------------------------------------------------------

## ✅ 13. Models Available (as of 2024-2025)

| Model              | Type       | Use Case                   |
|--------------------|------------|----------------------------|
| `gpt-4o`           | Chat       | Fast, cheap, vision, voice |
| `gpt-4`            | Chat       | Highest reasoning          |
| `gpt-3.5-turbo`    | Chat       | Cost-effective             |
| `dall-e-3`         | Image      | Prompt-to-image            |
| `whisper-1`        | Audio      | Speech to text             |
| `text-embedding-3` | Embeddings | Vector search              |

------------------------------------------------------------------------

## 🧰 14. Advanced Tips

- `response.usage.total_tokens` to track billing
- Use `.store=True` to save conversations if supported
- Use `logprobs` for token-level info (GPT-3 only)

------------------------------------------------------------------------

## 🧪 15. Testing in Notebooks

``` python
from IPython.display import Markdown
res = client.chat.completions.create(...)
display(Markdown(res.choices[0].message.content))
```

------------------------------------------------------------------------

## 🔚 Summary

| Task             | SDK Support                         |
|------------------|-------------------------------------|
| Chat Completion  | ✅ `.chat.completions.create()`     |
| Image Generation | ✅ `.images.generate()`             |
| Audio Transcribe | ✅ `.audio.transcriptions.create()` |
| Embeddings       | ✅ `.embeddings.create()`           |
| File Upload      | ✅ `.files.create()`                |
| Billing Info     | ❌ Use `requests` manually          |

------------------------------------------------------------------------

Back to top
