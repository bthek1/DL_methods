# Ollama

Ollama is a lightweight, extensible framework designed for building and running large language models (LLMs) on local machines. It provides a command-line interface (CLI) that facilitates model management, customization, and interaction. Here’s a comprehensive guide to using Ollama, including essential commands and examples.

Author

Benedict Thekkel

## **1. Installation**

- **For Linux:**

  Open your terminal and execute:

  ``` bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

  This command downloads and installs Ollama on your system.

------------------------------------------------------------------------

## **2. System Requirements**

- **Operating System:** macOS or Linux
- **Memory (RAM):** Minimum 8GB; 16GB or more recommended
- **Storage:** At least 10GB of free space
- **Processor:** Modern CPU from the last 5 years

------------------------------------------------------------------------

## **3. Basic CLI Commands**

- **Start the Ollama Server:**

  To run Ollama without the desktop application:

  ``` bash
  ollama serve
  ```

- **Download a Model:**

  To download a specific model:

  ``` bash
  ollama pull <model-name>
  ```

  Replace `<model-name>` with the desired model’s name, e.g., `llama3.2`.

- **List Downloaded Models:**

  To view all models available on your system:

  ``` bash
  ollama list
  ```

- **Run a Model:**

  To start a model and enter an interactive session:

  ``` bash
  ollama run <model-name>
  ```

  For example:

  ``` bash
  ollama run llama3.2
  ```

- **Stop a Running Model:**

  To stop a specific model:

  ``` bash
  ollama stop <model-name>
  ```

- **Remove a Model:**

  To delete a model from your system:

  ``` bash
  ollama rm <model-name>
  ```

- **Display Model Information:**

  To view details about a specific model:

  ``` bash
  ollama show <model-name>
  ```

- **List Running Models:**

  To see which models are currently active:

  ``` bash
  ollama ps
  ```

- **Access Help:**

  For a list of available commands and their descriptions:

  ``` bash
  ollama help
  ```

------------------------------------------------------------------------

## **4. Model Customization**

Ollama allows users to customize models using a `Modelfile`. This file specifies the base model and any modifications, such as system prompts or parameters.

- **Create a `Modelfile`:**

  Create a file named `Modelfile` with the following content:

      FROM llama3.2

      SYSTEM "You are an AI assistant specializing in environmental science. Answer all questions with a focus on sustainability."

      PARAMETER temperature 0.7

- **Build the Custom Model:**

  Use the `ollama create` command to build the model:

  ``` bash
  ollama create my_custom_model -f ./Modelfile
  ```

- **Run the Custom Model:**

  Start the customized model:

  ``` bash
  ollama run my_custom_model
  ```

------------------------------------------------------------------------

## **5. Using Ollama with Files**

- **Summarize Text from a File:**

  To summarize the content of `input.txt`:

  ``` bash
  ollama run llama3.2 "Summarize the content of this file." < input.txt
  ```

- **Save Model Responses to a File:**

  To save the model’s response to `output.txt`:

  ``` bash
  ollama run llama3.2 "Explain the concept of quantum computing." > output.txt
  ```

------------------------------------------------------------------------

## **6. Common Use Cases**

- **Text Generation:**

  - *Content Creation:*

    Generate an article on a specific topic:

    ``` bash
    ollama run llama3.2 "Write a short article on the benefits of renewable energy." > article.txt
    ```

  - *Question Answering:*

    Answer specific queries:

    ``` bash
    ollama run llama3.2 "What are the latest advancements in artificial intelligence?"
    ```

- **Data Analysis:**

  - *Sentiment Analysis:*

    Analyze the sentiment of a given text:

    ``` bash
    ollama run llama3.2 "Determine the sentiment of this review: 'The product exceeded my expectations.'"
    ```

------------------------------------------------------------------------

## **7. Python Integration**

### Ollama Python Library

The Ollama Python library provides the easiest way to integrate Python 3.8+ projects with [Ollama](https://github.com/ollama/ollama).

### Prerequisites

- [Ollama](https://ollama.com/download) should be installed and running
- Pull a model to use with the library: `ollama pull <model>` e.g. `ollama pull llama3.2`
  - See [Ollama.com](https://ollama.com/search) for more information on the models available.

### Install

``` sh
pip install ollama
```

### Usage

``` python
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
```

See [\_types.py](ollama/_types.py) for more information on the response types.

### Streaming responses

Response streaming can be enabled by setting `stream=True`.

``` python
from ollama import chat

stream = chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

### Custom client

A custom client can be created by instantiating `Client` or `AsyncClient` from `ollama`.

All extra keyword arguments are passed into the [`httpx.Client`](https://www.python-httpx.org/api/#client).

``` python
from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
```

### Async client

The `AsyncClient` class is used to make asynchronous requests. It can be configured with the same fields as the `Client` class.

``` python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='llama3.2', messages=[message])

asyncio.run(chat())
```

Setting `stream=True` modifies functions to return a Python asynchronous generator:

``` python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='llama3.2', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
```

### API

The Ollama Python library’s API is designed around the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)

#### Chat

``` python
ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

#### Generate

``` python
ollama.generate(model='llama3.2', prompt='Why is the sky blue?')
```

#### List

``` python
ollama.list()
```

#### Show

``` python
ollama.show('llama3.2')
```

#### Create

``` python
ollama.create(model='example', from_='llama3.2', system="You are Mario from Super Mario Bros.")
```

#### Copy

``` python
ollama.copy('llama3.2', 'user/llama3.2')
```

#### Delete

``` python
ollama.delete('llama3.2')
```

#### Pull

``` python
ollama.pull('llama3.2')
```

#### Push

``` python
ollama.push('user/llama3.2')
```

#### Embed

``` python
ollama.embed(model='llama3.2', input='The sky is blue because of rayleigh scattering')
```

#### Embed (batch)

``` python
ollama.embed(model='llama3.2', input=['The sky is blue because of rayleigh scattering', 'Grass is green because of chlorophyll'])
```

#### Ps

``` python
ollama.ps()
```

### Errors

Errors are raised if requests return an error status or if an error is detected while streaming.

``` python
model = 'does-not-yet-exist'

try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('Error:', e.error)
  if e.status_code == 404:
    ollama.pull(model)
```

``` python
import ollama
from IPython.display import Markdown, display

response = ollama.generate(model='deepseek-r1:7b', prompt='How to create a django project')

display(Markdown(response['response']))
```

Okay, so I need to figure out how to create a Django project. I’ve heard about Django before; it’s a Python framework for building web apps. But I’m not entirely sure where to start. Let me think through this step by step.

First, I know that Django requires some setup files and environment configuration. So maybe the first thing is to install Python on my machine if I don’t already have it. I remember that Python 3 is recommended for Django because it has better support from Django itself. So I’ll check if I have Python installed or not.

Next, after having Python installed, I need to make sure I’m using the correct version. I’ve heard about virtual environments being important for managing project dependencies. So maybe I should create a virtual environment with an activated one. That way, all my Django projects can use the same Python version without conflicts.

Once that’s set up, I need to install Django itself. From what I remember, you can install it using pip. So I’ll run ‘pip install django’ in my terminal or command prompt. After that, I should activate my virtual environment again because that’s where the installed packages will be located.

Now comes creating a new project. I think Django uses an app directory structure under settings.py. So to create a new project, I probably go into my project directory and then run some command related to Django projects. Maybe something like ‘django-admin startproject’ or similar? Oh wait, I think it’s just ‘python manage.py startproject’ because sometimes the admin interface isn’t available until after migrations are applied.

After starting the project, I should set up a settings file. The settings file is crucial because it configures various aspects of Django like middleware, URLs, templates, etc. So I need to create a new settings.py file in my project directory with some default configurations or maybe import an existing one if there’s a template.

I’m not entirely sure about all the parts that should be included in settings.py, but from what I remember, it includes things like ROOT_URLCONF, MiddlewareStack, Templates, TemplateSyntax, LoginView, etc. Maybe I can look up a basic Django settings file as a starting point to get an idea of how to structure mine.

Also, after creating the project and setting up the settings file, I probably need to create some initial apps or models. For example, maybe I should add users, orders, products, or other relevant apps depending on what kind of application I’m building. Each app would contain models, which define the database tables for my web application.

Wait, how do I create an app? I think it involves creating a new app directory in the apps folder and adding a main.py file that imports the models. Then, I can run migrations to adjust the database schema based on the models I’ve created. Oh right, after creating a model, I have to make sure to apply migrations so Django knows how to structure the database.

I also remember something about URLs. When you start a project with ‘startproject’, the URL configuration is already set up, but sometimes it’s not what you want. So maybe I need to go into urls.py and adjust the paths or include some custom apps there.

Another thing: permissions. Django handles user permissions, so I think after creating users app, I can assign superuser permissions to a specific user using ‘python manage.py migrate’ followed by adding the user as admin with ‘superuser’.

I’m a bit confused about what’s included in the default project. Maybe it includes a welcome app and other base apps that set up some initial functionality like logging in users or handling requests.

Oh, I also need to think about how to run migrations once the models are created. The Django console is helpful for managing projects, apps, models, and running migrations manually or automatically with ‘makemigrations’ commands.

Wait, sometimes when you start a project, the URLs might not point correctly because of an incorrect path in settings.py. So maybe I need to adjust that after creating it. Or perhaps I should use the ‘runserver’ command at some point to test if everything is working as expected.

I’m also wondering about templates and views. Once the models are set up, how do I create views for handling HTTP requests? Probably by creating view functions in urls.py or using DRF (Django REST Framework) if it’s a REST API. But maybe that’s beyond my current setup; perhaps I’ll focus on standard views first.

Testing is another aspect. I know Django has built-in testing tools, so after setting up everything correctly, I can run tests to make sure all parts of the application are working as intended.

Wait, but when creating a new project with ‘startproject’, it actually creates an app called ‘base’ by default. So maybe that’s where some initial configurations are placed. Do I need to modify anything in baseapp.py or other files?

I think I’m getting closer. Maybe the steps are: install Python 3, create a virtual environment and activate it, install Django, create a new project with ‘startproject’, set up settings, create apps/models, apply migrations, run tests, etc.

But what if something goes wrong? Like, maybe when I try to run migrations manually after creating models, it doesn’t work. Or perhaps the URLs aren’t set correctly and cause 404 errors. So debugging is important too.

I also heard about dev tools in Django for local development, like debug: true in settings.py or using the ‘dev’ server with host=‘localhost’ and port=8000. Maybe I should enable that to test my application without leaving the server running all the time.

In summary, my plan is:

1.  Install Python 3 if not already present.
2.  Create a virtual environment and activate it.
3.  Install Django using pip.
4.  Start a new Django project with ‘startproject’.
5.  Configure settings.py with necessary configurations.
6.  Create apps (e.g., users, products) with models.
7.  Run migrations to apply database changes.
8.  Set up URLs and tests as needed.
9.  Debug if issues arise.

I think I’ve covered the main steps. Now, I’ll try to write this out in a clear, step-by-step manner without being too vague or technical beyond my current understanding.

Creating a Django project involves several structured steps, each ensuring that your web application is set up correctly and efficiently. Here’s a concise guide based on the thought process:

### Step-by-Step Guide to Creating a Django Project

1.  **Install Python 3**
    - Ensure Python 3 is installed as Django is compatible with it.
2.  **Set Up Virtual Environment**
    - Create a virtual environment to manage project dependencies:

      ``` bash
      python -m venv myproject
      source myproject/bin/activate  # On Windows, use myproject\myproject.bat
      ```

    - Activate the virtual environment for all subsequent commands.
3.  **Install Django**
    - Install Django using pip within the activated environment:

      ``` bash
      pip install django
      ```
4.  **Start a New Django Project**
    - Create a new project by running:

      ``` bash
      python manage.py startproject myproject
      ```

    - This generates a `myproject` directory with initial files.
5.  **Configure Django Settings**
    - Create or modify `settings.py` in the `myproject/settings/` directory to set your environment variables and application URLs.

      ``` python
      from pathlib import Path

      BASE_DIR = str(Path(__file__).resolve().parent.parent)
      SECURITY middlewares are configured here.

      ROOT_URLCONF = 'myproject.urls'
      ```

    - Ensure necessary Django features like logging, middleware, and templates are enabled.
6.  **Create Applications/Models**
    - Add apps (e.g., `users`, `products`) by creating directories and initial Python files.

    - Create models for your database tables within each app directory:

      ``` python
      from django.db import models

      class User(models.Model):
          username = models.CharField(max_length=100)
          email = models.EmailField()
          ...
      ```
7.  **Run Migrations**
    - Apply database migrations after creating models:

      ``` bash
      python manage.py migrate
      ```

    - This adjusts the database schema based on your models.
8.  **Set Up URLs Configuration**
    - Configure URLs by editing `urls.py` or using Django’s admin interface.
    - Ensure paths are correctly set up, possibly adjusting after settings configuration adjustments.
9.  **Enable Developer Tools (Optional)**
    - Add `dev` middleware and enable debug mode in your settings for easier testing:

      ``` python
      MIDDLEWARE = (
          ...
          'django.middleware.clickjacking.D ClickJackingMiddleware',
          ...
      )
      ```

      or set `DEBUG: True` in `settings.py`.
10. **Run the Django Server**
    - Start the development server to test your application within a local web browser:

      ``` bash
      python manage.py runserver
      ```
11. **Test and Debug**
    - Use Django’s built-in tests or external tools like Postman, Firefox, or Chrome to access your application.
    - Address any issues arising from misconfigured URLs, missing models, or migration errors.

By following these steps, you’ll have a functional Django project ready for development. Each step ensures that your application is properly configured and set up for a smooth user experience.

Back to top
