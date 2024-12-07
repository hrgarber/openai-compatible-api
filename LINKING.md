# Linking to Non-OpenAI Compatible API

This guide explains how to connect your non-OpenAI compatible LLM API to this OpenAI-compatible wrapper.

## Architecture Overview

```
┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│              │    │  OpenAI-          │    │  Your              │    │  Your           │
│    Client    ├───►│  Compatible       ├───►│  Translation       ├───►│  Non-OpenAI     │
│  Application │    │  API Wrapper      │    │  Layer             │    │  LLM API        │
│              │◄───┤  (This Service)   │◄───┤                    │◄───┤                 │
└──────────────┘    └───────────────────┘    └────────────────────┘    └─────────────────┘
```

## Flow Description

1. **Client Request** → **OpenAI-Compatible Wrapper**
   - Client sends standard OpenAI-formatted request
   - Example: `/v1/chat/completions` with messages array

2. **Wrapper** → **Your Translation Layer**
   - Wrapper forwards request to your translation function
   - Translation function converts OpenAI format to your API format

3. **Translation Layer** → **Your LLM API**
   - Translated request is sent to your LLM API
   - Response is received from your LLM API

4. **Translation Layer** → **Wrapper** → **Client**
   - Response is converted back to OpenAI format
   - Wrapper sends OpenAI-compatible response to client

## Implementation Steps

1. **Configure Your API Connection**

Create a `.env` file in the project root:
```env
YOUR_API_BASE_URL=http://your-llm-api.example.com
YOUR_API_KEY=your_api_key_here
```

2. **Implement Translation Layer**

Create a new file `app/translation.py`:
```python
from typing import List, Dict, Any, AsyncGenerator
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("YOUR_API_BASE_URL")
API_KEY = os.getenv("YOUR_API_KEY")

async def translate_to_your_format(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """
    Translate OpenAI format to your API format.
    
    Example:
    OpenAI format:
    {
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7
    }
    
    Your API format:
    {
        "prompt": "Hello!",
        "params": {
            "temp": 0.7
        }
    }
    """
    # Extract the last message content as prompt
    prompt = messages[-1]["content"]
    
    # Map OpenAI parameters to your API parameters
    params = {
        "temp": kwargs.get("temperature", 1.0),
        # Add other parameter mappings
    }
    
    return {
        "prompt": prompt,
        "params": params
    }

async def translate_from_your_format(response: Dict[str, Any]) -> str:
    """
    Translate your API response format to OpenAI format.
    
    Example:
    Your API format:
    {
        "generated_text": "Hi there!",
        "metadata": {...}
    }
    
    Returns: "Hi there!"
    """
    return response.get("generated_text", "")

async def call_your_api(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make the actual API call to your LLM service."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/generate",  # Adjust endpoint
            json=data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        return response.json()

# For streaming responses
async def translate_stream_to_your_format(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Similar to translate_to_your_format but for streaming requests."""
    return await translate_to_your_format(messages, **kwargs)

async def stream_from_your_api(data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Stream responses from your API.
    Adjust the endpoint and response handling based on your API's streaming format.
    """
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{API_BASE_URL}/generate/stream",  # Adjust endpoint
            json=data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        ) as response:
            async for chunk in response.aiter_lines():
                if chunk:
                    # Parse your API's streaming format and yield text chunks
                    yield chunk  # Adjust based on your API's format
```

3. **Update Main Application**

Modify the placeholder functions in `app/main.py`:

```python
from .translation import (
    translate_to_your_format,
    translate_from_your_format,
    call_your_api,
    translate_stream_to_your_format,
    stream_from_your_api
)

async def call_llm_api(prompt: Union[str, List[ChatMessage]], **kwargs) -> str:
    """Non-streaming API call implementation."""
    if isinstance(prompt, list):  # Chat completion
        data = await translate_to_your_format(
            [m.model_dump() for m in prompt],
            **kwargs
        )
    else:  # Text completion
        data = await translate_to_your_format(
            [{"role": "user", "content": prompt}],
            **kwargs
        )
    
    response = await call_your_api(data)
    return await translate_from_your_format(response)

async def stream_llm_api(prompt: Union[str, List[ChatMessage]], **kwargs):
    """Streaming API call implementation."""
    if isinstance(prompt, list):  # Chat completion
        data = await translate_stream_to_your_format(
            [m.model_dump() for m in prompt],
            **kwargs
        )
    else:  # Text completion
        data = await translate_stream_to_your_format(
            [{"role": "user", "content": prompt}],
            **kwargs
        )
    
    async for chunk in stream_from_your_api(data):
        yield chunk
```

## Example API Formats

### Your Non-OpenAI API Format (Example)

```python
# Request format
{
    "prompt": "What is the capital of France?",
    "params": {
        "temp": 0.7,
        "max_length": 100
    }
}

# Response format
{
    "generated_text": "The capital of France is Paris.",
    "metadata": {
        "tokens_generated": 8,
        "generation_time": 0.5
    }
}

# Streaming format (example)
{
    "chunk": "The ",
    "finished": false
}
{
    "chunk": "capital ",
    "finished": false
}
{
    "chunk": "of France is Paris.",
    "finished": true
}
```

### OpenAI-Compatible Format (What Clients See)

```python
# Request format
{
    "model": "your-model",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

# Response format
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "your-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}
```

## Testing Your Integration

1. Start the OpenAI-compatible wrapper:
```bash
./run.sh
```

2. Test with curl:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

3. Test with Python OpenAI client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Common Issues and Solutions

1. **Token Counting**
   - If your API doesn't provide token counts, you can use `tiktoken` to estimate:
   ```python
   import tiktoken
   
   def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
       encoding = tiktoken.encoding_for_model(model)
       return len(encoding.encode(text))
   ```

2. **Error Handling**
   - Map your API's error responses to OpenAI-compatible error formats
   - Add try-catch blocks in translation functions

3. **Streaming Response Formatting**
   - Ensure each chunk is properly formatted as an SSE event
   - Include all required fields in streaming response objects