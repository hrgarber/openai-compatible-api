# OpenAI-Compatible API Layer

This project provides an OpenAI-compatible API layer that can wrap custom LLM APIs. It implements the same interface as OpenAI's API, making it easy to use existing OpenAI client libraries with your custom LLM implementation.

## Features

- Full OpenAI API compatibility for text-based endpoints
- Support for both streaming and non-streaming responses
- Chat completions endpoint (`/v1/chat/completions`)
- Text completions endpoint (`/v1/completions`)
- Models endpoint (`/v1/models`)
- Built with FastAPI for high performance and automatic API documentation

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
./run.sh
```

The server will start on `http://localhost:8000` with automatic reload enabled.

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation at `/docs`
- ReDoc documentation at `/redoc`

## Integration

To use this API with OpenAI client libraries, simply point them to your server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # API key can be any string as we're not using it
)

# Chat completion example
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Streaming example
for chunk in client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Implementation Notes

To integrate your custom LLM API:

1. Implement the `call_llm_api` function in `app/main.py` for non-streaming responses
2. Implement the `stream_llm_api` function for streaming responses
3. Add proper token counting in the Usage model
4. Update the models list in the `/v1/models` endpoint
5. Add any additional error handling specific to your API