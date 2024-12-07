from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
from datetime import datetime
import time

app = FastAPI(title="OpenAI Compatible API")

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: Usage

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

class CompletionStreamResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamResponseChoice]

async def call_llm_api(prompt: Union[str, List[ChatMessage]], **kwargs) -> str:
    """
    Placeholder function to call your actual LLM API.
    Implement this function to connect with your work's LLM API.
    """
    # TODO: Implement your actual API call here
    await asyncio.sleep(0.1)  # Simulate API latency
    return "This is a placeholder response from the LLM API."

async def stream_llm_api(prompt: Union[str, List[ChatMessage]], **kwargs):
    """
    Placeholder generator function for streaming responses from your LLM API.
    Implement this function to connect with your work's streaming LLM API.
    """
    # TODO: Implement your actual streaming API call here
    chunks = ["This ", "is ", "a ", "placeholder ", "streaming ", "response."]
    for chunk in chunks:
        await asyncio.sleep(0.1)  # Simulate API latency
        yield chunk

@app.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse])
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        async def generate_stream():
            stream_id = f"chatcmpl-{int(time.time()*1000)}"
            async for chunk in stream_llm_api(request.messages, **request.model_dump(exclude={'messages', 'stream'})):
                response = ChatCompletionStreamResponse(
                    id=stream_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=DeltaMessage(content=chunk),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {response.model_dump_json()}\n\n"
            
            # Send the final message
            yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return EventSourceResponse(generate_stream())
    
    response_text = await call_llm_api(request.messages, **request.model_dump(exclude={'messages'}))
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # Implement token counting as needed
    )

@app.post("/v1/completions", response_model=Union[CompletionResponse, CompletionStreamResponse])
async def create_completion(request: CompletionRequest):
    if request.stream:
        async def generate_stream():
            stream_id = f"cmpl-{int(time.time()*1000)}"
            async for chunk in stream_llm_api(request.prompt, **request.model_dump(exclude={'prompt', 'stream'})):
                response = CompletionStreamResponse(
                    id=stream_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        CompletionStreamResponseChoice(
                            text=chunk,
                            index=0,
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {response.model_dump_json()}\n\n"
            
            yield f"data: {json.dumps({'id': stream_id, 'object': 'text_completion', 'created': int(time.time()), 'model': request.model, 'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return EventSourceResponse(generate_stream())
    
    response_text = await call_llm_api(request.prompt, **request.model_dump(exclude={'prompt'}))
    
    return CompletionResponse(
        id=f"cmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionResponseChoice(
                text=response_text,
                index=0,
                finish_reason="stop"
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # Implement token counting as needed
    )

@app.get("/v1/models")
async def list_models():
    """
    Return a list of available models.
    Customize this list based on your available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "your-model-1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "your-organization"
            }
            # Add more models as needed
        ]
    }