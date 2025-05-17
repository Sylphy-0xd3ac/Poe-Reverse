import time
import uuid
import os
from typing import List, Optional, Dict, Any
import fastapi
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fastapi_poe as fp

app = FastAPI()

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POE_API_KEY = None

# 模型名称映射到Poe机器人
MODEL_MAPPING = {
    # OpenAI GPT series
    "gpt-4o": "GPT-4o",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gpt-4.1-nano": "GPT-4.1-nano",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4.5-preview": "GPT-4.5-Preview",
    "chatgpt-4o-latest": "ChatGPT-4o-Latest",
    "gpt-4o-search": "GPT-4o-Search",
    "gpt-4o-mini-search": "GPT-4o-mini-Search",
    # Anthropic Claude series
    "claude-3-7-sonnet": "Claude-3.7-Sonnet",
    "claude-3-7-sonnet-thinkng": "Claude-3.7-Sonnet-Reasoning",
    "claude-3-5-sonnet": "Claude-3.5-Sonnet",
    "claude-3-5-haiku-latest": "Claude-3.5-Haiku",
    "claude-3-opus": "Claude-3-Opus",
    # Google Gemini series
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini-2.5-pro-exp": "Gemini-2.5-Pro-Exp",
    "gemini-2.5-flash-preview": "Gemini-2.5-Flash-Preview",
    "gemini-2.0-flash-preview": "Gemini-2.0-Flash-Preview",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini-1.5-pro": "Gemini-1.5-Pro",
    # Llama series (Meta)
    "llama-4-scout-b10": "Llama-4-Scout-B10",
    "llama-4-scout-t": "Llama-4-Scout-T",
    "llama-4-scout-nitro": "Llama-4-Scout-nitro",
    "llama-4-maverick": "Llama-4-Maverick",
    "llama-4-maverick-t": "Llama-4-Maverick-T",
    "llama-3-70b-groq": "Llama-3-70b-Groq",
    "llama-3.3-70b": "Llama-3.3-70B",
    "llama-3.3-70b-nitro": "Llama-3.3-70B-nitro",
    "llama-3.3-70b-fp16": "Llama-3.3-70B-FP16",
    "llama-3.1-8b-nitro": "Llama-3.1-8B-nitro",
    "llama-4-scout": "Llama-4-Scout",
    "llama-3.3-70b-fw": "Llama-3.3-70B-FW",
    # Grok series (xAI)
    "grok-3": "Grok-3",
    "grok-3-mini": "Grok-3-Mini",
    # DeepSeek series
    "deepseek-v3": "DeepSeek-V3",
    "deepseek-v3-fw": "Deepseek-V3-FW",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-r1-fw": "DeepSeek-R1-FW",
    "deepseek-r1-distill": "DeepSeek-R1-Distill",
    "deepseek-r1-70b-nitro": "Deepseek-R1-70B-nitro",
    # Qwen series
    "qwen3-235b-a22b-fw": "Qwen3-235B-A22B-FW",
    "qwen-3-235b-t": "Qwen-3-235B-T",
    "qwen3-235b-a22b": "Qwen3-235B-A22B",
    "qwen3-235b-a22b-di": "Qwen3-235B-A22B-DI",
    "qwen-qwq-32b": "Qwen-QwQ-32b",
    "qwq-32b-t": "QwQ-32B-T",
    # Other OpenAI or related models
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o3-mini-high": "o3-mini-high",
    "o4-mini": "o4-mini",
    "o1": "o1",
    "o1-pro": "o1-pro",
    "o1-mini": "o1-mini",
    # GPT-Researcher series
    "gpt-researcher": "GPT-Researcher",
    # Perplexity Sonar series
    "perplexity-sonar": "Perplexity-Sonar",
    "perplexity-sonar-pro": "Perplexity-Sonar-Pro",
    "perplexity-sonar-rsn-pro": "Perplexity-Sonar-Rsn-Pro",
    "perplexity-sonar-reasoning": "Perplexity-Sonar-Reasoning",
    "perplexity-deep-research": "Perplexity-Deep-Research",
    # Other miscellaneous or less common models
    "web-search": "Web-Search",
    "mistral-medium": "Mistral-Medium",
    "mistral-small-3.1": "Mistral-Small-3.1",
    "App-Creator": "App-Creator",
    "Assistant": "Assistant",
}


# 支持的模型列表，用于 /v1/models 端点
MODELS_LIST = [
    {
        "id": model,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "poe",
    }
    for model, _ in MODEL_MAPPING.items()
]

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

@app.get("/")
async def root():
    return {"message": "OpenAI兼容API，基于Poe平台"}

@app.get("/v1/models")
async def get_models():
    return {
            "object": "list",
            "data": MODELS_LIST
        }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # 获取对应的Poe机器人名称
        bot_name = MODEL_MAPPING.get(request.model)
        if not bot_name:
            bot_name = MODEL_MAPPING.get("gpt-3.5-turbo")  # 默认
        
        # 获取APIKEY
        POE_API_KEY = request.api_key if hasattr(request, 'api_key') else os.getenv('POE_API_KEY')

        # 转换消息格式为Poe格式
        poe_messages = [
            fp.ProtocolMessage(role=msg.role, content=msg.content)
            for msg in request.messages
        ]

        # 如果是流式请求
        if request.stream:
            return fastapi.responses.StreamingResponse(
                stream_response(poe_messages, bot_name, request.model),
                media_type="text/event-stream"
            )
        else:
            # 非流式请求
            response_text = await get_full_response(poe_messages, bot_name)
            # 计算token数量（这是一个简单估计）
            prompt_tokens = sum(len(msg.content.split()) for msg in request.messages) * 4
            completion_tokens = len(response_text.split()) * 4

            # 构建OpenAI格式的响应
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message={
                            "role": "assistant",
                            "content": response_text
                        },
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_full_response(messages, bot_name):
    """获取完整的响应（非流式）"""
    response_text = ""
    async for partial in fp.get_bot_response(
        messages=messages,
        bot_name=bot_name,
        api_key=POE_API_KEY
    ):
        # 使用 str() 转换 PartialResponse 对象
        response_text += partial.text
    return response_text

async def stream_response(messages, bot_name, model_name):
    """流式响应生成器"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
    created_time = int(time.time())

    # 发送初始角色信息
    yield f"data: {{\n"
    yield f'"id": "{response_id}",\n'
    yield f'"object": "chat.completion.chunk",\n'
    yield f'"created": {created_time},\n'
    yield f'"model": "{model_name}",\n'
    yield '"choices": [\n'
    yield '{\n'
    yield '"index": 0,\n'
    yield '"delta": {"role": "assistant"},\n'
    yield '"finish_reason": null\n'
    yield '}\n'
    yield ']\n'
    yield '}\n\n'

    # 发送内容流
    try:
        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name=bot_name,
            api_key=POE_API_KEY
        ):
            # 使用 str() 转换并处理字符串
            content = partial.text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            yield f"data: {{\n"
            yield f'"id": "{response_id}",\n'
            yield f'"object": "chat.completion.chunk",\n'
            yield f'"created": {created_time},\n'
            yield f'"model": "{model_name}",\n'
            yield '"choices": [\n'
            yield '{\n'
            yield '"index": 0,\n'
            yield f'"delta": {{"content": "{content}"}},\n'
            yield '"finish_reason": null\n'
            yield '}\n'
            yield ']\n'
            yield '}\n\n'
        # 发送结束标记
        yield f"data: {{\n"
        yield f'"id": "{response_id}",\n'
        yield f'"object": "chat.completion.chunk",\n'
        yield f'"created": {created_time},\n'
        yield f'"model": "{model_name}",\n'
        yield '"choices": [\n'
        yield '{\n'
        yield '"index": 0,\n'
        yield '"delta": {},\n'
        yield '"finish_reason": "stop"\n'
        yield '}\n'
        yield ']\n'
        yield '}\n\n'
        # 发送完成标记
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_message = str(e).replace('"', '\\"')
        yield f"data: {{\n"
        yield f'"error": {{\n'
        yield f'"message": "{error_message}",\n'
        yield f'"type": "server_error"\n'
        yield '}\n'
        yield '}\n\n'

if __name__ == "__main__":
    print("启动OpenAI兼容API服务，基于Poe平台...")
    print("请确保已将代码中的POE_API_KEY替换为你的实际API密钥")
    uvicorn.run(app, host="0.0.0.0", port=8000)
