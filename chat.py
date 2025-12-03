"""
Chat completions route with memory enhancement.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from libmemory import save_memory, load_memory, process_conversation
from libproxy import get_proxy, LlamaProxy


logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Get user message and load relevant memories
    user_message = _get_user_message(messages)
    logger.debug(f"User message: {user_message}")
    
    memory_context = await load_memory(user_message)
    enhanced_messages = _cheat_messages(messages, memory_context)

    # Save user message to memory and get conversation_id (with messages for chain tracking)
    conversation_id = save_memory(user_message, messages=messages)
    logger.debug(f"Saved conversation: {conversation_id}")

    # Update body with enhanced messages
    enhanced_body = {**body, "messages": enhanced_messages}

    # Get proxy and forward request
    proxy = get_proxy()

    if stream:
        return await _handle_streaming_response(
            proxy, enhanced_body, user_message, conversation_id, request.headers
        )
    else:
        return await _handle_non_streaming_response(
            proxy, enhanced_body, user_message, conversation_id, request.headers
        )


async def _handle_non_streaming_response(
    proxy: LlamaProxy,
    body: dict[str, Any],
    user_message: str,
    conversation_id: str,
    headers,
) -> JSONResponse:
    """Handle non-streaming chat completion request."""
    # Forward to llama.cpp
    response = await proxy.forward_request(
        method="POST",
        path="/v1/chat/completions",
        headers=dict(headers),
        json_body=body,
    )

    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if response.content else {"error": "Proxy error"},
        )

    result = response.json()

    # Extract assistant's response and process conversation
    assistant_content = _extract_assistant_content(result)
    if assistant_content:
        await process_conversation(conversation_id, user_message, assistant_content)

    return JSONResponse(content=result)


async def _handle_streaming_response(
    proxy: LlamaProxy,
    body: dict[str, Any],
    user_message: str,
    conversation_id: str,
    headers,
) -> StreamingResponse:
    """Handle streaming chat completion request."""

    collected_content = []

    async def generate():
        nonlocal collected_content

        async for chunk in proxy.stream_request(
            method="POST",
            path="/v1/chat/completions",
            headers=dict(headers),
            json_body=body,
        ):
            # Try to extract content from SSE chunk for memory storage
            _collect_streaming_content(chunk, collected_content)
            yield chunk

        # After streaming completes, process conversation
        if collected_content:
            full_content = "".join(collected_content)
            if full_content:
                await process_conversation(conversation_id, user_message, full_content)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _get_user_message(messages: list[dict[str, Any]]) -> str:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            original_content = messages[i].get("content", "")

            if isinstance(original_content, str):
                return original_content
            elif isinstance(original_content, list):
                return "\n".join(map(lambda o: o["text"], original_content))
    return ""

def _cheat_messages(
    messages: list[dict[str, Any]],
    new_message,
) -> list[dict[str, Any]]:
    # Create enhanced messages
    enhanced_messages = list(messages)

    for i in range(len(enhanced_messages) - 1, -1, -1):
        if enhanced_messages[i].get("role") == "user":
            original_content = enhanced_messages[i].get("content", "")

            if isinstance(original_content, str):
                # Simple string content - append memories
                enhanced_messages[i] = {
                    **enhanced_messages[i],
                    "content": original_content + new_message,
                }
            elif isinstance(original_content, list):
                # Content array - append as new text part
                enhanced_messages[i] = {
                    **enhanced_messages[i],
                    "content": original_content
                    + [{"type": "text", "text": new_message}],
                }
            break

    return enhanced_messages


def _extract_assistant_content(response: dict[str, Any]) -> str:
    """Extract assistant's message content from response."""
    try:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
    except Exception:
        pass
    return ""


def _collect_streaming_content(chunk: bytes, collected: list[str]):
    """
    Extract content from SSE streaming chunk.

    SSE format: data: {...json...}\n\n
    """
    try:
        text = chunk.decode("utf-8")
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                json_str = line[6:]  # Remove "data: " prefix
                data = json.loads(json_str)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        collected.append(content)
    except Exception:
        # Ignore parsing errors in streaming chunks
        pass
