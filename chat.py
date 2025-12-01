"""
Chat completions route with memory enhancement.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

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

    # Enhance messages with relevant memories
    enhanced_messages = messages

    if messages:
        enhanced_messages, messages

    # Update body with enhanced messages
    enhanced_body = {**body, "messages": enhanced_messages}

    # Get proxy and forward request
    proxy = get_proxy()

    if stream:
        return await _handle_streaming_response(
            proxy, enhanced_body, messages, request.headers
        )
    else:
        return await _handle_non_streaming_response(
            proxy, enhanced_body, messages, request.headers
        )


async def _handle_non_streaming_response(
    proxy: LlamaProxy,
    body: dict[str, Any],
    original_messages: list[dict[str, Any]],
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

    # Extract assistant's response and store in memory
    assistant_content = _extract_assistant_content(result)
    if assistant_content:
        await _store_conversation(
            messages=original_messages,
            response_content=assistant_content,
        )

    return JSONResponse(content=result)


async def _handle_streaming_response(
    proxy: LlamaProxy,
    body: dict[str, Any],
    original_messages: list[dict[str, Any]],
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

        # After streaming completes, store in memory
        if collected_content:
            full_content = "".join(collected_content)
            if full_content:
                await _store_conversation(
                    messages=original_messages,
                    response_content=full_content,
                )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _store_conversation(
    messages: list[dict[str, Any]],
    response_content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    logger.debug(response_content)
    return ""


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
