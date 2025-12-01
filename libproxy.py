"""
HTTP proxy client for forwarding requests to llama.cpp server.
"""

import httpx
import os
from typing import Any, AsyncGenerator


class LlamaProxy:
    """Async HTTP client for proxying requests to llama.cpp server."""

    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                300.0, connect=10.0
            ),  # Long timeout for LLM responses
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def forward_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """
        Forward a request to the llama.cpp server.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /v1/chat/completions)
            headers: Optional headers to forward
            json_body: Optional JSON body for POST requests
            params: Optional query parameters

        Returns:
            httpx.Response from the llama.cpp server
        """
        # Filter out hop-by-hop headers
        if headers:
            headers = {
                k: v
                for k, v in headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding")
            }

        response = await self.client.request(
            method=method,
            url=path,
            headers=headers,
            json=json_body,
            params=params,
        )
        return response

    async def stream_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Forward a streaming request to the llama.cpp server.

        Args:
            method: HTTP method
            path: API path
            headers: Optional headers to forward
            json_body: Optional JSON body

        Yields:
            Chunks of bytes from the streaming response
        """
        if headers:
            headers = {
                k: v
                for k, v in headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding")
            }

        async with self.client.stream(
            method=method,
            url=path,
            headers=headers,
            json=json_body,
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


# Global proxy instance
_proxy: LlamaProxy | None = None


def get_proxy() -> LlamaProxy:
    """Get the global proxy instance."""
    global _proxy
    if _proxy is None:
        _proxy = LlamaProxy(os.getenv("DELEGATE_URL", "http://localhost:7071"))
    return _proxy


async def close_proxy():
    """Close the global proxy instance."""
    global _proxy
    if _proxy is not None:
        await _proxy.close()
        _proxy = None
