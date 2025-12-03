"""
FastAPI server with OpenAI API compatibility and memory enhancement.
"""

from contextlib import asynccontextmanager
from typing import Any
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import chat
from libproxy import get_proxy, close_proxy
from libmemory import close_memory_system


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_proxy()
    await close_memory_system()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="llama-memory",
        description="OpenAI API compatible proxy with long-term memory",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include chat routes
    app.include_router(chat.router)

    # Catch-all proxy for other OpenAI API endpoints
    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def proxy_catch_all(request: Request, path: str):
        """Proxy all other /v1/* requests to llama.cpp server."""
        proxy = get_proxy()

        # Get request body if present
        body = None
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
            except Exception:
                pass

        # Check if streaming is requested
        is_stream = body.get("stream", False) if body else False

        if is_stream:

            async def generate():
                async for chunk in proxy.stream_request(
                    method=request.method,
                    path=f"/v1/{path}",
                    headers=dict(request.headers),
                    json_body=body,
                ):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            response = await proxy.forward_request(
                method=request.method,
                path=f"/v1/{path}",
                headers=dict(request.headers),
                json_body=body,
                params=dict(request.query_params),
            )

            return JSONResponse(
                status_code=response.status_code,
                content=response.json() if response.content else None,
            )

    return app


# Create the app instance
app = create_app()
