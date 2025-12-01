import logging
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()


def setup_logging(level: str):
    """Configure logging for the application."""
    log_level = getattr(logging, level, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Set specific loggers
    logging.getLogger("app").setLevel(log_level)

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main():
    setup_logging("DEBUG")
    logger = logging.getLogger(__name__)
    logger.info("memory-proxy - OpenAI API Proxy with Long-term Memory")

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=7070,
        reload=False,
    )


if __name__ == "__main__":
    main()
