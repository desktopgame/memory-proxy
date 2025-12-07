import os
import asyncio
from libmemory import get_memory_system, close_memory_system
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("HIGH_SUPPORT_MODEL")


async def main():
    print("compress start.")
    with open("long_text.txt", "r", encoding="utf8") as fp:
        text = fp.read()
        print(await get_memory_system().compress_text(text))
    await close_memory_system()
    print("done.")


def start():
    asyncio.run(main())


if __name__ == "__main__":
    start()
