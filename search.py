import os
import asyncio
from libmemory import load_memory
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("HIGH_SUPPORT_MODEL")


async def main():
    with open("search.txt", "r", encoding="utf8") as fp:
        input = fp.read()
        print(await load_memory(input))


def start():
    asyncio.run(main())


if __name__ == "__main__":
    start()
