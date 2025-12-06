import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("HIGH_SUPPORT_MODEL")


async def main():
    client = AsyncOpenAI(base_url="http://localhost:7070/v1", api_key="not-needed")
    messages = []

    print("bake start.")
    with open("scenario.txt", "r", encoding="utf8") as fp:
        lines = fp.readlines()
        for line in lines:
            input = line.strip()
            message = {"role": "user", "content": input}
            messages.append(messages)
            resp = await client.chat.completions.create(model=MODEL, messages=messages)
            messages.append(resp.choices[0].message)
    print("done.")


def start():
    asyncio.run(main())


if __name__ == "__main__":
    start()
