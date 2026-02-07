import asyncio
import os
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from generic_llm_lib.llm_core.messages.models import BaseMessage
from generic_llm_lib.llm_impl import GenericOpenAI

# Load environment variables
load_dotenv()


async def main() -> None:
    """
    Main function to run the CLI chat using OpenAI.
    """
    print("Welcome to the CLI Chat (OpenAI)!")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    client = AsyncOpenAI(api_key=api_key)
    llm = GenericOpenAI(
        client=client,
        model_name="gpt-3.5-turbo",
        sys_instruction="You are a helpful assistant.",
    )
    print("Using OpenAI.")

    history: List[BaseMessage] = []

    print("\nStart chatting! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = await llm.chat(history=history, user_prompt=user_input)
            print(f"Assistant: {response.last_response.text}")
            history = response.history

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
