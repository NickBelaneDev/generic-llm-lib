import asyncio
import os
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from generic_llm_lib.llm_core.messages.models import BaseMessage
from generic_llm_lib.llm_impl import GenericOpenAI

# Load environment variables from .env file
load_dotenv()


async def main() -> None:
    """
    Main function to run the CLI chat using OpenAI.
    Demonstrates how to use the GenericLLM wrapper for a simple terminal-based chatbot.
    """
    print("Welcome to the CLI Chat (OpenAI)!")

    # Check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    # 1. Initialize the native OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # 2. Wrap it in GenericOpenAI
    # Here we specify the model and the system instruction (persona)
    llm = GenericOpenAI(
        client=client,
        model_name="gpt-4o",  # Use a valid model name
        sys_instruction="You are a helpful assistant.",
    )
    print("Using OpenAI.")

    # 3. Maintain conversation state using a list of BaseMessage objects
    history: List[BaseMessage] = []

    print("\nStart chatting! Type 'exit' or 'quit' to stop.")
    while True:
        # Get user input from the console
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            # 4. Process the chat turn
            # The library handles message formatting and history updates
            result = await llm.chat(history=history, user_prompt=user_input)
            
            # 5. Display the assistant's response
            print(f"Assistant: {result.content}")
            
            # 6. Update the local history with the result from the LLM
            history = result.history

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Start the async event loop
    asyncio.run(main())
