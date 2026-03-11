import asyncio
from dotenv import load_dotenv

from agents.coordinator import coordinator
from google.adk.runners import Runner

load_dotenv()

async def main():
    print("Initializing ADK Runner...")
    runner = Runner(agent=coordinator)
    
    query = "Find 2 recent papers on Large Language Models in healthcare and summarize finding"
    print(f"\nUser Query: {query}")
    print("Starting agent execution... (This will take a minute or two)")
    
    # Run the coordinator agent with our tools
    # runner.run() handles the agent lifecycle, calling LLM & using tools
    async for state in runner.run(query):
        # The ADK runner yields state updates as it interacts with tools
        if state.last_message:
            print(f"[Agent]: {state.last_message.content}")

if __name__ == "__main__":
    asyncio.run(main())
