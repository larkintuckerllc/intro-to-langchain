import asyncio

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

QUESTION = HumanMessage(content="What's the weather in San Francisco?")

client = MultiServerMCPClient(
    {
        "local_server": {
            "transport": "stdio",
            "command": "uv",
            "args": [
              "--directory",
              "/Users/jtucker/desktop/working/weather",
              "run",
              "weather.py"
            ],
        }
    }
)

async def main():
    tools = await client.get_tools()
    agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
    )
    response = await agent.ainvoke({
        "messages": [QUESTION],
    })
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())