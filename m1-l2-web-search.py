from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient

tavily_client = TavilyClient();

QUESTION = HumanMessage(content="Who is the current mayor of San Francisco?")

@tool
def web_search(query: str) -> str:
    """
    Search the web for information.
    """
    return tavily_client.search(query)

agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
)
response = agent.invoke({
    "messages": [QUESTION],
})
print(response["messages"][-1].content)
