from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient

tavily_client = TavilyClient();

CONFIG = {"configurable": {"thread_id": "1"}}
QUESTION = HumanMessage(content="I have some leftover chicken and rice in my fridge. What can I make?")
SYSTEM_PROMPT = """
You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.
"""

@tool
def web_search(query: str) -> str:
    """
    Search the web for information.
    """
    return tavily_client.search(query)
agent = create_agent(
    checkpointer=InMemorySaver(),
    model="gpt-5-nano",
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search],
)

response = agent.invoke(
    {"messages": [QUESTION],},
    config=CONFIG
)
print(response['messages'][-1].content)
