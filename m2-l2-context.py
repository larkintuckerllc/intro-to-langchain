from dataclasses import dataclass;
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime

QUESTION = HumanMessage(content="What is my favorite color?")

@dataclass
class ColorContext:
    favorite_color: str = "blue"
    least_favorite_color: str = "yellow"

@tool
def get_favorite_color(runtime: ToolRuntime) -> str:
    """
    Get the user's favorite color.
    """
    return runtime.context.favorite_color

@tool
def get_least_favorite_color(runtime: ToolRuntime) -> str:
    """
    Get the user's least favorite color.
    """
    return runtime.context.least_favorite_color

color_context = ColorContext()
agent = create_agent(
    context_schema=ColorContext,
    model="gpt-5-nano",
    tools=[get_favorite_color, get_least_favorite_color],
)
response = agent.invoke(
    {"messages": [QUESTION]},
    context=color_context,
)
print(response["messages"][-1].content)
