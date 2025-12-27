from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage, HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

CONFIG = {"configurable": {"thread_id": "1"}}
QUESTION_1 = HumanMessage(content="My favorite color is green.")
QUESTION_2 = HumanMessage(content="What is my favorite color?")

class CustomState(AgentState):
    favorite_color: str

@tool
def update_favorite_color(favorite_color: str, runtime: ToolRuntime) -> Command:
    """
    Update the favorite color of the user in the state once they've revealed it.
    """
    return Command(update={
        "favorite_color": favorite_color,
        "messages": [ToolMessage("Successfully updated favorite color", tool_call_id=runtime.tool_call_id)],
    }) 

@tool
def read_favorite_color(runtime: ToolRuntime) -> str:
    """
    Read the favorite color of the user from the state.
    """
    try:
        return runtime.state["favorite_color"]
    except KeyError:
        return "No favorite color found in state."

agent = create_agent(
    checkpointer=InMemorySaver(),
    model="gpt-5-nano",
    state_schema=CustomState,
    tools=[update_favorite_color, read_favorite_color],
)
response = agent.invoke(
    {"messages": [QUESTION_1]},
    config=CONFIG,
)
print(response["messages"][-1].content)
response = agent.invoke(
    {"messages": [QUESTION_2]},
    config=CONFIG,
)
print(response["messages"][-1].content)
