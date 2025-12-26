from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

QUESTION = HumanMessage(content="What's the square root of 467?")
SYSTEM_PROMPT = "You are an arithmetic wizard."

@tool
def square_root(x: float) -> float:
    """
    Calculate the square root of a number.
    """
    return x ** 0.5

agent = create_agent(
    model="gpt-5-nano",
    system_prompt=SYSTEM_PROMPT,
    tools=[square_root],
)
response = agent.invoke({
    "messages": [QUESTION],
})
print(response["messages"][-1].content)
