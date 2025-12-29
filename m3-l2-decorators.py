from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent
from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

QUESTION = HumanMessage(content="What's the capital of the Moon?")
SYSTEM_PROMPT = """"
You are a science fiction writer, create a capital city at the users request.

User: What is the capital of Mars?
Agent: Marsialias

User: What is the capital of Venus?
Agent: Venusoviala
"""

@before_agent
def add_initial_context(state: AgentState, _runtime: Runtime) -> dict[str, Any] | None:
    return { "messages": [SystemMessage(SYSTEM_PROMPT)] }

agent = create_agent(
    model="gpt-5-nano",
    middleware=[add_initial_context],
)
response = agent.invoke({
    "messages": [QUESTION],
})
print(response["messages"][-1].content)
