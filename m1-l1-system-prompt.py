from langchain.agents import create_agent
from langchain.messages import HumanMessage

SYSTEM_PROMPT = """"
You are a science fiction writer, create a capital city at the users request.

User: What is the capital of Mars?
Agent: Marsialias

User: What is the capital of Venus?
Agent: Venusoviala
"""
QUESTION = HumanMessage(content="What's the capital of the Moon?")

agent = create_agent(
    model="gpt-5-nano",
    system_prompt=SYSTEM_PROMPT,
)
response = agent.invoke({
    "messages": [QUESTION],
})
print(response["messages"][-1].content)
