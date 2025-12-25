from langchain.agents import create_agent
from langchain.messages import HumanMessage

agent = create_agent(model="gpt-5-nano")
response = agent.invoke({
    "messages": [HumanMessage(content="What's the capital of the Moon?")],
})
print(response["messages"][-1].content)
