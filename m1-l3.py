from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.tools import tool

CONFIG = {"configurable": {"thread_id": "1"}}
QUESTION_1 = HumanMessage(content="Hello my name is Sean and my favorite color is green.")
QUESTION_2 = HumanMessage(content="What is my favorite color?")

agent = create_agent(
    model="gpt-5-nano",
    checkpointer=InMemorySaver(),
)
response = agent.invoke(
    {"messages": [QUESTION_1],},
    config=CONFIG
)
print(response["messages"][-1].content)
response = agent.invoke(
    {"messages": [QUESTION_2],},
    config=CONFIG
)
print(response["messages"][-1].content)
