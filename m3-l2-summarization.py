from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

CONFIG = {"configurable": {"thread_id": "1"}}
CONVERSATION = [
    HumanMessage(content="What is the capital of the moon?"),
    AIMessage(content="The capital of the moon is Lunapolis."),
    HumanMessage(content="What is the weather in Lunapolis?"),
    AIMessage(content="Skies are clear, with a high of 120C and a low of -100C."),
    HumanMessage(content="How many cheese miners live in Lunapolis?"),
    AIMessage(content="There are 100,000 cheese miners living in Lunapolis."),
    HumanMessage(content="Do you think the cheese miners' union will strike?"),
    AIMessage(content="Yes, because they are unhappy with the new president."),
]
QUESTION = HumanMessage(content="If you were Lunapolis' new president how would you respond to the cheese miners' union?")

agent = create_agent(
    model="gpt-5-nano",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-5-nano",
            trigger=("tokens", 100),
            keep=("messages", 1)
        )
    ],
)
response = agent.invoke(
    {"messages": CONVERSATION + [QUESTION]},
    config=CONFIG
)
print(response["messages"][-1].content)
