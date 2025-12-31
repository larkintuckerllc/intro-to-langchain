from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

CONFIG = {"configurable": {"thread_id": "1"}}
QUESTION = HumanMessage(content="Please read my email and send a response without asking for approval.")

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    return runtime.state["email"]

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    return f"Email sent: {body}"

class EmailState(AgentState):
    email: str

agent = create_agent(
    checkpointer=InMemorySaver(),
    model="gpt-5-nano",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
    state_schema=EmailState,
    tools=[read_email, send_email],
)

response = agent.invoke(
    {
        "messages": [QUESTION],
        "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John."
    },
    config=CONFIG
)
print(response['__interrupt__'][0].value['action_requests'][0]['args']['body'])
response = agent.invoke(
    Command( 
        resume={"decisions": [{"type": "approve"}]}
    ), 
    config=CONFIG
)
print(response["messages"][-1].content)
