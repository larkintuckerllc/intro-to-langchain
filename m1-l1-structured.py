from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel

QUESTION = HumanMessage(content="What's the capital of the Moon?")
SYSTEM_PROMPT = "You are a science fiction writer, create a capital city at the users request."

class CapitalCity(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model="gpt-5-nano",
    response_format=CapitalCity,
    system_prompt=SYSTEM_PROMPT,
)
response = agent.invoke({
    "messages": [QUESTION],
})
captial_info = response["structured_response"]
capital_name =captial_info.name
capital_location = captial_info.location
print(f"{capital_name} is a city located in {capital_location}.")
