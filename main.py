from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)

llm = ChatOpenAI(model="gpt-5")

tools = [TavilySearch()]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)

def main():
    print("Running...")
    
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Find 2 AI engineer using langchain jobs in Vancouver")
        ]
    })
    
    print(result)

if __name__ == "__main__":
    main()