from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")

# Pull the prompt from the classic hub
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt) # it creates the chain
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # it runs the tools and decide what to do with the output of the tools, it also prints the intermediate steps because verbose is set to True

chain = agent_executor

def main(): 
    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for an ML engineer in the Vancouver,BC area in Canada on linkedin and list their details"
        }
    )
    print(result)

if __name__ == "__main__":
    main()