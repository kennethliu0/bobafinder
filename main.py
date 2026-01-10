import os
from dotenv import load_dotenv

from langchain_fireworks import ChatFireworks
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()


model = ChatFireworks(
    model="accounts/fireworks/models/minimax-m2p1", 
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

alice = create_agent(
    model,
    tools=[
        add,
        create_handoff_tool(
            agent_name="Bob",
            description="Transfer to Bob",
        ),
    ],
    system_prompt="You are Alice, an addition expert.",
    name="Alice",
)

bob = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="Alice",
            description="Transfer to Alice, she can help with math",
        ),
    ],
    system_prompt="You are Bob, you speak like a pirate.",
    name="Bob",
)
MONGO_KEY = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
    
checkpointer = MongoDBSaver(client)
workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
turn_1 = app.invoke(
    {"messages": [{"role": "user", "content": "i'd like to speak to Bob"}]},
    config,
)
print(turn_1)
turn_2 = app.invoke(
    {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
    config,
)
print(turn_2)