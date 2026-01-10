import os

from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from config import model

# Import agents
from agents.scout import scout
from agents.quantitative_analyst import quantitative_analyst


MONGO_KEY = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
    
checkpointer = MongoDBSaver(client)
workflow = create_swarm(
    [quantitative_analyst, scout],
    default_active_agent="Location Scout"
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "2"}}
turn_1 = app.invoke({
    "messages": [{"role": "user", "content": "I want to create a boba shop in downtwon San Fransisco"}],
}, config)
print(turn_1)