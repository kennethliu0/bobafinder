import os

from langgraph_swarm import create_swarm
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from agents.scout import scout


MONGO_KEY = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
    
checkpointer = MongoDBSaver(client)

workflow = create_swarm([scout], default_active_agent="Location Scout")
app = workflow.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": "1"}}
# turn_1 = app.invoke(
#     {"messages": [{"role": "user", "content": "I want to create a boba shop in downtwon San Fransisco"}]},
#     config,
# )

turn_1 = scout.invoke(
    {"messages": [{"role": "user", "content": "I want to create a boba shop in downtwon San Fransisco"}]},
    config,
)
print(turn_1)