import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables before importing agents

from langgraph_swarm import create_swarm
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

# Import agents
from agents.scout import scout
from agents.quantitative_analyst import quantitative_analyst
from agents.niche_finder import niche_finder
from agents.voice import voice

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(client)
db = client.get_database("langgraph")
collection = db.get_collection("workflows")

workflow = create_swarm(
    [quantitative_analyst, scout, niche_finder, voice],
    default_active_agent="Location Scout"
)
app = workflow.compile(checkpointer=checkpointer)
print(app.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}))