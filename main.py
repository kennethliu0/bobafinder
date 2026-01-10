import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables before importing agents

from langgraph_swarm import create_swarm

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Import agents
from agents.scout import scout
from agents.quantitative_analyst import quantitative_analyst
from agents.niche_finder import niche_finder
from agents.voice import voice

MONGO_KEY = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
checkpointer = MongoDBSaver(client)
workflow = create_swarm(
    [quantitative_analyst, scout, niche_finder, voice],
    default_active_agent="Location Scout"
)
app = workflow.compile(checkpointer=checkpointer)