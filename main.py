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
from agents.reporter import reporter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
import json

MONGO_KEY = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
checkpointer = MongoDBSaver(client)

workflow = create_swarm(
    [quantitative_analyst, scout, niche_finder, voice, reporter],
    default_active_agent="Location Scout"
)
graph = workflow.compile(checkpointer=checkpointer)

app = FastAPI()

async def stream_generator(prompt: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield f'0:{json.dumps(content)}\n'

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    prompt = messages[-1]["content"] if messages else ""
    thread_id = data.get("thread_id", "default")

    return StreamingResponse(
        stream_generator(prompt, thread_id),
        media_type="text/plain"
    )
@app.get("/")
async def root():
    return {"message": "Hello, World!"}