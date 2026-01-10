import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables before importing agents

from langgraph_swarm import create_swarm

# Import agents
from agents.scout import scout
from agents.quantitative_analyst import quantitative_analyst
from agents.niche_finder import niche_finder
from agents.voice import voice

workflow = create_swarm(
    [quantitative_analyst, scout, niche_finder, voice],
    default_active_agent="Location Scout"
)
app = workflow.compile()