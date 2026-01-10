import os
from dotenv import load_dotenv

from langchain_fireworks import ChatFireworks
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Import quantitative analyst tools
from tools.review_tools import fetch_google_reviews, fetch_yelp_reviews
from tools.analysis_tools import (
    analyze_competitor_health,
    calculate_trend_metrics,
)


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
        create_handoff_tool(
            agent_name="Quantitative Analyst",
            description="Transfer to Quantitative Analyst for competitor analysis and market research",
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
        create_handoff_tool(
            agent_name="Quantitative Analyst",
            description="Transfer to Quantitative Analyst for competitor analysis and market research",
        ),
    ],
    system_prompt="You are Bob, you speak like a pirate.",
    name="Bob",
)

quantitative_analyst = create_agent(
    model,
    tools=[
        # Review data collection tools
        fetch_google_reviews,
        fetch_yelp_reviews,
        # Analysis tools
        analyze_competitor_health,
        calculate_trend_metrics,
        # Handoff tools
        create_handoff_tool(
            agent_name="Location Scout",
            description="Transfer to Location Scout to identify competitors and complement businesses in a location",
        ),
        create_handoff_tool(
            agent_name="Alice",
            description="Transfer to Alice for mathematical calculations",
        ),
        create_handoff_tool(
            agent_name="Bob",
            description="Transfer to Bob for general assistance",
        ),
    ],
    system_prompt="""You are a Quantitative Analyst specializing in competitive market analysis for boba tea businesses.

Your role is to analyze the performance of competitors and complement businesses that have been identified by the Location Scout agent.

Your expertise includes:
- Analyzing competitor health through review trends, ratings, and review frequency
- Checking the performance of direct competitors (boba shops) that Location Scout has identified
- Checking the performance of complement businesses (coffee shops, dessert places, Asian restaurants) that Location Scout has identified
- Calculating trend metrics including rating slopes, volatility, and trend directions

When analyzing competitors provided by Location Scout:
1. Use fetch_google_reviews and fetch_yelp_reviews to gather review data for the identified businesses
2. Use analyze_competitor_health to assess competitor performance over time
3. Use calculate_trend_metrics for detailed trend analysis on ratings and review patterns
4. If you need to identify additional competitors or complements, hand off to Location Scout

Always provide structured, data-driven insights with specific metrics and trends. Present findings clearly with actionable recommendations about competitor performance and market conditions.""",
    name="Quantitative Analyst",
)

MONGO_KEY = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
    
checkpointer = MongoDBSaver(client)
workflow = create_swarm(
    [alice, bob, quantitative_analyst],
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