# BobaFinder

An AI-powered location intelligence system for analyzing and identifying optimal locations for boba tea shops. BobaFinder uses a multi-agent AI system to perform comprehensive market analysis by evaluating competitors, customer sentiment, demand indicators, and market positioning.

## Overview

BobaFinder helps entrepreneurs determine the best locations to open new boba tea shops based on data-driven insights. The system leverages multiple specialized AI agents that work together to analyze:

- Competitor landscape and market saturation
- Customer pain points and brand loyalty
- Complementary business demand indicators
- Niche positioning and differentiation opportunities

## Features

### Location Discovery
- Search nearby businesses using Google Places API
- Identify shopping centers and retail plazas
- Find competitor locations within configurable radius

### Competitor Analysis
- Detailed business profiles including price levels, ratings, and reviews
- Niche positioning analysis across multiple dimensions
- Direct business comparison with overlap scoring
- Market saturation assessment

### Customer Sentiment Analysis
- Review scraping from Google and Yelp
- Sentiment clustering by category (wait time, sweetness, pearl texture, staff friendliness)
- Pain point extraction and pattern matching
- Brand loyalty scoring

### Demand Indicator Analysis
- Business health status calculation
- Rating trend analysis using linear regression
- Review frequency tracking
- Market demand assessment

### Comprehensive Reporting
- Executive summary with opportunity scores
- Statistical breakdown of market composition
- Risk assessment with mitigation strategies
- Actionable differentiation strategies
- Final location recommendation with confidence levels

## Architecture

BobaFinder uses a multi-agent swarm architecture with the following specialized agents:

| Agent | Role |
|-------|------|
| Location Scout | Main orchestrator that coordinates the workflow and discovers locations |
| Niche Finder | Analyzes competitor positioning and market niches |
| Voice of Customer | Analyzes reviews to understand customer sentiment |
| Quantitative Analyst | Evaluates complementary business health as demand proxy |
| Reporter | Compiles all findings into comprehensive business reports |

## Tech Stack

- **AI Framework**: LangChain, LangGraph, LangGraph Swarm
- **LLM**: Fireworks API (Minimax M2P1 model)
- **Data Sources**: Google Places API, Yelp API, Google Maps API
- **Database**: MongoDB (with LangGraph checkpoint integration)
- **Runtime**: Python 3.11+

## Prerequisites

- Python 3.11 or higher
- API keys for:
  - Fireworks API
  - Google Places API
  - Yelp API
  - MongoDB Atlas connection string

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kennethliu0/bobafinder.git
   cd bobafinder
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # Using UV (recommended)
   uv sync

   # Or using pip
   pip install -e .
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your API keys:
   ```
   FIREWORKS_API_KEY=<your_fireworks_api_key>
   MONGODB_URI=<your_mongodb_atlas_connection_string>
   GOOGLE_PLACES_API_KEY=<your_google_places_api_key>
   YELP_API_KEY=<your_yelp_api_key>
   ```

## LangGraph API Deployment

Deploy using the LangGraph CLI:
```bash
langgraph deploy
```