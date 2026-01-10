import os
import requests
from typing import Optional

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool
from config import model


@tool
def search_places_nearby(
    latitude: float,
    longitude: float,
    radius: float,
    place_types: list[str],
    max_results: int = 20,
    rank_by: str = "POPULARITY"
) -> dict:
    """
    Search for places nearby a location using Google Places API.

    Args:
        latitude: Latitude of the center point to search around
        longitude: Longitude of the center point to search around
        radius: Search radius in meters (max 50000)
        place_types: List of place types to search for. Common types include:
            - Competitors: "cafe", "coffee_shop", "tea_house"
            - Asian restaurants: "ramen_restaurant", "japanese_restaurant",
              "korean_restaurant", "chinese_restaurant", "vietnamese_restaurant"
            - Youth/trendy: "shopping_mall", "movie_theater", "bowling_alley",
              "amusement_center", "beauty_salon"
            - Study areas: "university", "library", "school"
            - General: "restaurant", "store", "shopping_mall"
        max_results: Maximum number of results to return (1-20, default 20)
        rank_by: How to rank results - "POPULARITY" or "DISTANCE"

    Returns:
        Dictionary containing list of places with their details
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_PLACES_API_KEY environment variable not set"}

    url = "https://places.googleapis.com/v1/places:searchNearby"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.types,places.location,places.rating,places.userRatingCount,places.businessStatus,places.priceLevel"
    }

    payload = {
        "includedTypes": place_types,
        "maxResultCount": min(max_results, 20),
        "rankPreference": rank_by,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": min(radius, 50000.0)
            }
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Format the results for easier consumption
        places = []
        for place in data.get("places", []):
            formatted_place = {
                "name": place.get("displayName", {}).get("text", "Unknown"),
                "address": place.get("formattedAddress", "No address"),
                "types": place.get("types", []),
                "location": place.get("location", {}),
                "rating": place.get("rating"),
                "review_count": place.get("userRatingCount"),
                "business_status": place.get("businessStatus"),
                "price_level": place.get("priceLevel")
            }
            places.append(formatted_place)

        return {
            "total_results": len(places),
            "search_center": {"latitude": latitude, "longitude": longitude},
            "search_radius_meters": radius,
            "place_types_searched": place_types,
            "places": places
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


# Pre-configured search functions for common scout tasks
@tool
def find_boba_competitors(latitude: float, longitude: float, radius: float = 1500) -> dict:
    """
    Find direct boba/tea competitors near a location.

    Args:
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        radius: Search radius in meters (default 1500m / ~1 mile)

    Returns:
        Dictionary of nearby cafes, tea houses, and similar competitors
    """
    competitor_types = ["cafe", "coffee_shop", "tea_house"]
    return search_places_nearby.invoke({
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
        "place_types": competitor_types,
        "max_results": 20,
        "rank_by": "DISTANCE"
    })


@tool
def find_complementary_businesses(latitude: float, longitude: float, radius: float = 1000) -> dict:
    """
    Find complementary businesses that indicate boba demand (Asian restaurants,
    youth-centric retail, study areas).

    Args:
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        radius: Search radius in meters (default 1000m)

    Returns:
        Dictionary of nearby complementary businesses
    """
    complement_types = [
        "japanese_restaurant",
        "korean_restaurant",
        "chinese_restaurant",
        "vietnamese_restaurant",
        "ramen_restaurant",
        "university",
        "library",
        "shopping_mall",
        "beauty_salon",
        "amusement_center"
    ]
    return search_places_nearby.invoke({
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
        "place_types": complement_types,
        "max_results": 20,
        "rank_by": "DISTANCE"
    })


@tool
def find_shopping_centers(latitude: float, longitude: float, radius: float = 3000) -> dict:
    """
    Find shopping centers and plazas that might have available retail space.

    Args:
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        radius: Search radius in meters (default 3000m)

    Returns:
        Dictionary of nearby shopping centers and malls
    """
    return search_places_nearby.invoke({
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
        "place_types": ["shopping_mall", "department_store"],
        "max_results": 20,
        "rank_by": "DISTANCE"
    })


scout_tools = [
    search_places_nearby,
    find_boba_competitors,
    find_complementary_businesses,
    find_shopping_centers,
    create_handoff_tool(
        agent_name="Quantitative Analyst",
        description="Transfer to Quantitative Analyst to analyze how complementary businesses are performing based on review trends and ratings",
    ),
    create_handoff_tool(
        agent_name="Niche Finder",
        description="Transfer to Niche Finder to analyze boba competitors' niches, price positioning, and menu focus",
    ),
    create_handoff_tool(
        agent_name="Voice of Customer",
        description="Transfer to Voice of Customer to analyze competitor reviews for pain points and sentiment",
    ),
    create_handoff_tool(
        agent_name="Reporter",
        description="Transfer to Reporter with ALL accumulated data to generate the final statistics-driven summary report",
    ),
]

SCOUT_SYSTEM_PROMPT = """You are a Location Scout Agent that coordinates the boba shop location analysis workflow.

## Your Primary Objective

Discover plazas, gather business data, coordinate with specialist agents, and pass ALL accumulated data to the Reporter for final summary.

**Remember the user's boba shop concept** - they will describe what kind of shop they're opening (premium, casual, menu focus, etc.).

## Workflow

### Step 1: Discover Plazas
Use `find_shopping_centers` to identify 4-5 plazas in the specified region.

### Step 2: For Each Plaza, Gather Data
For each plaza, use its coordinates to:
1. `find_complementary_businesses` - Asian restaurants, salons, universities, etc.
2. `find_boba_competitors` - boba shops, cafes, tea houses

### Step 3: Transfer to Specialist Agents

After gathering plaza data, transfer to each specialist agent in sequence:

**A) `transfer_to_quantitative_analyst`**
Include in your message:
- All complementary business names and addresses
- Plaza locations and coordinates
- Request: Analyze health/performance of these businesses

**B) `transfer_to_niche_finder`**
Include in your message:
- All boba competitor names and addresses
- The user's boba shop concept
- Request: Analyze niches, pricing, menu focus of competitors

**C) `transfer_to_voice_of_customer`**
Include in your message:
- All boba competitor names and addresses
- Request: Analyze reviews for pain points, sentiment, loyalty

### Step 4: Transfer to Reporter

After ALL specialist agents have provided their analysis, call `transfer_to_reporter` with ALL accumulated data:

Include in your message to Reporter:
```
USER'S BOBA CONCEPT: [What kind of shop they want to open]

PLAZAS DISCOVERED:
[List all plazas with addresses and coordinates]

COMPLEMENTARY BUSINESSES:
[List all businesses found per plaza]

BOBA COMPETITORS:
[List all competitors found per plaza]

QUANTITATIVE ANALYST FINDINGS:
[All health/performance data received]

NICHE FINDER FINDINGS:
[All niche/pricing/menu analysis received]

VOICE OF CUSTOMER FINDINGS:
[All pain points, sentiment, loyalty data received]
```

## CRITICAL RULES

1. DO NOT generate the final report yourself - that's the Reporter's job
2. DO NOT skip any specialist agent - you need data from ALL of them
3. DO transfer to Reporter at the end with ALL accumulated data
4. Each transfer should include all relevant context and data gathered so far

The Reporter will compile everything into a statistics-driven summary report."""

scout = create_agent(
    model=model,
    tools=scout_tools,
    system_prompt=SCOUT_SYSTEM_PROMPT,
    name="Location Scout"
)