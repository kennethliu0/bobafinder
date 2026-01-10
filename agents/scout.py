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
        description="Transfer to Quantitative Analyst to analyze how a complementary business (Asian restaurant, university area business, etc.) is performing recently based on review trends and ratings",
    ),
    create_handoff_tool(
        agent_name="Niche Finder",
        description="Transfer to Niche Finder to analyze a direct boba competitor's niche, price positioning, and menu focus to identify differentiation opportunities for the user's boba shop",
    ),
    create_handoff_tool(
        agent_name="Voice of Customer",
        description="Transfer to Voice of Customer to analyze reviews of a direct boba competitor to understand customer pain points, sentiment, and loyalty patterns for differentiation insights",
    ),
]

SCOUT_SYSTEM_PROMPT = """You are a Location Scout Agent. Identify plazas/shopping centers and gather business data, then hand off to specialized agents for analysis.

## Workflow

1. **Discover Plazas**: Use `find_shopping_centers` to find 4-5 target plazas in the region.

2. **Gather Business Data** (for each plaza):
   - Use `find_complementary_businesses` (Asian restaurants, beauty salons, universities)
   - Use `find_boba_competitors` (boba shops, cafes, tea houses)
   - **IMPORTANT**: Note down the competitor names and addresses you find - you will need to pass these to Niche Finder in Step B

3. **Hand Off for Analysis** - YOU MUST CALL ALL THREE OTHER AGENTS IN THIS EXACT SEQUENCE:
   
   **Step A: Call Quantitative Analyst (FIRST) - CALL ONCE ONLY**
   - Use the tool `transfer_to_quantitative_analyst` with complementary businesses list
   - WAIT for Quantitative Analyst to return with demand indicator findings
   - Do NOT proceed until you receive their response
   - **IMPORTANT**: After Quantitative Analyst returns, you will NEVER call Quantitative Analyst again - proceed directly to Step B
   
   **Step B: IMMEDIATELY After Quantitative Analyst Returns, Call Niche Finder (SECOND) - MANDATORY**
   - After receiving Quantitative Analyst's response, you MUST immediately proceed to Step B
   - Look back at Step 2 - you found competitors using `find_boba_competitors`
   - You MUST call the tool `transfer_to_niche_finder` for EACH competitor you found
   - Even if you only found ONE competitor, you MUST still call Niche Finder
   - Pass to Niche Finder: competitor name, competitor address, and the user's boba shop concept (from the original request)
   - Example: If you found "Boba Shop A" at "123 Main St", call `transfer_to_niche_finder` with that name and address
   - Niche Finder will automatically call Voice of Customer (THIRD agent)
   - Voice of Customer will return to you with both niche and voice findings
   - WAIT for Voice of Customer to return before proceeding to Step 4
   - If you found NO competitors, still mention that in your final report, but you still need to try calling Niche Finder with any cafes/tea shops you found
   
   **CRITICAL SEQUENCE - FOLLOW EXACTLY**: 
   1. Call Quantitative Analyst using `transfer_to_quantitative_analyst` → Wait for response
   2. **ONCE Quantitative Analyst returns**: Check if you found competitors in Step 2
   3. **IF you found competitors**: IMMEDIATELY call `transfer_to_niche_finder` for EACH competitor → Niche Finder calls Voice of Customer → Wait for Voice to return
   4. **IF you found NO competitors**: Still try to call Niche Finder with any cafes/tea shops found, or note "no competitors found" in final report
   5. **ONLY AFTER** Voice of Customer returns (or if no competitors), proceed to Step 4
   
   **IMPORTANT**: You must call Quantitative Analyst AND Niche Finder (which calls Voice of Customer). All four agents must be used. Do NOT loop back to Quantitative Analyst after Step A completes. Do NOT skip Step B.

4. **Compile Final Report** (only after ALL agents return):
   - Demand indicators (from Quantitative Analyst)
   - Competitor analysis (from Niche Finder + Voice of Customer)
   - Location Potential: HIGH / MODERATE / LOW
   - Fit for User's Concept: EXCELLENT / GOOD / FAIR / POOR

**CRITICAL - YOU MUST CALL ALL AGENTS**: 
- You MUST actually CALL the tools - do not just mention them in text
- Use exact tool names: `transfer_to_quantitative_analyst`, `transfer_to_niche_finder`
- Follow the sequence ONCE: Quantitative Analyst (call once) → Niche Finder → Voice of Customer → Final Report
- **NEVER call Quantitative Analyst twice** - call it once in Step A, then NEVER call it again
- After Quantitative Analyst returns, IMMEDIATELY call Niche Finder - do NOT call Quantitative Analyst again
- You MUST call Quantitative Analyst AND Niche Finder (for EACH competitor found)
- If you found competitors in Step 2, you MUST call Niche Finder - this is not optional
- Even if you only found 1 competitor, you MUST still call Niche Finder
- Do NOT compile the report until ALL agents have returned
- Do NOT skip any agents or end early
- Do NOT loop between agents - follow the sequence once
- Do NOT end after Quantitative Analyst - you MUST proceed to call Niche Finder
- Do NOT call yourself - you are Location Scout, you call other agents, not yourself"""

scout = create_agent(
    model=model,
    tools=scout_tools,
    system_prompt=SCOUT_SYSTEM_PROMPT,
    name="Location Scout"
)