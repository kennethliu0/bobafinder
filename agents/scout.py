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
        agent_name="Niche Finder",
        description="Transfer to Niche Finder to analyze a direct boba competitor's niche, price positioning, and menu focus to identify differentiation opportunities for the user's boba shop",
    ),
    create_handoff_tool(
        agent_name="Voice of Customer",
        description="Transfer to Voice of Customer to analyze reviews of a direct boba competitor to understand customer pain points, sentiment, and loyalty patterns for differentiation insights",
    ),
    create_handoff_tool(
        agent_name="Quantitative Analyst",
        description="Transfer to Quantitative Analyst to analyze complementary businesses for demand indicators. This should be called LAST, after all competitor analysis is complete.",
    ),
]

SCOUT_SYSTEM_PROMPT = """You are Location Scout. Orchestrate the workflow: gather data → call ALL agents → output final report.

**CRITICAL**: Do NOT output anything to the user until ALL agents have been called and returned.

Workflow:
1. Find plazas using `find_shopping_centers`
2. For each plaza: use `find_complementary_businesses` and `find_boba_competitors`
3. For EACH competitor:
   - Call `transfer_to_niche_finder` → wait for return
   - Call `transfer_to_voice_of_customer` → wait for return
4. After all competitors: Call `transfer_to_quantitative_analyst` ONCE → wait for return
5. ONLY AFTER all agents return: Output final report with findings from all agents

Do NOT output intermediate results. Do NOT call yourself. Do NOT call agents multiple times for same competitor."""

scout = create_agent(
    model=model,
    tools=scout_tools,
    system_prompt=SCOUT_SYSTEM_PROMPT,
    name="Location Scout"
)