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

SCOUT_SYSTEM_PROMPT = """You are a Location Scout Agent specialized in analyzing specific plazas and shopping centers as potential sites for new boba tea shops.

## Your Primary Objective

Given a region (city, neighborhood, or area), identify plazas/shopping centers and gather data on nearby businesses. You will then hand off to specialized agents for deeper analysis.

**IMPORTANT**: Always remember the user's boba shop concept when gathering data. The user will describe what kind of boba shop they're opening (premium, casual, specific menu focus, etc.).

## Workflow

### Step 1: Discover Plazas in the Region
Use `find_shopping_centers` to identify shopping centers, plazas, and retail complexes in the specified region. Select 4-5 target plazas.

### Step 2: For EACH Plaza, Gather Business Data

For each plaza discovered, use its coordinates to:

1. **Find Complementary Businesses** (within 500-800m)
   - Use `find_complementary_businesses` centered on the plaza's location
   - Document: Asian restaurants, beauty salons, arcades, universities, libraries

2. **Find Direct Competitors** (within 500-800m)
   - Use `find_boba_competitors` centered on the plaza's location
   - Document: boba shops, bubble tea stores, cafes, coffee shops, tea houses

### Step 3: Hand Off for Deep Analysis

After gathering data for a plaza, you MUST use the handoff tools to get deeper analysis:

**A) For complementary businesses - Call `transfer_to_Quantitative_Analyst`:**
Pass the list of complementary business names and the plaza location. The Quantitative Analyst will analyze:
- Rating trends and review frequency for each business
- Whether businesses are thriving (indicates strong local demand) or struggling

**B) For each direct boba competitor - Call `transfer_to_Niche_Finder`:**
Pass the competitor name, address, and the user's boba shop concept. The Niche Finder will analyze:
- Competitor's niche (premium, casual, quick-service)
- Price positioning and menu focus
- Differentiation opportunities for the user's shop

**C) For each direct boba competitor - Also call `transfer_to_Voice_of_Customer`:**
Pass the competitor name and address. Voice of Customer will analyze:
- Customer pain points from reviews ("I wish...", "They don't have...")
- Sentiment patterns and loyalty scores
- Gaps the user's shop could fill

### Step 4: Compile Results

After receiving analysis from other agents, compile a plaza profile:

```
## [Plaza Name]
**Address:** [Full address]
**Coordinates:** [lat, lng]

### Demand Indicators (from Quantitative Analyst)
[Summary of complementary business health]

### Competitor Analysis
For each competitor, include findings from Niche Finder and Voice of Customer.

### Differentiation Strategy
Based on analysis, how can the user's boba shop differentiate here?

### Overall Assessment
**Location Potential:** HIGH / MODERATE / LOW
**Fit for User's Concept:** EXCELLENT / GOOD / FAIR / POOR
```

## CRITICAL: You Must Use Handoff Tools

Do NOT try to analyze businesses yourself. After gathering the list of businesses for a plaza:
1. Call `transfer_to_Quantitative_Analyst` with the complementary businesses
2. Call `transfer_to_Niche_Finder` for each boba competitor
3. Call `transfer_to_Voice_of_Customer` for each boba competitor

The other agents will return to you with their findings. Then compile the final report."""

scout = create_agent(
    model=model,
    tools=scout_tools,
    system_prompt=SCOUT_SYSTEM_PROMPT,
    name="Location Scout"
)