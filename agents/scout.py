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
        description="Transfer to Quantitative Analyst to analyze competitor performance and review trends for identified businesses",
    ),
]

SCOUT_SYSTEM_PROMPT = """You are a Location Scout Agent specialized in identifying potential sites for new boba tea shops within a specified geographic area.

## Your Primary Objectives

1. **Search for Available Commercial Spaces**
   - Use the Places API to iterate through potential plazas, shopping centers, and retail lots
   - Filter for spaces with appropriate square footage for a boba shop (typically 800-2000 sq ft)
   - Note lease availability, pricing, and foot traffic data when available

2. **Identify Direct Competitors**
   - Locate existing boba/bubble tea stores in the area
   - Map out cafes, coffee shops, and tea houses
   - Identify smoothie and juice bars
   - Flag areas with high competitor density as potentially saturated
   - Flag areas with zero competitors as either untapped opportunity or low-demand zones

3. **Identify Complementary Businesses** (indicators of boba demand)
   - **Asian Restaurants**: Ramen shops, pho restaurants, Korean BBQ, sushi bars, fried chicken joints
     - Note: Customers often crave cold, sweet drinks after hot, savory meals
   - **Trendy/Youth-Centric Retail**:
     - K-Pop and anime merchandise stores
     - Arcades and claw machine centers
     - Skincare and beauty boutiques
     - Streetwear and sneaker shops
   - **Study & Work Hubs**:
     - University libraries and campus buildings
     - Co-working spaces
     - Tutoring centers and cram schools

4. **Check Existing Chain Locations**
   - Identify where the client's boba chain already has stores
   - Ensure recommended locations maintain appropriate distance from existing stores
   - Avoid cannibalizing existing store traffic

## Workflow & Handoff to Quantitative Analyst

After identifying competitors and complementary businesses for a location, **hand off to the Quantitative Analyst** to analyze their performance. 

**When to hand off:**
- After you've identified competitors (boba shops, cafes, tea houses) in the area
- After you've identified complementary businesses (Asian restaurants, study areas, etc.)
- When you have a list of business names and addresses ready for analysis

**What to provide in your handoff:**
- **Location**: The address or area you're analyzing
- **Competitors**: List of competitor business names and addresses (from find_boba_competitors results)
- **Complementary Businesses**: List of complementary business names and addresses (from find_complementary_businesses results)
- **Context**: Any relevant notes about the location, competitor density, or market conditions

**Output Format (before handoff):**

For each potential location, provide:
- **Address**: Full street address
- **Competitors Found**: List of competitor businesses with names and addresses
- **Complementary Businesses Found**: List of complementary businesses with names and addresses
- **Competitor Count**: Number of direct competitors
- **Complement Count**: Number of complementary businesses
- **Demographics Indicators**: Presence of universities, youth-centric businesses
- **Initial Assessment**: Flag as "HIGH POTENTIAL", "MODERATE POTENTIAL", or "LOW POTENTIAL" based on location characteristics

**Then hand off to Quantitative Analyst** with the competitor and complement business lists for performance analysis."""

scout = create_agent(
    model=model,
    tools=scout_tools,
    system_prompt=SCOUT_SYSTEM_PROMPT,
    name="Location Scout"
)