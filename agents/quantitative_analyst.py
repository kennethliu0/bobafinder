import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool
import googlemaps
from yelpapi import YelpAPI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from config import model


def _geocode_address(address: str) -> Optional[Dict[str, float]]:
    """
    Helper function to geocode an address string to latitude/longitude coordinates.
    
    Args:
        address: Address string to geocode (e.g., "17 Powell St, San Francisco, CA")
    
    Returns:
        Dictionary with 'latitude' and 'longitude' keys, or None if geocoding fails
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return None
    
    # Strip quotes if present
    api_key = api_key.strip().strip('"').strip("'")
    
    if not api_key:
        return None
    
    try:
        gmaps = googlemaps.Client(key=api_key)
        geocode_result = gmaps.geocode(address)
        
        if geocode_result and len(geocode_result) > 0:
            location = geocode_result[0].get("geometry", {}).get("location", {})
            lat = location.get("lat")
            lng = location.get("lng")
            
            if lat is not None and lng is not None:
                return {"latitude": float(lat), "longitude": float(lng)}
    except Exception as e:
        # Silently fail - will fall back to using address string
        pass
    
    return None


@tool
def fetch_google_reviews(location: str, business_type: str, time_range: str) -> List[Dict[str, Any]]:
    """
    Fetch Google Places reviews with ratings and timestamps.
    
    Args:
        location: Address or place name to search
        business_type: Type of business (e.g., "boba tea", "bubble tea")
        time_range: Time range for reviews (e.g., "30d", "90d", "1y")
    
    Returns:
        List of dictionaries containing business name, ratings, review counts, and timestamps
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return [{"error": "GOOGLE_PLACES_API_KEY not set"}]
    
    # Strip quotes if present (common .env file issue)
    api_key = api_key.strip().strip('"').strip("'")
    
    if not api_key:
        return [{"error": "GOOGLE_PLACES_API_KEY is empty"}]
    
    gmaps = googlemaps.Client(key=api_key)
    
    # Search for places
    places_result = gmaps.places(query=f"{business_type} near {location}")
    
    results = []
    for place in places_result.get("results", [])[:10]:  # Limit to top 10 results
        place_id = place.get("place_id")
        if not place_id:
            continue
        
        # Get place details including reviews
        place_details = gmaps.place(place_id=place_id, fields=["name", "rating", "user_ratings_total", "reviews"])
        
        business_data = {
            "business_name": place_details.get("result", {}).get("name"),
            "rating": place_details.get("result", {}).get("rating"),
            "review_count": place_details.get("result", {}).get("user_ratings_total", 0),
            "reviews": []
        }
        
        # Get all reviews (time_range parameter kept for API compatibility but not used for filtering)
        for review in place_details.get("result", {}).get("reviews", []):
            review_time = datetime.fromtimestamp(review.get("time", 0))
            business_data["reviews"].append({
                "rating": review.get("rating"),
                "text": review.get("text", ""),
                "timestamp": review_time.isoformat(),
                "time": review.get("time", 0)
            })
        
        results.append(business_data)
    
    return results


@tool
def fetch_yelp_reviews(location: str, business_type: str, time_range: str = "90d", latitude: Optional[float] = None, longitude: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Fetch Yelp reviews with ratings and timestamps.
    
    This function can accept either:
    - An address string (will be geocoded to lat/lon automatically)
    - Latitude and longitude coordinates directly (via latitude/longitude parameters)
    
    Args:
        location: Address string (e.g., "17 Powell St, San Francisco, CA"). Ignored if latitude/longitude are provided.
        business_type: Type of business (e.g., "boba tea", "bubble tea")
        time_range: Time range for reviews (e.g., "30d", "90d", "1y") - not used by Yelp API but kept for compatibility
        latitude: Optional latitude coordinate (if provided along with longitude, location will be ignored and coordinates used instead)
        longitude: Optional longitude coordinate (if provided along with latitude, location will be ignored and coordinates used instead)
    
    Returns:
        List of dictionaries containing business name, ratings, review counts, and timestamps
    """
    api_key = os.getenv("YELP_API_KEY")
    if not api_key:
        return [{"error": "YELP_API_KEY not set"}]
    
    # Strip quotes if present (common .env file issue)
    api_key = api_key.strip().strip('"').strip("'")
    
    if not api_key or api_key.startswith("-") or len(api_key) < 10:
        return [{"error": "YELP_API_KEY appears to be invalid or incomplete"}]
    
    yelp_api = YelpAPI(api_key)
    
    # Determine if we have coordinates or need to geocode
    lat = None
    lng = None
    
    # If latitude and longitude are explicitly provided, use them (preferred method)
    if latitude is not None and longitude is not None:
        lat = float(latitude)
        lng = float(longitude)
    # Otherwise, geocode the address string to get coordinates
    elif location:
        geocode_result = _geocode_address(location)
        if geocode_result:
            lat = geocode_result["latitude"]
            lng = geocode_result["longitude"]
        # If geocoding fails, we'll fall back to using address string directly
    
    try:
        # Use latitude/longitude if available (preferred method)
        if lat is not None and lng is not None:
            # Validate coordinates
            if not (-90 <= lat <= 90):
                return [{"error": f"latitude must be between -90 and 90, got {lat}"}]
            if not (-180 <= lng <= 180):
                return [{"error": f"longitude must be between -180 and 180, got {lng}"}]
            
            search_results = yelp_api.search_query(
                term=business_type,
                latitude=lat,
                longitude=lng,
                limit=10
            )
        else:
            # Fallback to address string (may not work as well)
            if not isinstance(location, str):
                return [{"error": "location must be an address string if latitude/longitude are not provided"}]
            
            search_results = yelp_api.search_query(
                term=business_type,
                location=location,
                limit=10
            )
        
        results = []
        
        for business in search_results.get("businesses", []):
            business_id = business.get("id")
            if not business_id:
                continue
            
            # Get business details and reviews
            business_details = yelp_api.business_query(id=business_id)
            
            business_data = {
                "business_name": business_details.get("name"),
                "rating": business_details.get("rating"),
                "review_count": business_details.get("review_count", 0),
                "reviews": []
            }
            
            # Yelp API doesn't provide reviews in search results, so we'll use the rating and count
            # For full review text, would need Yelp Fusion API with reviews endpoint (if available)
            business_data["reviews"] = [{
                "rating": business.get("rating"),
                "timestamp": datetime.now().isoformat(),  # Placeholder - Yelp API limitations
            }]
            
            results.append(business_data)
    
    except Exception as e:
        return [{"error": f"Yelp API error: {str(e)}"}]
    
    return results


@tool
def analyze_competitor_health(business_data: List[Dict[str, Any]], time_period: str) -> Dict[str, Any]:
    """
    Calculate trends in ratings and review frequency over time.
    
    Args:
        business_data: List of business dictionaries with reviews
        time_period: Time period for analysis (e.g., "30d", "90d", "1y")
    
    Returns:
        Dictionary with health metrics including trend direction, volatility, and performance indicators
    """
    if not business_data:
        return {"error": "No business data provided"}
    
    results = []
    
    for business in business_data:
        if "error" in business:
            continue
        
        reviews = business.get("reviews", [])
        if not reviews:
            continue
        
        # Extract ratings and timestamps
        ratings = [r.get("rating", 0) for r in reviews]
        timestamps = [r.get("timestamp") for r in reviews if r.get("timestamp")]
        
        if len(ratings) < 2:
            continue
        
        # Calculate trend metrics using internal helper (not the tool wrapper)
        trend_metrics = _calculate_trend_metrics_internal(ratings, timestamps)
        
        # Calculate review frequency (simplified - assume time_period represents days)
        try:
            days = float(time_period.rstrip('dwmy')) if time_period[-1] in 'dwmy' else 90.0
            if time_period.endswith('w'):
                days *= 7
            elif time_period.endswith('m'):
                days *= 30
            elif time_period.endswith('y'):
                days *= 365
            review_frequency = len(reviews) / days * 30  # Reviews per month
        except:
            review_frequency = len(reviews) / 90.0 * 30  # Default to 90 days
        
        # Determine health status
        avg_rating = np.mean(ratings)
        rating_trend = trend_metrics.get("slope", 0)
        
        if avg_rating >= 4.5 and rating_trend >= 0:
            health_status = "strong"
        elif avg_rating >= 4.0 and rating_trend >= -0.1:
            health_status = "moderate"
        else:
            health_status = "weak"
        
        results.append({
            "business_name": business.get("business_name"),
            "average_rating": float(avg_rating),
            "rating_trend": float(rating_trend),
            "volatility": trend_metrics.get("volatility", 0),
            "review_frequency_per_month": float(review_frequency),
            "total_reviews": len(reviews),
            "health_status": health_status,
            "trend_direction": trend_metrics.get("trend_direction", "stable")
        })
    
    return {
        "competitors_analyzed": len(results),
        "analysis": results,
        "summary": {
            "strong_performers": len([r for r in results if r["health_status"] == "strong"]),
            "moderate_performers": len([r for r in results if r["health_status"] == "moderate"]),
            "weak_performers": len([r for r in results if r["health_status"] == "weak"])
        }
    }


def _calculate_trend_metrics_internal(ratings: List[float], timestamps: List[str]) -> Dict[str, Any]:
    """
    Internal helper function to compute slope, volatility, and trend direction from ratings over time.
    This can be called directly by other functions without going through the tool wrapper.
    
    Args:
        ratings: List of rating values
        timestamps: List of ISO format timestamp strings
    
    Returns:
        Dictionary with slope, volatility, and trend_direction
    """
    if len(ratings) < 2:
        return {
            "slope": 0.0,
            "volatility": 0.0,
            "trend_direction": "insufficient_data"
        }
    
    # Convert timestamps to numeric values (days since first review)
    try:
        parsed_times = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        first_time = min(parsed_times)
        time_deltas = [(t - first_time).days for t in parsed_times]
    except Exception:
        # Fallback to sequential indices if timestamp parsing fails
        time_deltas = list(range(len(ratings)))
    
    # Calculate linear regression slope
    if len(set(time_deltas)) > 1:
        slope = np.polyfit(time_deltas, ratings, 1)[0]
    else:
        slope = 0.0
    
    # Calculate volatility (standard deviation)
    volatility = float(np.std(ratings))
    
    # Determine trend direction
    if slope > 0.01:
        trend_direction = "improving"
    elif slope < -0.01:
        trend_direction = "declining"
    else:
        trend_direction = "stable"
    
    return {
        "slope": float(slope),
        "volatility": volatility,
        "trend_direction": trend_direction
    }


@tool
def calculate_trend_metrics(ratings: List[float], timestamps: List[str]) -> Dict[str, Any]:
    """
    Compute slope, volatility, and trend direction from ratings over time.
    
    Args:
        ratings: List of rating values
        timestamps: List of ISO format timestamp strings
    
    Returns:
        Dictionary with slope, volatility, and trend_direction
    """
    return _calculate_trend_metrics_internal(ratings, timestamps)


quantitative_analyst_tools = [
    fetch_google_reviews,
    fetch_yelp_reviews,
    analyze_competitor_health,
    calculate_trend_metrics,
    create_handoff_tool(
        agent_name="Location Scout",
        description="Transfer back to Location Scout with demand indicator findings after completing analysis",
    ),
]

QUANTITATIVE_ANALYST_SYSTEM_PROMPT = """You are a Quantitative Analyst specializing in competitive market analysis for boba tea businesses.

## Your Role

You receive competitor and complementary business data from the Location Scout agent and perform quantitative performance analysis.

## What You Receive from Location Scout

When Location Scout hands off to you, they will provide:
- **Location**: The address or area being analyzed (may include addresses or coordinates)
- **Competitors**: List of competitor businesses with their location data
- **Complementary Businesses**: List of complementary businesses with their location data
- **Context**: Notes about competitor density, market conditions, and location characteristics

**Location Data Formats**: Location Scout may provide data in different formats:
- **Address strings**: "17 Powell St, San Francisco, CA"
- **Coordinates**: `{"location": {"latitude": 37.7749, "longitude": -122.4194}}`
- **Business addresses**: Full street addresses in the business data

## Your Analysis Process

1. **Gather Review Data**
   - Use `fetch_google_reviews` with business names/addresses and business type (e.g., "boba tea", "cafe")
   - Use `fetch_yelp_reviews` with location data - the function is flexible and can handle:
     - **Address strings**: `fetch_yelp_reviews(location="17 Powell St, San Francisco, CA", business_type="boba tea")`
       - The function will automatically geocode the address to coordinates internally using a helper method
       - You can pass any address string from Location Scout's data
     - **Coordinates** (preferred for accuracy): `fetch_yelp_reviews(latitude=37.7749, longitude=-122.4194, business_type="boba tea", location="")`
       - If Location Scout provides coordinates in location objects, extract and use them directly
       - This is more accurate than geocoding addresses
     - **Automatic geocoding**: If you only have addresses, the function will geocode them automatically - no need to manually convert addresses to coordinates
   - **Flexibility**: You can use either addresses OR coordinates - choose based on what Location Scout provides. Coordinates are preferred when available.
   - Collect ratings, review counts, and review timestamps

2. **Analyze Competitor Health**
   - Use `analyze_competitor_health` to assess each competitor's performance over time
   - Calculate average ratings, rating trends, review frequency, and health status (strong/moderate/weak)

3. **Calculate Trend Metrics**
   - Use `calculate_trend_metrics` for detailed trend analysis on ratings and review patterns
   - Identify improving, declining, or stable trends

4. **Provide Insights**
   - Summarize competitor performance (strong vs weak competitors)
   - Assess market saturation based on competitor health
   - Evaluate complement business health as demand indicators
   - Provide actionable recommendations

## Output Format

Provide structured analysis with:
- **Complement Analysis**: Health of complementary businesses (indicates demand)
- **Demand Indicator Score**: HIGH / MODERATE / LOW based on business health
- **Summary**: Whether the area shows strong local demand

## Handoff Instructions - CRITICAL

After completing your analysis of complementary businesses:

Call `transfer_to_location_scout` with your complete findings:
- Health status for each business analyzed
- Rating trends and review frequencies
- Overall demand indicator score (HIGH/MODERATE/LOW)
- Summary statistics

You MUST transfer back to Location Scout after completing your analysis."""

quantitative_analyst = create_agent(
    model=model,
    tools=quantitative_analyst_tools,
    system_prompt=QUANTITATIVE_ANALYST_SYSTEM_PROMPT,
    name="Quantitative Analyst"
)
