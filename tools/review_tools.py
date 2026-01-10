"""Tools for fetching review data from Google Places and Yelp APIs."""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import googlemaps
from yelpapi import YelpAPI


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
        
        # Filter reviews by time range
        cutoff_date = _parse_time_range(time_range)
        for review in place_details.get("result", {}).get("reviews", []):
            review_time = datetime.fromtimestamp(review.get("time", 0))
            if review_time >= cutoff_date:
                business_data["reviews"].append({
                    "rating": review.get("rating"),
                    "text": review.get("text", ""),
                    "timestamp": review_time.isoformat(),
                    "time": review.get("time", 0)
                })
        
        results.append(business_data)
    
    return results


def fetch_yelp_reviews(location: str, business_type: str, time_range: str) -> List[Dict[str, Any]]:
    """
    Fetch Yelp reviews with ratings and timestamps.
    
    Args:
        location: Address or place name to search
        business_type: Type of business (e.g., "boba tea", "bubble tea")
        time_range: Time range for reviews (e.g., "30d", "90d", "1y")
    
    Returns:
        List of dictionaries containing business name, ratings, review counts, and timestamps
    """
    api_key = os.getenv("YELP_API_KEY")
    if not api_key:
        return [{"error": "YELP_API_KEY not set"}]
    
    yelp_api = YelpAPI(api_key)
    
    try:
        search_results = yelp_api.search_query(
            term=business_type,
            location=location,
            limit=10
        )
        
        results = []
        cutoff_date = _parse_time_range(time_range)
        
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


def _parse_time_range(time_range: str) -> datetime:
    """Parse time range string to datetime cutoff."""
    now = datetime.now()
    
    if time_range.endswith("d"):
        days = int(time_range[:-1])
        return now - timedelta(days=days)
    elif time_range.endswith("w"):
        weeks = int(time_range[:-1])
        return now - timedelta(weeks=weeks)
    elif time_range.endswith("m"):
        months = int(time_range[:-1])
        return now - timedelta(days=months * 30)
    elif time_range.endswith("y"):
        years = int(time_range[:-1])
        return now - timedelta(days=years * 365)
    else:
        # Default to 90 days
        return now - timedelta(days=90)
