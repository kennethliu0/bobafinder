"""Tools for quantitative analysis of competitor health, trends, and complements."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd


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
        
        # Calculate trend metrics
        trend_metrics = calculate_trend_metrics(ratings, timestamps)
        
        # Calculate review frequency
        review_frequency = len(reviews) / _parse_time_period_days(time_period) * 30  # Reviews per month
        
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


def identify_direct_competitors(location: str, radius: float) -> List[Dict[str, Any]]:
    """
    Find other boba shops in the area.
    
    Args:
        location: Address or place name
        radius: Search radius in meters
    
    Returns:
        List of competitor businesses with their details
    """
    # This is a placeholder - actual implementation would use Google Places API
    # with specific search for boba/bubble tea shops
    return [
        {
            "business_name": "Competitor Placeholder",
            "location": location,
            "radius_meters": radius,
            "note": "Use Google Places API with type='cafe' and keyword='boba' or 'bubble tea'"
        }
    ]


def identify_complements(location: str, categories: List[str]) -> List[Dict[str, Any]]:
    """
    Find complement businesses (coffee shops, dessert places, Asian restaurants, etc.).
    
    Args:
        location: Address or place name
        categories: List of complement categories (e.g., ["coffee", "dessert", "asian_restaurant"])
    
    Returns:
        List of complement businesses with their details
    """
    # This is a placeholder - actual implementation would use Google Places API
    # with searches for each category
    return [
        {
            "category": category,
            "location": location,
            "businesses": [],
            "note": f"Use Google Places API to search for {category} near {location}"
        }
        for category in categories
    ]


def calculate_trend_metrics(ratings: List[float], timestamps: List[str]) -> Dict[str, Any]:
    """
    Compute slope, volatility, and trend direction from ratings over time.
    
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


def _parse_time_period_days(time_period: str) -> float:
    """Parse time period string to number of days."""
    if time_period.endswith("d"):
        return float(time_period[:-1])
    elif time_period.endswith("w"):
        return float(time_period[:-1]) * 7
    elif time_period.endswith("m"):
        return float(time_period[:-1]) * 30
    elif time_period.endswith("y"):
        return float(time_period[:-1]) * 365
    else:
        return 90.0  # Default to 90 days
