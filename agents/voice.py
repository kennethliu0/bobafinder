import os
import re
from typing import List, Dict, Any

import googlemaps
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool

from config import model


def _scrape_google_reviews(place_name: str, location: str = "", max_reviews: int = 50) -> List[Dict[str, Any]]:
    """
    Scrape Google reviews for a given place.
    
    Args:
        place_name: Name of the place/business
        location: Location/address (optional)
        max_reviews: Maximum number of reviews to scrape
    
    Returns:
        List of review dictionaries
    """
    reviews = []
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    
    if not api_key:
        return reviews
    
    try:
        gmaps = googlemaps.Client(key=api_key)
        # Search for the place
        places_result = gmaps.places(query=f"{place_name} {location}")
        
        if places_result.get("results"):
            place_id = places_result["results"][0]["place_id"]
            
            # Get place details with reviews
            place_details = gmaps.place(
                place_id=place_id,
                fields=["reviews", "name", "rating"]
            )
            
            if "reviews" in place_details.get("result", {}):
                for review_data in place_details["result"]["reviews"][:max_reviews]:
                    author_name = review_data.get("author_name", "Anonymous")
                    reviews.append({
                        "text": review_data.get("text", ""),
                        "author": author_name,
                        "rating": review_data.get("rating", 0),
                        "date": str(review_data.get("time", 0)),
                        "is_local_guide": "Local Guide" in author_name.lower()
                    })
    except Exception as e:
        return [{"text": f"Error: {str(e)}", "author": "System", "rating": 0, "is_local_guide": False}]
    
    return reviews


def _cluster_reviews_by_sentiment(reviews: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Cluster reviews into categories with sentiment analysis.
    
    Categories: Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness
    
    Args:
        reviews: List of review dictionaries
    
    Returns:
        Dictionary mapping category names to sentiment cluster dictionaries
    """
    categories = {
        "Wait Time": [],
        "Sweetness Levels": [],
        "Pearl Texture": [],
        "Staff Friendliness": []
    }
    
    # Keywords for each category
    category_keywords = {
        "Wait Time": ["wait", "time", "fast", "slow", "quick", "minutes", "hours", "queue", "line", "busy"],
        "Sweetness Levels": ["sweet", "sugar", "sweetness", "too sweet", "not sweet", "perfectly sweet", "bland"],
        "Pearl Texture": ["pearl", "boba", "tapioca", "chewy", "soft", "hard", "texture", "q", "perfect"],
        "Staff Friendliness": ["staff", "service", "friendly", "rude", "nice", "helpful", "attitude", "customer service"]
    }
    
    # Classify reviews into categories
    for review in reviews:
        review_lower = review.get("text", "").lower()
        for category, keywords in category_keywords.items():
            if any(keyword in review_lower for keyword in keywords):
                categories[category].append(review)
    
    # Analyze sentiment for each category
    clusters = {}
    for category, category_reviews in categories.items():
        positive = 0
        negative = 0
        neutral = 0
        sample_texts = []
        
        for review in category_reviews[:5]:  # Sample first 5
            sample_texts.append(review.get("text", "")[:200])
            rating = review.get("rating", 0)
            if rating >= 4:
                positive += 1
            elif rating <= 2:
                negative += 1
            else:
                neutral += 1
        
        clusters[category] = {
            "category": category,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "sample_reviews": sample_texts
        }
    
    return clusters


def _extract_pain_points(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract pain points from reviews, specifically looking for "I wish" or "They don't have" statements.
    
    Args:
        reviews: List of review dictionaries
    
    Returns:
        List of pain point dictionaries
    """
    pain_points = []
    
    # Patterns to match
    wish_pattern = r"i wish[^.]*\.?"
    dont_have_pattern = r"they don't have[^.]*\.?|they do not have[^.]*\.?|it doesn't have[^.]*\.?"
    
    category_keywords = {
        "Wait Time": ["wait", "time", "slow", "queue"],
        "Sweetness Levels": ["sweet", "sugar"],
        "Pearl Texture": ["pearl", "boba", "texture"],
        "Staff Friendliness": ["staff", "service", "friendly"],
        "Menu": ["menu", "options", "variety", "flavors"],
        "Price": ["price", "expensive", "cheap", "cost"]
    }
    
    for review in reviews:
        text_lower = review.get("text", "").lower()
        
        # Find "I wish" statements
        wish_matches = re.findall(wish_pattern, text_lower, re.IGNORECASE)
        for match in wish_matches:
            category = None
            for cat, keywords in category_keywords.items():
                if any(keyword in match for keyword in keywords):
                    category = cat
                    break
            
            pain_points.append({
                "text": match.strip(),
                "category": category or "General",
                "review_source": review.get("author", "")
            })
        
        # Find "They don't have" statements
        dont_have_matches = re.findall(dont_have_pattern, text_lower, re.IGNORECASE)
        for match in dont_have_matches:
            category = None
            for cat, keywords in category_keywords.items():
                if any(keyword in match for keyword in keywords):
                    category = cat
                    break
            
            pain_points.append({
                "text": match.strip(),
                "category": category or "General",
                "review_source": review.get("author", "")
            })
    
    return pain_points


def _calculate_brand_loyalty_score(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate brand loyalty score based on Local Guides vs one-time tourists.
    
    Args:
        reviews: List of review dictionaries
    
    Returns:
        Dictionary with brand loyalty metrics
    """
    total_reviews = len(reviews)
    local_guide_count = sum(1 for r in reviews if r.get("is_local_guide", False))
    one_time_tourist_count = total_reviews - local_guide_count
    
    local_guide_percentage = (local_guide_count / total_reviews * 100) if total_reviews > 0 else 0
    
    # Heuristic: If >30% are Local Guides, it supports a "Regulars" business model
    supports_regulars_model = local_guide_percentage >= 30
    
    # Score calculation: higher percentage of local guides = higher score
    # Also consider repeat reviewers (multiple reviews from same author)
    authors = [r.get("author", "") for r in reviews]
    unique_authors = len(set(authors))
    repeat_customer_ratio = (total_reviews - unique_authors) / total_reviews if total_reviews > 0 else 0
    
    # Combined score: 60% local guides, 40% repeat customers
    score = (local_guide_percentage * 0.6) + (repeat_customer_ratio * 100 * 0.4)
    
    return {
        "total_reviews": total_reviews,
        "local_guide_count": local_guide_count,
        "one_time_tourist_count": one_time_tourist_count,
        "local_guide_percentage": round(local_guide_percentage, 2),
        "supports_regulars_model": supports_regulars_model,
        "score": round(score, 2)
    }


@tool
def scrape_reviews(place_name: str, location: str = "", max_reviews: int = 50) -> dict:
    """
    Scrape Google reviews for a boba shop or competitor.
    
    Args:
        place_name: Name of the boba shop/competitor
        location: Location/address (optional but recommended)
        max_reviews: Maximum number of reviews to scrape (default: 50)
    
    Returns:
        Dictionary with review data
    """
    reviews = _scrape_google_reviews(place_name, location, max_reviews)
    
    if not reviews or (len(reviews) == 1 and reviews[0].get("text", "").startswith("Error:")):
        return {"error": reviews[0].get("text", "") if reviews else "No reviews found"}
    
    return {
        "place_name": place_name,
        "location": location,
        "total_reviews": len(reviews),
        "reviews": reviews[:10]  # Return first 10 for brevity
    }


@tool
def analyze_sentiment_clusters(reviews_data: dict) -> dict:
    """
    Cluster reviews by sentiment into categories: Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness.
    
    Args:
        reviews_data: Dictionary containing reviews (from scrape_reviews tool)
    
    Returns:
        Dictionary with sentiment clusters
    """
    if "error" in reviews_data:
        return {"error": reviews_data["error"]}
    
    try:
        reviews_list = reviews_data.get("reviews", [])
        clusters = _cluster_reviews_by_sentiment(reviews_list)
        
        return {
            "categories": {
                cat: {
                    "positive_count": cluster["positive_count"],
                    "negative_count": cluster["negative_count"],
                    "neutral_count": cluster["neutral_count"],
                    "sample_reviews": cluster["sample_reviews"]
                }
                for cat, cluster in clusters.items()
            }
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def extract_pain_points(reviews_data: dict) -> dict:
    """
    Extract pain points from reviews, specifically "I wish" or "They don't have" statements.
    
    Args:
        reviews_data: Dictionary containing reviews (from scrape_reviews tool)
    
    Returns:
        Dictionary with extracted pain points
    """
    if "error" in reviews_data:
        return {"error": reviews_data["error"]}
    
    try:
        reviews_list = reviews_data.get("reviews", [])
        pain_points = _extract_pain_points(reviews_list)
        
        return {
            "total_pain_points": len(pain_points),
            "pain_points": pain_points
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def calculate_loyalty_score(reviews_data: dict) -> dict:
    """
    Calculate brand loyalty score based on Local Guides vs one-time tourists.
    Determines if the area supports a "Regulars" business model.
    
    Args:
        reviews_data: Dictionary containing reviews (from scrape_reviews tool)
    
    Returns:
        Dictionary with brand loyalty metrics
    """
    if "error" in reviews_data:
        return {"error": reviews_data["error"]}
    
    try:
        reviews_list = reviews_data.get("reviews", [])
        loyalty_score = _calculate_brand_loyalty_score(reviews_list)
        
        return {
            **loyalty_score,
            "interpretation": "High potential for regular customers" if loyalty_score["supports_regulars_model"] else "More tourist-driven market"
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def analyze_competitor_reviews(place_name: str, location: str = "", max_reviews: int = 50) -> dict:
    """
    Complete analysis of competitor reviews: scraping, sentiment clustering, pain points, and loyalty score.
    
    Args:
        place_name: Name of the competitor boba shop
        location: Location/address (optional but recommended)
        max_reviews: Maximum number of reviews to analyze (default: 50)
    
    Returns:
        Dictionary with complete analysis
    """
    # Scrape reviews
    reviews = _scrape_google_reviews(place_name, location, max_reviews)
    
    if not reviews or (len(reviews) == 1 and reviews[0].get("text", "").startswith("Error:")):
        return {"error": reviews[0].get("text", "") if reviews else "No reviews found. Make sure to provide a valid place name and location."}
    
    # Perform all analyses
    clusters = _cluster_reviews_by_sentiment(reviews)
    pain_points = _extract_pain_points(reviews)
    loyalty_score = _calculate_brand_loyalty_score(reviews)
    
    return {
        "place_name": place_name,
        "location": location,
        "total_reviews_analyzed": len(reviews),
        "sentiment_clusters": {
            cat: {
                "positive_count": cluster["positive_count"],
                "negative_count": cluster["negative_count"],
                "neutral_count": cluster["neutral_count"],
                "sample_reviews": cluster["sample_reviews"][:3]  # Top 3 samples
            }
            for cat, cluster in clusters.items()
        },
        "pain_points": {
            "total_count": len(pain_points),
            "by_category": {
                cat: [pp["text"] for pp in pain_points if pp.get("category") == cat]
                for cat in set(pp.get("category") for pp in pain_points if pp.get("category"))
            },
            "all_pain_points": [
                {"text": pp["text"], "category": pp.get("category")}
                for pp in pain_points[:10]  # Top 10
            ]
        },
        "brand_loyalty": {
            **loyalty_score,
            "recommendation": "Area supports a 'Regulars' business model - focus on building repeat customers" 
                            if loyalty_score["supports_regulars_model"] 
                            else "Area is more tourist-driven - focus on first impressions and marketing"
        }
    }


voice_tools = [
    scrape_reviews,
    analyze_sentiment_clusters,
    extract_pain_points,
    calculate_loyalty_score,
    analyze_competitor_reviews,
    create_handoff_tool(
        agent_name="Location Scout",
        description="Transfer back to Location Scout with customer voice findings (pain points, sentiment, loyalty) to compile the final plaza analysis",
    ),
    create_handoff_tool(
        agent_name="Niche Finder",
        description="Transfer to Niche Finder if additional niche analysis is needed for competitors",
    ),
    create_handoff_tool(
        agent_name="Quantitative Analyst",
        description="Transfer to Quantitative Analyst to analyze competitor performance metrics and review trends",
    ),
]

VOICE_SYSTEM_PROMPT = """You are the Voice of Customer Agent, specializing in analyzing competitor reviews 
for boba tea shops. Your expertise includes:

## Your Primary Objectives

1. **Scraping Google Reviews**: Extract reviews from Google for any boba shop or competitor
   - Use scrape_reviews tool to gather review data for a specific business
   - Provide place name and location for best results
   - Default to 50 reviews unless user specifies otherwise

2. **Sentiment Clustering**: Group reviews into categories and analyze sentiment for each category
   - Categories: Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness
   - Use analyze_sentiment_clusters tool to categorize and analyze sentiment
   - Identify which aspects customers love vs. complain about

3. **Pain Point Extraction**: Identify "I wish" and "They don't have" statements to uncover customer frustrations
   - Use extract_pain_points tool to find specific customer pain points
   - Categorize pain points: Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness, Menu, Price
   - Highlight opportunities for improvement based on customer feedback

4. **Brand Loyalty Analysis**: Calculate loyalty scores based on Local Guides vs tourists to determine if an area 
   supports a "Regulars" business model
   - Use calculate_loyalty_score tool to analyze customer loyalty patterns
   - Determine if area supports repeat customer business model (Local Guides >30%)
   - Provide recommendations based on customer base characteristics

## Workflow & Tool Usage

**For complete analysis:**
- Use analyze_competitor_reviews tool for one-shot complete analysis (scraping + sentiment + pain points + loyalty)

**For step-by-step analysis:**
- Step 1: Use scrape_reviews to gather review data
- Step 2: Use analyze_sentiment_clusters with the review data
- Step 3: Use extract_pain_points with the review data
- Step 4: Use calculate_loyalty_score with the review data

**Output Format:**
- Provide clear, actionable insights from the review data
- Highlight key pain points and opportunities
- Summarize sentiment by category
- Include loyalty score interpretation and business model recommendations
- Focus on helping identify market opportunities and customer pain points that could inform business strategy

## Handoff Instructions - CRITICAL

After completing your analysis of competitor reviews:

**ALWAYS call `transfer_to_Location_Scout`** with:
- Your complete sentiment analysis findings
- Key pain points discovered (what customers wish for)
- Loyalty score and business model recommendation
- Opportunities for the user's boba shop to differentiate

Location Scout will compile your findings with the niche analysis to create the final plaza assessment.

You MUST hand off after completing your analysis - do not end the conversation."""

voice = create_agent(
    model=model,
    tools=voice_tools,
    system_prompt=VOICE_SYSTEM_PROMPT,
    name="Voice of Customer"
)
