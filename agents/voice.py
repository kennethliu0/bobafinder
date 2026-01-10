"""
Voice of Customer Agent - NLP on reviews, sentiment clustering, pain point extraction, and brand loyalty scoring.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import googlemaps
from langchain_fireworks import ChatFireworks
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatFireworks(
    model="accounts/fireworks/models/minimax-m2p1",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize Google Maps client if API key is available
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
gmaps_client = googlemaps.Client(key=GOOGLE_PLACES_API_KEY) if GOOGLE_PLACES_API_KEY else None


@dataclass
class Review:
    """Represents a single review."""
    text: str
    author: str
    rating: int
    date: Optional[str] = None
    is_local_guide: bool = False


@dataclass
class SentimentCluster:
    """Represents a sentiment cluster for a specific category."""
    category: str
    positive_count: int
    negative_count: int
    neutral_count: int
    sample_reviews: List[str]


@dataclass
class PainPoint:
    """Represents an extracted pain point."""
    text: str
    category: Optional[str] = None
    review_source: Optional[str] = None


@dataclass
class BrandLoyaltyScore:
    """Represents brand loyalty metrics."""
    total_reviews: int
    local_guide_count: int
    one_time_tourist_count: int
    local_guide_percentage: float
    supports_regulars_model: bool
    score: float  # 0-100


def scrape_google_reviews(place_name: str, location: str = "", max_reviews: int = 50) -> List[Review]:
    """
    Scrape Google reviews for a given place.
    
    Args:
        place_name: Name of the place/business
        location: Location/address (optional)
        max_reviews: Maximum number of reviews to scrape
    
    Returns:
        List of Review objects
    """
    reviews = []
    
    # If Google Maps API is available, use it
    if gmaps_client:
        try:
            # Search for the place
            places_result = gmaps_client.places(query=f"{place_name} {location}")
            
            if places_result.get("results"):
                place_id = places_result["results"][0]["place_id"]
                
                # Get place details with reviews
                place_details = gmaps_client.place(
                    place_id=place_id,
                    fields=["reviews", "name", "rating"]
                )
                
                if "reviews" in place_details.get("result", {}):
                    for review_data in place_details["result"]["reviews"][:max_reviews]:
                        review = Review(
                            text=review_data.get("text", ""),
                            author=review_data.get("author_name", "Anonymous"),
                            rating=review_data.get("rating", 0),
                            date=review_data.get("time", 0),
                            is_local_guide="Local Guide" in review_data.get("author_name", "").lower()
                        )
                        reviews.append(review)
        except Exception as e:
            print(f"Error using Google Maps API: {e}")
    
    # Fallback: Use web scraping (limited functionality)
    if not reviews:
        try:
            # This is a simplified scraper - in production, you'd want more sophisticated scraping
            search_query = f"{place_name} {location} google reviews"
            url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                # Note: Google's HTML structure changes frequently, so this is a basic implementation
                # In production, use the Places API or a dedicated scraping service
                pass
        except Exception as e:
            print(f"Error scraping reviews: {e}")
    
    return reviews


def cluster_reviews_by_sentiment(reviews: List[Review]) -> Dict[str, SentimentCluster]:
    """
    Cluster reviews into categories with sentiment analysis.
    
    Categories: Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness
    
    Args:
        reviews: List of Review objects
    
    Returns:
        Dictionary mapping category names to SentimentCluster objects
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
        review_lower = review.text.lower()
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
            sample_texts.append(review.text[:200])
            if review.rating >= 4:
                positive += 1
            elif review.rating <= 2:
                negative += 1
            else:
                neutral += 1
        
        clusters[category] = SentimentCluster(
            category=category,
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            sample_reviews=sample_texts
        )
    
    return clusters


def extract_pain_points(reviews: List[Review]) -> List[PainPoint]:
    """
    Extract pain points from reviews, specifically looking for "I wish" or "They don't have" statements.
    
    Args:
        reviews: List of Review objects
    
    Returns:
        List of PainPoint objects
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
        text_lower = review.text.lower()
        
        # Find "I wish" statements
        wish_matches = re.findall(wish_pattern, text_lower, re.IGNORECASE)
        for match in wish_matches:
            category = None
            for cat, keywords in category_keywords.items():
                if any(keyword in match for keyword in keywords):
                    category = cat
                    break
            
            pain_points.append(PainPoint(
                text=match.strip(),
                category=category or "General",
                review_source=review.author
            ))
        
        # Find "They don't have" statements
        dont_have_matches = re.findall(dont_have_pattern, text_lower, re.IGNORECASE)
        for match in dont_have_matches:
            category = None
            for cat, keywords in category_keywords.items():
                if any(keyword in match for keyword in keywords):
                    category = cat
                    break
            
            pain_points.append(PainPoint(
                text=match.strip(),
                category=category or "General",
                review_source=review.author
            ))
    
    return pain_points


def calculate_brand_loyalty_score(reviews: List[Review]) -> BrandLoyaltyScore:
    """
    Calculate brand loyalty score based on Local Guides vs one-time tourists.
    
    Args:
        reviews: List of Review objects
    
    Returns:
        BrandLoyaltyScore object
    """
    total_reviews = len(reviews)
    local_guide_count = sum(1 for r in reviews if r.is_local_guide)
    one_time_tourist_count = total_reviews - local_guide_count
    
    local_guide_percentage = (local_guide_count / total_reviews * 100) if total_reviews > 0 else 0
    
    # Heuristic: If >30% are Local Guides, it supports a "Regulars" business model
    supports_regulars_model = local_guide_percentage >= 30
    
    # Score calculation: higher percentage of local guides = higher score
    # Also consider repeat reviewers (multiple reviews from same author)
    authors = [r.author for r in reviews]
    unique_authors = len(set(authors))
    repeat_customer_ratio = (total_reviews - unique_authors) / total_reviews if total_reviews > 0 else 0
    
    # Combined score: 60% local guides, 40% repeat customers
    score = (local_guide_percentage * 0.6) + (repeat_customer_ratio * 100 * 0.4)
    
    return BrandLoyaltyScore(
        total_reviews=total_reviews,
        local_guide_count=local_guide_count,
        one_time_tourist_count=one_time_tourist_count,
        local_guide_percentage=round(local_guide_percentage, 2),
        supports_regulars_model=supports_regulars_model,
        score=round(score, 2)
    )


# Tool functions for LangChain agent
def scrape_reviews_tool(place_name: str, location: str = "", max_reviews: int = 50) -> str:
    """
    Scrape Google reviews for a boba shop or competitor.
    
    Args:
        place_name: Name of the boba shop/competitor
        location: Location/address (optional but recommended)
        max_reviews: Maximum number of reviews to scrape (default: 50)
    
    Returns:
        JSON string with review data
    """
    reviews = scrape_google_reviews(place_name, location, max_reviews)
    
    result = {
        "place_name": place_name,
        "location": location,
        "total_reviews": len(reviews),
        "reviews": [
            {
                "text": r.text,
                "author": r.author,
                "rating": r.rating,
                "is_local_guide": r.is_local_guide
            }
            for r in reviews[:10]  # Return first 10 for brevity
        ]
    }
    
    return json.dumps(result, indent=2)


def analyze_sentiment_clusters_tool(reviews_json: str) -> str:
    """
    Cluster reviews by sentiment into categories: Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness.
    
    Args:
        reviews_json: JSON string containing reviews (from scrape_reviews_tool)
    
    Returns:
        JSON string with sentiment clusters
    """
    try:
        data = json.loads(reviews_json)
        reviews_data = data.get("reviews", [])
        
        reviews = [
            Review(
                text=r.get("text", ""),
                author=r.get("author", ""),
                rating=r.get("rating", 0),
                is_local_guide=r.get("is_local_guide", False)
            )
            for r in reviews_data
        ]
        
        clusters = cluster_reviews_by_sentiment(reviews)
        
        result = {
            "categories": {
                cat: {
                    "positive_count": cluster.positive_count,
                    "negative_count": cluster.negative_count,
                    "neutral_count": cluster.neutral_count,
                    "sample_reviews": cluster.sample_reviews
                }
                for cat, cluster in clusters.items()
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def extract_pain_points_tool(reviews_json: str) -> str:
    """
    Extract pain points from reviews, specifically "I wish" or "They don't have" statements.
    
    Args:
        reviews_json: JSON string containing reviews (from scrape_reviews_tool)
    
    Returns:
        JSON string with extracted pain points
    """
    try:
        data = json.loads(reviews_json)
        reviews_data = data.get("reviews", [])
        
        reviews = [
            Review(
                text=r.get("text", ""),
                author=r.get("author", ""),
                rating=r.get("rating", 0),
                is_local_guide=r.get("is_local_guide", False)
            )
            for r in reviews_data
        ]
        
        pain_points = extract_pain_points(reviews)
        
        result = {
            "total_pain_points": len(pain_points),
            "pain_points": [
                {
                    "text": pp.text,
                    "category": pp.category,
                    "review_source": pp.review_source
                }
                for pp in pain_points
            ]
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def calculate_loyalty_score_tool(reviews_json: str) -> str:
    """
    Calculate brand loyalty score based on Local Guides vs one-time tourists.
    Determines if the area supports a "Regulars" business model.
    
    Args:
        reviews_json: JSON string containing reviews (from scrape_reviews_tool)
    
    Returns:
        JSON string with brand loyalty metrics
    """
    try:
        data = json.loads(reviews_json)
        reviews_data = data.get("reviews", [])
        
        reviews = [
            Review(
                text=r.get("text", ""),
                author=r.get("author", ""),
                rating=r.get("rating", 0),
                is_local_guide=r.get("is_local_guide", False)
            )
            for r in reviews_data
        ]
        
        loyalty_score = calculate_brand_loyalty_score(reviews)
        
        result = {
            "total_reviews": loyalty_score.total_reviews,
            "local_guide_count": loyalty_score.local_guide_count,
            "one_time_tourist_count": loyalty_score.one_time_tourist_count,
            "local_guide_percentage": loyalty_score.local_guide_percentage,
            "supports_regulars_model": loyalty_score.supports_regulars_model,
            "loyalty_score": loyalty_score.score,
            "interpretation": "High potential for regular customers" if loyalty_score.supports_regulars_model else "More tourist-driven market"
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_competitor_reviews_tool(place_name: str, location: str = "", max_reviews: int = 50) -> str:
    """
    Complete analysis of competitor reviews: scraping, sentiment clustering, pain points, and loyalty score.
    
    Args:
        place_name: Name of the competitor boba shop
        location: Location/address (optional but recommended)
        max_reviews: Maximum number of reviews to analyze (default: 50)
    
    Returns:
        JSON string with complete analysis
    """
    # Scrape reviews
    reviews = scrape_google_reviews(place_name, location, max_reviews)
    
    if not reviews:
        return json.dumps({"error": "No reviews found. Make sure to provide a valid place name and location."})
    
    # Perform all analyses
    clusters = cluster_reviews_by_sentiment(reviews)
    pain_points = extract_pain_points(reviews)
    loyalty_score = calculate_brand_loyalty_score(reviews)
    
    result = {
        "place_name": place_name,
        "location": location,
        "total_reviews_analyzed": len(reviews),
        "sentiment_clusters": {
            cat: {
                "positive_count": cluster.positive_count,
                "negative_count": cluster.negative_count,
                "neutral_count": cluster.neutral_count,
                "sample_reviews": cluster.sample_reviews[:3]  # Top 3 samples
            }
            for cat, cluster in clusters.items()
        },
        "pain_points": {
            "total_count": len(pain_points),
            "by_category": {
                cat: [pp.text for pp in pain_points if pp.category == cat]
                for cat in set(pp.category for pp in pain_points)
            },
            "all_pain_points": [
                {"text": pp.text, "category": pp.category}
                for pp in pain_points[:10]  # Top 10
            ]
        },
        "brand_loyalty": {
            "total_reviews": loyalty_score.total_reviews,
            "local_guide_count": loyalty_score.local_guide_count,
            "one_time_tourist_count": loyalty_score.one_time_tourist_count,
            "local_guide_percentage": loyalty_score.local_guide_percentage,
            "supports_regulars_model": loyalty_score.supports_regulars_model,
            "loyalty_score": loyalty_score.score,
            "recommendation": "Area supports a 'Regulars' business model - focus on building repeat customers" 
                            if loyalty_score.supports_regulars_model 
                            else "Area is more tourist-driven - focus on first impressions and marketing"
        }
    }
    
    return json.dumps(result, indent=2)


# Create the Voice of Customer agent
voice_agent = create_agent(
    model,
    tools=[
        scrape_reviews_tool,
        analyze_sentiment_clusters_tool,
        extract_pain_points_tool,
        calculate_loyalty_score_tool,
        analyze_competitor_reviews_tool,
    ],
    system_prompt="""You are the Voice of Customer Agent, specializing in analyzing competitor reviews 
    for boba tea shops. Your expertise includes:

    1. **Scraping Google Reviews**: Extract reviews from Google for any boba shop or competitor
    2. **Sentiment Clustering**: Group reviews into categories (Wait Time, Sweetness Levels, Pearl Texture, Staff Friendliness) 
       and analyze sentiment for each category
    3. **Pain Point Extraction**: Identify "I wish" and "They don't have" statements to uncover customer frustrations
    4. **Brand Loyalty Analysis**: Calculate loyalty scores based on Local Guides vs tourists to determine if an area 
       supports a "Regulars" business model

    When a user asks you to analyze a competitor:
    - Use analyze_competitor_reviews_tool for complete analysis, OR
    - Use individual tools (scrape_reviews_tool, then analyze_sentiment_clusters_tool, etc.) for step-by-step analysis
    
    Always provide clear, actionable insights from the review data. Focus on helping identify market opportunities 
    and customer pain points that could inform business strategy.
    """,
    name="VoiceOfCustomer",
)


# For standalone usage, you can compile a swarm with just this agent
def create_voice_workflow():
    """Create a workflow with the Voice of Customer agent."""
    from langgraph.checkpoint.memory import InMemorySaver
    
    checkpointer = InMemorySaver()
    workflow = create_swarm(
        [voice_agent],
        default_active_agent="VoiceOfCustomer"
    )
    return workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # Example usage
    app = create_voice_workflow()
    config = {"configurable": {"thread_id": "voice_1"}}
    
    # Example: Analyze a competitor
    result = app.invoke(
        {"messages": [{"role": "user", "content": "Analyze reviews for 'Boba Guys' in San Francisco"}]},
        config,
    )
    print(result)
