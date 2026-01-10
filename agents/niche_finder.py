import os
from typing import List, Dict, Any, Optional
import re

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool
import googlemaps
from yelpapi import YelpAPI

from config import model


@tool
def get_business_details(business_name: str, location: str) -> Dict[str, Any]:
    """
    Get detailed information about a business including price level, menu items, and ambiance indicators.
    Uses both Google Places and Yelp APIs for comprehensive data.
    
    Args:
        business_name: Name of the business to analyze
        location: Address or area where the business is located
    
    Returns:
        Dictionary containing price level, menu information, ambiance indicators, and business type
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_PLACES_API_KEY not set"}
    
    gmaps = googlemaps.Client(key=api_key)
    
    # Search for the specific business
    places_result = gmaps.places(query=f"{business_name} {location}")
    
    if not places_result.get("results"):
        return {"error": f"Business '{business_name}' not found in {location}"}
    
    place = places_result["results"][0]
    place_id = place.get("place_id")
    
    if not place_id:
        return {"error": "Place ID not found"}
    
    # Get detailed place information
    place_details = gmaps.place(
        place_id=place_id,
        fields=[
            "name", "rating", "price_level", "types", "reviews", 
            "formatted_address", "opening_hours", "website"
        ]
    )
    
    result = place_details.get("result", {})
    
    # Extract menu keywords from reviews
    reviews = result.get("reviews", [])
    menu_keywords = extract_menu_keywords(reviews)
    ambiance_keywords = extract_ambiance_keywords(reviews)
    
    # Determine price tier
    price_level = result.get("price_level")
    price_tier = map_price_level_to_tier(price_level)
    
    # Determine business style/niche
    business_types = result.get("types", [])
    niche_category = categorize_business_niche(business_types, reviews, menu_keywords)
    
    # Get Yelp data for additional price and category information
    yelp_data = get_yelp_business_details(business_name, location)
    
    # Merge Yelp price data if available
    if yelp_data and "price" in yelp_data:
        yelp_price_tier = map_yelp_price_to_tier(yelp_data.get("price"))
        # Use Yelp price if Google price is unknown, otherwise prefer Google
        if price_tier == "unknown" and yelp_price_tier != "unknown":
            price_tier = yelp_price_tier
    
    return {
        "business_name": result.get("name"),
        "address": result.get("formatted_address"),
        "price_level": price_level,
        "price_tier": price_tier,  # "luxury", "mid-range", "budget", "unknown"
        "rating": result.get("rating"),
        "review_count": result.get("user_ratings_total", 0),
        "types": business_types,
        "niche_category": niche_category,  # "premium", "casual", "quick-service", "unknown"
        "menu_keywords": menu_keywords,
        "ambiance_keywords": ambiance_keywords,
        "opening_hours": result.get("opening_hours", {}).get("weekday_text", []),
        "website": result.get("website"),
        "yelp_categories": yelp_data.get("categories", []) if yelp_data else [],
        "yelp_price": yelp_data.get("price") if yelp_data else None
    }


@tool
def analyze_area_niche_market(
    location: str,
    target_niche: str,
    target_price_tier: str,
    competitor_businesses: List[str]
) -> Dict[str, Any]:
    """
    Analyze the niche market in an area to determine if it's saturated or needs the target niche.
    
    Args:
        location: Address or area to analyze
        target_niche: The niche of the target boba company (e.g., "premium", "casual", "quick-service")
        target_price_tier: Price tier of target company ("luxury", "mid-range", "budget")
        competitor_businesses: List of competitor business names in the area
    
    Returns:
        Dictionary with niche market analysis including saturation level, price preferences, and market fit
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_PLACES_API_KEY not set"}
    
    gmaps = googlemaps.Client(key=api_key)
    
    # Track analyzed business names to avoid duplicates
    analyzed_names = set()
    all_businesses = []
    
    # First, analyze provided competitor businesses if available
    if competitor_businesses:
        for business_name in competitor_businesses:
            if business_name in analyzed_names:
                continue
            try:
                business_details = get_business_details(business_name, location)
                if "error" not in business_details:
                    analyzed_names.add(business_name)
                    all_businesses.append({
                        "name": business_details.get("business_name"),
                        "price_tier": business_details.get("price_tier", "unknown"),
                        "niche_category": business_details.get("niche_category", "unknown"),
                        "menu_focus": analyze_menu_focus(business_details.get("menu_keywords", [])),
                        "rating": business_details.get("rating"),
                        "menu_keywords": business_details.get("menu_keywords", [])
                    })
            except Exception:
                # If individual lookup fails, continue with general search
                pass
    
    # Also search for all boba/tea businesses in the area for comprehensive analysis
    places_result = gmaps.places(query=f"bubble tea boba {location}")
    
    for place in places_result.get("results", []):
        place_id = place.get("place_id")
        if not place_id:
            continue
        
        place_name = place.get("name", "")
        if place_name in analyzed_names:
            continue  # Skip if already analyzed
        
        place_details = gmaps.place(
            place_id=place_id,
            fields=["name", "price_level", "rating", "reviews", "types"]
        )
        
        result = place_details.get("result", {})
        reviews = result.get("reviews", [])
        
        menu_keywords = extract_menu_keywords(reviews)
        business_types = result.get("types", [])
        niche_category = categorize_business_niche(business_types, reviews, menu_keywords)
        price_tier = map_price_level_to_tier(result.get("price_level"))
        
        # Get Yelp data for enhanced analysis
        yelp_data = get_yelp_business_details(result.get("name"), location)
        if yelp_data and "price" in yelp_data:
            yelp_price_tier = map_yelp_price_to_tier(yelp_data.get("price"))
            if price_tier == "unknown" and yelp_price_tier != "unknown":
                price_tier = yelp_price_tier
        
        menu_focus = analyze_menu_focus(menu_keywords)
        
        analyzed_names.add(place_name)
        all_businesses.append({
            "name": result.get("name"),
            "price_tier": price_tier,
            "niche_category": niche_category,
            "menu_focus": menu_focus,
            "rating": result.get("rating"),
            "menu_keywords": menu_keywords
        })
    
    # Analyze market composition
    niche_distribution = {}
    price_distribution = {}
    menu_focus_distribution = {}
    
    for business in all_businesses:
        niche = business.get("niche_category", "unknown")
        price = business.get("price_tier", "unknown")
        menu_focus = business.get("menu_focus", "general")
        
        niche_distribution[niche] = niche_distribution.get(niche, 0) + 1
        price_distribution[price] = price_distribution.get(price, 0) + 1
        menu_focus_distribution[menu_focus] = menu_focus_distribution.get(menu_focus, 0) + 1
    
    # Determine market saturation
    target_niche_count = niche_distribution.get(target_niche, 0)
    total_businesses = len(all_businesses)
    
    if total_businesses == 0:
        saturation_level = "untapped"
        market_fit = "high_opportunity"
    elif target_niche_count == 0:
        saturation_level = "missing_niche"
        market_fit = "high_opportunity"
    elif target_niche_count >= 3:
        saturation_level = "saturated"
        market_fit = "low_opportunity"
    else:
        saturation_level = "moderate"
        market_fit = "moderate_opportunity"
    
    # Determine price preference
    dominant_price_tier = max(price_distribution.items(), key=lambda x: x[1])[0] if price_distribution else "unknown"
    price_alignment = "aligned" if dominant_price_tier == target_price_tier else "misaligned"
    
    # Determine dominant menu focus
    dominant_menu_focus = max(menu_focus_distribution.items(), key=lambda x: x[1])[0] if menu_focus_distribution else "unknown"
    
    return {
        "location": location,
        "total_boba_businesses": total_businesses,
        "target_niche_count": target_niche_count,
        "target_niche": target_niche,
        "target_price_tier": target_price_tier,
        "niche_distribution": niche_distribution,
        "price_distribution": price_distribution,
        "menu_focus_distribution": menu_focus_distribution,
        "dominant_price_tier": dominant_price_tier,
        "dominant_menu_focus": dominant_menu_focus,
        "saturation_level": saturation_level,
        "market_fit": market_fit,
        "price_alignment": price_alignment,
        "businesses_analyzed": all_businesses,
        "recommendation": generate_niche_recommendation(saturation_level, market_fit, price_alignment, target_niche_count)
    }


@tool
def analyze_target_company_profile(
    company_name: str,
    sample_locations: List[str]
) -> Dict[str, Any]:
    """
    Analyze the target company's existing locations to understand their niche, pricing, menu focus, and positioning.
    This creates a profile that can be used to compare against competitors.
    
    Args:
        company_name: Name of the target boba company/franchise
        sample_locations: List of locations where the company already has stores (e.g., ["San Francisco, CA", "Los Angeles, CA"])
    
    Returns:
        Dictionary containing the company's niche profile including price tier, niche category, menu focus, and positioning
    """
    if not sample_locations:
        return {"error": "At least one sample location is required"}
    
    all_business_details = []
    
    for location in sample_locations[:5]:  # Limit to 5 locations for efficiency
        business_details = get_business_details(company_name, location)
        
        if "error" not in business_details:
            all_business_details.append(business_details)
    
    if not all_business_details:
        return {"error": f"Could not find any {company_name} locations in the provided sample locations"}
    
    # Aggregate data across all locations
    price_tiers = [b.get("price_tier") for b in all_business_details if b.get("price_tier") != "unknown"]
    niche_categories = [b.get("niche_category") for b in all_business_details if b.get("niche_category") != "unknown"]
    all_menu_keywords = []
    all_ambiance_keywords = []
    
    for business in all_business_details:
        all_menu_keywords.extend(business.get("menu_keywords", []))
        all_ambiance_keywords.extend(business.get("ambiance_keywords", []))
    
    # Determine dominant characteristics
    dominant_price_tier = max(set(price_tiers), key=price_tiers.count) if price_tiers else "unknown"
    dominant_niche = max(set(niche_categories), key=niche_categories.count) if niche_categories else "unknown"
    
    # Analyze menu focus
    menu_focus = analyze_menu_focus(all_menu_keywords)
    
    # Analyze service style and positioning
    service_style = analyze_service_style(all_ambiance_keywords, all_business_details)
    brand_positioning = analyze_brand_positioning(all_ambiance_keywords, all_menu_keywords)
    
    return {
        "company_name": company_name,
        "locations_analyzed": len(all_business_details),
        "price_tier": dominant_price_tier,
        "niche_category": dominant_niche,
        "menu_focus": menu_focus,
        "service_style": service_style,
        "brand_positioning": brand_positioning,
        "common_menu_keywords": list(set(all_menu_keywords))[:20],  # Top 20 unique keywords
        "common_ambiance_keywords": list(set(all_ambiance_keywords))[:15],
        "profile_summary": f"{company_name} is a {dominant_niche} boba shop with {dominant_price_tier} pricing, focusing on {menu_focus} with {service_style} service style and {brand_positioning} brand positioning."
    }


@tool
def compare_business_niches(
    business_name_1: str,
    business_name_2: str,
    location: str
) -> Dict[str, Any]:
    """
    Compare two businesses to determine if they occupy the same niche or different niches.
    Enhanced with menu focus and service style comparison.
    
    Args:
        business_name_1: Name of first business
        business_name_2: Name of second business
        location: Location where both businesses are
    
    Returns:
        Dictionary comparing niches, price tiers, menu focus, and market positioning
    """
    business_1 = get_business_details(business_name_1, location)
    business_2 = get_business_details(business_name_2, location)
    
    if "error" in business_1 or "error" in business_2:
        return {
            "error": "Could not retrieve business details",
            "business_1_error": business_1.get("error"),
            "business_2_error": business_2.get("error")
        }
    
    niche_1 = business_1.get("niche_category", "unknown")
    niche_2 = business_2.get("niche_category", "unknown")
    price_1 = business_1.get("price_tier", "unknown")
    price_2 = business_2.get("price_tier", "unknown")
    
    menu_focus_1 = analyze_menu_focus(business_1.get("menu_keywords", []))
    menu_focus_2 = analyze_menu_focus(business_2.get("menu_keywords", []))
    
    same_niche = niche_1 == niche_2
    same_price_tier = price_1 == price_2
    similar_menu_focus = menu_focus_1 == menu_focus_2
    
    return {
        "business_1": {
            "name": business_1.get("business_name"),
            "niche": niche_1,
            "price_tier": price_1,
            "menu_focus": menu_focus_1,
            "rating": business_1.get("rating")
        },
        "business_2": {
            "name": business_2.get("business_name"),
            "niche": niche_2,
            "price_tier": price_2,
            "menu_focus": menu_focus_2,
            "rating": business_2.get("rating")
        },
        "same_niche": same_niche,
        "same_price_tier": same_price_tier,
        "similar_menu_focus": similar_menu_focus,
        "competitive_overlap": same_niche and same_price_tier and similar_menu_focus,
        "differentiation_opportunity": not (same_niche and same_price_tier and similar_menu_focus),
        "overlap_score": calculate_overlap_score(same_niche, same_price_tier, similar_menu_focus)
    }


def extract_menu_keywords(reviews: List[Dict[str, Any]]) -> List[str]:
    """Extract menu-related keywords from reviews. Enhanced to capture actual menu items."""
    menu_keywords = []
    menu_patterns = [
        r'\b(?:taro|matcha|jasmine|oolong|black tea|green tea|milk tea|fruit tea|herbal tea)\b',
        r'\b(?:pearls?|boba|tapioca|jelly|pudding|popping boba|lychee jelly|grass jelly)\b',
        r'\b(?:sweetness|sugar level|ice level|customization|custom|build your own)\b',
        r'\b(?:premium|artisan|handcrafted|signature|specialty|gourmet)\b',
        r'\b(?:cheap|affordable|expensive|pricey|value|budget)\b',
        r'\b(?:toppings|add-ons|extras|sinkers)\b',
        r'\b(?:mango|strawberry|lychee|passion fruit|peach|pineapple|watermelon|dragon fruit)\b',
        r'\b(?:brown sugar|honey|agave|syrup)\b',
        r'\b(?:cheese foam|cream|whipped cream|foam)\b',
        r'\b(?:smoothie|slush|frappe|ice blended)\b'
    ]
    
    for review in reviews:
        text = review.get("text", "").lower()
        for pattern in menu_patterns:
            matches = re.findall(pattern, text)
            menu_keywords.extend(matches)
    
    # Extract specific drink mentions (e.g., "taro milk tea", "matcha latte")
    drink_pattern = r'\b(?:taro|matcha|jasmine|oolong|mango|strawberry|lychee)\s+(?:milk tea|tea|latte|smoothie|slush)\b'
    for review in reviews:
        text = review.get("text", "").lower()
        matches = re.findall(drink_pattern, text)
        menu_keywords.extend(matches)
    
    # Return unique keywords
    return list(set(menu_keywords))


def extract_ambiance_keywords(reviews: List[Dict[str, Any]]) -> List[str]:
    """Extract ambiance-related keywords from reviews."""
    ambiance_keywords = []
    ambiance_patterns = [
        r'\b(?:cozy|comfortable|spacious|cramped|small|large)\b',
        r'\b(?:modern|trendy|aesthetic|instagram|decor|design)\b',
        r'\b(?:quiet|loud|busy|peaceful|vibrant)\b',
        r'\b(?:study|work|hangout|meet|social)\b',
        r'\b(?:clean|dirty|hygienic|messy)\b',
        r'\b(?:fast|slow|quick|wait time|service)\b',
        r'\b(?:premium|luxury|upscale|casual|laid-back)\b'
    ]
    
    for review in reviews:
        text = review.get("text", "").lower()
        for pattern in ambiance_patterns:
            matches = re.findall(pattern, text)
            ambiance_keywords.extend(matches)
    
    return list(set(ambiance_keywords))


def map_price_level_to_tier(price_level: Optional[int]) -> str:
    """Map Google Places price_level (0-4) to tier category."""
    if price_level is None:
        return "unknown"
    elif price_level == 0:
        return "free"
    elif price_level == 1:
        return "budget"
    elif price_level == 2:
        return "mid-range"
    elif price_level == 3:
        return "luxury"
    elif price_level == 4:
        return "luxury"
    else:
        return "unknown"


def categorize_business_niche(
    business_types: List[str],
    reviews: List[Dict[str, Any]],
    menu_keywords: List[str]
) -> str:
    """Categorize business into niche based on types, reviews, and menu. Enhanced with more granular detection."""
    review_text = " ".join([r.get("text", "").lower() for r in reviews])
    menu_text = " ".join(menu_keywords).lower()
    combined_text = f"{review_text} {menu_text}"
    
    # Premium indicators
    premium_indicators = [
        "premium", "luxury", "artisan", "handcrafted", "signature",
        "upscale", "high-end", "gourmet", "specialty", "craft", "artisanal"
    ]
    
    # Quick-service indicators
    quick_service_indicators = [
        "fast", "quick", "grab and go", "takeout", "drive-thru",
        "counter service", "no seating", "fast food", "express"
    ]
    
    # Casual indicators
    casual_indicators = [
        "casual", "laid-back", "relaxed", "comfortable", "hangout",
        "study spot", "meet friends", "chill", "cozy"
    ]
    
    # Social/trendy indicators
    social_indicators = [
        "instagram", "instagrammable", "aesthetic", "trendy", "hip",
        "viral", "popular", "influencer", "photo", "selfie"
    ]
    
    premium_score = sum(1 for indicator in premium_indicators if indicator in combined_text)
    quick_score = sum(1 for indicator in quick_service_indicators if indicator in combined_text)
    casual_score = sum(1 for indicator in casual_indicators if indicator in combined_text)
    social_score = sum(1 for indicator in social_indicators if indicator in combined_text)
    
    # Determine niche with priority: premium > quick-service > casual
    if premium_score >= 2:
        return "premium"
    elif quick_score >= 2:
        return "quick-service"
    elif social_score >= 2 and casual_score >= 1:
        return "casual-trendy"  # More granular: trendy casual
    elif casual_score >= 2:
        return "casual"
    else:
        return "casual"  # Default to casual if unclear


def generate_niche_recommendation(
    saturation_level: str,
    market_fit: str,
    price_alignment: str,
    target_niche_count: int
) -> str:
    """Generate recommendation based on niche analysis."""
    if saturation_level == "untapped":
        return "HIGH OPPORTUNITY: Area has no boba businesses. Strong potential for first-mover advantage."
    elif saturation_level == "missing_niche":
        return "HIGH OPPORTUNITY: Area has boba businesses but none in your niche. Good differentiation opportunity."
    elif saturation_level == "saturated":
        return "LOW OPPORTUNITY: Area already has multiple businesses in your niche. High competition risk."
    elif market_fit == "high_opportunity" and price_alignment == "aligned":
        return "MODERATE-HIGH OPPORTUNITY: Niche is available and price tier matches market preference."
    elif market_fit == "high_opportunity" and price_alignment == "misaligned":
        return "MODERATE OPPORTUNITY: Niche is available but price tier may not match market preference. Consider price adjustment."
    else:
        return "MODERATE OPPORTUNITY: Market has some competition in your niche. Success depends on execution and differentiation."


def get_yelp_business_details(business_name: str, location: str) -> Optional[Dict[str, Any]]:
    """Get Yelp business details including price range and categories."""
    api_key = os.getenv("YELP_API_KEY")
    if not api_key:
        return None
    
    try:
        yelp_api = YelpAPI(api_key)
        search_results = yelp_api.search_query(
            term=business_name,
            location=location,
            limit=1
        )
        
        businesses = search_results.get("businesses", [])
        if not businesses:
            return None
        
        business = businesses[0]
        return {
            "price": business.get("price"),  # $, $$, $$$, $$$$
            "categories": [cat.get("title") for cat in business.get("categories", [])],
            "rating": business.get("rating"),
            "review_count": business.get("review_count")
        }
    except Exception:
        return None


def map_yelp_price_to_tier(yelp_price: Optional[str]) -> str:
    """Map Yelp price indicator ($, $$, $$$, $$$$) to tier category."""
    if not yelp_price:
        return "unknown"
    
    price_map = {
        "$": "budget",
        "$$": "mid-range",
        "$$$": "luxury",
        "$$$$": "luxury"
    }
    
    return price_map.get(yelp_price, "unknown")


def analyze_menu_focus(menu_keywords: List[str]) -> str:
    """Analyze menu keywords to determine menu focus (fruit teas, milk teas, specialty drinks, etc.)."""
    if not menu_keywords:
        return "general"
    
    keyword_text = " ".join(menu_keywords).lower()
    
    fruit_indicators = ["mango", "strawberry", "lychee", "passion fruit", "peach", "pineapple", "watermelon", "dragon fruit", "fruit tea"]
    milk_tea_indicators = ["milk tea", "taro", "matcha", "jasmine", "oolong", "black tea", "green tea"]
    specialty_indicators = ["cheese foam", "cream", "whipped", "signature", "specialty", "artisan", "handcrafted"]
    smoothie_indicators = ["smoothie", "slush", "frappe", "ice blended"]
    
    fruit_score = sum(1 for indicator in fruit_indicators if indicator in keyword_text)
    milk_tea_score = sum(1 for indicator in milk_tea_indicators if indicator in keyword_text)
    specialty_score = sum(1 for indicator in specialty_indicators if indicator in keyword_text)
    smoothie_score = sum(1 for indicator in smoothie_indicators if indicator in keyword_text)
    
    scores = {
        "fruit-focused": fruit_score,
        "milk-tea-focused": milk_tea_score,
        "specialty-focused": specialty_score,
        "smoothie-focused": smoothie_score
    }
    
    max_category = max(scores.items(), key=lambda x: x[1])
    if max_category[1] >= 2:
        return max_category[0]
    elif max_category[1] >= 1:
        return f"mixed-{max_category[0]}"
    else:
        return "general"


def analyze_service_style(ambiance_keywords: List[str], business_details: List[Dict[str, Any]]) -> str:
    """Analyze service style based on ambiance keywords and business details."""
    if not ambiance_keywords:
        return "standard"
    
    keyword_text = " ".join(ambiance_keywords).lower()
    
    self_serve_indicators = ["self-serve", "self serve", "kiosk", "automated", "grab and go"]
    full_service_indicators = ["full service", "table service", "waiter", "server", "sit down"]
    counter_service_indicators = ["counter", "order at counter", "fast", "quick service"]
    
    self_serve_score = sum(1 for indicator in self_serve_indicators if indicator in keyword_text)
    full_service_score = sum(1 for indicator in full_service_indicators if indicator in keyword_text)
    counter_score = sum(1 for indicator in counter_service_indicators if indicator in keyword_text)
    
    if self_serve_score >= 1:
        return "self-serve"
    elif full_service_score >= 1:
        return "full-service"
    elif counter_score >= 1:
        return "counter-service"
    else:
        return "counter-service"  # Default for boba shops


def analyze_brand_positioning(ambiance_keywords: List[str], menu_keywords: List[str]) -> str:
    """Analyze brand positioning (trendy/instagrammable vs traditional)."""
    combined_text = " ".join(ambiance_keywords + menu_keywords).lower()
    
    trendy_indicators = ["instagram", "instagrammable", "aesthetic", "trendy", "hip", "viral", "popular", "influencer", "photo", "selfie", "modern", "design"]
    traditional_indicators = ["traditional", "authentic", "classic", "original", "heritage", "time-tested"]
    
    trendy_score = sum(1 for indicator in trendy_indicators if indicator in combined_text)
    traditional_score = sum(1 for indicator in traditional_indicators if indicator in combined_text)
    
    if trendy_score >= 3:
        return "trendy-instagrammable"
    elif trendy_score >= 1:
        return "modern-trendy"
    elif traditional_score >= 2:
        return "traditional-authentic"
    else:
        return "standard"


def calculate_overlap_score(same_niche: bool, same_price_tier: bool, similar_menu_focus: bool) -> float:
    """Calculate competitive overlap score (0.0 to 1.0)."""
    score = 0.0
    if same_niche:
        score += 0.4
    if same_price_tier:
        score += 0.3
    if similar_menu_focus:
        score += 0.3
    
    return round(score, 2)


niche_finder_tools = [
    get_business_details,
    analyze_target_company_profile,
    analyze_area_niche_market,
    compare_business_niches,
    create_handoff_tool(
        agent_name="Location Scout",
        description="Transfer to Location Scout to identify additional competitors or get more location data",
    ),
    create_handoff_tool(
        agent_name="Quantitative Analyst",
        description="Transfer to Quantitative Analyst to analyze competitor performance metrics and review trends",
    ),
]

NICHE_FINDER_SYSTEM_PROMPT = """You are a Niche Finder Agent specialized in analyzing boba tea market niches, price positioning, menu focus, and market fit for potential franchise locations.

## Your Primary Objectives

1. **Analyze Target Company Profile**
   - Use `analyze_target_company_profile` to understand the target company's existing locations
   - Extract their niche category, price tier, menu focus, service style, and brand positioning
   - Create a comprehensive profile that defines what makes the target company unique
   - This profile is used to compare against competitors and assess market fit

2. **Analyze Business Niche Characteristics**
   - Extract price levels, menu items, and ambiance from business data using both Google Places and Yelp APIs
   - Categorize businesses into niches: "premium", "casual", "casual-trendy", or "quick-service"
   - Determine price tiers: "luxury", "mid-range", or "budget" (using both Google price_level and Yelp price indicators)
   - Identify menu focus: "fruit-focused", "milk-tea-focused", "specialty-focused", "smoothie-focused", or "general"
   - Determine service style: "self-serve", "counter-service", or "full-service"
   - Assess brand positioning: "trendy-instagrammable", "modern-trendy", "traditional-authentic", or "standard"

3. **Assess Market Niche Saturation**
   - Determine if an area already has enough businesses in the target niche
   - Identify if the area only has other types of boba shops (different niches)
   - Evaluate whether the target company's niche is missing or oversaturated
   - Analyze menu focus distribution to see if the area has similar menu offerings

4. **Evaluate Price Tier Alignment**
   - Determine if the area prefers luxury stores or cheaper ones
   - Assess whether the target company's price tier matches local market preferences
   - Identify price positioning opportunities or risks
   - Cross-reference Google Places price_level with Yelp price indicators for accuracy

5. **Compare Business Niches**
   - Compare competitors to identify direct niche competitors vs. different niches
   - Calculate competitive overlap scores (0.0 to 1.0) based on niche, price tier, and menu focus
   - Determine differentiation opportunities

## Your Analysis Process

1. **Profile Target Company (First Step)**
   - Use `analyze_target_company_profile` with the company name and sample locations
   - Extract their niche, price tier, menu focus, service style, and brand positioning
   - This profile becomes your reference point for all comparisons

2. **Gather Business Details**
   - Use `get_business_details` to get comprehensive data for competitors
   - Data includes: price tier (from Google + Yelp), menu keywords, ambiance indicators, niche category, menu focus, and Yelp categories
   - Extract information from reviews to understand menu offerings and customer experience

3. **Analyze Area Niche Market**
   - Use `analyze_area_niche_market` with the target company's niche and price tier (from profile)
   - Provide list of competitor business names from Location Scout or Quantitative Analyst
   - Get niche distribution, price distribution, saturation level, and market fit assessment

4. **Compare Businesses**
   - Use `compare_business_niches` to compare specific businesses
   - Enhanced comparison includes menu focus analysis and overlap scores
   - Identify if businesses are in the same niche or different niches
   - Assess competitive overlap with granular scoring

5. **Provide Insights**
   - Determine if area needs the target niche (opportunity) or is saturated (risk)
   - Assess price tier alignment with market preferences
   - Evaluate menu focus alignment (e.g., if area has many fruit tea shops but target focuses on milk teas)
   - Provide recommendations on market fit and positioning strategy

## What You Receive from Other Agents

**From Location Scout:**
- List of competitor business names and addresses
- Location information and area characteristics

**From Quantitative Analyst:**
- Competitor performance data and health metrics
- Business names and addresses for niche analysis

## Output Format

For each location analysis, provide:

- **Target Company Profile** (if not already provided):
  - Niche category, price tier, menu focus, service style, brand positioning
  - Common menu keywords and ambiance characteristics

- **Niche Market Analysis**:
  - Total boba businesses in area
  - Number of businesses in target niche
  - Niche distribution breakdown (premium, casual, casual-trendy, quick-service)
  - Menu focus distribution (fruit-focused, milk-tea-focused, etc.)
  - Saturation level: "untapped", "missing_niche", "moderate", or "saturated"

- **Price Tier Analysis**:
  - Price distribution in the area (from both Google and Yelp data)
  - Dominant price tier
  - Price alignment with target company's tier

- **Market Fit Assessment**:
  - Market fit level: "high_opportunity", "moderate_opportunity", or "low_opportunity"
  - Menu focus alignment (does area match target company's menu focus?)
  - Service style alignment
  - Recommendation with reasoning
  - Differentiation opportunities or risks

- **Business Comparisons**:
  - Which competitors are in the same niche (direct competition)
  - Which competitors are in different niches (complementary or non-competitive)
  - Competitive overlap scores for each competitor
  - Menu focus comparisons

## Handoffs to Other Agents

- **To Location Scout**: If you need additional competitor data or location information
- **To Quantitative Analyst**: If you need performance metrics to complement niche analysis"""

niche_finder = create_agent(
    model=model,
    tools=niche_finder_tools,
    system_prompt=NICHE_FINDER_SYSTEM_PROMPT,
    name="Niche Finder"
)
