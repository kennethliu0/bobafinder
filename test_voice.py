"""
Self-contained test file for voice.py agent.

This file tests all voice agent tools without any handoffs to other agents.
Tests both direct tool invocation and agent workflow usage.
"""

import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_swarm import create_swarm

from agents.voice import (
    scrape_reviews,
    analyze_sentiment_clusters,
    extract_pain_points,
    calculate_loyalty_score,
    analyze_competitor_reviews,
    voice
)

load_dotenv()


def test_direct_tools():
    """Test all voice agent tools directly."""
    print("=" * 80)
    print("TESTING DIRECT TOOL INVOCATION")
    print("=" * 80)
    
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        print("⚠️  Warning: GOOGLE_PLACES_API_KEY not found. Some tests will be skipped.")
        print()
        return
    
    # Test 1: Scrape reviews
    print("\n[Test 1] scrape_reviews tool")
    print("-" * 80)
    try:
        result = scrape_reviews.invoke({
            "place_name": "Boba Guys",
            "location": "San Francisco, CA",
            "max_reviews": 10
        })
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success: Found {result.get('total_reviews', 0)} reviews")
            print(f"   Place: {result.get('place_name')} in {result.get('location')}")
            print(f"   Reviews returned: {len(result.get('reviews', []))}")
            
            # Use this result for subsequent tests
            return result
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None
    
    print("\n" + "-" * 80)


def test_sentiment_analysis(reviews_data):
    """Test sentiment analysis with sample reviews."""
    print("\n[Test 2] analyze_sentiment_clusters tool")
    print("-" * 80)
    
    if not reviews_data:
        print("⏭️  Skipping: No reviews data available")
        return
    
    try:
        result = analyze_sentiment_clusters.invoke({
            "reviews_data": reviews_data
        })
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success: Sentiment clusters generated")
            categories = result.get("categories", {})
            print(f"   Categories found: {list(categories.keys())}")
            for cat, data in categories.items():
                pos = data.get("positive_count", 0)
                neg = data.get("negative_count", 0)
                neu = data.get("neutral_count", 0)
                print(f"   {cat}: +{pos} / -{neg} / ~{neu}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "-" * 80)


def test_pain_point_extraction(reviews_data):
    """Test pain point extraction with sample reviews."""
    print("\n[Test 3] extract_pain_points tool")
    print("-" * 80)
    
    if not reviews_data:
        print("⏭️  Skipping: No reviews data available")
        return
    
    try:
        result = extract_pain_points.invoke({
            "reviews_data": reviews_data
        })
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            total = result.get("total_pain_points", 0)
            print(f"✅ Success: Found {total} pain points")
            
            pain_points = result.get("pain_points", [])
            if pain_points:
                print(f"   Sample pain points:")
                for pp in pain_points[:3]:  # Show first 3
                    cat = pp.get("category", "General")
                    text = pp.get("text", "")[:60]
                    print(f"     [{cat}] {text}...")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "-" * 80)


def test_loyalty_score(reviews_data):
    """Test loyalty score calculation with sample reviews."""
    print("\n[Test 4] calculate_loyalty_score tool")
    print("-" * 80)
    
    if not reviews_data:
        print("⏭️  Skipping: No reviews data available")
        return
    
    try:
        result = calculate_loyalty_score.invoke({
            "reviews_data": reviews_data
        })
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success: Loyalty score calculated")
            print(f"   Total reviews: {result.get('total_reviews', 0)}")
            print(f"   Local guides: {result.get('local_guide_count', 0)} ({result.get('local_guide_percentage', 0)}%)")
            print(f"   One-time tourists: {result.get('one_time_tourist_count', 0)}")
            print(f"   Loyalty score: {result.get('loyalty_score', 0)}/100")
            print(f"   Supports regulars model: {result.get('supports_regulars_model', False)}")
            print(f"   Interpretation: {result.get('interpretation', 'N/A')}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "-" * 80)


def test_complete_analysis():
    """Test the complete analysis tool."""
    print("\n[Test 5] analyze_competitor_reviews tool (complete analysis)")
    print("-" * 80)
    
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        print("⏭️  Skipping: GOOGLE_PLACES_API_KEY not found")
        return
    
    try:
        result = analyze_competitor_reviews.invoke({
            "place_name": "Gong Cha",
            "location": "San Francisco, CA",
            "max_reviews": 15
        })
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success: Complete analysis generated")
            print(f"   Place: {result.get('place_name')} in {result.get('location')}")
            print(f"   Total reviews analyzed: {result.get('total_reviews_analyzed', 0)}")
            
            # Sentiment clusters
            clusters = result.get("sentiment_clusters", {})
            print(f"   Sentiment categories: {list(clusters.keys())}")
            
            # Pain points
            pain_points = result.get("pain_points", {})
            print(f"   Total pain points: {pain_points.get('total_count', 0)}")
            
            # Brand loyalty
            loyalty = result.get("brand_loyalty", {})
            print(f"   Loyalty score: {loyalty.get('loyalty_score', 0)}/100")
            print(f"   Recommendation: {loyalty.get('recommendation', 'N/A')[:60]}...")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "-" * 80)


def test_agent_workflow():
    """Test the voice agent through workflow (without handoffs)."""
    print("\n" + "=" * 80)
    print("TESTING AGENT WORKFLOW (NO HANDOFFS)")
    print("=" * 80)
    
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        print("⚠️  Warning: GOOGLE_PLACES_API_KEY not found. Skipping agent workflow tests.")
        return
    
    # Create standalone workflow with only voice agent (no handoffs)
    # We'll create tools list without handoff tool
    from langchain.agents import create_agent
    from config import model
    
    voice_tools_only = [
        scrape_reviews,
        analyze_sentiment_clusters,
        extract_pain_points,
        calculate_loyalty_score,
        analyze_competitor_reviews,
    ]
    
    # Import the system prompt
    from agents.voice import VOICE_SYSTEM_PROMPT
    
    # Create agent without handoff
    voice_standalone = create_agent(
        model=model,
        tools=voice_tools_only,
        system_prompt=VOICE_SYSTEM_PROMPT,
        name="Voice of Customer"
    )
    
    # Create workflow
    checkpointer = InMemorySaver()
    workflow = create_swarm(
        [voice_standalone],
        default_active_agent="Voice of Customer"
    )
    app = workflow.compile(checkpointer=checkpointer)
    
    # Test queries
    test_queries = [
        {
            "query": "Scrape 10 reviews for 'Boba Guys' in San Francisco",
            "description": "Simple review scraping"
        },
        {
            "query": "Analyze reviews for 'Gong Cha' in San Francisco. Give me sentiment clusters.",
            "description": "Sentiment analysis request"
        },
        {
            "query": "What are the pain points from reviews of 'Tpumps' in San Francisco?",
            "description": "Pain point extraction request"
        },
        {
            "query": "Calculate the loyalty score for 'Boba Guys' in San Francisco. Does it support a regulars model?",
            "description": "Loyalty score calculation"
        },
        {
            "query": "Give me a complete analysis of reviews for 'Sharetea' in San Francisco, including sentiment, pain points, and loyalty score.",
            "description": "Complete analysis request"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n[Agent Test {i}] {test['description']}")
        print("-" * 80)
        print(f"Query: {test['query']}")
        print()
        
        try:
            config = {"configurable": {"thread_id": f"test_{i}"}}
            result = app.invoke({
                "messages": [{"role": "user", "content": test["query"]}],
            }, config)
            
            # Extract and print agent response
            if isinstance(result, dict) and "messages" in result:
                # Get the last assistant message
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "role") and msg.role == "assistant":
                        if hasattr(msg, "content"):
                            content = msg.content
                            if content:
                                print("Agent Response:")
                                print(content[:500] + "..." if len(content) > 500 else content)
                                break
                    elif isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content:
                            print("Agent Response:")
                            print(content[:500] + "..." if len(content) > 500 else content)
                            break
                
                print("✅ Agent completed request")
            else:
                print("⚠️  Unexpected result format")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
        
        print()


def test_with_mock_data():
    """Test tools with mock review data (no API key needed)."""
    print("\n" + "=" * 80)
    print("TESTING WITH MOCK DATA (NO API KEY NEEDED)")
    print("=" * 80)
    
    # Create mock review data
    mock_reviews = {
        "place_name": "Test Boba Shop",
        "location": "Test Location",
        "total_reviews": 5,
        "reviews": [
            {
                "text": "Great boba! The pearls are perfectly chewy and the sweetness is just right. Staff is super friendly. I wish they had more flavors though.",
                "author": "Local Guide John",
                "rating": 5,
                "is_local_guide": True
            },
            {
                "text": "The wait time was too long, almost 20 minutes. They don't have enough staff. But the boba texture is good.",
                "author": "Sarah M.",
                "rating": 3,
                "is_local_guide": False
            },
            {
                "text": "Too sweet for my taste. The pearls were a bit too soft. Customer service was okay. I wish the sweetness levels were customizable.",
                "author": "Mike T.",
                "rating": 2,
                "is_local_guide": False
            },
            {
                "text": "Fast service, friendly staff, perfect pearl texture. Highly recommend!",
                "author": "Local Guide Emma",
                "rating": 5,
                "is_local_guide": True
            },
            {
                "text": "The staff was rude and the queue was very long. They don't have a variety of toppings. I wish they would improve their service.",
                "author": "Alex K.",
                "rating": 2,
                "is_local_guide": False
            }
        ]
    }
    
    print("\n[Mock Test 1] Sentiment Analysis with Mock Data")
    print("-" * 80)
    try:
        result = analyze_sentiment_clusters.invoke({"reviews_data": mock_reviews})
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success: Sentiment clusters generated from mock data")
            categories = result.get("categories", {})
            for cat, data in categories.items():
                print(f"   {cat}: +{data.get('positive_count', 0)} / -{data.get('negative_count', 0)} / ~{data.get('neutral_count', 0)}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n[Mock Test 2] Pain Point Extraction with Mock Data")
    print("-" * 80)
    try:
        result = extract_pain_points.invoke({"reviews_data": mock_reviews})
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success: Found {result.get('total_pain_points', 0)} pain points")
            for pp in result.get("pain_points", [])[:5]:
                print(f"   [{pp.get('category', 'General')}] {pp.get('text', '')[:60]}...")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n[Mock Test 3] Loyalty Score with Mock Data")
    print("-" * 80)
    try:
        result = calculate_loyalty_score.invoke({"reviews_data": mock_reviews})
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success: Loyalty score calculated")
            print(f"   Local guides: {result.get('local_guide_count', 0)}/{result.get('total_reviews', 0)} ({result.get('local_guide_percentage', 0)}%)")
            print(f"   Loyalty score: {result.get('loyalty_score', 0)}/100")
            print(f"   Supports regulars model: {result.get('supports_regulars_model', False)}")
    except Exception as e:
        print(f"❌ Exception: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VOICE AGENT TEST SUITE")
    print("=" * 80)
    print("\nThis test suite tests all voice agent tools independently")
    print("without any handoffs to other agents.\n")
    
    # Test with real API (if available)
    reviews_data = test_direct_tools()
    
    # Test analysis tools with real data
    if reviews_data and "error" not in reviews_data:
        test_sentiment_analysis(reviews_data)
        test_pain_point_extraction(reviews_data)
        test_loyalty_score(reviews_data)
    
    # Test complete analysis
    test_complete_analysis()
    
    # Test with mock data (always works)
    test_with_mock_data()
    
    # Test agent workflow
    test_agent_workflow()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nNote: Some tests require GOOGLE_PLACES_API_KEY to be set in .env file")
    print("      Mock data tests should always work without an API key")


if __name__ == "__main__":
    main()
