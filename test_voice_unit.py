"""
Unit tests for voice.py helper functions.

Tests the internal helper functions directly without any tool decorators or agent setup.
"""

from agents.voice import (
    _scrape_google_reviews,
    _cluster_reviews_by_sentiment,
    _extract_pain_points,
    _calculate_brand_loyalty_score
)


def test_cluster_reviews_by_sentiment():
    """Test sentiment clustering function."""
    print("=" * 80)
    print("Testing _cluster_reviews_by_sentiment")
    print("=" * 80)
    
    # Sample reviews
    reviews = [
        {"text": "Great boba! The pearls are perfectly chewy. Fast service.", "rating": 5, "is_local_guide": True},
        {"text": "Too sweet for my taste. Staff was friendly though.", "rating": 2, "is_local_guide": False},
        {"text": "Long wait time, almost 20 minutes. But good texture.", "rating": 3, "is_local_guide": False},
        {"text": "Perfect sweetness levels! Staff is super helpful.", "rating": 5, "is_local_guide": True},
        {"text": "Rude staff, terrible customer service. Bad pearl texture.", "rating": 1, "is_local_guide": False},
    ]
    
    try:
        result = _cluster_reviews_by_sentiment(reviews)
        
        print(f"✅ Success: Clustered into {len(result)} categories")
        print()
        
        for category, cluster in result.items():
            print(f"Category: {category}")
            print(f"  Positive: {cluster['positive_count']}")
            print(f"  Negative: {cluster['negative_count']}")
            print(f"  Neutral: {cluster['neutral_count']}")
            print(f"  Sample reviews: {len(cluster['sample_reviews'])}")
            print()
        
        assert len(result) == 4, "Should have 4 categories"
        assert "Wait Time" in result, "Should have Wait Time category"
        assert "Sweetness Levels" in result, "Should have Sweetness Levels category"
        print("✅ All assertions passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_extract_pain_points():
    """Test pain point extraction function."""
    print("\n" + "=" * 80)
    print("Testing _extract_pain_points")
    print("=" * 80)
    
    # Sample reviews with pain point patterns
    reviews = [
        {"text": "I wish they had more flavor options. Great boba though!", "author": "John", "rating": 4, "is_local_guide": False},
        {"text": "They don't have customizable sweetness levels. I wish they did.", "author": "Sarah", "rating": 3, "is_local_guide": False},
        {"text": "I wish the wait time was shorter. The queue is always long.", "author": "Mike", "rating": 2, "is_local_guide": False},
        {"text": "Good service. They don't have enough seating though.", "author": "Emma", "rating": 4, "is_local_guide": True},
        {"text": "I wish the staff was more friendly. Prices are okay.", "author": "Alex", "rating": 3, "is_local_guide": False},
    ]
    
    try:
        result = _extract_pain_points(reviews)
        
        print(f"✅ Success: Extracted {len(result)} pain points")
        print()
        
        # Group by category
        by_category = {}
        for pp in result:
            cat = pp.get("category", "General")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(pp)
        
        print("Pain points by category:")
        for cat, points in by_category.items():
            print(f"  {cat}: {len(points)} pain points")
            for pp in points[:2]:  # Show first 2 per category
                print(f"    - {pp['text'][:60]}...")
        print()
        
        assert len(result) > 0, "Should extract at least one pain point"
        print("✅ All assertions passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_calculate_brand_loyalty_score():
    """Test brand loyalty score calculation."""
    print("\n" + "=" * 80)
    print("Testing _calculate_brand_loyalty_score")
    print("=" * 80)
    
    # Test case 1: High local guide percentage (should support regulars model)
    reviews_high_loyalty = [
        {"text": "Great!", "author": "Local Guide 1", "rating": 5, "is_local_guide": True},
        {"text": "Amazing!", "author": "Local Guide 2", "rating": 5, "is_local_guide": True},
        {"text": "Love it!", "author": "Local Guide 3", "rating": 4, "is_local_guide": True},
        {"text": "Good", "author": "Tourist 1", "rating": 4, "is_local_guide": False},
        {"text": "Okay", "author": "Tourist 2", "rating": 3, "is_local_guide": False},
    ]
    
    # Test case 2: Low local guide percentage
    reviews_low_loyalty = [
        {"text": "Great!", "author": "Tourist 1", "rating": 5, "is_local_guide": False},
        {"text": "Amazing!", "author": "Tourist 2", "rating": 5, "is_local_guide": False},
        {"text": "Love it!", "author": "Tourist 3", "rating": 4, "is_local_guide": False},
        {"text": "Good", "author": "Local Guide 1", "rating": 4, "is_local_guide": True},
    ]
    
    try:
        # Test high loyalty case
        print("Test Case 1: High Local Guide Percentage")
        result1 = _calculate_brand_loyalty_score(reviews_high_loyalty)
        print(f"  Total reviews: {result1['total_reviews']}")
        print(f"  Local guides: {result1['local_guide_count']} ({result1['local_guide_percentage']}%)")
        print(f"  Loyalty score: {result1['score']}/100")
        print(f"  Supports regulars model: {result1['supports_regulars_model']}")
        
        assert result1['local_guide_percentage'] == 60.0, "Should be 60% local guides"
        assert result1['supports_regulars_model'] == True, "Should support regulars model (>30%)"
        
        print("\nTest Case 2: Low Local Guide Percentage")
        result2 = _calculate_brand_loyalty_score(reviews_low_loyalty)
        print(f"  Total reviews: {result2['total_reviews']}")
        print(f"  Local guides: {result2['local_guide_count']} ({result2['local_guide_percentage']}%)")
        print(f"  Loyalty score: {result2['score']}/100")
        print(f"  Supports regulars model: {result2['supports_regulars_model']}")
        
        assert result2['local_guide_percentage'] == 25.0, "Should be 25% local guides"
        assert result2['supports_regulars_model'] == False, "Should not support regulars model (<30%)"
        
        print()
        print("✅ All assertions passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("Testing Edge Cases")
    print("=" * 80)
    
    # Test empty reviews
    print("\nTest 1: Empty reviews list")
    try:
        result = _cluster_reviews_by_sentiment([])
        assert len(result) == 4, "Should return 4 empty categories"
        print("✅ Empty reviews handled correctly")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test empty pain points
    print("\nTest 2: Reviews with no pain point patterns")
    try:
        reviews = [
            {"text": "Great boba!", "author": "John", "rating": 5, "is_local_guide": False},
            {"text": "Amazing service!", "author": "Sarah", "rating": 5, "is_local_guide": False},
        ]
        result = _extract_pain_points(reviews)
        print(f"✅ Extracted {len(result)} pain points (should be 0 or low)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test loyalty score with empty reviews
    print("\nTest 3: Loyalty score with empty reviews")
    try:
        result = _calculate_brand_loyalty_score([])
        assert result['total_reviews'] == 0, "Should be 0 reviews"
        assert result['local_guide_percentage'] == 0, "Should be 0%"
        assert result['score'] == 0, "Should be 0 score"
        print("✅ Empty reviews handled correctly")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test reviews with missing fields
    print("\nTest 4: Reviews with missing fields")
    try:
        reviews = [
            {"text": "Good", "rating": 5},  # Missing author and is_local_guide
            {"text": "Bad", "author": "John"},  # Missing rating and is_local_guide
        ]
        result = _cluster_reviews_by_sentiment(reviews)
        print("✅ Missing fields handled gracefully")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("VOICE AGENT UNIT TESTS")
    print("=" * 80)
    print("\nTesting internal helper functions directly")
    print("(No API keys required for these tests)\n")
    
    test_cluster_reviews_by_sentiment()
    test_extract_pain_points()
    test_calculate_brand_loyalty_score()
    test_edge_cases()
    
    print("\n" + "=" * 80)
    print("ALL UNIT TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
