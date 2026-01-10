import os
import requests
from typing import Optional, Dict, Any, List
import googlemaps
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool
from config import model


def _geocode_to_coordinates(address: str) -> Optional[Dict[str, float]]:
    """
    Geocode an address to latitude/longitude coordinates using Google Maps API.
    
    Args:
        address: Address string to geocode
    
    Returns:
        Dictionary with 'latitude' and 'longitude' keys, or None if geocoding fails
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return None
    
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
        pass
    
    return None


def _coords_to_census_tract(latitude: float, longitude: float) -> Optional[Dict[str, str]]:
    """
    Convert coordinates to Census tract FIPS code using Census Geocoder API.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Dictionary with state_fips, county_fips, tract_fips, or None if conversion fails
    """
    try:
        url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        params = {
            "x": longitude,
            "y": latitude,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "format": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "result" in data and "geographies" in data["result"]:
            geographies = data["result"]["geographies"]
            
            # Try to get tract-level data
            if "Census Tracts" in geographies and len(geographies["Census Tracts"]) > 0:
                tract = geographies["Census Tracts"][0]
                state_fips = tract.get("STATE")
                county_fips = tract.get("COUNTY")
                tract_fips = tract.get("TRACT")
                
                if state_fips and county_fips and tract_fips:
                    return {
                        "state_fips": state_fips,
                        "county_fips": county_fips,
                        "tract_fips": tract_fips
                    }
    except Exception as e:
        pass
    
    return None


def _fetch_census_data(state_fips: str, county_fips: str, tract_fips: str, variables: List[str]) -> Optional[Dict[str, Any]]:
    """
    Fetch Census ACS 5-year data for a specific tract.
    
    Args:
        state_fips: State FIPS code
        county_fips: County FIPS code
        tract_fips: Tract FIPS code
        variables: List of Census variable codes to fetch
    
    Returns:
        Dictionary with Census data or None if fetch fails
    """
    census_api_key = os.getenv("CENSUS_API_KEY", "")
    
    try:
        # Use ACS 5-year data (most recent available)
        url = "https://api.census.gov/data/2022/acs/acs5"
        params = {
            "get": ",".join(variables),
            "for": f"tract:{tract_fips}",
            "in": f"state:{state_fips} county:{county_fips}",
        }
        
        if census_api_key:
            params["key"] = census_api_key.strip().strip('"').strip("'")
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if len(data) > 1:
            # Convert to dictionary format
            headers = data[0]
            values = data[1]
            result = dict(zip(headers, values))
            return result
    except Exception as e:
        pass
    
    return None


def _analyze_age_income(census_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze age and income data to find 18-34 year olds with discretionary income.
    
    Args:
        census_data: Dictionary with Census variable data
    
    Returns:
        Dictionary with age/income analysis results
    """
    # Age variables (18-34 years old)
    # B01001_007E: Male 18-19
    # B01001_008E: Male 20
    # B01001_009E: Male 21
    # B01001_010E: Male 22-24
    # B01001_011E: Male 25-29
    # B01001_012E: Male 30-34
    # B01001_031E: Female 18-19
    # B01001_032E: Female 20
    # B01001_033E: Female 21
    # B01001_034E: Female 22-24
    # B01001_035E: Female 25-29
    # B01001_036E: Female 30-34
    
    age_variables = [
        "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",
        "B01001_011E", "B01001_012E", "B01001_031E", "B01001_032E",
        "B01001_033E", "B01001_034E", "B01001_035E", "B01001_036E"
    ]
    
    # Income variables
    # B19013_001E: Median household income
    # B08301_021E: Workers 16+ who worked from home (proxy for flexible income)
    # B19001_014E through B19001_017E: Households with income $75k-$200k+ (discretionary income range)
    
    income_variables = [
        "B19013_001E",  # Median household income
        "B19001_014E",  # $75k-$99k
        "B19001_015E",  # $100k-$124k
        "B19001_016E",  # $125k-$149k
        "B19001_017E",  # $150k-$199k
        "B19001_018E",  # $200k+
    ]
    
    try:
        # Calculate 18-34 population
        age_18_34 = 0
        for var in age_variables:
            value = census_data.get(var, "0")
            try:
                age_18_34 += int(value) if value and value != "null" else 0
            except (ValueError, TypeError):
                pass
        
        # Calculate total population
        total_pop = census_data.get("B01001_001E", "0")
        try:
            total_pop = int(total_pop) if total_pop and total_pop != "null" else 0
        except (ValueError, TypeError):
            total_pop = 0
        
        # Calculate discretionary income households
        discretionary_income = 0
        for var in income_variables[1:]:  # Skip median income
            value = census_data.get(var, "0")
            try:
                discretionary_income += int(value) if value and value != "null" else 0
            except (ValueError, TypeError):
                pass
        
        # Total households
        total_households = census_data.get("B19001_001E", "0")
        try:
            total_households = int(total_households) if total_households and total_households != "null" else 0
        except (ValueError, TypeError):
            total_households = 1  # Avoid division by zero
        
        # Median household income
        median_income = census_data.get("B19013_001E", "0")
        try:
            median_income = int(median_income) if median_income and median_income != "null" else 0
        except (ValueError, TypeError):
            median_income = 0
        
        # Calculate percentages
        age_18_34_pct = (age_18_34 / total_pop * 100) if total_pop > 0 else 0
        discretionary_income_pct = (discretionary_income / total_households * 100) if total_households > 0 else 0
        
        # Score calculation: combination of age concentration and discretionary income
        # Higher concentration of 18-34 AND higher discretionary income = higher score
        age_score = min(age_18_34_pct / 25.0, 1.0) * 50  # Normalize to 25% baseline
        income_score = min(discretionary_income_pct / 40.0, 1.0) * 50  # Normalize to 40% baseline
        combined_score = age_score + income_score
        
        return {
            "age_18_34_count": age_18_34,
            "age_18_34_percentage": round(age_18_34_pct, 2),
            "total_population": total_pop,
            "median_household_income": median_income,
            "discretionary_income_households": discretionary_income,
            "discretionary_income_percentage": round(discretionary_income_pct, 2),
            "total_households": total_households,
            "age_income_score": round(combined_score, 2),
            "score_interpretation": "HIGH" if combined_score >= 70 else "MODERATE" if combined_score >= 40 else "LOW"
        }
    except Exception as e:
        return {"error": f"Error analyzing age/income data: {str(e)}"}


def _analyze_ethnicity_cultural_alignment(census_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze ethnicity data to score locations based on cultural alignment with boba tea.
    Boba has high adoption rates in Asian populations.
    
    Args:
        census_data: Dictionary with Census variable data
    
    Returns:
        Dictionary with ethnicity analysis and cultural alignment score
    """
    try:
        # Ethnicity variables from B03002 (Hispanic or Latino by Race)
        # B03002_003E: White alone
        # B03002_004E: Black or African American alone
        # B03002_005E: American Indian and Alaska Native alone
        # B03002_006E: Asian alone
        # B03002_007E: Native Hawaiian and Other Pacific Islander alone
        # B03002_008E: Some other race alone
        # B03002_009E: Two or more races
        # B03002_012E: Hispanic or Latino (of any race)
        
        total_pop = census_data.get("B03002_001E", "0")
        try:
            total_pop = int(total_pop) if total_pop and total_pop != "null" else 0
        except (ValueError, TypeError):
            total_pop = 0
        
        asian_pop = census_data.get("B03002_006E", "0")
        try:
            asian_pop = int(asian_pop) if asian_pop and asian_pop != "null" else 0
        except (ValueError, TypeError):
            asian_pop = 0
        
        hispanic_pop = census_data.get("B03002_012E", "0")
        try:
            hispanic_pop = int(hispanic_pop) if hispanic_pop and hispanic_pop != "null" else 0
        except (ValueError, TypeError):
            hispanic_pop = 0
        
        # Also check detailed Asian subgroups if available (B02015 series)
        # But for tract-level, B03002_006E should be sufficient
        
        # Calculate percentages
        asian_pct = (asian_pop / total_pop * 100) if total_pop > 0 else 0
        hispanic_pct = (hispanic_pop / total_pop * 100) if total_pop > 0 else 0
        
        # Cultural alignment score: Higher Asian population = higher boba adoption potential
        # Asian population is primary indicator (boba originated in Taiwan)
        # Hispanic population is secondary (boba popular in some Hispanic communities)
        asian_score = min(asian_pct / 15.0, 1.0) * 70  # Normalize to 15% baseline
        hispanic_score = min(hispanic_pct / 30.0, 1.0) * 30  # Normalize to 30% baseline
        cultural_score = asian_score + hispanic_score
        
        return {
            "total_population": total_pop,
            "asian_population": asian_pop,
            "asian_percentage": round(asian_pct, 2),
            "hispanic_population": hispanic_pop,
            "hispanic_percentage": round(hispanic_pct, 2),
            "cultural_alignment_score": round(cultural_score, 2),
            "score_interpretation": "HIGH" if cultural_score >= 70 else "MODERATE" if cultural_score >= 40 else "LOW",
            "boba_adoption_potential": "HIGH" if asian_pct >= 10 else "MODERATE" if asian_pct >= 5 else "LOW"
        }
    except Exception as e:
        return {"error": f"Error analyzing ethnicity data: {str(e)}"}


def _analyze_daytime_nighttime_population(latitude: float, longitude: float, radius: float = 500) -> Dict[str, Any]:
    """
    Analyze daytime vs nighttime population using business hours and density as proxy.
    Uses Google Places API to check if area stays active after 5 PM.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        radius: Search radius in meters (default 500m)
    
    Returns:
        Dictionary with daytime/nighttime activity analysis
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_PLACES_API_KEY not set"}
    
    api_key = api_key.strip().strip('"').strip("'")
    
    try:
        # Search for businesses that typically stay open late
        # Types: restaurants, cafes, bars, entertainment
        url = "https://places.googleapis.com/v1/places:searchNearby"
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.types,places.businessStatus,places.regularOpeningHours"
        }
        
        # Search for evening-active businesses
        evening_types = ["restaurant", "cafe", "bar", "night_club", "movie_theater", "shopping_mall"]
        
        payload = {
            "includedTypes": evening_types,
            "maxResultCount": 20,
            "rankPreference": "POPULARITY",
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "radius": radius
                }
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        places = data.get("places", [])
        
        # Count places that are likely open late (after 9 PM)
        # Check opening hours if available
        evening_places = 0
        late_night_places = 0
        total_active_places = 0
        
        for place in places:
            if place.get("businessStatus") == "OPERATIONAL":
                total_active_places += 1
                opening_hours = place.get("regularOpeningHours", {})
                
                # If we have opening hours data, check if they close after 9 PM
                # Otherwise, assume restaurants/bars/cafes in dense areas stay open late
                if opening_hours:
                    # Opening hours format is complex, for simplicity assume if we have data, check weekdayHours
                    weekday_hours = opening_hours.get("weekdayDescriptions", [])
                    # If any weekday has hours past 9 PM, count as late night
                    for hours_str in weekday_hours:
                        if "21:00" in hours_str or "10:00 PM" in hours_str or "11:00 PM" in hours_str or "12:00 AM" in hours_str:
                            late_night_places += 1
                            break
                else:
                    # Heuristic: bars, night_clubs, and some restaurants likely stay open late
                    types = place.get("types", [])
                    if any(t in ["bar", "night_club"] for t in types):
                        late_night_places += 1
                        evening_places += 1
                    elif "restaurant" in types or "cafe" in types:
                        evening_places += 1
        
        # Calculate scores
        evening_activity_score = (evening_places / max(total_active_places, 1)) * 50 if total_active_places > 0 else 0
        late_night_activity_score = (late_night_places / max(total_active_places, 1)) * 50 if total_active_places > 0 else 0
        total_activity_score = evening_activity_score + late_night_activity_score
        
        # Determine if area "dies at 5 PM" or "stays alive until 11 PM"
        stays_alive = late_night_places >= 3 or (total_active_places >= 10 and late_night_places >= 1)
        
        return {
            "total_active_businesses": total_active_places,
            "evening_places_count": evening_places,
            "late_night_places_count": late_night_places,
            "evening_activity_score": round(evening_activity_score, 2),
            "late_night_activity_score": round(late_night_activity_score, 2),
            "total_activity_score": round(total_activity_score, 2),
            "stays_alive_after_5pm": stays_alive,
            "stays_alive_until_11pm": late_night_places >= 3,
            "activity_interpretation": "STAYS ALIVE (active until 11 PM+)" if stays_alive else "DIES AT 5 PM (limited evening activity)"
        }
    except Exception as e:
        return {"error": f"Error analyzing daytime/nighttime population: {str(e)}"}


@tool
def analyze_age_income_demographics(location: str, latitude: Optional[float] = None, longitude: Optional[float] = None) -> dict:
    """
    Analyze age and income demographics for a location using Census data.
    Finds high concentrations of 18-34 year olds with discretionary income.
    
    Args:
        location: Address string (e.g., "123 Main St, San Francisco, CA")
        latitude: Optional latitude coordinate (if provided, location is ignored)
        longitude: Optional longitude coordinate (if provided, location is ignored)
    
    Returns:
        Dictionary with age/income analysis including scores and interpretations
    """
    # Get coordinates
    if latitude is None or longitude is None:
        coords = _geocode_to_coordinates(location)
        if not coords:
            return {"error": f"Could not geocode location: {location}"}
        latitude = coords["latitude"]
        longitude = coords["longitude"]
    
    # Convert to Census tract
    tract_info = _coords_to_census_tract(latitude, longitude)
    if not tract_info:
        return {"error": f"Could not convert coordinates to Census tract for {latitude}, {longitude}"}
    
    # Define variables needed for age/income analysis
    variables = [
        "B01001_001E",  # Total population
        "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",  # Male 18-24
        "B01001_011E", "B01001_012E",  # Male 25-34
        "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",  # Female 18-24
        "B01001_035E", "B01001_036E",  # Female 25-34
        "B19001_001E",  # Total households
        "B19013_001E",  # Median household income
        "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E", "B19001_018E",  # Discretionary income ranges
    ]
    
    # Fetch Census data
    census_data = _fetch_census_data(
        tract_info["state_fips"],
        tract_info["county_fips"],
        tract_info["tract_fips"],
        variables
    )
    
    if not census_data:
        return {"error": "Could not fetch Census data for this location"}
    
    # Analyze age/income
    analysis = _analyze_age_income(census_data)
    
    return {
        "location": location if latitude is None else f"{latitude}, {longitude}",
        "census_tract": f"{tract_info['state_fips']}{tract_info['county_fips']}{tract_info['tract_fips']}",
        "analysis": analysis
    }


@tool
def analyze_ethnicity_cultural_alignment(location: str, latitude: Optional[float] = None, longitude: Optional[float] = None) -> dict:
    """
    Analyze ethnicity demographics to score cultural alignment with boba tea.
    Scores locations based on presence of populations familiar with boba (primarily Asian populations).
    
    Args:
        location: Address string (e.g., "123 Main St, San Francisco, CA")
        latitude: Optional latitude coordinate (if provided, location is ignored)
        longitude: Optional longitude coordinate (if provided, location is ignored)
    
    Returns:
        Dictionary with ethnicity analysis and cultural alignment scores
    """
    # Get coordinates
    if latitude is None or longitude is None:
        coords = _geocode_to_coordinates(location)
        if not coords:
            return {"error": f"Could not geocode location: {location}"}
        latitude = coords["latitude"]
        longitude = coords["longitude"]
    
    # Convert to Census tract
    tract_info = _coords_to_census_tract(latitude, longitude)
    if not tract_info:
        return {"error": f"Could not convert coordinates to Census tract for {latitude}, {longitude}"}
    
    # Define variables needed for ethnicity analysis
    variables = [
        "B03002_001E",  # Total population
        "B03002_003E",  # White alone
        "B03002_004E",  # Black or African American alone
        "B03002_006E",  # Asian alone
        "B03002_012E",  # Hispanic or Latino (of any race)
    ]
    
    # Fetch Census data
    census_data = _fetch_census_data(
        tract_info["state_fips"],
        tract_info["county_fips"],
        tract_info["tract_fips"],
        variables
    )
    
    if not census_data:
        return {"error": "Could not fetch Census data for this location"}
    
    # Analyze ethnicity
    analysis = _analyze_ethnicity_cultural_alignment(census_data)
    
    return {
        "location": location if latitude is None else f"{latitude}, {longitude}",
        "census_tract": f"{tract_info['state_fips']}{tract_info['county_fips']}{tract_info['tract_fips']}",
        "analysis": analysis
    }


@tool
def analyze_daytime_nighttime_population(location: str, latitude: Optional[float] = None, longitude: Optional[float] = None, radius: float = 500) -> dict:
    """
    Analyze daytime vs nighttime population using business hours and activity as proxy.
    Determines if area dies at 5 PM or stays alive until 11 PM using cell phone mobility proxies
    (business hours data from Google Places).
    
    Args:
        location: Address string (e.g., "123 Main St, San Francisco, CA")
        latitude: Optional latitude coordinate (if provided, location is ignored)
        longitude: Optional longitude coordinate (if provided, location is ignored)
        radius: Search radius in meters for nearby businesses (default 500m)
    
    Returns:
        Dictionary with daytime/nighttime activity analysis
    """
    # Get coordinates
    if latitude is None or longitude is None:
        coords = _geocode_to_coordinates(location)
        if not coords:
            return {"error": f"Could not geocode location: {location}"}
        latitude = coords["latitude"]
        longitude = coords["longitude"]
    
    # Analyze activity patterns
    analysis = _analyze_daytime_nighttime_population(latitude, longitude, radius)
    
    return {
        "location": location if latitude is None else f"{latitude}, {longitude}",
        "search_radius_meters": radius,
        "analysis": analysis
    }


@tool
def analyze_demographics_complete(location: str, latitude: Optional[float] = None, longitude: Optional[float] = None, radius: float = 500) -> dict:
    """
    Complete demographics analysis combining age/income, ethnicity/cultural alignment, and daytime/nighttime population.
    Provides comprehensive demographic profile for boba shop location evaluation.
    
    Args:
        location: Address string (e.g., "123 Main St, San Francisco, CA")
        latitude: Optional latitude coordinate (if provided, location is ignored)
        longitude: Optional longitude coordinate (if provided, location is ignored)
        radius: Search radius in meters for nighttime activity analysis (default 500m)
    
    Returns:
        Dictionary with complete demographic analysis
    """
    # Get coordinates
    if latitude is None or longitude is None:
        coords = _geocode_to_coordinates(location)
        if not coords:
            return {"error": f"Could not geocode location: {location}"}
        latitude = coords["latitude"]
        longitude = coords["longitude"]
    
    # Convert to Census tract
    tract_info = _coords_to_census_tract(latitude, longitude)
    if not tract_info:
        return {"error": f"Could not convert coordinates to Census tract for {latitude}, {longitude}"}
    
    # Fetch all Census variables needed
    variables = [
        # Population and age (18-34)
        "B01001_001E",  # Total population
        "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",  # Male 18-24
        "B01001_011E", "B01001_012E",  # Male 25-34
        "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",  # Female 18-24
        "B01001_035E", "B01001_036E",  # Female 25-34
        # Income
        "B19001_001E",  # Total households
        "B19013_001E",  # Median household income
        "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E", "B19001_018E",  # Discretionary income
        # Ethnicity
        "B03002_001E",  # Total population
        "B03002_006E",  # Asian alone
        "B03002_012E",  # Hispanic or Latino
    ]
    
    # Fetch Census data
    census_data = _fetch_census_data(
        tract_info["state_fips"],
        tract_info["county_fips"],
        tract_info["tract_fips"],
        variables
    )
    
    if not census_data:
        return {"error": "Could not fetch Census data for this location"}
    
    # Perform all analyses
    age_income = _analyze_age_income(census_data)
    ethnicity = _analyze_ethnicity_cultural_alignment(census_data)
    activity = _analyze_daytime_nighttime_population(latitude, longitude, radius)
    
    # Calculate overall score
    age_income_score = age_income.get("age_income_score", 0) if "error" not in age_income else 0
    cultural_score = ethnicity.get("cultural_alignment_score", 0) if "error" not in ethnicity else 0
    activity_score = activity.get("total_activity_score", 0) if "error" not in activity else 0
    
    # Weighted overall score (adjust weights as needed)
    overall_score = (age_income_score * 0.4) + (cultural_score * 0.4) + (activity_score * 0.2)
    
    return {
        "location": location if latitude is None else f"{latitude}, {longitude}",
        "census_tract": f"{tract_info['state_fips']}{tract_info['county_fips']}{tract_info['tract_fips']}",
        "coordinates": {"latitude": latitude, "longitude": longitude},
        "age_income_analysis": age_income,
        "ethnicity_cultural_analysis": ethnicity,
        "daytime_nighttime_analysis": activity,
        "overall_demographic_score": round(overall_score, 2),
        "overall_interpretation": "HIGH" if overall_score >= 70 else "MODERATE" if overall_score >= 40 else "LOW",
        "recommendation": "STRONG DEMOGRAPHIC FIT" if overall_score >= 70 
                         else "MODERATE DEMOGRAPHIC FIT" if overall_score >= 40 
                         else "WEAK DEMOGRAPHIC FIT"
    }


demographics_tools = [
    analyze_age_income_demographics,
    analyze_ethnicity_cultural_alignment,
    analyze_daytime_nighttime_population,
    analyze_demographics_complete,
    create_handoff_tool(
        agent_name="Location Scout",
        description="Transfer to Location Scout to identify competitor locations and complementary businesses in the analyzed area",
    ),
    create_handoff_tool(
        agent_name="Quantitative Analyst",
        description="Transfer to Quantitative Analyst to analyze competitor performance metrics and validate demographic findings",
    ),
]

DEMOGRAPHICS_SYSTEM_PROMPT = """You are the Demographics Agent, specializing in analyzing demographic data for boba tea shop location evaluation using US Census data.

## Your Primary Objectives

1. **Age/Income Analysis**: Use Census data to find high concentrations of 18-34-year-olds with "Discretionary Income"
   - Use `analyze_age_income_demographics` to analyze age distribution and household income
   - Focus on identifying areas with high percentages of young adults (18-34) and households with discretionary income ($75k+)
   - Score locations based on the combination of age concentration and income levels

2. **Ethnicity & Cultural Alignment**: Score locations based on the presence of populations already familiar with boba tea
   - Use `analyze_ethnicity_cultural_alignment` to analyze ethnic demographics
   - Boba has high adoption rates in Asian populations (primary indicator) and some Hispanic communities (secondary)
   - Calculate cultural alignment scores based on Asian population percentage
   - Higher Asian population = higher boba adoption potential

3. **Daytime vs. Nighttime Population**: Use business activity data to see if the area dies at 5 PM or stays alive until 11 PM
   - Use `analyze_daytime_nighttime_population` to assess evening activity levels
   - Uses Google Places API to check business hours and density as a proxy for mobility patterns
   - Identifies if area has active evening businesses (restaurants, cafes, bars, entertainment) that stay open late
   - Critical for boba shops that benefit from evening foot traffic

## Workflow & Tool Usage

**For complete analysis (recommended):**
- Use `analyze_demographics_complete` for one-shot comprehensive analysis
  - Combines all three demographic dimensions
  - Provides overall demographic score and recommendation
  - Most efficient for initial location evaluation

**For targeted analysis:**
- Use individual tools when you need to focus on specific demographic dimensions:
  - `analyze_age_income_demographics`: Focus on age and income only
  - `analyze_ethnicity_cultural_alignment`: Focus on cultural alignment only
  - `analyze_daytime_nighttime_population`: Focus on activity patterns only

**Location Input Formats:**
- Address strings: `analyze_demographics_complete(location="123 Main St, San Francisco, CA")`
- Coordinates (preferred for accuracy): `analyze_demographics_complete(latitude=37.7749, longitude=-122.4194, location="")`

## Output Format

Provide structured demographic insights:

**Age/Income Analysis:**
- 18-34 population count and percentage
- Median household income
- Discretionary income households percentage
- Age/Income score and interpretation (HIGH/MODERATE/LOW)

**Ethnicity & Cultural Alignment:**
- Asian population count and percentage
- Hispanic population count and percentage
- Cultural alignment score
- Boba adoption potential (HIGH/MODERATE/LOW)

**Daytime/Nighttime Activity:**
- Total active businesses in area
- Evening and late-night business counts
- Activity scores
- Interpretation: "STAYS ALIVE" vs "DIES AT 5 PM"

**Overall Assessment:**
- Overall demographic score (0-100)
- Overall interpretation
- Recommendation: STRONG/MODERATE/WEAK DEMOGRAPHIC FIT

## Handoff to Other Agents

**Hand off to Location Scout when:**
- After identifying high-scoring demographic areas
- Need to find competitor locations and complementary businesses in the analyzed area
- Want to validate demographic findings with actual business presence

**Hand off to Quantitative Analyst when:**
- Need to analyze competitor performance metrics
- Want to validate demographic findings with actual business performance data
- Need quantitative validation of market potential

## Data Sources & Notes

- **Census Data**: Uses ACS 5-year estimates (most recent available)
- **Geocoding**: Automatically converts addresses to coordinates and Census tracts
- **Activity Proxy**: Uses Google Places business hours as proxy for mobility data (cell phone mobility data would require specialized APIs)
- **Census API Key**: Optional but recommended (set CENSUS_API_KEY environment variable for higher rate limits)"""

demographics = create_agent(
    model=model,
    tools=demographics_tools,
    system_prompt=DEMOGRAPHICS_SYSTEM_PROMPT,
    name="Demographics Analyst"
)