import sys
import requests
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def get_location_coords(location_name: str):
    """
    Converts a location name into latitude and longitude coordinates.
    """
    try:
        geolocator = Nominatim(user_agent="heartwise_er_finder_v1")
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return f"Could not find coordinates for '{location_name}'."
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        # Catch a wider range of geocoding API errors
        print(f"Error: Geocoding service failed: {e}")
        return f"Geocoding service failed: {e}"


def find_nearest_ers_expanding(lat: float, lon: float):

    overpass_url = "http://overpass-api.de/api/interpreter"

    # Start with a 10km radius and expand until we find enough results
    radius_km = 20  # Start with a slightly larger radius
    max_radius_km = 300  # Set a reasonable max limit to avoid overly long searches
    ers = []

    # --- NEW: Define a list of queries, from most specific to most general ---
    query_templates = [
        # 1. Hospitals with explicit emergency services
        'node["amenity"="hospital"]["emergency"="yes"](around:{radius_m},{lat},{lon});way["amenity"="hospital"]["emergency"="yes"](around:{radius_m},{lat},{lon});relation["amenity"="hospital"]["emergency"="yes"](around:{radius_m},{lat},{lon});',
        # 2. Any hospital (emergency service implied or not tagged)
        'node["amenity"="hospital"](around:{radius_m},{lat},{lon});way["amenity"="hospital"](around:{radius_m},{lat},{lon});relation["amenity"="hospital"](around:{radius_m},{lat},{lon});',
        # 3. Clinics, as a last resort
        'node["amenity"="clinic"](around:{radius_m},{lat},{lon});way["amenity"="clinic"](around:{radius_m},{lat},{lon});relation["amenity"="clinic"](around:{radius_m},{lat},{lon});'
    ]

    for query_template in query_templates:
        # If we already found results, we don't need to run broader queries
        if ers:
            break

        # Reset radius for each new query type
        radius_km = 20
        while len(ers) < 10 and radius_km <= max_radius_km:
            radius_m = radius_km * 1000

            query = f"""
            [out:json];
            (
              {query_template.format(radius_m=radius_m, lat=lat, lon=lon)}
            );
            out center;
            """

            try:
                response = requests.get(overpass_url, params={'data': query}, timeout=25) # Add a 25-second timeout
                response.raise_for_status()
                data = response.json()

                current_results = []
                for element in data.get('elements', []):
                    tags = element.get('tags', {})
                    name = tags.get('name', 'N/A')

                    if 'center' in element:
                        er_lat, er_lon = element['center']['lat'], element['center']['lon']
                    else:
                        er_lat, er_lon = element.get('lat'), element.get('lon')

                    if name != 'N/A' and er_lat is not None and er_lon is not None:
                        # Avoid adding duplicates from previous searches
                        if not any(d['name'] == name for d in ers):
                            current_results.append({'name': name, 'lat': er_lat, 'lon': er_lon})

                ers.extend(current_results)

                # If we haven't found enough, double the search radius for the next attempt
                if len(ers) < 10:
                    radius_km *= 2

            except requests.exceptions.RequestException as e:
                # Handle timeouts and other network errors
                error_message = f"Error querying the location service: {e}. The service may be slow or the location is too broad."
                print(error_message)
                # Return the error message instead of an empty list to provide feedback
                return error_message

    return ers


def FindER(location_input):
    ER = dict()

    start_coords = get_location_coords(location_input)
    # Check if get_location_coords returned an error string or is otherwise invalid
    if not isinstance(start_coords, tuple):
        return f"Could not find coordinates for the provided location: '{location_input}'.", None

    nearby_ers = find_nearest_ers_expanding(start_coords[0], start_coords[1])
    if not nearby_ers:
        return "Could not find any nearby emergency rooms, even after expanding the search.", None
    
    # If the search function returned an error message, pass it along.
    if isinstance(nearby_ers, str):
        return nearby_ers, None

    for er in nearby_ers:
        er_coords = (er['lat'], er['lon'])
        distance = great_circle(start_coords, er_coords).miles
        er['distance_miles'] = distance

    nearby_ers.sort(key=lambda x: x.get('distance_miles', 9999))

    # Return the top 5 results as a list of dictionaries, preserving coordinates
    top_ers = nearby_ers[:5]

    # For email formatting, create the simple dictionary as before
    for er in top_ers:
        ER[er['name']] = f"{er.get('distance_miles', 0):.2f} miles away"
    return ER, top_ers