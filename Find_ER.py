import sys
import requests
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut


def get_location_coords(location_name: str):
    """
    Converts a location name into latitude and longitude coordinates.
    """
    try:
        geolocator = Nominatim(user_agent="er_finder_script_unbounded")
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return f"Could not find coordinates for '{location_name}'."
    except GeocoderTimedOut:
        print("Error: Geocoding service timed out. Please try again.")
        return None


def find_nearest_ers_expanding(lat: float, lon: float):

    overpass_url = "http://overpass-api.de/api/interpreter"

    # Start with a 10km radius and expand until we find enough results
    radius_km = 10
    max_radius_km = 300  # Set a reasonable max limit to avoid overly long searches
    ers = []

    while len(ers) < 5 and radius_km <= max_radius_km:
        radius_m = radius_km * 1000

        query = f"""
        [out:json];
        (
          node["amenity"="hospital"]["emergency"="yes"](around:{radius_m},{lat},{lon});
          way["amenity"="hospital"]["emergency"="yes"](around:{radius_m},{lat},{lon});
          relation["amenity"="hospital"]["emergency"="yes"](around:{radius_m},{lat},{lon});
        );
        out center;
        """

        try:
            response = requests.get(overpass_url, params={'data': query})
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

            # If we haven't found at least 10, double the search radius for the next attempt
            if len(ers) < 5:
                radius_km *= 2

        except requests.exceptions.RequestException as e:
            print(f"Error querying the Overpass API: {e}")
            return []

    return ers


def FindER(location_input):
    ER = dict()

    start_coords = get_location_coords(location_input)
    if not start_coords:
        return

    nearby_ers = find_nearest_ers_expanding(start_coords[0], start_coords[1])
    if not nearby_ers:
        return "Could not find any ERs, even after expanding the search."

    for er in nearby_ers:
        er_coords = (er['lat'], er['lon'])
        distance = great_circle(start_coords, er_coords).miles
        er['distance_miles'] = distance

    nearby_ers.sort(key=lambda x: x['distance_miles'])

    for i, er in enumerate(nearby_ers[:10], 1):  # Display top 5 results
        ER[er['name']] = f"{er['distance_miles']:.2f} miles away"
    return ER