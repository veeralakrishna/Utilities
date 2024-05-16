import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3959  # Radius of Earth in miles
    return r * c

def filter_within_radius(dataframe, center_lat, center_lon, radius_miles=50):
    """
    Filter the dataframe to include only data points within a certain radius
    of a given latitude and longitude.

    Parameters:
    dataframe (pandas.DataFrame): Input DataFrame containing latitude and longitude columns.
    center_lat (float): Latitude of the center point.
    center_lon (float): Longitude of the center point.
    radius_miles (float): Radius in miles (default is 50 miles).

    Returns:
    pandas.DataFrame: Filtered DataFrame containing only data points within the specified radius.
    """
    # Calculate the distances from each point to the center point
    distances = haversine(dataframe['latitude'], dataframe['longitude'], center_lat, center_lon)
    
    # Filter the DataFrame to include only data points within the radius
    filtered_df = dataframe[distances <= radius_miles]

    return filtered_df


# Example usage
import pandas as pd

# Sample DataFrame
data = {
    'usage_id': [1, 2, 3, 4],
    'state': ['NY', 'CA', 'TX', 'FL'],
    'county': ['New York', 'Los Angeles', 'Harris', 'Miami-Dade'],
    'latitude': [40.7128, 34.0522, 29.7604, 25.7617],
    'longitude': [-74.0060, -118.2437, -95.3698, -80.1918]
}
df = pd.DataFrame(data)

# Latitude and longitude of the center point
center_lat = 34.0522
center_lon = -118.2437

# Call the function
filtered_df = filter_within_radius(df, center_lat, center_lon)

# Print the filtered DataFrame
print(filtered_df.shape)
