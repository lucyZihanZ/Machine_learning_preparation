import json
import plotly.express as px

# Load the GeoJSON file containing geographic data for Illinois counties
with open('illinois-counties.geojson') as f:
    geojson_data = json.load(f)

# Assuming df_avg is a DataFrame that includes the average housing price for each county in Illinois
fig = px.choropleth(
    df_avg,  # DataFrame containing the average sold price for each county
    geojson=geojson_data,  # GeoJSON data
    locations="county",  # Column in df_avg that matches GeoJSON features
    featureidkey="properties.name",  # Path in GeoJSON to match with locations
    color="sold_price",  # Column in df_avg used to determine the color of the counties
    color_continuous_scale="Viridis",  # Color scale for the Choropleth map
    scope="usa",
    labels={"sold_price": "Average Sold Price"}  # Label for the color scale
)

# Update layout to minimize white space around the map
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# Ensure that the map scales and centers correctly around the locations
fig.update_geos(fitbounds="locations", visible=True)
# Display the map
fig.show()
