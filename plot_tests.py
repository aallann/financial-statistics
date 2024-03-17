import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the built-in world dataset
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Plot the world map
world.plot()
plt.show()

# Assume 'my_data' is a DataFrame that contains a 'country_code' column and a 'value' column
my_data = pd.DataFrame({"country_code": ["FRA", "GBR", "USA"], "value": [1, 2, 3]})

# Merge 'my_data' with the 'world' GeoDataFrame
merged = world.set_index("iso_a3").join(my_data.set_index("country_code"))

# Plot the world map with the 'value' column used to determine the color of each country
merged.plot(column="value", legend=True)
plt.show()
