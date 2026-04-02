# import rasterio
# import numpy as np
# import pandas as pd
#
#
# def tif_to_population_df(tif_path):
#     with rasterio.open(tif_path) as src:
#         # Read the population data (band 1)
#         data = src.read(1)
#
#         # Get the affine transform (mapping pixel to coordinate)
#         transform = src.transform
#         nodata = src.nodata
#
#         # Create a grid of row and column indices
#         rows, cols = np.indices(data.shape)
#
#         # Convert pixel indices to longitude and latitude
#         lons, lats = rasterio.transform.xy(transform, rows, cols)
#
#         # Flatten arrays for DataFrame creation
#         lons = np.array(lons).flatten()
#         lats = np.array(lats).flatten()
#         pop = data.flatten()
#
#         # Filter out 'No Data' values and zeros to save memory/space
#         mask = (pop != nodata) & (pop > 0)
#
#         # Create the DataFrame
#         df = pd.DataFrame(
#             {
#                 'lat': np.round(lats[mask], 2),
#                 'lon': np.round(lons[mask], 2),
#                 'population_count': pop[mask]
#             }
#         )
#
#         # Since we rounded to 2 decimals, multiple pixels may now
#         # share the same lat/lon. We aggregate them by summing.
#         df = df.groupby(['lat', 'lon'], as_index=False)['population_count'].sum()
#
#         return df
# df = tif_to_population_df('data/shapes_2026/global_pop_2026_CN_1km_R2025A_UA_v1.tif')
# # TODO join with _grid_data.csv
