import os
import glob
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr

# Define file paths (Modify these paths before running)
input_dir = "path/to/data"  # Change this to the directory containing your files
shp_files = glob.glob(os.path.join(input_dir, "pink2018/spectxtgeo2/*.shp"))
fie_files = glob.glob(os.path.join(input_dir, "object_newresults/pink2018/*.shp"))

# Initialize lists to store results
area_avg_list = []
file_names = []

# Loop through each shapefile
for i, shp_path in enumerate(shp_files):
    file_name = os.path.basename(shp_path).split('.')
    print("Processing:", file_name)
    
    # Load shapefiles
    feature_gdf = gpd.read_file(fie_files[0])  # Assuming same reference for all
    shape_gdf = gpd.read_file(shp_path)
    
    # Perform spatial join (object values with area information)
    joined_df = gpd.sjoin(feature_gdf, shape_gdf)
    area_df = gpd.sjoin(shape_gdf, feature_gdf)
    
    # Compute centroids and distances
    centroids = area_df.centroid
    area_df = pd.concat([area_df, centroids], axis=1)
    
    distances = joined_df.geometry.distance(area_df.geometry)
    joined_df = pd.concat([joined_df, distances], axis=1)
    
    # Sort by distance and retain closest objects
    joined_df = joined_df.sort_values(by=joined_df.columns[-1], ascending=True).groupby("samples_cs").first()
    joined_df = joined_df.drop(columns=[joined_df.columns[-1]])
    
    # Organize columns
    cols = joined_df.columns.tolist()
    joined_df = joined_df[cols[:-7] + cols[-3:] + cols[-7:-3]]
    
    # Compute area and save
    area_values = area_df.area / 10**6  # Convert to square km
    joined_df.to_csv(os.path.join(input_dir, f"processed_{file_name[0]}.csv"), header=True)
    
    # Store results
    area_avg_list.append(area_values.mean())
    file_names.append(file_name[0])

# Save area summary
area_summary = pd.DataFrame(zip(file_names, area_avg_list), columns=["filename", "area_objects"])
area_summary.to_csv(os.path.join(input_dir, "area_summary.csv"), header=True)
print("Processing complete. Results saved.")