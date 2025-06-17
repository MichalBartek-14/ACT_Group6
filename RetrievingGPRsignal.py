import geopandas as gpd
import shapely
import json
import subprocess
import os

# --- Input Files ---
input_laz = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\WUR_ACT_PG_250515\WUR_ACT_PG_250515\LAZ_Euroradar\TvGPark.laz"
gpkg = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Valid_loc_3.gpkg"

output_las = "Proefsleuf_3.las"  # output file name
target_crs = "EPSG:28992"  # Set to your desired CRS (e.g., UTM zone 33N)

# --- Load Shapefile ---
gdf = gpd.read_file(gpkg)
gdf = gdf.to_crs(target_crs)

# --- Construct PDAL Pipeline ---
pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": input_laz,
            "spatialreference": target_crs  # Set CRS since original has none
        },
        {
            "type": "filters.crop",
            "polygon": gdf.geometry.union_all().wkt
        },
        {
            "type": "writers.las",
            "filename": output_las,
            "compression": "none"
        }
    ]
}

# --- Write and Run PDAL Pipeline ---
pipeline_path = "pipeline.json"
with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=4)

subprocess.run(["pdal", "pipeline", pipeline_path])

# --- Clean up ---
os.remove(pipeline_path)
