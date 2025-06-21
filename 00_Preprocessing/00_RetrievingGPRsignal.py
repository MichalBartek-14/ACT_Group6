import geopandas as gpd
import json
import subprocess
import os

def process_lidar_data(input_laz, gpkg, output_las, target_crs="EPSG:28992"):
    """
    Processes LiDAR data by cropping it to a specified polygon and saving the output.

    :param input_laz: Path to the input LAZ file.
    :param gpkg: Path to the GeoPackage file containing the polygon for cropping.
    :param output_las: Path to the output LAS file.
    :param target_crs: Coordinate Reference System to use for the output.
    """

    # Load Shapefile
    gdf = gpd.read_file(gpkg)
    gdf = gdf.to_crs(target_crs)

    # PDAL Pipeline to read the point cloud of the location
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_laz,
                "spatialreference": target_crs
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

    # Write and Run PDAL Pipeline
    pipeline_path = "pipeline.json"
    with open(pipeline_path, "w") as f:
        json.dump(pipeline, f, indent=4)

    subprocess.run(["pdal", "pipeline", pipeline_path])

    # Clean up
    os.remove(pipeline_path)

#usage:
"""
Note: user must have the GPR point cloud of the larger area in the LAZ file format alongside
 the gpkg file of the desired location they intend to intersect.
 The function produces a new .las file with coordinate system assigned.
"""
process_lidar_data(
     input_laz=r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\WUR_ACT_PG_250515\WUR_ACT_PG_250515\LAZ_Euroradar\TvGPark.laz",
     gpkg=r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Valid_loc_3.gpkg",
     output_las="Proefsleuf_test.las"
 )
