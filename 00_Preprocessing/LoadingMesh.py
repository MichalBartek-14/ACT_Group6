import open3d as o3d
import numpy as np


mesh_path = "C:/Users/mees2/Downloads/WUR_ACT_PG_250515/WUR_ACT_PG_250515/Trial_trenches/3D_modellen/51025869_Proefsleuf1_DaCosta_16042025_JCTSimons.ply"
GPR_data_path = ...


def load_GPR_point_cloud(file_path):
    """Load point cloud data from a LAZ file using PDAL."""
    pipeline_json = {
        "pipeline": [
            file_path,
            {
                "type": "filters.range",
                "limits": "Z[-1000:1000]"
            }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    arrays = pipeline.arrays
    return arrays[0]


def preprocess_trench(path):
    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(path)
    # Check if the mesh has vertex normals and colors
    mesh.compute_vertex_normals()
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], window_name="Open3D Viewer")


def preprocess_ground_data(mesh_path):



def garbagemethod():
    # Load as point cloud
    pcd = o3d.io.read_point_cloud(
        "C:/Users/mees2\Downloads/WUR_ACT_PG_250515/WUR_ACT_PG_250515/Trial_trenches/3D_modellen/51025869_Proefsleuf1_DaCosta_16042025_JCTSimons.ply")
    # Optional: estimate normals (if needed for rendering or processing)
    pcd.estimate_normals()
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Open3D Point Cloud Viewer")


# garbagemethod()
