import open3d as o3d
import numpy as np
import pdal
import json

mesh_path = "C:/Users/mees2/Downloads/WUR_ACT_PG_250515/WUR_ACT_PG_250515/Trial_trenches/3D_modellen/51025869_Proefsleuf1_DaCosta_16042025_JCTSimons.ply"


def vertex_visualisation(path):
    """ visualize a photogrammetry mesh with computed vertexes for visualisation.
    @param path: expects ply file
    """
    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(path)
    # Check if the mesh has vertex normals and colors
    mesh.compute_vertex_normals()
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], window_name="Open3D Viewer")


def normals_visualisation(path):
    """visualize a photogrammetry mesh with computed estimated normals for visualisation.
    @param path: expects ply file
    """
    # Load as point cloud
    pcd = o3d.io.read_point_cloud(path)
    # estimate normals
    pcd.estimate_normals()
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Open3D Point Cloud Viewer")


vertex_visualisation(mesh_path)

normals_visualisation(mesh_path)
