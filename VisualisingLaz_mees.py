import pdal
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import pyvista as pv


def load_point_cloud(file_path):
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


def print_point_cloud_attributes(points):
    """Print attributes of the point cloud data."""
    print(points.dtype.names)


def visualize_high_intensity(points):
    """Visualize high intensity points in the point cloud."""
    # compress the z-scale
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    #z = xyz[:, 2]
    # 10 default
    z = (xyz[:, 2])/50
    xyz[:, 2] = z

    r = points['Red'].astype(np.float32) / 65535
    g = points['Green'].astype(np.float32) / 65535
    b = points['Blue'].astype(np.float32) / 65535
    rgb = np.vstack((r, g, b)).T
    gray = r.astype(float)

    # Mask for strong signals (e.g., > 0.8) deep
    mask_strong = (gray > 0.8) & (z < -0.01)
    xyz_strong = xyz[mask_strong]

    pcd_strong = o3d.geometry.PointCloud()
    pcd_strong.points = o3d.utility.Vector3dVector(xyz_strong)
    pcd_strong.paint_uniform_color([1.0, 0.0, 0.0])

    # Overlay with full point cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd_all, pcd_strong])


def visualize_parabolas(points):
    """ Visualize high intensity points in the point cloud. """
    # compress the z-scale
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    print(f"xyz min/max: \n"
          f"x: {xyz[:,0].min():.4f}, \t{xyz[:,0].max():.4f}  \n"
          f"y: {xyz[:,1].min():.4f}, \t{xyz[:,1].max():.4f} \n"
          f"z: {xyz[:,2].min():.4f}, \t\t{xyz[:,2].max():.4f}\n")

    print("shape: ", np.shape(xyz))

    z = (xyz[:, 2]) / 100
    xyz[:, 2] = z

    r = (points['Red'].astype(np.float32) / 65535)                              # .astype(float)
    print(f"Red intensity range: min = {r.min():.4f}, max = {r.max():.4f}")
    g = (points['Green'].astype(np.float32) / 65535)                            # .astype(float)
    print(f"Green intensity range: min = {g.min():.4f}, max = {g.max():.4f}")
    b = (points['Blue'].astype(np.float32) / 65535)                             # .astype(float)
    print(f"Blue intensity range: min = {b.min():.4f}, max = {b.max():.4f}")

    rgb = np.vstack((r, g, b)).T

    gray = (r + g + b)/3
    print(gray.min(), gray.max())

    # Mask for certain signals (e.g., > 0.8)
    mask_strong = (gray > 0.75) & (z < -0.01)
    mask_weak = (gray < 0.75) & (z < -0.01)

    print(np.shape(mask_strong))
    xyz_strong = xyz[mask_strong]
    xyz_weak = xyz[mask_weak]

    # mask_strong = (r > 0)
    # xyz_strong = xyz[mask_strong]

    pcd_strong = o3d.geometry.PointCloud()
    pcd_strong.points = o3d.utility.Vector3dVector(xyz_strong)
    pcd_strong.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_weak = o3d.geometry.PointCloud()
    pcd_weak.points = o3d.utility.Vector3dVector(xyz_weak)
    pcd_weak.paint_uniform_color([0.0, 1.0, 0.0])

    # Overlay with full point cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    mask_1 = (gray > 0.2) & (gray <= 0.4) & (z < -0.2)
    mask_2 = (gray > 0.6) & (gray <= 0.7) & (z < -0.2)
    mask_3 = (gray > 0.7) & (gray <= 0.8) & (z < -0.2)
    mask_4 = (gray > 0.8) & (gray <= 0.9) & (z < -0.2)
    mask_5 = (gray > 0.9) & (z < -0.01)

    # TODO fix gray values to pull apart signal and also choose better value for z values (should be higher/lower)

    print("mask_1 count:", np.sum(mask_1))
    print("mask_2 count:", np.sum(mask_2))
    print("mask_3 count:", np.sum(mask_3))
    print("mask_4 count:", np.sum(mask_4))
    print("mask_5 count:", np.sum(mask_5))

    xyz_1 = xyz[mask_1]
    xyz_2 = xyz[mask_2]
    xyz_3 = xyz[mask_3]
    xyz_4 = xyz[mask_4]
    xyz_5 = xyz[mask_5]

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(xyz_1)
    pcd_1.paint_uniform_color([1.0, 0.5, 0.0])  # Orange

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(xyz_2)
    pcd_2.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow

    pcd_3 = o3d.geometry.PointCloud()
    pcd_3.points = o3d.utility.Vector3dVector(xyz_3)
    pcd_3.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan

    pcd_4 = o3d.geometry.PointCloud()
    pcd_4.points = o3d.utility.Vector3dVector(xyz_4)
    pcd_4.paint_uniform_color([0.5, 0.0, 1.0])  # Purple

    pcd_5 = o3d.geometry.PointCloud()
    pcd_5.points = o3d.utility.Vector3dVector(xyz_5)
    pcd_5.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta

    # o3d.visualization.draw_geometries([pcd_weak, pcd_strong])
    o3d.visualization.draw_geometries([pcd_2, pcd_3, pcd_4])


def main():
    # filepath for the valid data
    file_path = r"C:\Users\mees2\Downloads\P1_10mBuffer.las"
    # "C:\Users\mees2\Downloads\Proefsleuf_1.las"
    # C:\Users\mees2\Downloads\WUR_ACT_PG_250515\WUR_ACT_PG_250515\LAZ_Euroradar\Bomen-23-37.laz
    points = load_point_cloud(file_path)

    print_point_cloud_attributes(points)
    visualize_parabolas(points)
    #visualize_high_intensity(points)


if __name__ == "__main__":
    main()

# 23 - 37