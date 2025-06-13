import pdal
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


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
    rgb_transparant = ...  # TODO
    gray = (r + g + b)/3
    print(gray.min(), gray.max())
    # Mask for strong signals (e.g., > 0.8) deep
    mask_strong = (gray > 0.75) & (z < -0.01)
    xyz_strong = xyz[mask_strong]

    # mask_strong = (r > 0)
    # xyz_strong = xyz[mask_strong]

    pcd_strong = o3d.geometry.PointCloud()
    pcd_strong.points = o3d.utility.Vector3dVector(xyz_strong)
    pcd_strong.paint_uniform_color([1.0, 0.0, 0.0])

    # Overlay with full point cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)


    o3d.visualization.draw_geometries([pcd_all, pcd_strong])


def main():
    # filepath for the valid data
    file_path = r"C:\Users\mees2\Downloads\Proefsleuf_1.las"

    points = load_point_cloud(file_path)

    print_point_cloud_attributes(points)
    visualize_parabolas(points)


if __name__ == "__main__":
    main()
