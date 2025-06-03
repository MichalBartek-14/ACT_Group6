import pdal
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    r = points['Red'].astype(np.float32) / 65535
    g = points['Green'].astype(np.float32) / 65535
    b = points['Blue'].astype(np.float32) / 65535
    rgb = np.vstack((r, g, b)).T

    # Create point cloud and assign color
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    # Thresholding for strong signals deeper than -50
    mask_strong = (r > 0.8) & (points['Z'] < -50)
    xyz_strong = xyz[mask_strong]

    pcd_strong = o3d.geometry.PointCloud()
    pcd_strong.points = o3d.utility.Vector3dVector(xyz_strong)
    pcd_strong.paint_uniform_color([1.0, 0.0, 0.0])  # Highlight with red color

    # Visualize the high intensity
    o3d.visualization.draw_geometries([pcd_all, pcd_strong], window_name="High Intensity Visualization")

def visualize_dbscan_segmentation(points):
    """Visualize DBSCAN segmentation of high intensity points."""
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    r = points['Red'].astype(np.float32) / 65535

    # Thresholding for strong signals deeper than -50
    mask_strong = (r > 0.8) & (points['Z'] < -50)
    xyz_strong = xyz[mask_strong]

    # DBSCAN clustering to group reflectors
    clustering = DBSCAN(eps=0.2, min_samples=10).fit(xyz_strong)
    labels = clustering.labels_

    # Assign color to each cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # Exclude noise label (-1)

    colors = plt.get_cmap('tab10')((labels % 10) / 10.0)  # Map clusters to colors (cycling if >10)
    pcd_clustered = o3d.geometry.PointCloud()
    pcd_clustered.points = o3d.utility.Vector3dVector(xyz_strong)
    pcd_clustered.colors = o3d.utility.Vector3dVector(colors[:, :3])  # RGB only

    # Visualize clusters
    print(f"Clusters found (excluding noise): {n_clusters}")
    o3d.visualization.draw_geometries([pcd_clustered], window_name="DBSCAN Segmentation Visualization")

def main():
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\WUR_ACT_PG_250515\WUR_ACT_PG_250515\LAZ_Euroradar\Bomen-1-6.laz"
    points = load_point_cloud(file_path)

    print_point_cloud_attributes(points)
    visualize_high_intensity(points)
    visualize_dbscan_segmentation(points)

if __name__ == "__main__":
    main()
