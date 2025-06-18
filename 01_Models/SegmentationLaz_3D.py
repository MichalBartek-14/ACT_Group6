import pdal
import json
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import open3d as o3d

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
    return pipeline.arrays[0]

def compute_pca_features(points_xyz, k_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points_xyz)
    _, indices = nbrs.kneighbors(points_xyz)

    linearity, planarity, curvature = [], [], []

    for idx_list in indices:
        neighbors = points_xyz[idx_list]
        cov = np.cov(neighbors.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1] + 1e-10
        l1, l2, l3 = eigvals
        linearity.append((l1 - l2) / l1)
        planarity.append((l2 - l3) / l1)
        curvature.append(l3 / (l1 + l2 + l3))

    return {
        "linearity": np.array(linearity),
        "planarity": np.array(planarity),
        "curvature": np.array(curvature)
    }

def voxel_downsample(points_xyz, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)

def run_hdbscan(points_xyz, reflectivity, min_cluster_size, min_samples):
    pca_features = compute_pca_features(points_xyz)
    features = np.column_stack((
        points_xyz,
        reflectivity[:len(points_xyz)],
        pca_features["linearity"],
        pca_features["planarity"],
        pca_features["curvature"]
    ))

    scaled_features = StandardScaler().fit_transform(features)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        max_cluster_size = 40
    )
    labels = clusterer.fit_predict(scaled_features)
    return labels

def visualize_clusters_3d(xyz, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    unique_labels = np.unique(labels)
    color_map = plt.get_cmap("tab20", len(unique_labels))
    colors = np.array([color_map(l % 20)[:3] if l != -1 else [0, 0, 0] for l in labels])

    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 10.0  #15 works well for visualisation
    opt.background_color = np.asarray([1, 1, 1])  # White background (optional)

    vis.run()
    vis.destroy_window()

def main():
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\P1_voxel_rootsegmentation.las"
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\Proefsleuf_1.las"

    points = load_point_cloud(file_path)

    # Extract XYZ and reflectivity (Red channel as proxy)
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    reflectivity = points['Red'].astype(np.float32) / 65535.0

    #scale z for visualisation
    xyz[:, 2] /= 30.0

    # Downsample to speed up clustering
    xyz_down = voxel_downsample(xyz, voxel_size=0.03)

    # Match reflectivity length (approximate — better would be remapping)
    reflectivity_down = reflectivity[:len(xyz_down)]

    min_cluster_size = 20
    min_samples = 50 #higher = more clusters

#run and visualise
    labels = run_hdbscan(xyz_down, reflectivity_down,min_cluster_size,min_samples)

    visualize_clusters_3d(xyz_down, labels)

if __name__ == "__main__":
    main()

