import numpy as np
import open3d as o3d
from laspy import read
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
import os

def load_las_point_cloud(file_path):
    las = read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(las.point_format)
    reflectivity = las.red / np.max(las.red)
    return points, reflectivity

def voxelize_reflectivity(points, reflectivity, voxel_size):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    voxel_grid = np.full(dims, np.nan, dtype=np.float32)
    count_grid = np.zeros(dims, dtype=int)
    indices = ((points - min_bound) / voxel_size).astype(int)

    for i, idx in enumerate(indices):
        x, y, z = idx
        if all(0 <= idx[j] < dims[j] for j in range(3)):
            if np.isnan(voxel_grid[x, y, z]):
                voxel_grid[x, y, z] = reflectivity[i]
            else:
                voxel_grid[x, y, z] += reflectivity[i]
            count_grid[x, y, z] += 1

    # Average reflectivity per voxel
    with np.errstate(invalid='ignore'):
        voxel_grid = np.divide(voxel_grid, count_grid, where=count_grid != 0)

    return voxel_grid, min_bound, voxel_size

def compute_z_gradient(voxel_grid, compensation_power, base_threshold):
    kernel_z = np.zeros((3, 3, 3))
    kernel_z[1, 1, 0] = -1
    kernel_z[1, 1, 2] = 1

    filled = np.nan_to_num(voxel_grid, nan=0.0)
    grad_z = convolve(filled, kernel_z, mode='constant', cval=0.0)

    # Depth compensation
    z_indices = np.arange(filled.shape[2])[np.newaxis, np.newaxis, :]
    compensation = np.power((z_indices + 1), compensation_power)
    grad_z = np.abs(grad_z) * compensation

    edges = grad_z > base_threshold
    print(f"grad_z stats: min={np.min(grad_z):.6f}, max={np.max(grad_z):.6f}, mean={np.mean(grad_z):.6f}")
    return np.argwhere(edges)

def cluster_voxels(voxel_coords, eps, min_samples):
    ### 2
    if len(voxel_coords) == 0:
        return np.array([]), np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(voxel_coords)
    labels = clustering.labels_

    return voxel_coords, labels

def create_labeled_voxel_grid(shape, coords, labels):
    label_grid = np.zeros(shape, dtype=np.uint8)
    for i, idx in enumerate(coords):
        x, y, z = idx
        if labels[i] != -1:
            label_grid[x, y, z] = 1  # Mark root cluster
    return label_grid

def visualize_clusters(coords, labels, min_bound, voxel_size):
    colors = plt.cm.jet((labels % 20) / 20)[:, :3]
    points = coords * voxel_size + min_bound
    points[:, 2] *= 0.03
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Root Clusters")

def main():
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\Proefsleuf_4.las"
    #file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\P1_1mBuffer.las"
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found.")

    """"the parameters to alter:"""
    defined_voxel_size = 0.02
    #-----------z-gradient values-------#
    defined_compensation_power = 1.05 #1.05 default
    defined_base_threshold = 20000 #20k for TvG, 15k for Costakade
    #-----------clustering--------------#
    defined_eps = 4
    defined_min_samples = 10


    print("Loading point cloud...")
    points, reflectivity = load_las_point_cloud(file_path)
    print(
        f"Reflectivity stats: min={np.min(reflectivity)}, max={np.max(reflectivity)}, mean={np.mean(reflectivity):.4f}")

    print("Voxelizing reflectivity...")
    voxel_grid, min_bound, voxel_size = voxelize_reflectivity(points, reflectivity, voxel_size=defined_voxel_size)

    non_empty_voxels = np.count_nonzero(~np.isnan(voxel_grid))
    total_voxels = voxel_grid.size
    print(f"Non-empty voxels: {non_empty_voxels} / {total_voxels} ({100 * non_empty_voxels / total_voxels:.2f}%)")

    print("Computing Z-gradient (with depth compensation)...")
    edge_voxels = compute_z_gradient(voxel_grid,
                                     compensation_power=defined_compensation_power,
                                     base_threshold=defined_base_threshold)
    print(f"Detected {len(edge_voxels)} potential root edges.")

    if len(edge_voxels) == 0:
        print("No edge voxels detected. Adjust gradient threshold or check data.")

    print("Clustering edge voxels with DBSCAN...")
    clustered_coords, labels = cluster_voxels(edge_voxels,
                                              eps=defined_eps,
                                              min_samples=defined_min_samples)
    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")

    voxel_labels = create_labeled_voxel_grid(voxel_grid.shape, clustered_coords, labels)
    print("Visualizing clusters...")
    visualize_clusters(clustered_coords, labels, min_bound, voxel_size)

    non_empty_voxels = np.count_nonzero(~np.isnan(voxel_grid))
    print(f"Non-empty voxels: {non_empty_voxels} / {voxel_grid.size}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
