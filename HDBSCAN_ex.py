import pdal
import json
import numpy as np
import hdbscan
import numpy as np
import open3d as o3d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from laspy import read

def voxelize_point_cloud(points, voxel_size=0.03):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    voxel_grid = np.zeros(dims, dtype=np.float32)

    # Fill voxel grid with point counts (density)
    indices = ((points - min_bound) / voxel_size).astype(int)
    for idx in indices:
        if all(0 <= idx[i] < dims[i] for i in range(3)):
            voxel_grid[tuple(idx)] += 1.0  # accumulate density

    return voxel_grid, min_bound, voxel_size


def create_z_voxel_grid(points, voxel_size):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    z_grid = np.full(dims, np.nan)  # to store mean Z
    count_grid = np.zeros(dims, dtype=int)
    sum_grid = np.zeros(dims, dtype=np.float32)

    indices = ((points - min_bound) / voxel_size).astype(int)

    for point, idx in zip(points, indices):
        if all(0 <= idx[i] < dims[i] for i in range(3)):
            sum_grid[tuple(idx)] += point[2]
            count_grid[tuple(idx)] += 1

    mask = count_grid > 0
    z_grid[mask] = sum_grid[mask] / count_grid[mask]

    return z_grid, min_bound
def detect_edges(voxel_grid, threshold=1.5):
    # Z-gradient kernel: detects vertical signal difference (horizontal edge)
    z_gradient_kernel = np.zeros((3, 3, 3))
    z_gradient_kernel[0, 1, 1] = -1  # below
    z_gradient_kernel[2, 1, 1] = 1   # above

    # Convolve to compute difference in Z direction
    gradient_z = convolve(voxel_grid, z_gradient_kernel, mode='constant', cval=0.0)

    # Threshold the gradient to find strong edges
    edges = np.abs(gradient_z) > threshold

    return np.argwhere(edges)

def visualize_edges(edge_voxels_world):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(edge_voxels_world)
    pcd.paint_uniform_color([1, 0, 0])  # red points

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Edge Voxels (Z Gradient)",
        point_show_normal=False
    )

def main():
    # ---- Load LAS file ----
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\Proefsleuf_1.las"
    las = read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # ---- Optional Z scaling ----
    points[:, 2] /= 30.0

    # ---- Voxelization ----
    voxel_size = 0.03
    voxel_grid, min_bound, voxel_size = voxelize_point_cloud(points, voxel_size)

    # ---- Detect edges based on Z-gradient ----
    edge_voxels = detect_edges(voxel_grid, threshold=1.5)
    edge_voxels_world = edge_voxels * voxel_size + min_bound

    print(f"Detected {edge_voxels.shape[0]} edge voxels.")
    print("Sample world coordinates:", edge_voxels_world[:5])

    # ---- Visualize ----
    visualize_edges(edge_voxels_world)

if __name__ == "__main__":
    main()



