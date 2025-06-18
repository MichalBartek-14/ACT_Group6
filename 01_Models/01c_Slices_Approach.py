import numpy as np
import open3d as o3d
from laspy import read
from scipy.ndimage import sobel
import os

# ======================
# 1. Load & Voxelize
# ======================
def load_las_point_cloud(file_path):
    las = read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
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

    with np.errstate(invalid='ignore'):
        voxel_grid = np.divide(voxel_grid, count_grid, where=count_grid != 0)
    return voxel_grid, min_bound, voxel_size

# ======================
# 2. Slice & Detect
# ======================
### old v2
def detect_roots_by_vertical_slicing(voxel_grid, gradient_threshold):
    filled = np.nan_to_num(voxel_grid, nan=0.0)
    root_voxels = []

    # --- YZ slicing (fixed X) ---
    for x in range(filled.shape[0]):
        yz_slice = filled[x, :, :]  # shape: (Y, Z)

        grad_y = sobel(yz_slice, axis=0)
        grad_z = sobel(yz_slice, axis=1)
        magnitude = np.hypot(grad_y, grad_z)

        edges = magnitude > gradient_threshold * np.max(magnitude)

        y_coords, z_coords = np.argwhere(edges).T
        x_coords = np.full_like(y_coords, x)
        yz_voxels = np.stack([x_coords, y_coords, z_coords], axis=1)
        root_voxels.extend(yz_voxels)

    # --- XZ slicing (fixed Y) ---
    for y in range(filled.shape[1]):
        xz_slice = filled[:, y, :]  # shape: (X, Z)

        grad_x = sobel(xz_slice, axis=0)
        grad_z = sobel(xz_slice, axis=1)
        magnitude = np.hypot(grad_x, grad_z)

        edges = magnitude > gradient_threshold * np.max(magnitude)

        x_coords, z_coords = np.argwhere(edges).T
        y_coords = np.full_like(x_coords, y)
        xz_voxels = np.stack([x_coords, y_coords, z_coords], axis=1)
        root_voxels.extend(xz_voxels)

    return np.array(root_voxels)

# ======================
# 3. Visualize
# ======================
def visualize_voxels(voxel_coords, min_bound, voxel_size):
    if len(voxel_coords) == 0:
        print("No root voxels found.")
        return

    points = voxel_coords * voxel_size + min_bound
    points[:, 2] *= 0.03  # optional exaggeration of height

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.array([[0.2, 0.8, 0.1] for _ in range(len(points))])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Vertical Slice Root Detection")

# ======================
# 4. Run
# ======================
def main():
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\Proefsleuf_1.las"
    if not os.path.exists(file_path):
        raise FileNotFoundError("LAS file not found.")

    voxel_size = 0.03
    print("Loading LAS...")
    points, reflectivity = load_las_point_cloud(file_path)

    print("Voxelizing...")
    voxel_grid, min_bound, voxel_size = voxelize_reflectivity(points, reflectivity, voxel_size)

    print("Slicing and detecting roots...")
    root_voxels = detect_roots_by_vertical_slicing(voxel_grid, gradient_threshold=0.8)
    print(f"Detected {len(root_voxels)} root candidate voxels.")

    print("Visualizing...")
    visualize_voxels(root_voxels, min_bound, voxel_size)

if __name__ == "__main__":
    main()
