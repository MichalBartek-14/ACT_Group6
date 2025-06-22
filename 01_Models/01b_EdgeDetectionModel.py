import numpy as np
import open3d as o3d
from laspy import read
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt

def load_las_point_cloud(file_path):
    """
    this function uses laspy to load the point cloud of GPR. It returns
    the individual points and its reflectivity from a point cloud dataset.
    :param file_path: expects a path to point cloud data in .LAZ format
    :return: points (coordinates) and their reflectivity values
    """
    las = read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # The reflectivity is stored in rgb values where r=b=g. Here we extract r as reflectivity
    # and the reflectivity is normalised
    reflectivity = las.red / np.max(las.red)
    return points, reflectivity

def voxelize_reflectivity(points, reflectivity, voxel_size):
    """
    This function voxelizes the reflectivity. This is done in order to:
     a) smoothen the computation process
     b) deal with the noisy GPR data.
     Function creates the voxels which are later used for detection of root-like structures in gpr data.
    :param points: the float array of the xyz coordinates for each point of the point cloud
    :param reflectivity: the reflection parameter retrieved from the data for each point
    :param voxel_size: define a voxel size for which the values will be interpolated
    (0.02, 0.03 are usually best options for the size of the validation trench)
    :return:
    - min_bound = the min extent of the data
    - voxel_grid = the averrgae wvalue of reflectivity of each of the voxels from the GPR point cloud
    - voxel_size = returns the voxel size for more efficient calling of the function
    """
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # the rounding is applied on the dimensions
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Fills the numpy array with NaNs
    voxel_grid = np.full(dims, np.nan, dtype=np.float32)
    count_grid = np.zeros(dims, dtype=int)

    # Attributes each voxel an index
    indices = ((points - min_bound) / voxel_size).astype(int)

    for i, idx in enumerate(indices):
        x, y, z = idx
        # Makes sure that the point is within the dimensions
        if all(0 <= idx[j] < dims[j] for j in range(3)):
            # Stores the values of the pixels in the voxels
            if np.isnan(voxel_grid[x, y, z]):
                voxel_grid[x, y, z] = reflectivity[i]
            else:
                voxel_grid[x, y, z] += reflectivity[i]
            count_grid[x, y, z] += 1

    # Averages reflectivity per voxel
    with np.errstate(invalid='ignore'):
        voxel_grid = np.divide(voxel_grid, count_grid, where=count_grid != 0)

    return voxel_grid, min_bound, voxel_size

def compute_z_gradient(voxel_grid, compensation_power, base_threshold):
    """
       Calculates the gradient of a 3D voxel grid along the z-axis
       and applies a depth compensation to highlight edges in the grid.
       :param voxel_grid: the voxel reflectivity values
       :param compensation_power: Depth compensation
       :param base_threshold: Threshold for a voxel z-gradient to be considered edge
       :return: Coordinates of the edge voxels
       """
    # Kernel to detect the edges on the z-axis
    kernel_z = np.zeros((3, 3, 3))
    kernel_z[1, 1, 0] = -1
    kernel_z[1, 1, 2] = 1

    filled = np.nan_to_num(voxel_grid, nan=0.0)
    # Kernel that uses 0-padding at the boundaries
    grad_z = convolve(filled, kernel_z, mode='constant', cval=0.0)

    # Depth compensation
    z_indices = np.arange(filled.shape[2])[np.newaxis, np.newaxis, :]
    compensation = np.power((z_indices + 1), compensation_power)
    # The gradient on z is rescaled based on the depth with the predefined compensation
    grad_z = np.abs(grad_z) * compensation
    # Edges detected if the gradient_z is higher the threshold

    edges = grad_z > base_threshold
    print(f"grad_z stats: min={np.min(grad_z):.6f}, max={np.max(grad_z):.6f}, mean={np.mean(grad_z):.6f}")
    return np.argwhere(edges)

def cluster_voxels(voxel_coords, eps, min_samples):
    """
    Function clusters the voxels using the DBSCAN clustering method from sklearn
    :param voxel_coords: coordinates of the identified edges retrieved from edge_voxels
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: The function returns the voxel coordinates and their corresponding cluster labels.
    """
    # Clusters the voxels based on the asigned values
    if len(voxel_coords) == 0:
        return np.array([]), np.array([])
    # EPS and min_samples are the parameters for the DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(voxel_coords)
    labels = clustering.labels_

    return voxel_coords, labels

def visualize_clusters(coords, labels, min_bound, voxel_size):
    """
    This function visualizes the clusters in the pointcloud
    :param coords: coordinates xyz for the voxels
    :param labels: Labels of the clusters
    :param min_bound: from the voxelization
    :param voxel_size: predefined voxel size
    :return: o3d visualisation of clusters
    """
    #coloring options (copper, jet, Reds, gray)
    # %int modulo operand can be changed for visualisations with fewer colors.
    colors = plt.cm.copper((labels % 20) / 20)[:, :3]
    points = coords * voxel_size + min_bound
    # for better visualisation the z-axis is shrunk
    points[:, 2] *= 0.03
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Root Clusters")

def main():
    # --- FILE PATH to desired LAS/LAZ file --- #
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\Proefsleuf_4.las"
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found.")

    # --- PARAMETERS to alter to fine-tune on desired point cloud dataset --- #

    defined_voxel_size = 0.02
    #-----------   z-gradient values    -------#
    defined_compensation_power = 1.05 #1.05 default
    defined_base_threshold = 20000 #20k for TvG, 15k for Costakade
    #-----------   clustering    --------------#
    defined_eps = 4 # increase
    defined_min_samples = 10

    # --- Calling the functions --- #
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

    print("Visualizing clusters...")
    visualize_clusters(clustered_coords, labels, min_bound, voxel_size)

    non_empty_voxels = np.count_nonzero(~np.isnan(voxel_grid))
    print(f"Non-empty voxels: {non_empty_voxels} / {voxel_grid.size}")

if __name__ == "__main__":
    main()
