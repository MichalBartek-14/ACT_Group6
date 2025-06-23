import numpy as np
import open3d as o3d
from laspy import read
from scipy.ndimage import sobel
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter

def load_las_point_cloud(file_path):
    """
    Loads the las files using laspy library
    :param file_path: Path to the las file to read
    :return: xyz coordinates of the pixels from the pointcloud and the reflectivity
    """
    las = read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    # Reflectivity is normalised
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

def detect_roots_by_vertical_slicing(voxel_grid, gradient_threshold):
    """
    Function applies the Sobel (from scipy) operator to detect edges in vertical slices
    of the voxel grid (both YZ and XZ planes). The edges are identified
    based on a gradient threshold. Voxels with gradients above this threshold
    are considered potential root locations.
    :param voxel_grid: values of voxels to be processed
    :param gradient_threshold: threshold from which the voxel is identified as a root
    :return: array of the voxels considered as roots
    """
    filled = np.nan_to_num(voxel_grid, nan=0.0)
    root_voxels = []

    # --- YZ slicing (fixed X) ---
    for x in range(filled.shape[0]):
        yz_slice = filled[x, :, :] # Y,Z Plane

        grad_y = sobel(yz_slice, axis=0)
        grad_z = sobel(yz_slice, axis=1)
        #
        magnitude = np.hypot(grad_y, grad_z)

        edges = magnitude > gradient_threshold * np.max(magnitude)

        y_coords, z_coords = np.argwhere(edges).T
        x_coords = np.full_like(y_coords, x)
        yz_voxels = np.stack([x_coords, y_coords, z_coords], axis=1)
        root_voxels.extend(yz_voxels)

    # --- XZ slicing (fixed Y) ---
    for y in range(filled.shape[1]):
        xz_slice = filled[:, y, :]  # X, Z plane

        grad_x = sobel(xz_slice, axis=0)
        grad_z = sobel(xz_slice, axis=1)
        magnitude = np.hypot(grad_x, grad_z)

        edges = magnitude > gradient_threshold * np.max(magnitude)

        x_coords, z_coords = np.argwhere(edges).T
        y_coords = np.full_like(x_coords, y)
        xz_voxels = np.stack([x_coords, y_coords, z_coords], axis=1)
        root_voxels.extend(xz_voxels)

    return np.array(root_voxels)

def visualize_voxels(voxel_coords, min_bound, voxel_size):
    """
    Function visualises the detected roots. Converts the voxel coordinates
    back to their original scale and creates a point cloud using Open3D.
    :param voxel_coords: the coordinates of the voxels
    :param min_bound: the minimum bounds of the area
    :param voxel_size: the size of voxels, which is visualised
    :return: visualisation of the voxels detected by this method.
    """
    if len(voxel_coords) == 0:
        print("No root voxels found.")
        return

    points = voxel_coords * voxel_size + min_bound
    points[:, 2] *= 0.03  # optional visualisation of z-axis into more realistic point cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #colors = plt.cm.copper(np.linspace(0, 1, len(points)))
    colors = np.array([[0.6, 0.3, 0.1] for _ in range(len(points))])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Vertical Slice Root Detection")

#--- Visualising slices ---#
def interpolate_raster(slice_data, method='gaussian'):
    """
    Interpolates a 2D slice to fill gaps between the layers.
    This function is then appplied in Slicing of the YZ and XZ planes.
    :param slice_data: sliced data without the applied slicing
    :param method: method that is desired for this application (gaussian worked the best)
    :return: interpolated data for the slicing
    """
    if method == 'gaussian':
        return gaussian_filter(slice_data, sigma=1)
    else:
        return slice_data

def browse_yz_slices(voxel_grid):
    """
    Slicing function for the YZ slicing along X
    :param voxel_grid: The values of the voxel
    :return: visualisation of the YZ slices
    """
    filled = np.nan_to_num(voxel_grid, nan=0.0)
    max_x = filled.shape[0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    def get_slice_image(x_idx):
        slice_data = filled[x_idx, :, :]

        #Remove empty Z-layers (without this the visualisation is unreadable)
        non_empty_z = np.any(slice_data > 0, axis=0)
        if not np.any(non_empty_z):
            return np.zeros_like(slice_data.T)

        slice_data = slice_data[:, non_empty_z]

        # Contrast normalization for enhanced visualisation
        nonzero = slice_data[slice_data > 0]
        low = np.percentile(nonzero, 8)
        high = np.percentile(nonzero, 92)
        slice_data = np.clip(slice_data, low, high)
        slice_norm = (slice_data - low) / (high - low + 1e-6)
        gamma = 0.5
        slice_norm = slice_norm ** gamma

        #raster interpolation
        slice_norm = interpolate_raster(slice_norm, method='gaussian')

        return slice_norm.T  # Transpose for imshow (shape: Z, Y)

    initial_slice = get_slice_image(0)
    slice_img = ax.imshow(initial_slice, cmap='OrRd', origin='lower',
                          aspect='auto', vmin=0, vmax=1)

    ax.invert_yaxis()
    ax.set_title("YZ Slice Viewer (Slice X=0)")
    ax.set_xlabel("Y axis")
    ax.set_ylabel("Z axis")

    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'X Slice', 0, max_x - 1, valinit=0, valstep=1)

    def update(val):
        x_idx = int(slider.val)
        slice_data = get_slice_image(x_idx)
        slice_img.set_data(slice_data)
        slice_img.set_extent([0, slice_data.shape[1], 0, slice_data.shape[0]])
        ax.set_title(f"YZ Slice Viewer (Slice X={x_idx})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def browse_xz_slices(voxel_grid):
    """
    Slicing function for the XZ slicing along Y
    :param voxel_grid: The values of the voxel
    :return: visualisation of the XZ slices
    """
    filled = np.nan_to_num(voxel_grid, nan=0.0)
    max_y = filled.shape[1]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    def get_slice_image(y_idx):
        slice_data = filled[:, y_idx, :]  # shape: (X, Z)

        # Remove empty Z-layers (without this the visualisation is unreadable)
        non_empty_z = np.any(slice_data > 0, axis=0)
        slice_data = slice_data[:, non_empty_z]

        # Contrast normalization (for enhanced visualisation results)
        nonzero = slice_data[slice_data > 0]
        low = np.percentile(nonzero, 8)
        high = np.percentile(nonzero, 92)
        slice_data = np.clip(slice_data, low, high)
        slice_norm = (slice_data - low) / (high - low + 1e-6)
        gamma = 0.5
        slice_norm = slice_norm ** gamma

        # raster interpolation
        slice_norm = interpolate_raster(slice_norm, method='gaussian')

        return slice_norm.T  # Transpose for imshow (shape: Z, X)

    initial_slice = get_slice_image(0)
    slice_img = ax.imshow(initial_slice, cmap='OrRd', origin='lower', aspect='auto', vmin=0, vmax=1)

    ax.invert_yaxis()
    ax.set_title("XZ Slice Viewer (Slice Y=0)")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Z axis")

    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Y Slice', 0, max_y - 1, valinit=0, valstep=1)

    def update(val):
        y_idx = int(slider.val)
        slice_img.set_data(get_slice_image(y_idx))
        ax.set_title(f"XZ Slice Viewer (Slice Y={y_idx})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

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

    print("Browsing slices...")
    browse_yz_slices(voxel_grid)
    browse_xz_slices(voxel_grid)

if __name__ == "__main__":
    main()
