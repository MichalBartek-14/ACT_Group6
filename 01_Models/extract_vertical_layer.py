import pdal
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
import os


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

    # Overlay with full point cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    mask_1 = (gray > 0.65) & (gray <= 0.7) & (z < -1.0)
    mask_2 = (gray > 0.7) & (gray <= 0.75) & (z < -1.0)
    mask_3 = (gray > 0.75) & (gray <= 0.8) & (z < -1.0)
    mask_4 = (gray > 0.8) & (gray <= 0.85) & (z < -1.0)
    mask_5 = (gray > 0.85) & (gray <= 0.9) & (z < -1.0)

    # TODO fix gray values to pull apart signal and also
    #  choose better value for z values (should be higher/lower)

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

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(xyz_2)

    pcd_3 = o3d.geometry.PointCloud()
    pcd_3.points = o3d.utility.Vector3dVector(xyz_3)


    pcd_4 = o3d.geometry.PointCloud()
    pcd_4.points = o3d.utility.Vector3dVector(xyz_4)

    pcd_5 = o3d.geometry.PointCloud()
    pcd_5.points = o3d.utility.Vector3dVector(xyz_5)

    pcd_1.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    pcd_2.paint_uniform_color([0.5, 1.0, 0.0])  # Yellow-Green
    pcd_3.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    pcd_4.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
    pcd_5.paint_uniform_color([1.0, 0.0, 0.0])  # Red

    # o3d.visualization.draw_geometries([pcd_weak, pcd_strong])
    o3d.visualization.draw_geometries([pcd_1, pcd_2, pcd_3, pcd_4, pcd_5])

    # TODO @Michal Bartek MGI I was thinking, I would like to try to
    #  denoise my results by removing points that have very little points around them
    #  (this is like clustering i guess?)


def visualize_vertical_layer(points, z_min, z_max, grid_size=0.1, value='gray'):
    """
    Visualize a vertical slice of the point cloud as a 2D raster image.

    Args:
        points (np.ndarray): PDAL point cloud array.
        z_min (float): Minimum Z value for vertical slice.
        z_max (float): Maximum Z value for vertical slice.
        grid_size (float): Size of each raster grid cell.
        value (str): What to display in the image: 'gray', 'count', 'r', 'g', 'b'
    """
    # Create basic XYZ and intensity arrays
    x = points['X']
    y = points['Y']
    z = points['Z']

    # Normalize colors to [0, 1]
    r = points['Red'].astype(np.float32) / 65535
    g = points['Green'].astype(np.float32) / 65535
    b = points['Blue'].astype(np.float32) / 65535
    gray = (r + g + b) / 3

    # Filter by height range
    mask = (z >= z_min) & (z <= z_max)
    x, y, z = x[mask], y[mask], z[mask]
    r, g, b, gray = r[mask], g[mask], b[mask], gray[mask]

    # Translate XY to raster grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    width = int(np.ceil((x_max - x_min) / grid_size))
    height = int(np.ceil((y_max - y_min) / grid_size))

    img = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)

    # Map points to raster
    col = ((x - x_min) / grid_size).astype(int)
    row = ((y - y_min) / grid_size).astype(int)

    # Choose value for pixel
    if value == 'gray':
        val = gray
    elif value == 'r':
        val = r
    elif value == 'g':
        val = g
    elif value == 'b':
        val = b
    elif value == 'count':
        val = np.ones_like(x)
    else:
        raise ValueError("Unsupported value type. Choose from 'gray', 'r', 'g', 'b', 'count'.")

    for i in range(len(x)):
        img[row[i], col[i]] += val[i]
        count[row[i], col[i]] += 1

    # Average value per cell
    with np.errstate(divide='ignore', invalid='ignore'):
        img = np.divide(img, count, where=(count > 0))

    # Optional: mask empty cells
    img[count == 0] = np.nan

    img = interpolate_raster(img, method='gaussian')  # or 'linear', 'gaussian'

    # Display image
    plt.figure(figsize=(10, 8))
    cmap = 'gray' if value in ['gray', 'count'] else None
    plt.imshow(img, origin='lower', cmap=cmap, extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label=value)
    plt.title(f"Vertical Layer Visualization (Z = {z_min} to {z_max})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def interpolate_raster(raster, method='gaussian'):
    """
    Interpolates missing (NaN) values in a raster using specified method.

    Args:
        raster (np.ndarray): 2D array with NaN as missing values.
        method (str): Interpolation method: 'nearest', 'linear', 'gaussian'

    Returns:
        np.ndarray: Interpolated raster image.
    """
    mask = np.isnan(raster)

    if not np.any(mask):
        return raster  # No interpolation needed

    if method == 'nearest':
        # Use distance transform to find nearest known-value pixel
        filled = raster.copy()
        idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
        filled[mask] = raster[tuple(idx[:, mask])]
        return filled

    elif method == 'linear':
        # Use scipy's griddata for linear interpolation
        from scipy.interpolate import griddata

        y, x = np.indices(raster.shape)
        known = ~mask
        points = np.stack((x[known], y[known]), axis=-1)
        values = raster[known]

        interpolated = griddata(points, values, (x, y), method='linear')

        # Optionally fall back to nearest for any remaining NaNs
        if np.isnan(interpolated).any():
            interpolated = interpolate_raster(interpolated, method='nearest')

        return interpolated

    elif method == 'gaussian':
        # Simple Gaussian blur-based fill (not true interpolation)
        filled = raster.copy()
        filled[mask] = 0
        blurred = ndimage.gaussian_filter(filled, sigma=1)
        norm = ndimage.gaussian_filter((~mask).astype(float), sigma=1)
        with np.errstate(invalid='ignore'):
            result = blurred / norm
        return result

    else:
        raise ValueError("Unsupported interpolation method. Use 'nearest', 'linear', or 'gaussian'.")


def save_vertical_layer_image(points, z_min, z_max, grid_size, value_type, save_path, interpolation_method='nearest'):
    """
    Creates a vertical slice image from point cloud, interpolates missing pixels, and saves as PNG.

    Args:
        points (np.ndarray): Point cloud data from PDAL.
        z_min (float): Minimum Z height to include.
        z_max (float): Maximum Z height to include.
        grid_size (float): Resolution of the raster grid in XY.
        value_type (str): One of 'gray', 'r', 'g', 'b', or 'count'.
        save_path (str): Full path where PNG image will be saved.
        interpolation_method (str): 'nearest', 'linear', or 'gaussian'.
    """
    # Create basic XYZ and intensity arrays
    x = points['X']
    y = points['Y']
    z = points['Z']

    r = points['Red'].astype(np.float32) / 65535
    g = points['Green'].astype(np.float32) / 65535
    b = points['Blue'].astype(np.float32) / 65535
    gray = (r + g + b) / 3

    # Filter by height range
    mask = (z >= z_min) & (z <= z_max)
    x, y = x[mask], y[mask]
    r, g, b, gray = r[mask], g[mask], b[mask], gray[mask]

    # Translate XY to raster grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    width = int(np.ceil((x_max - x_min) / grid_size))
    height = int(np.ceil((y_max - y_min) / grid_size))

    img = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)

    # Map to grid
    col = ((x - x_min) / grid_size).astype(int)
    row = ((y - y_min) / grid_size).astype(int)

    if value_type == 'gray':
        val = gray
    elif value_type == 'r':
        val = r
    elif value_type == 'g':
        val = g
    elif value_type == 'b':
        val = b
    elif value_type == 'count':
        val = np.ones_like(x)
    else:
        raise ValueError("value_type must be 'gray', 'r', 'g', 'b', or 'count'.")

    for i in range(len(x)):
        img[row[i], col[i]] += val[i]
        count[row[i], col[i]] += 1

    # Average value per pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        img = np.divide(img, count, where=(count > 0))
    img[count == 0] = np.nan

    # Interpolate
    img = interpolate_raster(img, method=interpolation_method)

    # Plot and save
    plt.figure(figsize=(10, 8))
    cmap = 'gray' if value_type in ['gray', 'count'] else None
    extent = [x_min, x_max, y_min, y_max]
    plt.imshow(img, origin='lower', cmap=cmap, extent=extent)
    plt.title(f"Z Slice: {z_min} to {z_max}, Value: {value_type}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label=value_type)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved image to: {save_path}")


def main():
    # filepath for the valid data
    file_path = r"C:\Users\mees2\Downloads\P1_10mBuffer.las"
    # "C:\Users\mees2\Downloads\Proefsleuf_1.las"
    # C:\Users\mees2\Downloads\WUR_ACT_PG_250515\WUR_ACT_PG_250515\LAZ_Euroradar\Bomen-23-37.laz

    # create points
    points = load_point_cloud(file_path)

    # histogram
    # plot_backscatter_intensity_distribution(points)

    # print_point_cloud_attributes(points)
    # visualize_parabolas(points)

    # Inside main or your visualizer
    img = visualize_vertical_layer(points, z_min=-150, z_max=-100, grid_size=0.1, value='gray')

    # Then interpolate
    # img_interp = interpolate_raster(img, method='gaussian')
    #
    # # Visualize interpolated
    # plt.imshow(img_interp, origin='lower', cmap='gray')
    # plt.title("Interpolated Vertical Slice")
    # plt.colorbar()
    # plt.show()

    save_vertical_layer_image(
        points=points,
        z_min=-130,
        z_max=-120,
        grid_size=0.05,
        value_type='gray',
        save_path=r"C:\Users\mees2\OneDrive\Bureaublad\slices\vertical_slice_gray2.png",
        interpolation_method='gaussian'
    )


if __name__ == "__main__":
    main()
