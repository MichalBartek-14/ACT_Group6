import pdal
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def plot_backscatter_intensity_distribution(points):
    """Plot distribution of backscatter intensity.
    @param points: expects pointcloud file created using PDAL
    """
    r = points['Red'].astype(np.float32) / 65535

    plt.hist(r, bins=100, color='gray', edgecolor='black')
    plt.title("Distribution of Backscatter Intensity")
    plt.xlabel("Grayscale Value")
    plt.ylabel("Number of Points")
    plt.show()


def load_point_cloud(file_path):
    """Load point cloud data from a LAZ file using PDAL.
    @param file_path: expects local absolute file path to LAZ file
    @return: numpy ndarray (array for multidimensional data)
    """
    pipeline_json = {"pipeline": [file_path, {"type": "filters.range", "limits": "Z[-1000:1000]"}]}

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    arrays = pipeline.arrays
    return arrays[0]


def visualize_high_intensity(points, z_scale_compression: int = 50):
    """Visualize  points in the point cloud of specific intensities. The points are filtered by using multiple masks.
    The aim is to create the best visualisation of points that belong to roots.
    The masks are actually lists that contain true/false data for every point in the cloud, which
    can be used to only load the points with specific intensity values and z-values.
    The selections of points are then colored.
    @param z_scale_compression: integer that determines space between points along z-axis.
    Tweakable for visualization purposes only
    @param points: expects pointcloud file created using PDAL
    """
    # compress the z-scale
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    z = (xyz[:, 2]) / z_scale_compression
    xyz[:, 2] = z

    # change value range of color values to 0-1
    r = points['Red'].astype(np.float32) / 65535
    g = points['Green'].astype(np.float32) / 65535
    b = points['Blue'].astype(np.float32) / 65535

    rgb = np.vstack((r, g, b)).T
    gray = r.astype(float)

    # Mask for strong signals (e.g., gray > 0.8) at time/depth (z < -0.01)
    mask_high = (gray > 0.8) & (z < -0.01)
    xyz_high = xyz[mask_high]

    pcd_high = o3d.geometry.PointCloud()
    pcd_high.points = o3d.utility.Vector3dVector(xyz_high)
    pcd_high.paint_uniform_color([1.0, 0.0, 0.0])  # red

    # Overlay with full point cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    # this next line sometimes gives an error about "expected types", but this is no problem
    o3d.visualization.draw_geometries([pcd_all, pcd_high])


def roots_value_filter(points, z_scale_compression: int = 50):
    """ Visualize points in the point cloud of certain depth and intensity.
    There are five masks created that attempt to filter out values that correspond to roots.
    The method for this is visually trying to get the best results. The testsite for the current values are based on
    the proefsleuf_1.laz file. Be aware that these values might not translate to other test sites because of
    differences in soil composition and hence the strength of the returned signal for the roots.
    @param z_scale_compression: integer that determines space between points along z-axis.
    Tweakable for visualization purposes only
    @param points: expects pointcloud file created using PDAL
    """
    # compress the z-scale
    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    print(f"xyz min/max: \n"
          f"x: {xyz[:, 0].min():.4f}, \t{xyz[:, 0].max():.4f}  \n"
          f"y: {xyz[:, 1].min():.4f}, \t{xyz[:, 1].max():.4f} \n"
          f"z: {xyz[:, 2].min():.4f}, \t\t{xyz[:, 2].max():.4f}\n")

    print("shape: ", np.shape(xyz))

    z = (xyz[:, 2]) / z_scale_compression
    xyz[:, 2] = z
    print("zmin, zmax are: ", z.min(), z.max())

    r = (points['Red'].astype(np.float32) / 65535)
    g = (points['Green'].astype(np.float32) / 65535)
    b = (points['Blue'].astype(np.float32) / 65535)

    rgb = np.vstack((r, g, b)).T

    gray = (r + g + b) / 3
    print(gray.min(), gray.max())

    # Overlay with full point cloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    mask_1 = (gray > 0.8) & (gray <= 0.82) & (z < -0.00)
    mask_2 = (gray > 0.82) & (gray <= 0.84) & (z < -0.00)
    mask_3 = (gray > 0.84) & (gray <= 0.86) & (z < -0.00)
    mask_4 = (gray > 0.86) & (gray <= 0.88) & (z < -0.00)
    mask_5 = (gray > 0.8) & (gray <= 0.9) & (z < -0.00)

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

    # mask_all = (gray > 0.8) & (gray <= 0.9) & (z < -0.0)
    # xyz_all = xyz[mask_all]
    # pcd_all = o3d.geometry.PointCloud()
    # pcd_all.points = o3d.utility.Vector3dVector(xyz_all)

    # o3d.visualization.draw_geometries([pcd_weak, pcd_strong])
    o3d.visualization.draw_geometries([pcd_1, pcd_2, pcd_3, pcd_4, pcd_5])


def main():
    file_path = r"C:\Users\mees2\Downloads\P1_10mBuffer.las"

    points = load_point_cloud(file_path)

    # visualize_high_intensity(points)

    # plot_backscatter_intensity_distribution(points)

    roots_value_filter(points)


if __name__ == "__main__":
    main()
