import pdal
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


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
    print(pipeline.metadata)
    arrays = pipeline.arrays
    return arrays[0]


def print_point_cloud_attributes(points):
    """Print attributes of the point cloud data."""
    print(points.dtype.names)


def investigate_data(points):
    """Investigate and print various attributes of the point cloud data.
    @param points: The points data"""
    x, y, z = points['X'], points['Y'], points['Z']
    intensity = points['Intensity']
    return_number = points['ReturnNumber']
    num_returns = points['NumberOfReturns']

    print(f"Y range: {y.min()} to {y.max()}")
    print(f"X range: {x.min()} to {x.max()}")
    print(f"Z range: {z.min()} to {z.max()}")
    print(f"Intensity: {intensity.min()} to {intensity.max()}")
    print(f"ReturnNumber: {return_number.min()} to {return_number.max()}")
    print(f"NumberOfReturns: {num_returns.min()} to {num_returns.max()}")

    for attr in ['ReturnNumber', 'NumberOfReturns', 'ScanAngleRank', 'Classification', 'UserData', 'Red', 'Green', 'Blue']:
        values = points[attr]
        print(f"{attr}: min={values.min()}, max={values.max()}, unique={np.unique(values)[:10]}")


def plot_signal_intensity_vs_depth(points):
    """Plot signal intensity vs depth."""
    r = points['Red'].astype(np.float32) / 65535
    z = points['Z']

    plt.figure(figsize=(10, 6))
    plt.scatter(r, z, s=1, c=r, cmap='gray')
    plt.xlabel("Grayscale Intensity")
    plt.ylabel("Z (Depth or Elevation)")
    plt.title("Signal Intensity vs Depth")
    plt.grid(True)
    plt.show()
    print("This plot loads for a long time - Estimated time 5-10minutes")


def plot_backscatter_intensity_distribution(points):
    """Plot distribution of backscatter intensity."""
    r = points['Red'].astype(np.float32) / 65535

    plt.hist(r, bins=100, color='gray', edgecolor='black')
    plt.title("Distribution of Backscatter Intensity")
    plt.xlabel("Grayscale Value")
    plt.ylabel("Number of Points")
    plt.show()

def visualize_high_intensity(points):
    """Visualize high intensity points in the point cloud."""

    xyz = np.vstack((points['X'], points['Y'], points['Z'])).T
    # compress the z-scale
    # 50 default
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
def main():
    # filepath for the intersected location with valid data
    # define your file path to the .las file here
    file_path = r"C:\Users\mees2\Downloads\Proefsleuf_1.las"

    points = load_point_cloud(file_path)

    print_point_cloud_attributes(points)
    investigate_data(points)
    # plot_signal_intensity_vs_depth(points) # --uncomment for this function to run -> takes a long time to run

    #Visualise the backscatter and high values in .laz file

    plot_backscatter_intensity_distribution(points)
    visualize_high_intensity(points)


if __name__ == "__main__":
    main()
