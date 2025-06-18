import open3d as o3d
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    scaled_points = StandardScaler().fit_transform(points)
    return scaled_points, pcd


def dbscan(file_path):
    scaled_points, pcd = load_ply(file_path)
    model = DBSCAN(eps=0.15, min_samples=10)
    model.fit(scaled_points)

    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cmap = plt.get_cmap("tab20")
    colors = cmap((labels % 20) / 20)
    colors[labels < 0] = [0, 0, 0, 1]

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Use RGB, drop alpha
    return pcd

def visualize_clusters_3d(pcd):
    o3d.visualization.draw_geometries([pcd])

def main():
    file_path = r"C:\Users\misko\Documents\Michal\Master\RS Integration\ACT_6\Data\Proefsleuf_1.las"
    #Large files 100mb < freeze the computer dont run
    clustered_pcd = dbscan(file_path)
    visualize_clusters_3d(clustered_pcd)


if __name__ == "__main__":
    main()
