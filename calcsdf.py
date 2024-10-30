import open3d as o3d
import pandas as pd
import numpy as np

# Load the CSV file
file_path = "/home/adahdah/datasurfaceSurface17.csv"
data = pd.read_csv(file_path, header=None)

# Assuming your CSV contains only X, Y, Z columns, load them into a numpy array
# If there are headers in the CSV, replace 'header=None' with the correct header or remove it.
points = data.values

# Create a PointCloud object
pcd = o3d.geometry.PointCloud()

# Assign points to the PointCloud
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the Point Cloud
o3d.visualization.draw_geometries([pcd])
