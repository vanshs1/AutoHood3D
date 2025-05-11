#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"AutoHood3D: A Multi-Modal Benchmark for Automotive Hood Design and Fluid–Structure Interaction".

This is a demonstration intended to provide a working example with data provided in the repo.
For application on other datasets, the requirement is to configure the settings. 

Dependencies: 
    - Python package list provided: package.list

Running the code: 
    - assuming above dependencies are configured, "python <this_file.py>" will run the demo code. 
    NOTE - It is important to check README prior to running this code.

Contact: 
    - Vansh Sharma at vanshs@umich.edu
    - Venkat Raman at ramanvr@umich.edu

Affiliation: 
    - APCL Group 
    - Dept. of Aerospace Engg., University of Michigan, Ann Arbor
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.interpolate
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.rcParams["figure.dpi"] = 500
# Function to set the style for each axes object
def set_axes_style(ax):
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    # ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6, pad=8, direction='in', top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='minor', width=1, length=3, direction='in', top=True, bottom=True, left=True, right=True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

#%%% Functions

def arrange_points_in_circular_order(points, clockwise=True):
    """
    Arrange points (x, y, z) in clockwise or counterclockwise order around their centroid.

    :param points: Array of shape (n, 3) with columns representing x, y, z coordinates.
    :param clockwise: If True, sort points in clockwise order. If False, sort counterclockwise.
    :return: Array of points sorted in the desired order.
    """
    # Calculate the centroid of the (x, y) coordinates
    centroid = np.mean(points[:, :2], axis=0)
    
    # Calculate the angles of each point relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort the points by angle
    if clockwise:
        sorted_indices = np.argsort(-angles)
    else:
        sorted_indices = np.argsort(angles)
    
    # Reorder the points
    sorted_points = points[sorted_indices]
    
    return sorted_points

def calculate_perimeter(points):
    """Computes the perimeter of a closed 3D curve."""
    distances = np.linalg.norm(np.diff(points, axis=0, append=points[:1]), axis=1)
    return np.sum(distances)

def calculate_area(points):
    """Computes the projected 2D area of a closed 3D curve using the Convex Hull."""
    if points.shape[1] == 3:
        # Project onto the plane with largest variance
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        points_2d = points - np.outer(points @ normal, normal)  # Project onto plane
        points_2d = points_2d[:, :2]  # Take first two dimensions for 2D hull
    else:
        points_2d = points  # If already 2D

    hull = ConvexHull(points_2d)
    return hull.volume  # Convex hull area

def fit_closed_bspline(points, degree=3, num_samples=100):
    """ Fits a closed B-spline to the given 3D outer boundary points. """
    # Ensure periodic boundary condition
    points = np.vstack([points, points[0]])  # Append first point to the end
    
    # Create periodic B-spline
    tck, _ = scipy.interpolate.splprep(
        [points[:, 0], points[:, 1], points[:, 2]], 
        k=degree, s=0, per=True  # `per=True` ensures closure
    )
    u_fine = np.linspace(0, 1, num_samples)
    fitted_points = np.array(scipy.interpolate.splev(u_fine, tck)).T
    return fitted_points


#%% USER INPUT
local = os.getcwd()  ## Get local dir
os.chdir(local)      ## shift the work dir to local dir
print('\nWork Directory: {}'.format(local))

points_path = "PATH_TO_CURVE_DATABASE"
filePath = [f.path for f in os.scandir(points_path) if '.txt' in f.name] 
filePath.sort()

curves_list=[]
for path in filePath:
    mask_pts = np.loadtxt(path, skiprows=1, delimiter=',') * 0.001  ## Units adjustment
    mask_pts[:, -1] = mask_pts[:, -1]*0 + -0.055                    ## Linear translation for brevity
    sorted_points_ckw = arrange_points_in_circular_order(mask_pts, clockwise=True)
    curves_list.append(sorted_points_ckw) 

#%% USER INPUTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
list_peri=[]
list_area=[]
list_fit=[]
counter=0
for curve_points in curves_list:
    perimeter = calculate_perimeter(curve_points)
    area = calculate_area(curve_points)   
    # Fit a B-spline to the outer loop
    fitted_curve = fit_closed_bspline(curve_points)
    list_peri.append(perimeter)
    list_area.append(area)
    list_fit.append( fitted_curve.flatten() )  # Flatten to a single vector)
    
    counter+=1


num_curves = len(list_area)
# Create feature matrix
X = np.column_stack([list_peri, list_area, np.array(list_fit)])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

######   Apply K-Means clustering on t-SNE embeddings

# # Plot Elbow Curve
# plt.figure(figsize=(8, 5))
# # plt.plot(K, wcss, marker='o', linestyle='-')
# plt.plot(K, sil_db, marker='o', linestyle='-')
# plt.xlabel("Number of Clusters (k)")
# # plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
# plt.ylabel("SIL (Within-Cluster Sum of Squares)")
# plt.title("Elbow Method for Optimal Clusters")
# plt.show()

########################## KMEANS
num_clusters = 9
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)

tsne = TSNE(perplexity=80, learning_rate=100, init='pca')
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=500)
set_axes_style(ax)

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab20", edgecolor='k', alpha=0.75)
ax.set_title("t-SNE Projection of Curve Features")
# Add a legend with cluster colors
legend_labels = [f"Cluster {i}" for i in range(num_clusters)]

# Place the legend outside the axes (to the right in this example)
ax.legend(handles=scatter.legend_elements()[0],
          labels=legend_labels,
          title="Clusters",
          loc='upper left',
          bbox_to_anchor=(1.05, 1),
          borderaxespad=0.)

plt.show()
########################## KMEANS

#### save cluster IDs and the file paths
import json

outfile = f"./clustering_output_clusters_{num_clusters}.json"
### create a dictionary to store both pieces of data
data_to_save = {
    "labels": labels.tolist(),
    "file_path": filePath,
    "X_tsne": X_tsne.tolist(),
    "list_peri": list_peri,
    "list_area": list_area,
    "list_fit": [arr.tolist() for arr in list_fit],
}

# Save the data as a formatted JSON 
with open(outfile, 'w') as f:
    json.dump(data_to_save, f, indent=4)

f.close()




