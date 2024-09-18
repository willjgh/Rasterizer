import pandas as pd
import numpy as np
from scipy.spatial import distance

# read file
file = pd.read_csv("Rasterizer/scan_data_20230526100037.csv")

# remove points with only depth information
file = file.dropna()

# extract min and max point values
min_lat = file['latitude'].min()
max_lat = file['latitude'].max()

min_long = file['longtitude'].min()
max_long = file['longtitude'].max()

min_depth = file['depth'].min()
max_depth = file['depth'].max()

points = []

# loop over points
for index, row in file.iterrows():

    # normalize points
    x = (row['latitude'] - min_lat) / (max_lat - min_lat)
    y = (row['depth'] - min_depth) / (max_depth - min_depth)
    z = (row['longtitude'] - min_long) / (max_long - min_long)

    points.append([x, y, z])

n = len(points)
points = np.array(points)

# compute euclidean distance matrix
#x2 = np.sum(points**2, axis=1)
#xy = np.matmul(points, points.T)
#x2 = x2.reshape(-1, 1)
#dist = np.sqrt(2*x2 - 2*xy)

# set radius
radius = 0.1

# store triangles
triangles = set()

# loop over triplets of points
for i in range(100):
    for j in range(100):
        for k in range(100):

            if (i == j) or (i == k) or (j == k):
                continue

            d_ij = np.linalg.norm((points[i] - points[j]))
            d_ik = np.linalg.norm((points[i] - points[k]))
            d_jk = np.linalg.norm((points[j] - points[k]))

            # select triplets of adjacent points
            if (d_ij < radius) and (d_ik < radius) and (d_jk < radius):

                # add to set
                triangles.add((i, j, k))

# write to model file
file = open('point_cloud.obj', 'w+')

# add points
for p in points:
    file.write(f"v {p[0]} {p[1]} {p[2]}\n")

# add triangles
for t in triangles:
    file.write(f"f {t[0]} {t[1]} {t[2]}\n")










