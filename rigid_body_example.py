# Script to try to find the rigid body transformation that aligns a square floating in 3D space
# To another square floating in 3D space.
#
# Kabsch failed on this test case, So  used Monte Carlo.

import numpy as np
from skspatial.objects import Plane, Points
import cv2

def is_coplanar(points):
    """
    taken from the idea here: https://stackoverflow.com/a/72384583
    returns True or False depending if the points are too coplanar or not.
    """

    best_fit = Plane.best_fit(points)
    distances = np.empty((points.shape[0]))

    for i in range(points.shape[0]):
        distances[i] = best_fit.distance_point(points[i])

    error = distances.mean()
    return error

def check_error(A, B):

    return np.linalg.norm(A-B, axis=1).sum()

def random_rotation_matrix(mean_x=0, mean_y=0, mean_z=0, std_dev=5):

    # Convert mean angles from degrees to radians
    mean_x_rad = np.deg2rad(mean_x)
    mean_y_rad = np.deg2rad(mean_y)
    mean_z_rad = np.deg2rad(mean_z)

    # Convert standard deviation from degrees to radians
    std_dev_rad = np.deg2rad(std_dev)

    # Generate normally distributed angles for rotation about x, y, and z
    rx = np.random.normal(mean_x_rad, std_dev_rad)
    ry = np.random.normal(mean_y_rad, std_dev_rad)
    rz = np.random.normal(mean_z_rad, std_dev_rad)

    # Compute individual rotation matrices
    R = cv2.Rodrigues(np.array([rx, ry, rz]))[0]

    return R

def kabsch_rigid_transform(A, B):
    # Ensure A and B are numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center the points
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Compute cross-covariance matrix
    H = np.dot(A_centered.T, B_centered)
    
    # Compute SVD
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute translation
    t = centroid_B - np.dot(centroid_A, R)
    
    return R, t


# Data to transform
# A = np.array([  [ 0., 40.,  0.],
#                 [40., 40.,  0.],
#                 [ 0.,  0.,  0.],
#                 [40.,  0.,  0.]])

# A = np.array([[ -54.50592394,   56.11328204, -105.72469213],
#               [ -14.93819274,   60.22187404, -101.53970293],
#               [ -49.48336714,   47.11116124, -144.37356973],
#               [  -9.91563594,   51.21975324, -140.18858053]])
A = np.array([[ -54.35040672,   56.52564748, -105.58820791],
              [ -14.68552029,   59.85329247, -101.63554113],
              [ -49.73603958,   47.47974282, -144.27773152],
              [ -10.07115316,   50.8073878 , -140.32506474]])
A = A - A.mean(axis=0) # Center data around the origin

# Target for the transformation
B = np.array([[ -53.93304374,   56.61932146, -105.32713253], 
              [ -15.23000162,   59.8388264 , -101.55474765], 
              [ -49.58298373,   47.47285135, -144.35019901],
              [ -10.09709066,   50.73507135, -140.59446611]])
B = B - B.mean(axis=0) # Center data around the origin

# Make the points column vectors
At = A.T


# First round. Normal distribution, from -180 to +180 degrees
n_iteration = int(1e7)
best_R = None
best_error = 1000000
for i in range(n_iteration):

    R = random_rotation_matrix(mean_x=0, mean_y=0, mean_z=0, std_dev=2)
    C = (R @ At).T
    error = check_error(C, B)

    if error < best_error:
        best_error = error
        best_R = R
        print(f"best error = {best_error}")


print(f"best R = {best_R}")
print(f"A = {A}")
print(f"B = {B}")
print(f"(R @ At) = {(best_R @ At).T}")


# Last best R
last_R = np.array([[ 0.99452382, -0.05808328, -0.08688332],
                   [ 0.07663475,  0.97055163,  0.22837832],
                   [ 0.07105978, -0.23378596,  0.9696879 ]])

