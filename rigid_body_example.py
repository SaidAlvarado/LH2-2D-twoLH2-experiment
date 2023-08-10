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

# Example points
# A = np.array([[304.22258184, 169.35194119, 0.], 
#               [264.18925565, 168.28511852, 0.], 
#               [303.56316036, 209.4004265 , 0.],
#               [263.32708102, 208.43592785, 0.]])

# Simplified problem
# A = np.array([    [ 1,   -1,   0.],
#                   [-1,   -1,   0.],
#                   [ 1,    1,   0.],
#                   [-1 ,   1,   0.]])  + np.array([283.82551972, 188.86835351,   0.])

# Pre-transformed data.
A = np.array([[ -52.60784206,   53.66651764, -103.44022401],
            [ -12.57451587,   53.66651764, -102.37340134],
            [ -51.94842058,   53.66651764, -143.48870932],
            [ -11.71234124,   53.66651764, -142.52421067]]) - np.array([ -32.21077994,   53.66651764, -122.95663633])


B = np.array([[ -53.93304374,   56.61932146, -105.32713253], 
              [ -15.23000162,   59.8388264 , -101.55474765], 
              [ -49.58298373,   47.47285135, -144.35019901],
              [ -10.09709066,   50.73507135, -140.59446611]]) - np.array([ -32.21077994,   53.66651764, -122.95663633])

# B = np.array([[-1 ,  0,  1],
#               [ 1,   0,  1],
#               [-1,   0, -1],
#               [ 1,   0, -1]]) + np.array([ -32.21077994,   53.66651764, -122.95663633])

# Compute transformation
# R, t = kabsch_rigid_transform(A, B)
# print("Rotation Matrix:\n", R)
# print("Translation Vector:\n", t)
# t = t.reshape((3,1))

At = A.T

print(f"Coplanar error = {is_coplanar(B)}")

best_R = None
best_error = 1000000
for i in range(int(1e6)):

    R = random_rotation_matrix(mean_x=-13.385892116357232, mean_y=-4.574583938784697, mean_z=3.901904790013345, std_dev=2)

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

