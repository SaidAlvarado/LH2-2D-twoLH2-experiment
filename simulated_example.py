import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import json
import pandas as pd
from scipy.spatial.transform import Rotation

def solve_2d_scene_get_Rtn(pts_a, pts_b):
    """
    Use SVD to recover R,t,star from LHA to LHB, as well the 'n' of the plane.
    """
    # Obtain translation and rotation vectors
    """
    Use the projected LH2-camera points to triangulate the position of the LH2  basestations and of the LH2 receiver
    """
    # Obtain translation and rotation vectors
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.FM_LMEDS)

    # Now do the math stuff from the Paper, with the SVD and stuff,
    U, S, Vt = np.linalg.svd(H)
    # Scale the singular values
    S *= 1 / S[1] 

    # Compute scaling factor between LH camera to the Robot plane
    zeta = S[0]/S[2]

    # Calculate n_star
    a = np.sqrt( (1 - S[2]**2) / (S[0]**2 - S[2]**2) )
    b = np.sqrt( (S[0]**2 - 1) / (S[0]**2 - S[2]**2) )
    n_star = b*Vt[0] - a*Vt[2]
    # Use asumption about the LH pointing towards the robot plane from above to resolve the ambiguity in the calculation.
    if n_star[1] < 0: 
        n_star = b*Vt[0] + a*Vt[2]

    # Calculate R_star
    c = 1 + S[0]*S[2]
    d = a*b
    # scale c and d
    cd_scale = 1 / np.sqrt(c**2 + d**2)
    c *= cd_scale
    d *= cd_scale
    R_star = U @ np.array([[c, 0, d], [0, 1, 0], [-d, 0, c]]) @ Vt

    # Calculate t_star
    e = -b / S[0]
    f = -a / S[2]
    # scale e and f
    ef_scale = 1 / np.sqrt(e**2 + f**2)
    e *= ef_scale
    f *= ef_scale
    # Then, t_Star becomes
    t_star = e * Vt[0] + f * Vt[2]

    # Check that both LH are looking at each other.
    # If not, recalculate R,t with the other option
    if t_star[2] < 0:
        # Calculate R_star
        c = 1 + S[0]*S[2]
        d = a*b
        # scale c and d
        cd_scale = 1 / np.sqrt(c**2 + d**2)
        c *= cd_scale
        d *= cd_scale
        R_star = U @ np.array([[c, 0, -d], [0, 1, 0], [d, 0, c]]) @ Vt

        # Calculate t_star
        e = -b / S[0]
        f = -a / S[2]
        # scale e and f
        ef_scale = 1 / np.sqrt(e**2 + f**2)
        e *= ef_scale
        f *= ef_scale
        # Then, t_Star becomes
        t_star = e * Vt[0] - f * Vt[2]

    # Make t_Star a column vector, because the rest of the code wxpectes it to be so.
    t_star = t_star.reshape((3,1))

    return t_star, R_star, n_star, zeta


def fil_solve_2d(pts_a, pts_b):

    # gg=cv2.findHomography(pts_a,pts_b, method = cv2.RANSAC, ransacReprojThreshold = 3)
    gg=cv2.findHomography(pts_a, pts_b, cv2.FM_LMEDS)

    homography_mat = gg[0]
    U, S, V = np.linalg.svd(homography_mat)

    V = V.T

    s1 = S[0]/S[1]
    s3 = S[2]/S[1]
    zeta = s1-s3
    a1 = np.sqrt(1-s3**2)
    b1 = np.sqrt(s1**2-1)

    def unitize(x_in,y_in):
        magnitude = np.sqrt(x_in**2 + y_in**2)
        x_out = x_in/magnitude
        y_out = y_in/magnitude

        return x_out, y_out  

    a, b = unitize(a1,b1)
    c, d = unitize(1+s1*s3,a1*b1)
    e, f = unitize(-b/s1,-a/s3)

    v1 = np.array(V[:,0])
    v3 = np.array(V[:,2])
    n1 = b*v1-a*v3
    n2 = b*v1+a*v3

    R1 = np.matmul(np.matmul(U,np.array([[c,0,d], [0,1,0], [-d,0,c]])),V.T)
    R2 = np.matmul(np.matmul(U,np.array([[c,0,-d], [0,1,0], [d,0,c]])),V.T)
    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    print(R1)

    t1 = (e*v1+f*v3).reshape((3,1))
    t2 = (e*v1-f*v3).reshape((3,1))

    if (n1[2]<0):
        t1 = -t1
        n1 = -n1
    elif (n2[2]<0):
        t2 = -t2
        n2 = -n2

    fil = {'R1' : R1,
           'R2' : R2,
           'n1' : n1,
           'n2' : n2,
           't1' : t1,
           't2' : t2,
           'zeta': zeta}

    # Check which of the solutions is the correct one. 

    solution_1 = (t1, R1, n1)   
    solution_2 = (t2, R2, n2)   
    

    return solution_1, solution_2, zeta


def solve_point_plane(n_star, zeta, pts):
    
    # Extend the points to homogeneous coordinates.
    pts_hom = np.hstack((pts, np.ones((len(pts),1))))

    # Get the scaling factor for every point in the LH image plane.
    scales = (1/zeta) / (n_star @ pts_hom.T)
    # scale the points
    scales_matrix = np.vstack((scales,scales,scales))
    final_points = scales_matrix*pts_hom.T
    final_points = final_points.T

    return final_points

#############################################################################
###                Define LH2 and point positions                         ###
#############################################################################
# LH2 A, pos and rotation
lha_t = np.array([0,0,0])
lha_R, _ = cv2.Rodrigues(np.array([0, np.pi/4, 0]))    # tilted right (towards X+)
# LH2 B, pos and rotation
lhb_t = np.array([6,0,0])
lhb_R, _ = cv2.Rodrigues(np.array([0, -np.pi/4, 0 ]))  # tilted left (towards X-)
points = np.array([[2,0,2],
                   [2,2,2],
                   [3,-2,2],
                   [3,-2,2],
                   [3,2,2],
                   [3,3,2],
                   [4,0,2],
                   [4,2,2],
                   [4,-2,2]], dtype=float)


obj_points = points - np.array([2,0,2])

#############################################################################
###                   Elevation and Azimuth angle                         ###
#############################################################################

# lhx_R.T is the inverse rotation matrix
# (points - lha_t).T is just for making them column vectors for correctly multiplying witht the rotation matrix.
p_a = lha_R.T @ (points - lha_t).T
p_b = lhb_R.T @ (points - lhb_t).T

elevation_a = np.arctan2( p_a[1], np.sqrt(p_a[0]**2 + p_a[2]**2))
elevation_b = np.arctan2( p_b[1], np.sqrt(p_b[0]**2 + p_b[2]**2))

azimuth_a = np.arctan2(p_a[0], p_a[2]) # XZ plan angle, 0 == +Z, positive numbers goes to +X
azimuth_b = np.arctan2(p_b[0], p_b[2])

#############################################################################
###                             Cristobal Code                            ###
#############################################################################

pts_lighthouse_A = np.array([np.tan(azimuth_a),       # horizontal pixel  
                             np.tan(elevation_a) * 1/np.cos(azimuth_a)]).T  # vertical   pixel 

pts_lighthouse_B = np.array([np.tan(azimuth_b),       # horizontal pixel 
                             np.tan(elevation_b) * 1/np.cos(azimuth_b)]).T  # vertical   pixel


## 2D SCENE SOLVING
# t_star, R_star, n_star, zeta = solve_2d_scene_get_Rtn(pts_lighthouse_A, pts_lighthouse_B)
solution_1, solution_2, zeta = fil_solve_2d(pts_lighthouse_A, pts_lighthouse_B)
t_star, R_star, n_star = solution_1


# point3D = solve_point_plane(n_star, zeta, pts_lighthouse_A)
point3D = solve_point_plane(n_star, zeta, pts_lighthouse_A)

# Triangulate the points
R_1 = np.eye(3,dtype='float64')
t_1 = np.zeros((3,1),dtype='float64')

# Normalize the size of the triangulation to the real size of the dataset, to better compare.
# scale = np.linalg.norm(lhb_t - lha_t) / np.linalg.norm( t_star - t_1) 
# t_star *= scale
# point3D *= scale*3

# Rotate the dataset to superimpose the triangulation, and the ground truth.
# rot, _ = Rotation.align_vectors(lhb_t.reshape((1,-1)), t_star.T)  # align_vectors expects 2 (N,3), so we need to reshape and transform htem a bit.
# t_star = rot.as_matrix() @ t_star
# point3D = (rot.as_matrix() @ point3D.T).T

#############################################################################
###                                 PNP                                   ###
#############################################################################

img_points = pts_lighthouse_A
# retval, r_pnp, t_pnp = cv2.solvePnP(points, img_points, np.eye(3), np.zeros((4,1)), flags=cv2.SOLVEPNP_EPNP)
retval, r_pnp, t_pnp = cv2.solvePnP(obj_points, img_points, np.eye(3), np.zeros((4,1)), flags=cv2.SOLVEPNP_EPNP)

# Calculate the projection using the real Camera A pose.
Pw_a = np.vstack([np.hstack([lha_R.T, -lha_R.T @ lha_t.reshape((-1,1))]), [0,0,0,1]])   # Pw_a - transformation matrix: World -> Camera A
Pw_b = np.vstack([np.hstack([lhb_R.T, -lhb_R.T @ lhb_t.reshape((-1,1))]), [0,0,0,1]])   # Pw_b - transformation matrix: World -> Camera B
                                                                                        # Pa_b - transformation matrix: Camera A -> Camera B
Pa_b = Pw_b @ np.linalg.inv(Pw_a)                                                       # Pw_b - transformation matrix: World -> Camera B



#############################################################################
###                             Plotting                                  ###
#############################################################################
############################# 2D projection #################################  
# Plot the results
fig = plt.figure(layout="constrained")
gs = GridSpec(6, 3, figure = fig)
lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
lh2_ax = fig.add_subplot(gs[3:6, 0:3])
axs = (lh1_ax, lh2_ax)

# 2D plots - LH2 perspective
lh1_ax.scatter(pts_lighthouse_A[:,0], pts_lighthouse_A[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1")
lh2_ax.scatter(pts_lighthouse_B[:,0], pts_lighthouse_B[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2")
# Plot one synchronized point to check for a delay.

# Add labels and grids
for ax in axs:
    ax.grid()
    ax.legend()
lh1_ax.axis('equal')
lh2_ax.axis('equal')
# 
lh1_ax.set_xlabel('U [px]')
lh1_ax.set_ylabel('V [px]')
#
lh2_ax.set_xlabel('U [px]')
lh2_ax.set_ylabel('V [px]')
#
# plt.show()

######################################### 3D Plotting #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_proj_type('ortho')

# Plot the lighthouse orientation
arrow = np.array([0,0,1]).reshape((-1,1))
ax2.quiver(lha_t[0],lha_t[1],lha_t[2], (lha_R @ arrow)[0], (lha_R @ arrow)[1], (lha_R @ arrow)[2], length=0.4, color='xkcd:red')
ax2.quiver(lhb_t[0],lhb_t[1],lhb_t[2], (lhb_R @ arrow)[0], (lhb_R @ arrow)[1], (lhb_R @ arrow)[2], length=0.4, color='xkcd:red')
ax2.quiver(t_1[0],t_1[1],t_1[2], (R_1 @ arrow)[0], (R_1 @ arrow)[1], (R_1 @ arrow)[2], length=0.2, color='xkcd:orange' )
ax2.quiver(t_star[0],t_star[1],t_star[2], (R_star @ arrow)[0], (R_star @ arrow)[1], (R_star @ arrow)[2], length=0.2, color='xkcd:orange' )

ax2.scatter(points[:,0],points[:,1],points[:,2], color='xkcd:blue', label='ground truth', s=50)
ax2.scatter(point3D[:,0],point3D[:,1],point3D[:,2], color='xkcd:green', alpha=0.5, label='triangulated')
ax2.scatter(lha_t[0],lha_t[1],lha_t[2], color='xkcd:red', label='LH1', s=50)
ax2.scatter(lhb_t[0],lhb_t[1],lhb_t[2], color='xkcd:red', label='LH2', s=50)
ax2.scatter(t_1[0],t_1[1],t_1[2], color='xkcd:orange', label='triang LH1')
ax2.scatter(t_star[0],t_star[1],t_star[2], color='xkcd:orange', label='triang LH2')

ax2.axis('equal')
ax2.legend()

ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')
ax2.set_zlabel('Z [mm]')

plt.show()