import pandas as pd
from datetime import datetime
from dateutil import parser
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2


#############################################################################
###                                Options                                ###
#############################################################################

# file with the data to analyze
data_file = './data.csv'
experiment_file = './video_track/4_blender.csv'  
calib_file = './video_track/calibration.json'

#############################################################################
###                                Functions                              ###
#############################################################################

def import_data(data_file, experiment_file, calib_file):

    # Read the files.
    data = pd.read_csv(data_file, index_col=0, parse_dates=['timestamp'])
    exp_data = pd.read_csv(experiment_file, parse_dates=['timestamp'])
    with open(calib_file, 'r') as json_file:
        calib_data = json.load(json_file)
    lh2_calib_time = calib_data["GX010146.MP4"]["timestamps_lh2"]

    # Add a Z=0 axis to the Camera (X,Y) coordinates.
    exp_data['z'] = 0.0

    # Convert the strings to datetime objects
    for key in lh2_calib_time:
        lh2_calib_time[key] = [parser.parse(ts) for ts in lh2_calib_time[key]]

    # Get a timestamp column of the datetime.
    for df in [data, exp_data]:
        df['time_s'] = df['timestamp'].apply(lambda x: x.timestamp() )

    # Convert pixel corners to numpy arrays
    for corner in ['tl', 'tr', 'bl', 'br']:
        calib_data["GX010146.MP4"]['corners_px'][corner] = np.array(calib_data["GX010146.MP4"]['corners_px'][corner])

    # Slice the calibration data and add it to the  data dataframe.
    tl = data.loc[ (data['timestamp'] > lh2_calib_time["tl"][0]) & (data['timestamp'] < lh2_calib_time["tl"][1])].mean(axis=0, numeric_only=True)
    tr = data.loc[ (data['timestamp'] > lh2_calib_time["tr"][0]) & (data['timestamp'] < lh2_calib_time["tr"][1])].mean(axis=0, numeric_only=True)
    bl = data.loc[ (data['timestamp'] > lh2_calib_time["bl"][0]) & (data['timestamp'] < lh2_calib_time["bl"][1])].mean(axis=0, numeric_only=True)
    br = data.loc[ (data['timestamp'] > lh2_calib_time["br"][0]) & (data['timestamp'] < lh2_calib_time["br"][1])].mean(axis=0, numeric_only=True)
    
    # Slice the lh2 data to match the timestamps on the blender experiment
    start = exp_data['timestamp'].iloc[0 + 18]  # Use the a point about 250ms later than the start of the dataset, to address the time delay correction that we will do later when we interpolate the data.
    end   = exp_data['timestamp'].iloc[-1]
    data = data.loc[ (data['timestamp'] > start) & (data['timestamp'] < end)]

    # Save the calibration data.
    calib_data["GX010146.MP4"]['corners_lh2_count'] = {'tl':tl,
                                 'tr':tr,
                                 'bl':bl,
                                 'br':br,
                                 }

    return exp_data, data, calib_data["GX010146.MP4"]


def LH2_count_to_pixels(count_1, count_2, mode):
    """
    Convert the sweep count from a single lighthouse into pixel projected onto the LH2 image plane
    ---
    count_1 - int - polinomial count of the first sweep of the lighthouse
    count_2 - int - polinomial count of the second sweep of the lighthouse
    mode - int [0,1] - mode of the LH2, let's you know which polynomials are used for the LSFR. and at which speed the LH2 is rotating.
    """
    periods = [959000, 957000]

    # Translate points into position from each camera
    a1 = (count_1*8/periods[mode])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
    a2 = (count_2*8/periods[mode])*2*np.pi   

    # Transfor sweep angles to azimuth and elevation coordinates
    azimuth   = (a1+a2)/2 
    elevation = np.pi/2 - np.arctan2(np.sin(a2/2-a1/2-60*np.pi/180),np.tan(np.pi/6)) 

    # Project the angles into the z=1 image plane
    pts_lighthouse = np.zeros((len(count_1),2))
    for i in range(len(count_1)):
        pts_lighthouse[i,0] = -np.tan(azimuth[i])
        pts_lighthouse[i,1] = -np.sin(a2[i]/2-a1[i]/2-60*np.pi/180)/np.tan(np.pi/6) * 1/np.cos(azimuth[i])

    # Return the projected points
    return pts_lighthouse

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

    # print(R1)

    t1 = (e*v1+f*v3).reshape((3,1))
    t2 = (e*v1-f*v3).reshape((3,1))

    # if (n1[2]<0):
    #     t1 = -t1
    #     n1 = -n1
    # elif (n2[2]<0):
    #     t2 = -t2
    #     n2 = -n2

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

def compute_rando_rodriguez(n_star):

    rando_rodriguez = np.array([
    [-n_star[1]/np.sqrt(n_star[0]*n_star[0] + n_star[1]*n_star[1]), n_star[0]/np.sqrt(n_star[0]*n_star[0] + n_star[1]*n_star[1]), 0],
    [n_star[0]*n_star[2]/np.sqrt(n_star[0]*n_star[0] + n_star[1]*n_star[1]),n_star[1]*n_star[2]/np.sqrt(n_star[0]*n_star[0] + n_star[1]*n_star[1]),-np.sqrt(n_star[0]*n_star[0] + n_star[1]*n_star[1])],
    [-n_star[0],-n_star[1],-n_star[2]]]);

    return rando_rodriguez

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

def process_calibration(n_star, zeta, calib_data):
    
    # Create the nested dictionary structure needed
    calib_data['corners_lh2_proj'] = {}
    calib_data['corners_lh2_proj']['LHA'] = {}
    calib_data['corners_lh2_proj']['LHB'] = {}

    calib_data['corners_lh2_3D'] = {}


    # Project calibration points 
    for corner in ['tl','tr','bl','br']:
        # Project the points
        c1a = np.array([calib_data['corners_lh2_count'][corner]['LHA_count_1']])
        c2a = np.array([calib_data['corners_lh2_count'][corner]['LHA_count_2']])
        c1b = np.array([calib_data['corners_lh2_count'][corner]['LHB_count_1']])
        c2b = np.array([calib_data['corners_lh2_count'][corner]['LHB_count_2']])
        pts_A = LH2_count_to_pixels(c1a, c2a, 0)
        pts_B = LH2_count_to_pixels(c1b, c2b, 1)

        # Add it back to the calib dictionary
        calib_data['corners_lh2_proj']['LHA'][corner] = pts_A
        calib_data['corners_lh2_proj']['LHB'][corner] = pts_B

        # Reconstruct the 3D points
        point3D = solve_point_plane(n_star, zeta, pts_A)

        # Add the 3D points back to the  dictionary
        calib_data['corners_lh2_3D'][corner] = point3D

    return calib_data

def scale_LH2_to_real_size(calib_data, point3D):
    
    # Grab the point at top-left and bottom-right and scale them to be the corners of a 40cm square and use them to calibrate/scale the system.
    scale_p1 = calib_data['corners_lh2_3D']['tl']
    scale_p2 = calib_data['corners_lh2_3D']['br']
    scale = np.sqrt(2) * 40 / np.linalg.norm(scale_p2 - scale_p1)
    # Scale all the 3D points
    point3D *= scale
    
    # scale the calibration points
    calib_data['corners_lh2_3D_scaled'] = {}
    calib_data['corners_lh2_3D_scaled']['tl'] = calib_data['corners_lh2_3D']['tl'] * scale
    calib_data['corners_lh2_3D_scaled']['tr'] = calib_data['corners_lh2_3D']['tr'] * scale
    calib_data['corners_lh2_3D_scaled']['bl'] = calib_data['corners_lh2_3D']['bl'] * scale
    calib_data['corners_lh2_3D_scaled']['br'] = calib_data['corners_lh2_3D']['br'] * scale

    # Return scaled up scene
    return scale, calib_data, point3D

def scale_cam_to_real_size(calib_data, exp_data):
    
    pts_src = np.array([calib_data['corners_px']['tl'], calib_data['corners_px']['tr'], calib_data['corners_px']['br'], calib_data['corners_px']['bl']])
    pts_dst = np.array([[0.0, 40.0, 0.0], [40.0, 40.0, 0.0], [40.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # h, status = cv2.getPerspectiveTransform(pts_src, pts_dst)
    h, status = cv2.findHomography(pts_src[:,:2], pts_dst[:,:2]) # Only grab X,Y axis, because findhomography needs 2d points

    homo_pts = cv2.perspectiveTransform(exp_data[['x', 'y']].values.reshape((-1,1,2)), h).reshape((-1,2))

    # Update data frame of experiment data.
    exp_data['x'] = homo_pts[:,0]
    exp_data['y'] = homo_pts[:,1]
    
    # scale the calibration points
    calib_data['corners_px_scaled'] = {}
    calib_data['corners_px_scaled']['tl'] = pts_dst[0]
    calib_data['corners_px_scaled']['tr'] = pts_dst[1]
    calib_data['corners_px_scaled']['br'] = pts_dst[2]
    calib_data['corners_px_scaled']['bl'] = pts_dst[3]

    # Return scaled up scene
    return calib_data, exp_data

def interpolate_cam_data(lh_data, exp_data):


    camera_np = {'time': exp_data['time_s'].to_numpy(),
                'x':    exp_data['x'].to_numpy(),
                'y':    exp_data['y'].to_numpy(),}


    lh2_np = {'time':   lh_data['time_s'].to_numpy(),
                'x':    lh_data['LH_x'],
                'y':    lh_data['LH_y'],}


    # Offset the camera timestamp to get rid of the communication delay.
    camera_np['time'] += 265000e-6 # seconds
    

    camera_np['x_interp_lh2'] = np.interp(lh2_np['time'], camera_np['time'],  camera_np['x'])
    camera_np['y_interp_lh2'] = np.interp(lh2_np['time'], camera_np['time'],  camera_np['y'])


    exp_data_interp = pd.DataFrame({
                          'time_s' : lh2_np['time'],

                          'source': exp_data['source'].iloc[0],

                          'x': camera_np['x_interp_lh2'],

                          'y': camera_np['y_interp_lh2'],

                          'z': 0.0}
                          )

    return exp_data_interp

def correct_perspective(calib_data, exp_data):
    """
    THE SVD TECHNIQUE  FAILED, SO I HARDCODED T,R THAT I COMPUTED ELSEWHERE
    Create a rotation and translation vector to move the reconstructed grid onto the origin for better comparison.
    Using an SVD, according to: https://nghiaho.com/?page_id=671
    """
    
    A = np.array([calib_data['corners_px_scaled'][corner] for corner in ['tl', 'tr', 'bl', 'br']]).reshape((4,3))
    B = np.array([calib_data['corners_lh2_3D_scaled'][corner] for corner in ['tl', 'tr', 'bl', 'br']]).reshape((4,3))

    # Get  all the reconstructed points
    A2 = exp_data[['x','y','z']].to_numpy().T

    # Convert the point to column vectors,
    # to match twhat the SVD algorithm expects
    A = A.T
    B = B.T


    # Get the centroids
    A_centroid = A.mean(axis=1).reshape((-1,1))
    B_centroid = B.mean(axis=1).reshape((-1,1))

    # Get H
    H = (A - A_centroid) @ (B - B_centroid).T

    # Do the SVD
    U, S, V = np.linalg.svd(H)

    # Get the rotation matrix
    R = V @ U.T

    # check for errors, and run the correction
    if np.linalg.det(R) < 0:
        U, S, V = np.linalg.svd(R)
        V[:,2] = -1*V[:,2]
        R = V @ U.T

    # Get the ideal translation
    t = B_centroid - R @ A_centroid

    # Try tranformation you found elsewhere
    R1 = np.array(  [[ 0.98919328, -0.12556392,  0.07569918,],
                    [ 0.1027148 ,  0.22505302, -0.96891734,],
                    [ 0.10462473,  0.96622194,  0.23551821,]])
    
    R2 = np.array([[ 0.99979044,  0.01958034,  0.00597359],
                   [-0.01957252,  0.99980751, -0.00136582],
                   [-0.00599918,  0.00124862,  0.99998123]])
    
    R3 = np.array([[ 9.99999540e-01, -1.49610272e-04, -9.47844489e-04],
                   [ 1.49625869e-04,  9.99999989e-01,  1.63844060e-05],
                   [ 9.47842027e-04, -1.65262205e-05,  9.99999551e-01]])

    R = R3 @ R2 @ R1
    
    t = B_centroid - R @ A_centroid

    correct_points = (R@A2 + t)
    correct_points = correct_points.T

    correct_corners = (R@A + t).T
    # correct_corners = (A).T

    # Update dataframe
    exp_data['Rt_x'] = correct_points[:,0]
    exp_data['Rt_y'] = correct_points[:,1]
    exp_data['Rt_z'] = correct_points[:,2]

    # Add information to the calib data dictionary
    calib_data['corners_px_Rt'] = {}
    calib_data['corners_px_Rt']['tl'] = correct_corners[0]
    calib_data['corners_px_Rt']['tr'] = correct_corners[1]
    calib_data['corners_px_Rt']['bl'] = correct_corners[2]
    calib_data['corners_px_Rt']['br'] = correct_corners[3]

    return exp_data


def plot_error_histogram(lh_data, exp_data):
    """ 
    Calculate and plot a histogram  of the error of the reconstructed points, vs. 
    the ground truth.
    """
    # Extract needed data from the main dataframe
    points = lh_data[['LH_x', 'LH_y', 'LH_z']].to_numpy()
    ground_truth = exp_data[['Rt_x', 'Rt_y', 'Rt_z']].to_numpy()

    # Calculate distance between points and their ground truth
    errors =  np.linalg.norm(ground_truth - points, axis=1) * 10 # x10 To have the errors in milimeters
    # print the mean and standard deviation
    print(f"Mean Absolute Error = {errors.mean()} mm")
    print(f"Root Mean Square Error = {np.sqrt((errors**2).mean())} mm")
    print(f"Error Standard Deviation = {errors.std()} mm")

    # prepare the plot
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 3, figure = fig)
    hist_ax    = fig.add_subplot(gs[0:3, 0:3])
    axs = (hist_ax,)

    # Plot the error histogram
    n, bins, patches = hist_ax.hist(errors, 50, density=False)
    hist_ax.axvline(x=errors.mean(), color='red', label="Mean")

    for ax in axs:
        ax.grid()
        ax.legend()
    
    hist_ax.set_xlabel('Distance Error [mm]')
    hist_ax.set_ylabel('Measurements')

    plt.show()

    return

def plot_reconstructed_3D_scene(point3D, t_star, R_star, calib_data=None, exp_data=None):
    """
    Plot a 3D scene with the traingulated points previously calculated
    ---
    input:
    point3D - array [3,N] - triangulated points of the positions of the LH2 receveier
    t_star  - array [3,1] - Translation vector between the first and the second lighthouse basestation
    R_star  - array [3,3] - Rotation matrix between the first and the second lighthouse basestation
    point3D - array [3,N] - second set of pointstriangulated points of the positions of the LH2 receveier
    """
    ## Plot the two coordinate systems
    #  x is blue, y is red, z is green

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    # First lighthouse:
    arrow_size = 10
    ax.quiver(0,0,0, -arrow_size,0,0, color='xkcd:blue',lw=3)
    ax.quiver(0,0,0, 0,arrow_size,0, color='xkcd:red',lw=3)
    ax.quiver(0,0,0, 0,0,-arrow_size, color='xkcd:green',lw=3)

    # Second lighthouse:
    t_star_rotated = np.array([t_star.item(0), t_star.item(1), t_star.item(2)])
    # print(R_star)
    # print(t_star_rotated)
    x_axis = np.array([-arrow_size,0,0])@np.linalg.inv(R_star)
    y_axis = np.array([0,arrow_size,0])@np.linalg.inv(R_star)
    z_axis = np.array([0,0,-arrow_size])@np.linalg.inv(R_star)
    ax.quiver(t_star_rotated[0],t_star_rotated[1],t_star_rotated[2],x_axis[0],x_axis[1],x_axis[2], color='xkcd:blue',lw=3)
    ax.quiver(t_star_rotated[0],t_star_rotated[1],t_star_rotated[2],y_axis[0],y_axis[1],y_axis[2],color='xkcd:red',lw=3)
    ax.quiver(t_star_rotated[0],t_star_rotated[1],t_star_rotated[2],z_axis[0],z_axis[1],z_axis[2],color='xkcd:green',lw=3)
    ax.scatter(point3D[:,0],point3D[:,1],point3D[:,2], alpha=0.1, label="lh2")

    # Plot the calibration points in the LHA reference frame
    if calib_data is not None:
        calib_lh2 = np.array([calib_data['corners_lh2_3D_scaled'][corner] for corner in ['tl','tr','bl','br']]).reshape((4,3)) # originally it came out as shape=(3,1,4), I'm removing th uneeded dimension
        # ax.scatter(calib_lh2[:,0],calib_lh2[:,1],calib_lh2[:,2], alpha=0.5, color="xkcd:red")
        ax.scatter(calib_lh2[0,0],calib_lh2[0,1],calib_lh2[0,2], alpha=1, color="xkcd:red")
        ax.scatter(calib_lh2[1,0],calib_lh2[1,1],calib_lh2[1,2], alpha=1, color="xkcd:cyan")
        ax.scatter(calib_lh2[2,0],calib_lh2[2,1],calib_lh2[2,2], alpha=1, color="xkcd:green")
        ax.scatter(calib_lh2[3,0],calib_lh2[3,1],calib_lh2[3,2], alpha=1, color="xkcd:black")

    # Plot the Camera points, calibration and data
    if calib_data is not None and exp_data is not None:
        calib_cam = np.array([calib_data['corners_px_Rt'][corner] for corner in ['tl','tr','bl','br']])
        # ax.scatter(calib_cam[:,0],calib_cam[:,1],calib_cam[:,2], alpha=0.5, color="xkcd:orange")

        ax.scatter(calib_cam[0,0],calib_cam[0,1],calib_cam[0,2], alpha=1, color="xkcd:red")
        ax.scatter(calib_cam[1,0],calib_cam[1,1],calib_cam[1,2], alpha=1, color="xkcd:cyan")
        ax.scatter(calib_cam[2,0],calib_cam[2,1],calib_cam[2,2], alpha=1, color="xkcd:green")
        ax.scatter(calib_cam[3,0],calib_cam[3,1],calib_cam[3,2], alpha=1, color="xkcd:black")

        # cam_point3D = exp_data[['x','y','z']].values
        # ax.scatter(cam_point3D[:,0], cam_point3D[:,1], cam_point3D[:,2], alpha=1, color="xkcd:gray", label="camera")

        cam_point3D_Rt = exp_data[['Rt_x','Rt_y','Rt_z']].values
        ax.scatter(cam_point3D_Rt[:,0], cam_point3D_Rt[:,1], cam_point3D_Rt[:,2], alpha=0.1, color="xkcd:red", label="camera")


    # R_1 = np.eye(3,dtype='float64')
    # t_1 = np.zeros((3,1),dtype='float64')

    # # Plot the lighthouse orientation
    # arrow = np.array([0,0,1]).reshape((-1,1))
    # ax.quiver(t_1[0],t_1[1],t_1[2], (R_1 @ arrow)[0], (R_1 @ arrow)[1], (R_1 @ arrow)[2], length=0.2, color='xkcd:orange' )
    # ax.quiver(t_star[0],t_star[1],t_star[2], (R_star @ arrow)[0], (R_star @ arrow)[1], (R_star @ arrow)[2], length=0.2, color='xkcd:orange' )

    # ax.scatter(point3D[:,0],point3D[:,1],point3D[:,2], color='xkcd:green', alpha=0.5, label='triangulated')
    # ax.scatter(t_1[0],t_1[1],t_1[2], color='xkcd:orange', label='triang LH1')
    # ax.scatter(t_star[0],t_star[1],t_star[2], color='xkcd:orange', label='triang LH2')    

    # Plot the real 
    ax.text(-0.18,-0.1,0,s='LHA')
    ax.text(t_star_rotated[0], t_star_rotated[1], t_star_rotated[2],s='LHB')

    ax.axis('equal')
    ax.legend()
    ax.set_title('2D solved scene - 3D triangulated Points')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')   

    plt.show()

def plot_ALL_reconstructed_3D_scene(df, exp_data, t_star, R_star, calib_data):
    """

    """
    ## Plot the two coordinate systems
    #  x is blue, y is red, z is green

    fig = plt.figure(layout="constrained", figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    # First lighthouse:
    arrow_size = 10
    # ax.quiver(0,0,0, -arrow_size,0,0, color='xkcd:blue',lw=3)   # Previous before changin the the reference frame to get a prettier looking plot
    # ax.quiver(0,0,0, 0,-arrow_size,0, color='xkcd:red',lw=3)
    # ax.quiver(0,0,0, 0,0,arrow_size, color='xkcd:green',lw=3)
    ax.quiver(0,0,0, -arrow_size,0,0, color='xkcd:green',lw=3)
    ax.quiver(0,0,0, 0,-arrow_size,0, color='xkcd:blue',lw=3)
    ax.quiver(0,0,0, 0,0,arrow_size, color='xkcd:red',lw=3)

    # Second lighthouse:
    t_star_rotated = np.array([t_star.item(0), t_star.item(1), t_star.item(2)])
    # print(R_star)
    # print(t_star_rotated)
    x_axis = np.array([-arrow_size,0,0])@np.linalg.inv(R_star)
    y_axis = np.array([0,arrow_size,0])@np.linalg.inv(R_star)
    z_axis = np.array([0,0,-arrow_size])@np.linalg.inv(R_star)
    ax.quiver(t_star_rotated[2],t_star_rotated[0],t_star_rotated[1],x_axis[2],x_axis[0],x_axis[1], color='xkcd:blue',lw=3)
    ax.quiver(t_star_rotated[2],t_star_rotated[0],t_star_rotated[1],y_axis[2],y_axis[0],y_axis[1],color='xkcd:red',lw=3)
    ax.quiver(t_star_rotated[2],t_star_rotated[0],t_star_rotated[1],z_axis[2],z_axis[0],z_axis[1],color='xkcd:green',lw=3)

    # Plot the calibration points in the LHA reference frame
    if calib_data is not None:
        calib_lh2 = np.array([calib_data['corners_lh2_3D_scaled'][corner] for corner in ['tl','tr','bl','br']]).reshape((4,3)) # originally it came out as shape=(3,1,4), I'm removing th uneeded dimension
        # ax.scatter(calib_lh2[:,0],calib_lh2[:,1],calib_lh2[:,2], alpha=0.5, color="xkcd:red")
        # ax.scatter(calib_lh2[0,2],calib_lh2[0,0],calib_lh2[0,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_lh2[1,2],calib_lh2[1,0],calib_lh2[1,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_lh2[2,2],calib_lh2[2,0],calib_lh2[2,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_lh2[3,2],calib_lh2[3,0],calib_lh2[3,1], alpha=1, s=4, color="xkcd:black")

    # Plot the Camera points, calibration and data
    if calib_data is not None and exp_data is not None:
        calib_cam = np.array([calib_data['corners_px_Rt'][corner] for corner in ['tl','tr','bl','br']])
        # ax.scatter(calib_cam[:,2],calib_cam[:,0],calib_cam[:,1], alpha=0.5, color="xkcd:orange")

        # ax.scatter(calib_cam[0,2],calib_cam[0,0],calib_cam[0,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_cam[1,2],calib_cam[1,0],calib_cam[1,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_cam[2,2],calib_cam[2,0],calib_cam[2,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_cam[3,2],calib_cam[3,0],calib_cam[3,1], alpha=1, s=4, color="xkcd:black")

        # cam_point3D = exp_data[['x','y','z']].values
        # ax.scatter(cam_point3D[:,0], cam_point3D[:,1], cam_point3D[:,2], alpha=1, color="xkcd:gray", label="camera")


    for color in ['R', 'G', 'B']:

        point3D = df[color][['LH_x', 'LH_y','LH_z']].values
        cam_point3D_Rt = exp_data[color][['Rt_x','Rt_y','Rt_z']].values

        if color == 'R': 
            c = 'xkcd:red'
            label_lh = "Robot #1"
            label_c  = None
        if color == 'G': 
            c = 'xkcd:green'
            label_lh = "Robot #2"
            label_c  = None
        if color == 'B': 
            c = 'xkcd:blue'
            label_lh = "Robot #3"
            label_c  = "Ground Truth"


        if '4' in experiment_file and color == 'B':
            ax.scatter(point3D[60:,2],point3D[60:,0],point3D[60:,1], alpha=0.5, color=c, s=4, label=label_lh)
            ax.plot(cam_point3D_Rt[60:,2], cam_point3D_Rt[60:,0], cam_point3D_Rt[60:,1], alpha=0.5, color="xkcd:black", label=label_c)
        else:
            ax.scatter(point3D[:,2],point3D[:,0],point3D[:,1], alpha=0.5, color=c, s=4, label=label_lh)
            ax.plot(cam_point3D_Rt[:,2], cam_point3D_Rt[:,0], cam_point3D_Rt[:,1], alpha=0.5, color="xkcd:black", label=label_c)

        # ax.scatter(cam_point3D_Rt[:,0], cam_point3D_Rt[:,1], cam_point3D_Rt[:,2], alpha=0.2, color="xkcd:black", label=label_c)


    # Plot the real 
    ax.text(-0.18,-0.1,0,s='LHA')
    ax.text(t_star_rotated[2], t_star_rotated[0], t_star_rotated[1],s='LHB')


    ax.legend()
    ax.axis('equal')
    # ax.set_title('2D solved scene - 3D triangulated Points')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')   

    # Set Viewing orientation and zoom of the plot
    ax.view_init(-142,38, 0)
    ax.set_xlim3d((-144.424, -26.672))
    ax.set_ylim3d((-105, 13.6))
    ax.set_zlim3d((-20, 80))

    plt.savefig('Result-C-2lh_2d-solvedscene.pdf')
    plt.show()

def plot_projected_LH_views(pts_a, pts_b):
    """
    Plot the projected views from each of the lighthouse
    """

    fig = plt.figure(layout="constrained")
    gs = GridSpec(6, 3, figure = fig)
    lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
    lh2_ax = fig.add_subplot(gs[3:6, 0:3])
    axs = (lh1_ax, lh2_ax)

    # 2D plots - LH2 perspective
    lh1_ax.scatter(pts_a[:,0], pts_a[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1")
    lh2_ax.scatter(pts_b[:,0], pts_b[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2")

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
    lh1_ax.invert_yaxis()
    lh2_ax.invert_yaxis()

    plt.show()

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    cam_data, data, calib_data = import_data(data_file, experiment_file, calib_file)

    # Separate the data from the 3 DotBots
    df = {  'R': data.loc[data['source'] == 'R'],
            'B': data.loc[data['source'] == 'B'],
            'G': data.loc[data['source'] == 'G']}
    
    # Separate the data from the 3 DotBots
    exp_data = {'R': cam_data.loc[cam_data['source'] == 'R'],
                'B': cam_data.loc[cam_data['source'] == 'B'],
                'G': cam_data.loc[cam_data['source'] == 'G']}
    
    # Iterate over all the available colors on the dataframe, ignore colors that do not appear in the experiment.
    for color in [c for c in df.keys() if not df[c].empty]:
        # Use real dataset directly from lfsr counts
        pts_A = LH2_count_to_pixels(df[color]['LHA_count_1'].values, df[color]['LHA_count_2'].values, 0)
        pts_B = LH2_count_to_pixels(df[color]['LHB_count_1'].values, df[color]['LHB_count_2'].values, 1)
    
        # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
        df[color].loc[:,'LHA_proj_x'] = pts_A[:,0]
        df[color].loc[:,'LHA_proj_y'] = pts_A[:,1]
        df[color].loc[:,'LHB_proj_x'] = pts_B[:,0]
        df[color].loc[:,'LHB_proj_y'] = pts_B[:,1]

        # Solve the scene to find the transformation R,t from LHA to LHB
        # t_star, R_star, n_star, zeta = solve_2d_scene_get_Rtn(pts_A, pts_B)
        # solution_1, solution_2, zeta = fil_solve_2d(pts_A, pts_B)
        # t_star, R_star, n_star = solution_1

        # Solve the scene witht the blender_1.csv file, and then reuse that solution for the rest of the experiment
        t_star = np.array([[-0.99037204], [-0.04119019], [-0.13216125]])
        R_star =  np.array([[ 0.79756361, -0.28569461, -0.53129171],
                            [ 0.23989689,  0.95831365, -0.1551916 ],
                            [ 0.5534815 , -0.00368005,  0.83285334]])
        n_star = np.array([-0.05719626,  0.97055711, -0.23398181])
        zeta = 1.2583023961808537

        # Convert the for 4 calibration points from a LH projection to a 3D point
        calib_data = process_calibration(n_star, zeta, calib_data)

        # Transform LH projected points into 3D
        point3D = solve_point_plane(n_star, zeta, pts_A)

        # Scale up the LH2 points
        lh2_scale, calib_data, point3D = scale_LH2_to_real_size(calib_data, point3D)
        # Scale up the camera points
        calib_data, exp_data[color] = scale_cam_to_real_size(calib_data, exp_data[color])

        df[color]['LH_x'] = point3D[:,0]
        df[color]['LH_y'] = point3D[:,1]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
        df[color]['LH_z'] = point3D[:,2]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height
        
        # Interpolate Camera data to match the time base of the LH2 data
        exp_data[color] = interpolate_cam_data(df[color], exp_data[color]) 

        # Find the transform that superimposes one dataset over the other.
        exp_data[color] = correct_perspective(calib_data, exp_data[color])

        # Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
        # This will help correlate which point are supposed to go where.

        #############################################################################
        ###                             Plotting                                  ###
        #############################################################################
        # Plot Error Histogram
        # plot_error_histogram(df[color], exp_data[color])

        # Plot 3D reconstructed scene
        # plot_reconstructed_3D_scene(point3D, t_star * lh2_scale, R_star, calib_data, exp_data[color])

        # Plot projected views of the lighthouse
        # plot_projected_LH_views(pts_A, pts_B)

    # Plot all the data at the same time.
    plot_ALL_reconstructed_3D_scene(df, exp_data, t_star * lh2_scale, R_star, calib_data)
