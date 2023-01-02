import numpy as np 
import os 
from skimage.io import imread
from cv_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def linear_estimate_RT(E):
    U, S, VT = np.linalg.svd(E)
    Z = np.array([[0,  1, 0],
                  [-1, 0, 0],
                  [0,  0, 0]])
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    M = U @ Z @ U.T
    Q1 = U @ W.T @ VT
    R1 = np.linalg.det(Q1) * Q1 
    Q2 = U @ W @ VT 
    R2 = np.linalg.det(Q2) * Q2 
    T1 = U[:,-1]
    T2 = -U[:,-1]

    RT = np.zeros((4, 3, 4))
    last_row = np.array([0,0,0,1])
    RT[0,:,:] = np.vstack((np.hstack((R1, T1)), last_row))
    RT[1,:,:] = np.vstack((np.hstack((R1, T2)), last_row))
    RT[2,:,:] = np.vstack((np.hstack((R2, T1)), last_row))
    RT[3,:,:] = np.vstack((np.hstack((R2, T2)), last_row))
    return RT


'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor) i.e. M1 and M2
Returns:
    point_3d - the 3D point (3x1 vector)
''' 
def linear_estimate_3d_point(image_points, camera_matrices):
    N = image_points.shape[0]
    A = np.zeros((2*N, 4))

    for i in range(N):
        p = image_points[i]
        M = camera_matrices[i]
        A[2*i,:] = p[0] * M[2] - M[0]
        A[2*1+1,:] = p[1] * M[2] - M[1]

    U, S, VT = np.linalg.svd(A)
    P = VT[-1]
    P /= P[-1]
    scene_point = P[:3]
    return scene_point


'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    scene_point - the 3D point corresponding to points in the image, X
    image_points - the measured points in each of the M images, x and x' (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(scene_point, image_points, camera_matrices):
    x1 = image_points[0,:]
    x2 = image_points[1,:]
    x1_hat = dehomogenize(camera_matrices[0,:,:] @ homogenize(scene_point)) 
    x2_hat = dehomogenize(camera_matrices[1,:,:] @ homogenize(scene_point))
    err1 = x1 - x1_hat
    err2 = x2 - x2_hat 
    error = np.vstack((err1, err2))
    return error
    

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(scene_point, camera_matrices):
    J = np.zeros((2*camera_matrices.shape[0], 3))
    X = homogenize(scene_point)

    for i in range(camera_matrices.shape[0]):
        M = camera_matrices[i]
        p = dehomogenize(M @ X)
        Jx = (p[2]*np.array([M[0, 0], M[0, 1], M[0, 2]]) \
              - p[0]*np.array([M[2, 0], M[2, 1], M[2, 2]])) / p[2]**2
        Jy = (p[2]*np.array([M[1, 0], M[1, 1], M[1, 2]]) \
              - p[1]*np.array([M[2, 0], M[2, 1], M[2, 2]])) / p[2]**2


    return J



def SFM():
    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Compute initial estimate of R,T transformations from Essential Matrix
    # 4 pairs of R,T configurations result from decompositions of E but only 
    # 1 pair is valid , i.e. in front of the cameras.
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length  
    E = K.T @ fundamental_matrices[0] @ K
    img0 = imread(image_paths[0])
    H, W, _ = img0.shape
    RT = linear_estimate_RT(E)

    # Compute linear estimate of the 3D scene points 
    camera_matrices = np.zeros((2, 3, 4))
    P0 = np.hstack((np.eye(3), np.zeros((3,1))))
    camera_matrices[0, :, :] =  K @ P0
    camera_matrices[1, :, :] = K @ RT[0,:,:]
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    linear_estimate_x3d = linear_estimate_3d_point(unit_test_matches.copy(), camera_matrices.copy())
    
    # Calcuate the reprojection error and its Jacobian