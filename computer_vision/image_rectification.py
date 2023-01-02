import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt 
from cv_utils import * 



def compute_matching_homographies(e2, F, im2, points1, points2):
    '''This function computes the homographies to get the rectified images
    input:
    e2--> epipole in image 2
    F--> the Fundamental matrix (Think about what you should be passing F or F.T!)
    im2--> image2
    points1 --> corner points in image1
    points2--> corresponding corner points in image2
    output:
    H1--> Homography for image 1
    H2--> Homography for image 2
    '''
    # calculate H2
    width = im2.shape[1]
    height = im2.shape[0]

    T = np.identity(3)
    T[0][2] = -1.0 * width / 2
    T[1][2] = -1.0 * height / 2

    e = T.dot(e2)
    e1_prime = e[0]
    e2_prime = e[1]
    if e1_prime >= 0:
        alpha = 1.0
    else:
        alpha = -1.0

    R = np.identity(3)
    R[0][0] = alpha * e1_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[0][1] = alpha * e2_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[1][0] = - alpha * e2_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[1][1] = alpha * e1_prime / np.sqrt(e1_prime**2 + e2_prime**2)

    f = R.dot(e)[0]
    G = np.identity(3)
    G[2][0] = - 1.0 / f

    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    # calculate H1
    e_prime = np.zeros((3, 3))
    e_prime[0][1] = -e2[2]
    e_prime[0][2] = e2[1]
    e_prime[1][0] = e2[2]
    e_prime[1][2] = -e2[0]
    e_prime[2][0] = -e2[1]
    e_prime[2][1] = e2[0]

    v = np.array([1, 1, 1])
    M = e_prime.dot(F) + np.outer(e2, v)

    points1_hat = H2.dot(M.dot(points1)).T
    points2_hat = H2.dot(points2).T
   
    W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
    b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]

    # least square problem
    a1, a2, a3 = np.linalg.lstsq(W, b)[0]
    HA = np.identity(3)
    HA[0] = np.array([a1, a2, a3])

    H1 = HA.dot(H2).dot(M)
    return H1, H2


def rectify(H, img):
    rect_img = np.zeros(img.shape).astype('uint8')
    Hinv = np.linalg.inv(H)
    for i in range(rect_img.shape[0]):
        for j in range(rect_img.shape[1]):
            q = np.array([j,i]).T    
            p = dehomogenize(Hinv @ homogenize(q))
            pi = int(p[1])
            pj = int(p[0])
            if 0 <= pi <= rect_img.shape[0] and 0 <= pj <= rect_img.shape[1]:
                rect_img[i,j,:] = img[pi,pj,:]
    return rect_img


def image_rectification(img1, img2, x1, x2):
    F = normalized_eight_pt_alg(x1, x2)
    e1, e2 = compute_epipole(F)
    H1, H2 = compute_matching_homographies(e2, F, img2, x1, x2)

    img1_rectified = rectify(H1, img1)
    img2_rectified = rectify(H2, img2)

    x1_rectified = dehomogenize(H1 @ homogenize(x1))
    x2_rectified = dehomogenize(H2 @ homogenize(x2))

    plot_epipolar_lines(img1_rectified, img2_rectified, x1_rectified, x2_rectified)
    
