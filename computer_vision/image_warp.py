import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt 
import cv2 as cv 
from scipy import interpolate, ndimage
from cv_utils import * 

# https://github.com/lesialin/image_warp/blob/master/image_warping.cpp
# https://www.reddit.com/r/learnpython/comments/dbbpoi/different_answers_with_same_data_matlabs_interp2/
# https://scipython.com/book/chapter-8-scipy/additional-examples/interpolation-of-an-image/

def compute_homography(pts_source, pts_target):
    assert pts_source.shape[0]==3 and pts_target.shape[0]==3
    n = pts_source.shape[1]
    A = np.zeros((n*2, 9))

    for i in range(n):
        x1, y1, _ = pts_source[:, i]
        x2, y2, _ = pts_target[:, i]
        A[2*i,:] = np.array([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A[2*i+1,:] = np.array([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H = np.reshape(h, (3,3))
    return H

def test_compute_homography():
    pts_source = np.array([[0,0.5],[1,0.5],[1,1.5],[0,1.5]]).T
    pts_target = np.array([[0,0],[1,0],[2,1],[-1,1]]).T
    H = compute_homography(homogenize(pts_source), homogenize(pts_target))
    mapped_points = dehomogenize(np.matmul(H, homogenize(pts_source)))
    error = np.sum(pts_target - mapped_points)
    assert error < 1e-5, "error between mapped and target points is too large, check calculation"

def warp(source, pts_source, target):
    # q = H * p
    assert source.shape[2] == target.shape[2]
    r, c = source.shape[0], source.shape[1]
    rt, ct = target.shape[0], target.shape[1]

    pts_target = np.array([[0, 0], [0, rt-1], [ct-1, rt-1], [ct-1, 0]]).T
    H = compute_homography(homogenize(pts_source), homogenize(pts_target))

    for i in range(r):
        for j in range(c):
            p = np.array([[j,i]]).T
            q = dehomogenize(H @ homogenize(p))
            qi = int(q[1])
            qj = int(q[0])
            target[qi,qj,:] = source[i,j,:]
    return target

def inverse_warp(source, pts_source, target):
    # p = inv(H) @ q
    assert source.shape[2] == target.shape[2]
    rt, ct = target.shape[0], target.shape[1]

    pts_target = np.array([[0, 0], [0, rt-1], [ct-1, rt-1], [ct-1, 0]]).T
    H = compute_homography(homogenize(pts_source), homogenize(pts_target))
    Hinv = np.linalg.inv(H)

    for i in range(rt):
        for j in range(ct):
            q = np.array([j,i]).T      
            p = dehomogenize(Hinv @ homogenize(q))
            pi = int(p[1])
            pj = int(p[0])
            target[i,j,:] = source[pi,pj,:]
    return target

def inverse_warp_interp(source, pts_source, target):
    # p = inv(H) @ q
    assert source.shape[2] == target.shape[2]
    rt, ct = target.shape[0], target.shape[1]
    B, G, R = cv.split(source)

    pts_target = np.array([[0, 0], [0, rt-1], [ct-1, rt-1], [ct-1, 0]]).T
    H = compute_homography(homogenize(pts_source), homogenize(pts_target))
    Hinv = np.linalg.inv(H)

    x = np.arange(ct)
    y = np.arange(rt)
    xx, yy = np.meshgrid(x,y)
    
    xx = xx.reshape(1,-1)
    yy = yy.reshape(1,-1)

    q = np.vstack((xx, yy))
    p = dehomogenize(Hinv @ homogenize(q))

    px = p[0,:]
    py = p[1,:]

    Vqb = ndimage.map_coordinates(B, [px.ravel(), py.ravel()], order=1, mode='nearest').reshape(rt,ct)
    Vqg = ndimage.map_coordinates(G, [px.ravel(), py.ravel()], order=1, mode='nearest').reshape(rt,ct)
    Vqr = ndimage.map_coordinates(R, [px.ravel(), py.ravel()], order=1, mode='nearest').reshape(rt,ct)

    target[:,:,0] = Vqb
    target[:,:,1] = Vqg
    target[:,:,2] = Vqr
    return target


def image_warp():
    img_path = Path("/Users/jdeguzman/workspace/photo.jpeg")
    image = cv.imread(str(img_path))
    pts_source = np.array([[102,1154],[1044,1493],[511,121],[1941,211]]).T
    target = np.zeros((1200,840,3))
    inverse_warp_interp(image, pts_source, target)


image_warp()