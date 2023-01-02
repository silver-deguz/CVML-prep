import numpy as np 


def homogenize(pts):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((pts, np.ones((1,pts.shape[1]))))


def dehomogenize(pts):
    # converts points from homogeneous to inhomogeneous coordinates   
    return pts[:-1]/pts[-1]


def compute_fundamental_matrix(x1, x2):
    """    
    Computes the fundamental matrix from corresponding points 
    (x1,x2 3*n arrays) using the 8 point algorithm.
    Each row in the A matrix below is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] 

    """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    A = np.zeros((n,9))
    for i in range(n):
        A[i,:] = np.kron(x1[:,i], x2[:,i])
  
    _, _, vh = np.linalg.svd(A)
    Fs = vh[-1,:].reshape((3, 3))

    # Enforcing rank 2 constraint
    U, S, VT = np.linalg.svd(Fs)
    S[-1] = 0
    F = U @ np.diag(S) @ VT
    return F/F[2,2]


def compute_epipole(F):
    """
    This function computes the epipoles for a given fundamental matrix 
    
    Returns:
    e1 epipole in image 1
    e2 epipole in image 2
    """
    
    U, S, VT = np.linalg.svd(F)
    e1 = VT[-1,:]
    e2 = -U[:,-1]
    return dehomogenize(e1), homogenize(e2)


def normalized_eight_pt_alg(x1, x2):
    # Normalization of the corner points is handled here
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array(
        [[S1, 0, -S1 * mean_1[0]],
         [0, S1, -S1 * mean_1[1]],
         [0, 0, 1]
        ])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array(
        [[S2, 0, -S2 * mean_2[0]],
         [0, S2, -S2 * mean_2[1]],
         [0, 0, 1]
        ])
    x2 = np.dot(T2,x2)

    # compute F with the normalized coordinates
    F = compute_fundamental_matrix(x1, x2)

    # undo the normalization
    F = np.dot(T1.T, np.dot(F, T2))
    return F / F[2,2]


def plot_epipolar_lines(img1, img2, cor1, cor2):
    """
    Plot epipolar lines on image given images, corners
    """
    def build_line(line, m, n):
        a, b, c = line
        t = np.linspace(0,n,100)
        lt = np.array([(c + a*tt) / (-b) for tt in t])

        # take only line points inside the image
        ndx = (lt>=0) & (lt<m)
        return t[ndx], lt[ndx]
        
    F = normalized_eight_pt_alg(cor2, cor1)
    e1, e2 = compute_epipole(F)
    print(e1[:2],e2[:2])
    assert cor1.shape[1] == cor2.shape[1]
    num_corners = cor1.shape[1]
    m, n = img1.shape[:2]

    l1 = []
    l2 = []
    for i in range(num_corners):
        l1.append(F.T @ cor2[:,i])
        l2.append(F @ cor1[:,i])
 
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax1.imshow(img1, cmap='gray')
    ax1.scatter(cor1[0], cor1[1], s=35, edgecolors='r', facecolors='none')
    
    ax2 = fig.add_subplot(222)
    ax2.imshow(img2, cmap='gray')
    ax2.scatter(cor2[0], cor2[1], s=35, edgecolors='r', facecolors='none')
                
    for i in range(num_corners):
        x1, y1 = build_line(l1[i], m-1, n-1)
        ax1.plot(x1, y1)
        x2, y2 = build_line(l2[i], m-1, n-1)
        ax2.plot(x2, y2)
    plt.show()