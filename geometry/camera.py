# useful functions to work with camera in computer vision

import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la
from . import utils as ut

def decomposeCameraMatrix(P):
    p1 = P[:,[0]]
    p2 = P[:,[1]]
    p3 = P[:,[2]]
    p4 = P[:,[3]]

    M = np.hstack((p1, p2, p3))

    X = np.linalg.det(np.hstack((p2, p3, p4)))
    Y = -np.linalg.det(np.hstack((p1, p3, p4)))
    Z = np.linalg.det(np.hstack((p1, p2, p4)))
    T = -np.linalg.det(np.hstack((p1, p2, p3)))

    # camera centre
    C = ut.h2a(np.array((X,Y,Z,T)).reshape(4,1))

    K, R = la.rq(M)
    K = K / np.abs(K[2,2])

    if K[0,0] < 0:
        D = np.diag([-1, -1, 1])
        K = K.dot(D)
        R = D.dot(R)

    if K[1,1] < 0:
        D = np.diag([1, -1, -1])
        K = K.dot(D)
        R = D.dot(R)

    if np.dot(np.cross(R[:,0], R[:,1]), R[:,2]) < 0:
        print('Warning: R is left handed')

    return K, R, C

def plotCamera(ax,P,color='blue',size=1):
    K, R, C = decomposeCameraMatrix(P)
    ptsCam = (np.array([[-1, -1, 1],[ 1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]]).transpose())*size
    ptsWorld = ut.h2a(np.vstack((np.hstack((R.transpose(), C)),np.array((0, 0, 0, 1)))).dot(ut.a2h(ptsCam)))
    ptsIdxs = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 4, 2, 4, 3, 4])
    lines = ptsWorld[:,ptsIdxs]
    ax.plot(lines[0,:],lines[1,:],lines[2,:],color)




