# useful functions to work with camera in computer vision

import numpy as np
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

    if np.dot(np.cross(R[:,0], R[:,1]), R[:,2]) < 0:
        print('Warning: R is left handed')

    return K, R, C



