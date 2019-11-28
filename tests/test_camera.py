# unit tests for camera functions
import numpy as np
from geometry import camera
import pytest
import scipy.linalg as la

eps = 1e-10

def test_decomposeCameraMatrix():
    for i in range(0,10000):
        M = np.random.rand(3,3)
        K, R = la.rq(M)
        C = np.random.rand(3,1)
        P = K.dot(np.hstack((R, -R.dot(C))))
        Kr, Rr, Cr = camera.decomposeCameraMatrix(P)
        Pr = Kr.dot(np.hstack((Rr, -Rr.dot(Cr))))
        assert (np.all((P - Pr) < eps)) & (np.all((K - Kr) < eps)) & (np.all((R - Rr) < eps)) & (np.all((C - Cr) < eps))
