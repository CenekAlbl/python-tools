import numpy as np

def triangulateTwoView(P1, P2, u1, u2):
    assert P1.shape[0] is 3 and P1.shape[1] is 4 , "shape of the P1 matrix should be 3x4" 
    assert P2.shape[0] is 3 and P2.shape[1] is 4 , "shape of the P2 matrix should be 3x4" 
    assert u1.shape[0] is 2 and u1.shape[1] is 1 , "shape of u1 should be 2x1" 
    assert u2.shape[0] is 2 and u2.shape[1] is 1 , "shape of u2 should be 2x1" 
    x1 = u1[0]
    y1 = u1[1]
    x2 = u2[0]
    y2 = u2[1]
    A = np.array(((x1*P1[2,:]-P1[0,:]), (y1*P1[2,:]-P1[1,:]), (x2*P2[2,:]-P2[0,:]), (y2*P2[2,:]-P2[1,:])))
    U,D,V = np.linalg.svd(A)
    X = V[-1,0:3]/V[-1,3]
    return X

