import numpy as np
import math as m

def eax2R(t):
    """Converts angle-axis vector to rotation matrix"""
    theta = np.linalg.norm(t)
    if theta < np.finfo(float).eps:   # If the rotation is very small...
        return np.array((
            (1 , -t[2, 0], t[1, 0]),
            (t[2, 0], 1, -t[0, 0]),
            (-t[1, 0], t[0, 0], 1))
            )
    
    # Otherwise set up standard matrix, first setting up some convenience
    # variables
    t = t/theta;  x = t[0, 0]; y = t[1, 0]; z = t[2, 0]
    
    c = m.cos(theta); s = m.sin(theta); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    return np.array((
        (x*xC+c,xyC-zs,zxC+ys),
        (xyC+zs, y*yC+c, yzC-xs),
        (zxC-ys, yzC+xs, z*zC+c))
        )
    