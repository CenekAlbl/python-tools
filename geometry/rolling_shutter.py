import numpy as np
from . import utils as ut
from . import rotation as rot
from scipy.linalg import null_space

# angle-axis to rotation matrix fast version for RS projection

def eax2RforRS(axis,angles):
        if(len(angles.shape)>1):
                n = angles.shape[1]
        else:
                n = angles.shape[0]
        theta = np.linalg.norm(axis)
        if theta < np.finfo(float).eps:   # If the rotation is very small...
                return np.tile(np.array((
                (1 , -axis[2, 0], axis[1, 0]),
                (axis[2, 0], 1, -axis[0, 0]),
                (-axis[1, 0], axis[0, 0], 1))
                ), (1,n))
    
        # Otherwise set up standard matrix, first setting up some convenience
        # variables
        axis = axis/theta
        
        vx = np.tile(ut.x_(axis),(n,1))
        vx2 = np.tile(ut.x_(axis).dot(ut.x_(axis)),(n,1))
        s = np.tile(np.sin(angles),(3,1)).transpose()
        c = np.tile(np.cos(angles),(3,1)).transpose()
        sM = np.tile(s.ravel().reshape(n*3,1),(1,3))
        cM = np.tile(c.ravel().reshape(n*3,1),(1,3))
        I = np.tile(np.eye(3),(n,1))

        Rw = I + vx*sM + (1-cM)*vx2
        return Rw

# RS epipolar error for 2cams wt model
def rs2CamEpipolarErrorWTModelLinearized(M,data):
    u1 = data[0:2,:]
    u2 = data[2:4,:]
    Rr = np.diag(np.array((-1, -1, 1)))
    w = M[0:3]
    t = np.array([1-M[3],M[3],M[4]]).reshape(3,1)
    err = np.zeros(data.shape[1])
    for i in range(0,u1.shape[1]):
        R = Rr.dot(np.eye(3)+ut.x_(w*u2[1,i])).dot(np.eye(3)-ut.x_(w*u1[1,i]))
        E = ut.x_(u2[1,i]*Rr.dot(t)-u1[1,i]*R.dot(t)).dot(R)
        Ex = E.dot(np.array((u2[0,i], u2[0,i], 1)).reshape(3,1))
        xpt = np.array((u2[0,i],u2[1,i],1))
        Etxp = E.T.dot(xpt.reshape(3,1))
        err[i] = np.power(xpt.dot(Ex),2)/(np.power(Ex[0],2)+np.power(Ex[1],2)+np.power(Etxp[0],2)+np.power(Etxp[1],2))
        if(np.isnan(err[i])):
            err[i] = 0
    return err

def lin_wtvC(X,u,vk):    
        A = np.zeros((12,13))
        A[:,[0]] = np.vstack((X[:,[2]] + X[:,[1]]*u[:,[1]], -X[:,[1]]*u[:,[0]]))
        A[:,[1]] = np.vstack((-X[:,[0]]*u[:,[1]], X[:,[2]] + X[:,[0]]*u[:,[0]]))
        A[:,[2]] = np.vstack((-X[:,[0]], -X[:,[1]]))
        A[:,[3]] = np.vstack((X[:,[0]]*vk[1]*( - u[:,[0]]) - X[:,[2]]*( - u[:,[0]]) - u[:,[1]]*(X[:,[1]]*( - u[:,[0]]) + X[:,[0]]*vk[2]*( - u[:,[0]]) - X[:,[2]]*vk[0]*( - u[:,[0]])) - X[:,[1]]*vk[0]*( - u[:,[0]]), u[:,[0]]*(X[:,[1]]*( - u[:,[0]]) + X[:,[0]]*vk[2]*( - u[:,[0]]) - X[:,[2]]*vk[0]*( - u[:,[0]]))))
        A[:,[4]] = np.vstack((u[:,[1]]*(X[:,[0]]*(-u[:,[0]]) - X[:,[1]]*vk[2]*( - u[:,[0]]) + X[:,[2]]*vk[1]*( - u[:,[0]])), X[:,[0]]*vk[1]*( - u[:,[0]]) - X[:,[2]]*( - u[:,[0]]) - u[:,[0]]*(X[:,[0]]*( - u[:,[0]]) - X[:,[1]]*vk[2]*( - u[:,[0]]) + X[:,[2]]*vk[1]*( - u[:,[0]])) - X[:,[1]]*vk[0]*( - u[:,[0]])))
        A[:,[5]] = np.vstack((X[:,[0]]*( - u[:,[0]]) - X[:,[1]]*vk[2]*( - u[:,[0]]) + X[:,[2]]*vk[1]*( - u[:,[0]]), X[:,[1]]*( - u[:,[0]]) + X[:,[0]]*vk[2]*( - u[:,[0]]) - X[:,[2]]*vk[0]*( - u[:,[0]])))
        A[:,[6]] = np.vstack((np.zeros((6,1)), np.ones((6,1))))
        A[:,[7]] = np.vstack((-np.ones((6,1)), np.zeros((6,1))))
        A[:,[8]] = np.vstack((u[:,[1]], -u[:,[0]]))
        A[:,[9]] = np.vstack((np.zeros((6,1)), u[:,[0]]))
        A[:,[10]] = np.vstack((-u[:,[0]], np.zeros((6,1))))
        A[:,[11]] = np.vstack((-u[:,[1]]*( - u[:,[0]]), u[:,[0]]*( - u[:,[0]])))
        A[:,[12]] = np.vstack((X[:,[2]]*u[:,[1]] - X[:,[1]], X[:,[0]] - X[:,[2]]*u[:,[0]]))
        n = null_space(A)
        s = n[12,-1]
        v = n[0:3,[-1]]/s
        w = n[3:6,[-1]]/s
        C = n[6:9,[-1]]/s
        t = n[9:12,[-1]]/s
        return w,t,v,C

def proj_R6P_eax(M,X,d):
        w = M[0:3].reshape(3,1)
        t = M[3:6].reshape(3,1)
        v = M[6:9].reshape(3,1)
        C = M[9:12].reshape(3,1)
        u = np.array((0,0)).reshape(2,1)
        projs = []
        for XX in X.transpose():
                for i in range(50):
                        XX = XX.reshape(3,1)
                        unew = ut.h2a(np.matmul(rot.eax2R(u[d]*w),np.matmul(rot.eax2R(v),XX))+C+u[d]*t)
                        if(np.linalg.norm(unew-u)<1e-8):
                                u = unew
                                break
                        u = unew
                projs.append(u)
        return np.squeeze(np.array(projs),axis=2).transpose()

def calc_R6P_eax_err(M,data):
        n = data.shape[1]
        w = M[0:3].reshape(3,1)
        t = M[3:6].reshape(3,1)
        v = M[6:9].reshape(3,1)
        C = M[9:12].reshape(3,1)
        u = data[3:5,:]
        X = data[0:3,:]
        col = u[0,:]
        r = np.tile(col,(3,1))
        Rw = eax2RforRS(w,col*np.linalg.norm(w))
        Rv = rot.eax2R(v)
        Rt = Rw.dot(Rv)
        Xrep = np.tile(X,(3,1))
        RX = np.sum(Rt.transpose()*Xrep.ravel(order='F').reshape((3,3*n),order='F'),axis=0).reshape((3,n),order='F')
        Z = RX + np.tile(C,(1,n)) + r*np.tile(t,(1,n))
        u_rs = ut.h2a(Z)
        err  = np.sqrt(np.sum((u_rs-u)**2,axis=0))
        return err

def calc_R6P_eax_err_penalize_motion(M,data):
        weight = 1000000
        n = data.shape[1]
        w = M[0:3].reshape(3,1)
        t = M[3:6].reshape(3,1)
        v = M[6:9].reshape(3,1)
        C = M[9:12].reshape(3,1)
        u = data[3:5,:]
        X = data[0:3,:]
        col = u[0,:]
        r = np.tile(col,(3,1))
        Rw = eax2RforRS(w,col*np.linalg.norm(w))
        Rv = rot.eax2R(v)
        Rt = Rw.dot(Rv)
        Xrep = np.tile(X,(3,1))
        RX = np.sum(Rt.transpose()*Xrep.ravel(order='F').reshape((3,3*n),order='F'),axis=0).reshape((3,n),order='F')
        Z = RX + np.tile(C,(1,n)) + r*np.tile(t,(1,n))
        u_rs = ut.h2a(Z)
        err  = np.hstack(((u_rs-u).ravel(),np.abs(w.ravel()*weight,np.abs(t.ravel())*weight)))

        return err

def calc_R6P_eax_err_for_lsq(M,data):
        n = data.shape[1]
        w = M[0:3].reshape(3,1)
        t = M[3:6].reshape(3,1)
        v = M[6:9].reshape(3,1)
        C = M[9:12].reshape(3,1)
        u = data[3:5,:]
        X = data[0:3,:]
        col = u[0,:]
        r = np.tile(col,(3,1))
        Rw = eax2RforRS(w,col*np.linalg.norm(w))
        Rv = rot.eax2R(v)
        Rt = Rw.dot(Rv)
        Xrep = np.tile(X,(3,1))
        RX = np.sum(Rt.transpose()*Xrep.ravel(order='F').reshape((3,3*n),order='F'),axis=0).reshape((3,n),order='F')
        Z = RX + np.tile(C,(1,n)) + r*np.tile(t,(1,n))
        u_rs = ut.h2a(Z)
        err  = (u_rs-u).ravel()
        return err

def calc_R6P_2lin_err(M,data):
        w = M[0:3].reshape(3,1)
        t = M[3:6].reshape(3,1)
        v = M[6:9].reshape(3,1)
        C = M[9:12].reshape(3,1)
        err = np.zeros(data.shape[1])
        i = 0
        for temp in data.transpose():
                X = temp[0:3,None]
                u = temp[3:5,None]
                proj = ut.h2a(np.matmul((np.eye(3)+u[0]*ut.x_(w)),np.matmul((np.eye(3)+ut.x_(v)),X))+C+u[0]*t)
                #print(proj.shape)
                err[i] = np.linalg.norm(proj-u)
                i+=1
        return err

def calc_R6P_2lin_err_eq(M,data):
        w = M[0:3]
        t = M[3:6]
        v = M[6:9]
        C = M[9:12]
        err = np.zeros(data.shape[1])
        i = 0
        for temp in data.transpose():
                X = temp[0:3,None]
                u = temp[3:5,None]
                u = ut.a2h(u)
                eq = np.matmul(ut.x_(u),np.matmul((np.eye(3)+u[0]*ut.x_(w)),np.matmul((np.eye(3)+ut.x_(v)),X))+C+u[0]*t)
                err[i] = np.sum(np.absolute(eq))
                i+=1
        return err

def r6p_lin(data):
        maxiter = 5
        eps = 1e-10
        X = data[0:3,0:6]
        u = data[3:5,0:6]
        v = np.zeros((3,1))
        maxiter = 5
        k = 0
        while k<maxiter:
                w,t,v,C = lin_wtvC(X.transpose(),u.transpose(),v)
                M = np.vstack((w,t,v,C))
                err = np.sum(calc_R6P_2lin_err_eq(M,data))
                if err < eps:
                        return [M]
                else:
                        k += 1
        return []

def testR6PLin(X,u):
        # perfect perspective data
        X = np.array([[-0.735203763665459, -0.496627573213431, -0.872484275284515, -0.427093140715349, -0.834433230284650,  -0.510357709134182], [0.238783627918357, -0.331469664458513, -0.774067735937978, -0.330087062598541, -0.532562868622104, -0.175836814526873], [0.348232082844991, 0.496786171364075, 0.842812754497372, 0.900023147138860, 0.062454513759772,0.920453051685240]])
        u = np.array([[0.087573419907225, 0.234591535755890, -0.008470881920492,0.221656649106303, 0.020112962609198, 0.176817312337601], [0.377812297395972, -0.030443875680965, -0.260227638405494, -0.023394282943956, -0.225285886765297,  0.055631494170819]])
        return r6p_lin(np.vstack((X,u)))
