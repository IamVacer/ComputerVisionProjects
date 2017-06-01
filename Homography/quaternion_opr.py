import numpy as np
import math as m

def quatmult(p, q):
    pq = np.zeros([4])
    pq[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    pq[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]
    pq[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    pq[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    return pq


def conjugate(q):
    qNew = [q[0],-q[1],-q[2],-q[3]]
    return qNew

def position_quat(rCoord):
    ans = [0, rCoord[0], rCoord[1], rCoord[2]]
    return ans

def rotate_quat_vec(x, y, z, theta):
    q = [0,0,0,0]
    t = m.radians(theta)/2.0
    q[0] = m.cos(t)
    q[1] = m.sin(t) * x
    q[2] = m.sin(t) * y
    q[3] = m.sin(t) * z
    return q

def rotate_quat(r, q):
    p = position_quat(r)
    pPrime = quatmult(quatmult(q, p), conjugate(q))
    return np.delete(pPrime,0)