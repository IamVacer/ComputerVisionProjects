import numpy as np
import numpy.linalg as la
import Input
import quaternion_opr as quat_opr

from matplotlib import pyplot as plt



persPts = np.zeros([11*4, 2])
orthoPts = np.zeros([11*4, 2])

p0 = [0,0,-5]
q = quat_opr.rotate_quat_vec(0, 1, 0, -30)
p1 = quat_opr.rotate_quat(p0, q)
p2 = quat_opr.rotate_quat(p1, q)
p3 = quat_opr.rotate_quat(p2, q)

print ('Part 1.2     ===============', '\n')
print (np.round(p1, 2), '\n')
print (np.round(p2, 2), '\n')
print (np.round(p3, 2), '\n')



print ('Part 1.3 ===============', '\n')
def quat2rot(q):
    r = np.zeros([3,3])
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    r[0][0] = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
    r[0][1] = 2 * (q1 * q2 - q0 * q3)
    r[0][2] = 2 * (q1 * q3 + q0 * q2)
    r[1][0] = 2 * (q1 * q2 + q0 * q3)
    r[1][1] = q0 ** 2 + q2** 2 - q1 ** 2 - q3 ** 2
    r[1][2] = 2 * (q2 * q3 - q0 * q1)
    r[2][0] = 2 * (q1 * q3 - q0 * q2)
    r[2][1] = 2 * (q2 * q3 + q0 * q1)
    r[2][2] = q0 ** 2 + q3 ** 2 - q1 ** 2 - q2 ** 2
    return np.matrix(r)

q2 = quat_opr.rotate_quat_vec(0, 1, 0, 30)

rotateMatrix = quat2rot(q2)
quatmat_0 = np.identity(3)
quatmat_1 = rotateMatrix * quatmat_0
quatmat_2 = rotateMatrix * quatmat_1
quatmat_3 = rotateMatrix * quatmat_2

print(np.round(quatmat_0, 2), '\n')
print(np.round(quatmat_1, 2), '\n')
print(np.round(quatmat_2, 2), '\n')
print(np.round(quatmat_3, 2), '\n')

print ('Part 2.1 ===============', '\n')

def calU(t, i, j, k, pt):

    top = (pt - t)* i.transpose()
    bottom = (pt - t)* k.transpose()
    return top, top/bottom

def calV(t, i, j, k, pt):
    top = (pt - t) * j.transpose()
    bottom = (pt - t)* k.transpose()
    return top, top/bottom


def drawPoints(ptList, t, orient, mode):

    i = np.matrix(orient[0])
    j = np.matrix(orient[1])
    k = np.matrix(orient[2])

    for x in range(len(ptList)):
        #print ('first point', ptList[x])
        pt = np.matrix(ptList[x])
        uO, uP = calU(t, i, j, k, pt)
        vO, vP = calV(t, i, j, k, pt)
        y = mode * 11 + x
        persPts[y][0] = uP.item(0)
        persPts[y][1] = vP.item(0)
        orthoPts[y][0] = uO.item(0)
        orthoPts[y][1] = vO.item(0)

drawPoints(Input.pts, p0 ,quatmat_0, 0)
drawPoints(Input.pts, p1 ,quatmat_1, 1)
drawPoints(Input.pts, p2 ,quatmat_2, 2)
drawPoints(Input.pts, p3 ,quatmat_3, 3)

#print (persPts)


print ('Part 2.2 ===============', '\n')
fig = plt.figure(1)
fig.suptitle('Perspective', fontsize=14, fontweight='bold')

for j in range (0, 4):
    i = 11*j
    ax = fig.add_subplot(2, 2, j+1)
    ax.set_title(j, fontsize=12)
    pers2 = [[persPts[4 + i][0], persPts[4 + i][1]],
             [persPts[5 + i][0], persPts[5 + i][1]],
             [persPts[6 + i][0], persPts[6 + i][1]],
             [persPts[7 + i][0], persPts[7 + i][1]]]
    polygon2 = plt.Polygon(pers2, edgecolor='blue',  fill = False)
    plt.gca().add_patch(polygon2)

    pers1 = [[persPts[0 + i][0], persPts[0 + i][1]],
            [persPts[1 + i][0], persPts[1 + i][1]],
            [persPts[2 + i][0], persPts[2 + i][1]],
            [persPts[3 + i][0], persPts[3 + i][1]]]
    polygon1 = plt.Polygon(pers1, edgecolor='red',  fill = False)
    plt.gca().add_patch(polygon1)

    pers3 = [[persPts[8 + i][0], persPts[8 + i][1]],
             [persPts[9 + i][0], persPts[9 + i][1]],
             [persPts[10+ i][0], persPts[10+ i][1]]]
    polygon3 = plt.Polygon(pers3, edgecolor='green',  fill = False)
    plt.gca().add_patch(polygon3)
    plt.axis([-0.5, 0.5, -0.5, 0.5])

fig = plt.figure(2)
fig.suptitle('Orthographic', fontsize=14, fontweight='bold')

for j in range (0, 4):
    i = 11*j
    ax = fig.add_subplot(2, 2, j+1)
    ax.set_title(j, fontsize=12)

    orth2 = [[orthoPts[4 + i][0], orthoPts[4 + i][1]],
             [orthoPts[5 + i][0], orthoPts[5 + i][1]],
             [orthoPts[6 + i][0], orthoPts[6 + i][1]],
             [orthoPts[7 + i][0], orthoPts[7 + i][1]]]
    polygon = plt.Polygon(orth2, edgecolor='blue',  fill = False)
    plt.gca().add_patch(polygon)

    orth1 = [[orthoPts[0 + i][0], orthoPts[0 + i][1]],
            [orthoPts[1 + i][0], orthoPts[1 + i][1]],
            [orthoPts[2 + i][0], orthoPts[2 + i][1]],
            [orthoPts[3 + i][0], orthoPts[3 + i][1]]]
    polygon = plt.Polygon(orth1, edgecolor='red',  fill = False)
    plt.gca().add_patch(polygon)

    orth3 = [[orthoPts[8 + i][0], orthoPts[8 + i][1]],
             [orthoPts[9 + i][0], orthoPts[9 + i][1]],
             [orthoPts[10+ i][0], orthoPts[10+ i][1]]]
    polygon = plt.Polygon(orth3, edgecolor='green',  fill = False)
    plt.gca().add_patch(polygon)
    plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.show()

print ('Part 3 ===============', '\n')

camPt0 = persPts[11*2 + 0]
camPt1 = persPts[11*2 + 1]
camPt2 = persPts[11*2 + 2]
camPt3 = persPts[11*2 + 3]
camPt8 = persPts[11*2 + 8]

patPt0 = Input.pts[0]
patPt1 = Input.pts[1]
patPt2 = Input.pts[2]
patPt3 = Input.pts[3]
patPt8 = Input.pts[8]

pat = np.matrix([patPt0, patPt1, patPt2, patPt3, patPt8])
cam = np.matrix([camPt0, camPt1, camPt2, camPt3, camPt8])

m = np.zeros([10, 9])
for i in range (10):
    index = int(i / 2)
    u_p = pat[index].item(0)
    v_p = pat[index].item(1)
    u_c = cam[index].item(0)
    v_c = cam[index].item(1)
    if (i%2 == 0):
        m[i][0] = u_p
        m[i][1] = v_p
        m[i][2] = 1
        m[i][6] = -u_c * u_p
        m[i][7] = -u_c * v_p
        m[i][8] = -u_c
    else :
        m[i][3] = u_p
        m[i][4] = v_p
        m[i][5] = 1
        m[i][6] = -v_c * u_p
        m[i][7] = -v_c * v_p
        m[i][8] = -v_c

m = np.matrix(m)
u,s,v = la.svd(m)
min_index = -1
minimun_eigen = 99999
# Find the index with zero as the eigenvalue
for i in range(9):
	if abs(s[i]) < minimun_eigen:
		minimun_eigen = abs(s[i])
		min_index = i


homography_array = np.array(v[i])[0]
homography = homography_array / homography_array[-1]

homography.resize([3, 3])
print ('homography')
print (homography)


