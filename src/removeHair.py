import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import utils.pointVisualise as pv


max_pixel_value = np.array([256, 256, 256])
origin = np.array([0, 0, 0])
def differenceRatioBetweenColours(rgb1, rgb2):
    diff_rgb = rgb1 - rgb2
    diff_dist = distance.euclidean(diff_rgb, origin)
    max_dist = distance.euclidean(max_pixel_value, origin)
    return diff_dist/max_dist


max_dist_value_for_neighbour = 0.1
def computeDistanceScaledByColourRatio(dp1, dp2):
    xyz1 = dp1[:3]
    rgb1 = dp1[3:]
    xyz2 = dp2[:3]
    rgb2 = dp2[3:]
    spatial_dist = distance.euclidean(xyz1, xyz2)
    col_dist = differenceRatioBetweenColours(rgb1, rgb2)
    if spatial_dist > max_dist_value_for_neighbour:
        return 100
    else:
        return ((col_dist**2) * spatial_dist)


#pv.showPointCloud(pv.parsePtsFile("data/matthew.pts"), plotColour=True)



#np_obj = pv.parseObjFile("data/matthew.obj")
np_pts = pv.parsePtsFile("data/matthew.pts")

#np_obj = np_obj[np_obj[:, 0].argsort()]
#np_pts = np_pts[np_pts[:, 0].argsort()]


#a = np.array([[1, 234, 534, 123, 1345, 123, 2], [0, 234, 534, 123, 1345, 123, 2]])
#a = a[a[:, 1].argsort()]
#print(a.shape)

#for i in range(20):
#    print(np_obj[i, :])
#    print(np_pts[i, :])




def constructDistanceColourMatrix(ptsData):
    return distance.pdist(ptsData, metric=computeDistanceScaledByColourRatio)
    

def addClusterIndexToNPArrayPTS(ptsData):
    print(ptsData.shape)
    ptsData = ptsData[::1000]
    print(ptsData.shape)
    clustered_indexes = DBSCAN(eps=0.000065, metric=computeDistanceScaledByColourRatio).fit_predict(ptsData)
    print(clustered_indexes)
    return np.concatenate((ptsData, clustered_indexes), axis = 1)



clustered_face = addClusterIndexToNPArrayPTS(np_pts)
