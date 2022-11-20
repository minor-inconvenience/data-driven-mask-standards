
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import utils as pv


max_pixel_value = np.array([256, 256, 256])
origin = np.array([0, 0, 0])
def differenceRatioBetweenColours(rgb1, rgb2):
    diff_rgb = rgb1 - rgb2
    diff_dist = distance.euclidean(diff_rgb, origin)
    max_dist = distance.euclidean(max_pixel_value, origin)
    return diff_dist/max_dist


max_dist_value_for_neighbour = 0.03
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


pv.showPointCloud(pv.parsePtsFile("data/matthew.pts"))


#def constructDistanceColourMatrix(ptsData):
#    return distance.pdist(ptsData, metric=computeDistanceScaledByColourRatio)
    



