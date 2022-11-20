import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance, KDTree
from statistics import mode
import utils.pointVisualise as pv



max_pixel_value = np.array([256, 256, 256])
pale_skin_colour = np.array([232, 190, 172])
origin = np.array([0, 0, 0])
def differenceRatioBetweenColours(rgb1, rgb2):
    return distance.euclidean(np.cross((pale_skin_colour/np.linalg.norm(pale_skin_colour)), ((rgb1/(np.linalg.norm(rgb1)+0.0001)) - (rgb2/(np.linalg.norm(rgb2)+0.0001)))), origin)


max_dist_value_for_neighbour = 0.095
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
        return ((col_dist) * spatial_dist)


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
    clustered_indexes = DBSCAN(eps=0.0006, min_samples=5, metric=computeDistanceScaledByColourRatio).fit_predict(ptsData)
    print(clustered_indexes)
    clustered_indexes = clustered_indexes.reshape((len(clustered_indexes), 1))
    return np.concatenate((ptsData, clustered_indexes), axis = 1)

def upscaleImage(clustered_face, ptsData):
    search_radius = 0.1
    kdTree = KDTree(clustered_face[:, :3])
    nearest_neighbours_list = kdTree.query_ball_point(ptsData[:, :3], search_radius)
    ptsData = np.concatenate(ptsData, np.zeros(ptsData.shape[0]), axis=0)
    print(ptsData.shape)
    unlabeledPtsData = np.array([])
    labeledPtsData = np.array([])
    for i in range(len(nearest_neighbours_list)):
        if nearest_neighbours_list[i] is None:
            unlabeledPtsData.append(ptsData[i])
            continue
        ptsData[i, -1] = checkNeighbourCluster(nearest_neighbours_list[i]) + 1
        labeledPtsData.append(ptsData[i])
    clustered_point_cloud = np.concatenate(clustered_face, labeledPtsData)
    print("completed first sweep")

    kdTree = KDTree(labeledPtsData[:, :3])
    nearest_neighbours_list = kdTree.query_ball_point(unlabeledPtsData[:, :3], search_radius)
    newLabeledData = np.array([])
    for i in range(len(unlabeledPtsData)):
        if nearest_neighbours_list[i] is None:
            continue
        unlabeledPtsData[i:, -1] = checkNeighbourCluster(nearest_neighbours_list[i]) + 1
        clustered_point_cloud.append(unlabeledPtsData[i])
    print("sweep 2 finished")
    return clustered_point_cloud



def checkNeighbourCluster(neihgbourIndexes):
    cluster_list = []
    for n_i in neihgbourIndexes:
        cluster_list.append(clustered_face[n_i, -1])
    return mode(cluster_list)


clustered_face = addClusterIndexToNPArrayPTS(np_pts)
print("clustered!")
clustered_point_cloud = upscaleImage(clustered_face, np_pts)
print(clustered_point_cloud.shape)
pv.showPointCloud(clustered_point_cloud, plotColourForMyFunction=True, resolution=1)
