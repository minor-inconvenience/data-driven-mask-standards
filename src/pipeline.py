import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import itertools
import utils.pointVisualise as pv
import matplotlib.pyplot as mpl

# Read in data
dir = os.listdir("data/faces")
faces = []
for file in dir:
    if file[-3:] == "obj":
        read = pv.parseObjFile(f"data/faces/{file}")
        colour = False
        padding = [0, 0, 0]
    elif file[-3:] == "pts":
        read = pv.parsePtsFile(f"data/faces/{file}")
        colour = True
        padding = [0, 0, 0, 0, 0, 0]
    else:
        continue
    faces.append(list(read))

# I can't reshape this shit without breaking it bigtime so just remember it's faces[<pointNumber>,<faceNumber>,<xyz>]
# so to get all points from face 13, do faces[:,13,:]
faces = np.array(list(itertools.zip_longest(*faces, fillvalue=padding)))
# Example extraction of one face:
# pv.showPointCloud(faces[:, 0, :])

### Extract variables from face data
features = []
for i in range(faces.shape[1]):
    featureset = faces[:, i, :]  # get features
    features.append(featureset)

# Gather principal components to cover 95% of the variance, then discard
pca_coverage = 0.95
# pca
pca = PCA(n_components=pca_coverage)
transformed_features = pca.fit_transform(features)

# k means clustering
inertia = []
for i in range(1,11):
    fitted_cluster_estimator = KMeans(n_clusters=i).fit(transformed_features)
    cluster_labeled_data = fitted_cluster_estimator.predict(transformed_features)
    inertia.append(fitted_cluster_estimator.interia_)
mpl.plot(range(1, 11), inertia)
mpl.show()

cluster_means = fitted_cluster_estimator.cluster_centers_
cluster_means.reshape(pca.n_samples_, pca.n_components_)
retransformed = pca.inverse_transform(cluster_means)
print(retransformed)



