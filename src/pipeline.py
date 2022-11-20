import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# this is the pca stuff and kmeans clustering stuff, it would be a joke to make a module including it so just writing what is needed in pipeline below:

# assuming that it is getting effectively a list of n-sized lists

temp_n = 7
temp_num_data = 20
temp_test_data = [np.random.rand(temp_n) for _ in range(temp_num_data)]

num_new_features = 0.90
num_clusters = 2

data_features = temp_test_data

# pca
pca = PCA(n_components=num_new_features)
transformed_features = pca.fit_transform(data_features)
print(transformed_features)

# k means clustering
fitted_cluster_estimator = KMeans(n_clusters=num_clusters).fit(transformed_features)
cluster_labeled_data = fitted_cluster_estimator.predict(transformed_features)

print(cluster_labeled_data)
