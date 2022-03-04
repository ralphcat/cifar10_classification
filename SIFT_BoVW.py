# Create Bag of visual word by SIFT
import cv2
from cuml.cluster import KMeans
import numpy as np
from tqdm import tqdm


def img2sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    key_points, feature_vectors = sift.detectAndCompute(img, None)
    return feature_vectors


def create_BoVW(imgs, n_visual_words):
    print('Create Bag of Visual Words by SIFT')
    # SIFT descriptors of all images
    all_feature_vectors = []
    for img in tqdm(imgs):
        feature_vectors = img2sift(img)
        if feature_vectors is not None:
            all_feature_vectors.extend(feature_vectors)

    all_feature_vectors = np.asarray(all_feature_vectors)
    print('The number of features is extracted from all images = ',all_feature_vectors.shape)
    # Clustering to create n_visual_words
    km = KMeans(n_clusters = n_visual_words)
    km.fit(all_feature_vectors)
    return km


def imgs2BoVW(km, imgs):

    def img2BoVW(km, img):
        n_visual_words = km.n_clusters
        vector = np.zeros(n_visual_words)
        feature_vectors = img2sift(img)
        if feature_vectors is not None:
            for feature_vector in feature_vectors:
                feature_vector = feature_vector.reshape(1, 128)
                id_cluster = km.predict(feature_vector)
                vector[id_cluster] += 1
        return vector

    vectors = []
    for img in tqdm(imgs):
        vector = img2BoVW(km, img)
        vectors.append(vector)
    vectors = np.asarray(vectors)
    print(vectors.shape)
    return vectors
