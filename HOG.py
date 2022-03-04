# extracting image feature by HOG global descriptor
from skimage.feature import hog
from tqdm import tqdm
import numpy as np


def imgs2hogs(imgs):

    def img2hog(img):
        feature_vector = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), channel_axis=-1)
        return feature_vector     

    feature_vectors = []
    for img in tqdm(imgs):
        feature_vector = img2hog(img)
        feature_vectors.append(feature_vector)
    feature_vectors = np.asarray(feature_vectors)
    return feature_vectors