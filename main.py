from HOG import imgs2hogs
from SIFT_BoVW import create_BoVW, imgs2BoVW
from cuml.svm import SVC
from keras.datasets import cifar10
from sklearn.metrics import classification_report
import numpy as np

def load_cifar10():
    clss = ['plane', 'car' , 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train.shape, y_train.shape, x_test.shape, y_test.shape
    return clss, (x_train, y_train), (x_test, y_test)

def hog_and_bovw(hogs, bovw):
    combine = np.concatenate((hogs, bovw), axis=1)
    return combine

if __name__=='__main__':

    # load cifar10
    clss, (x_train, y_train), (x_test, y_test) = load_cifar10()

    # HOG feature 
    hog_x_train = imgs2hogs(x_train)
    hog_x_test = imgs2hogs(x_test)
    print('Shape hog x_train and x_test = ', hog_x_train.shape, hog_x_test.shape)

    # SIFT_BoVW feature
    km = create_BoVW(x_train, n_visual_words=1024)
    print('Embedding train_imgs and test_imgs to BoVW')
    bovw_x_train = imgs2BoVW(km, x_train)
    bovw_x_test = imgs2BoVW(km, x_test)
    print('Shape SIFT_BoVW x_train and x_test = ', bovw_x_train.shape, bovw_x_test.shape)

    #Combine HOG feature and BoVW feature
    HaB_x_train = hog_and_bovw(hog_x_train, bovw_x_train)
    HaB_x_test  = hog_and_bovw(hog_x_test, bovw_x_test)    
    print('Shape combine HOG and BoVW x_train and x_test = ', HaB_x_train.shape, HaB_x_test.shape)

    # SVM Classifier
    classifier = SVC()
    classifier.fit(HaB_x_train, y_train)
    y_predict = classifier.predcit(HaB_x_test, y_test)

    # Evaluation
    print(classification_report(y_test, y_predict))
