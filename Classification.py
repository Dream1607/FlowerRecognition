# -*- coding: utf-8 -*-
import cv2
import cPickle as pickle
import numpy as np
import os
import sys
import math
import datetime
from pylab import *
from scipy import signal
from sklearn import svm
from sklearn import datasets,metrics
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.cross_validation import train_test_split


def Color_Features_Extract(img_folder):
    print "Color_Features_Extract Start"
    starttime = datetime.datetime.now()

    back = np.array([255,128,128])

    image_num = len(os.listdir(seg_img_folder))

    Color_Features = []

    for index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)
        rows, columns, lab = image.shape

        # Make densely-sampling color features
        pixel_index = 0

        for x in range(rows):
            for y in range(columns):
                if pixel_index % 3 == 0 and np.array_equal(image[x][y],back) == False:
                    Color_Features.append(image[x][y].tolist())
                pixel_index += 1

    # Get CodeBook of Color_Features
    Color_Features = np.float32(Color_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Color_Features,800,None,criteria,10,flags)

    Image_Color_Features = [[0 for x in range(800)] for y in range(image_num)]

    color_index = 0

    for image_index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)
        rows, columns, lab = image.shape

        pixel_index = 0

        for x in range(rows):
            for y in range(columns):
                if pixel_index % 3 == 0 and np.array_equal(image[x][y],back) == False:
                    Image_Color_Features[image_index][labels[color_index]] += 1
                    color_index += 1
                pixel_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Color_Features_Extract End"
    print Image_Color_Features
    return Image_Color_Features

# Dense SIFT Features
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles

def Gen_DGauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y
    direction.
    '''
    fwid = np.int(2*np.ceil(sigma))
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor:
    '''
    The class that does dense sift feature extractor.
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        '''
        gridSpacing: the spacing for sampling dense descriptors
        patchSize: the size for each sift patch
        nrml_thres: low contrast normalization threshold
        sigma_edge: the standard deviation for the gaussian smoothing
            before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on
            Lowe's SIFT paper)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        #pyplot.imshow(self.weights)
        #pyplot.show()

    def process_image(self, image, positionNormalize = True,\
                       verbose = True):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. If you 
            pass a color image, it will automatically be converted
            to a grayscale image.
        positionNormalize: whether to normalize the positions
            to [0,1]. If False, the pixel-based positions of the
            top-right position of the patches is returned.
        
        Return values:
        feaArr: the feature array, each row is a feature
        positions: the positions of the features
        '''

        image = image.astype(np.double)

        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH/2
        offsetW = remW/2
        gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        if verbose:
            print 'Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
                    format(W,H,gS,pS,gridH.size)
        feaArr = self.calculate_sift_grid(image,gridH,gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features
        It is called by process_image().
        '''
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches,Nsamples*Nangles))

        # calculate gradient
        GH,GW = Gen_DGauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((Nangles,H,W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((Nangles,Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

def Multi_Scale_Dense_SIFT_Features_Extract(seg_img_folder):
    print "Multi_Scale_Dense_SIFT_Features_Extract Start"
    starttime = datetime.datetime.now()

    image_num = len(os.listdir(seg_img_folder))

    Image_KeyPoints_Num = [0 for y in range(image_num)]
    Multi_Scale_Dense_SIFT_Features = []

    for index, image_name in enumerate(os.listdir(seg_img_folder)):
        image_path = seg_img_folder + str("/") +image_name
        img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2GRAY)

        k = 0
        for scale in [4,6,8,10]:
            extractor = DsiftExtractor(gridSpacing = scale,patchSize = 5,sigma_edge = 6)
            feaArr,positions = extractor.process_image(img)

            Multi_Scale_Dense_SIFT_Features += feaArr.tolist()
            k += feaArr.shape[0]

        Image_KeyPoints_Num[index] = k

    # Get CodeBook of Multi_Scale_Dense_SIFT_Features
    Multi_Scale_Dense_SIFT_Features = np.float32(Multi_Scale_Dense_SIFT_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 2, 0.0001)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Multi_Scale_Dense_SIFT_Features,8000,None,criteria,2,flags)

    Image_Multi_Scale_Dense_SIFT_Features = [[0 for x in range(8000)] for y in range(image_num)]

    image_index = 0
    label_index = 0

    while image_index < image_num:
        Image_Multi_Scale_Dense_SIFT_Features[image_index][labels[label_index]] += 1
        label_index += 1

        if label_index == sum(Image_KeyPoints_Num[0:image_index + 1]):
            image_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Multi_Scale_Dense_SIFT_Features_Extract End"

    return Image_Multi_Scale_Dense_SIFT_Features


def Interest_Point_SIFT_Features_Extract(seg_img_folder):
    print "Interest_Point_SIFT_Features_Extract Start"
    starttime = datetime.datetime.now()

    image_num = len(os.listdir(seg_img_folder))

    Image_KeyPoints_Num = [0 for y in range(image_num)]
    Interest_Point_SIFT_Features = []

    for index, image_name in enumerate(os.listdir(seg_img_folder)):
        image_path = seg_img_folder + str("/") +image_name
        img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints,des = sift.detectAndCompute(img,None)

        k = 0

        for point in keypoints:
            Interest_Point_SIFT_Features.append(des[k])
            k += 1

        Image_KeyPoints_Num[index] = k

    # Get CodeBook of Interest_Point_SIFT_Features
    Interest_Point_SIFT_Features = np.float32(Interest_Point_SIFT_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Interest_Point_SIFT_Features,8000,None,criteria,10,flags)

    Image_Interest_Point_SIFT_Features = [[0 for x in range(8000)] for y in range(image_num)]

    image_index = 0
    label_index = 0

    while image_index < image_num:
    	Image_Interest_Point_SIFT_Features[image_index][labels[label_index]] += 1
        label_index += 1

        if label_index == sum(Image_KeyPoints_Num[0:image_index + 1]):
        	image_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Interest_Point_SIFT_Features_Extract End"

    return Image_Interest_Point_SIFT_Features

def Boundary_SIFT_Features_Extract(seg_img_folder):
    print "Boundary_SIFT_Features_Extract Start"
    starttime = datetime.datetime.now()

    image_num = len(os.listdir(seg_img_folder))

    Image_KeyPoints_Num = [0 for y in range(image_num)]
    Boundary_SIFT_Features = []

    for index, image_name in enumerate(os.listdir(seg_img_folder)):
        image_path = seg_img_folder + str("/") +image_name
        img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img,(3,3),0)
        canny = cv2.Canny(img, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        dilated = cv2.dilate(canny,kernel)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints,des = sift.detectAndCompute(img,dilated)

        Image_KeyPoints_Num[index] = len(keypoints)
        Boundary_SIFT_Features += des.tolist()

    # Get CodeBook of Boundary_SIFT_Features
    Boundary_SIFT_Features = np.float32(Boundary_SIFT_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Boundary_SIFT_Features,1000,None,criteria,10,flags)

    Image_Boundary_SIFT_Features = [[0 for x in range(1000)] for y in range(image_num)]

    image_index = 0
    label_index = 0

    while image_index < image_num:
        Image_Boundary_SIFT_Features[image_index][labels[label_index]] += 1
        label_index += 1

        if label_index == sum(Image_KeyPoints_Num[0:image_index + 1]):
            image_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Boundary_SIFT_Features_Extract End"

    return Image_Boundary_SIFT_Features

def Get_Features(seg_img_folder):
    Color_Features = Color_Features_Extract(seg_img_folder)
    Multi_Scale_Dense_SIFT_Features = Multi_Scale_Dense_SIFT_Features_Extract(seg_img_folder)
    Interest_Point_SIFT_Features = Interest_Point_SIFT_Features_Extract(seg_img_folder)
    Boundary_SIFT_Features = Boundary_SIFT_Features_Extract(seg_img_folder)

    Features = [ x + y + z + w for x,y,z,w in zip(Color_Features,Multi_Scale_Dense_SIFT_Features,Interest_Point_SIFT_Features,Boundary_SIFT_Features)]

    print np.array(Features).shape
    return Features

# main function
if __name__ == "__main__":
    seg_img_folder = 'seg_image'

    features_file = open('features.txt','w+')
    trans_features_file = open('trans_features.txt','w+')
    label_file = open('label.txt','r+')

    ### read label from label_file

    features = Get_Features(seg_img_folder)

    for img_feature in features:
        features_file.write(','.join(map(str,img_feature))+'\n')

    ### features.shape is (n_samples,n_features)
    ### trans_features.shape is (n_samples,3*n_features)
    transformer = AdditiveChi2Sampler(sample_steps=2,sample_interval=0.7)
    trans_features = transformer.fit_transform(features)
    # print trans_features
    for trans in trans_features:
        trans_features_file.write(','.join(map(str,trans))+'\n')


    rng = np.random.RandomState(42)
    train_data, test_data, train_label, test_label = train_test_split(trans_features, label, test_size=0.7, random_state=rng)

    clf = svm.LinearSVC()
    clf.fit(train_data,train_label)

    ### STILL NEED TO SEPARATE TRAIN AND TEST DATA ACCORDING TO CLASS

    # predict itself
    predicted = clf.predict(test_data)

    # report
    print "Classification report for classifier %s:\n%s\n" % (
    clf, metrics.classification_report(test_label, predicted))
    print "Confusion matrix:\n%s" % metrics.confusion_matrix(test_label, predicted)
