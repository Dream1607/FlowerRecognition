import cv2
import cPickle as pickle

import numpy as np

import math

from PIL import Image
from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.measure import block_reduce

from sklearn import svm
from sklearn import datasets,metrics

def draw(img,mask):
    rows,columns,rgb = img.shape

    for i in range(rows):
        for j in range(columns):
            img[i][j]*=mask[i][j]
    plt.imshow(img),plt.show()

def Grab_Cut(img):
    # Loading image
    img = cv2.imread(img)
    height, weight, rgb = img.shape

    # mask initialized to PR_BG
    mask = np.zeros(img.shape[:2],np.uint8)

    # the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
    rect = (50,50,weight - 150,height - 100)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    return mask2

def Super_Pixels(img):
    # load the image and convert it to a floating point data type
    image = img_as_float(io.imread(img))

	# apply SLIC and extract (approximately) the supplied number
    numSegments = 100
    
    # of segments
    segments = slic(image, n_segments = numSegments, sigma = 5)

    return segments

def Label_Super_Pixels(segments, grabcut):
    segments_num = max(max(row) for row in segments) + 1
    segments_cnt = np.zeros(segments_num)

    # count the majority of 0/1
    for seg, value in zip(np.array(segments).flatten(),grabcut.flatten()):
        if value==0:
            segments_cnt[seg-1]-=1
        else:
            segments_cnt[seg-1]+=1
    segments_label = [1 if cnt>0 else 0 for cnt in segments_cnt]

    rows,columns = np.array(segments).shape
    segments_pixels = [[0 for col in range(columns)] for row in range(rows)]
    for i in range(rows):
        for j in range(columns):
            segments_pixels[i][j] = segments_label[segments[i][j]-1]

    return segments,segments_pixels,segments_label

def SuperPixels_Segmentation_Adjust(features, label):
    # features are all the superpixels' features of the same class
    clf = svm.LinearSVC(C=10, loss='hinge')
    clf.fit(features,label)

    # predict itself
    predicted = clf.predict(features)

    # report
    print "Classification report for classifier %s:\n%s\n" % (
    clf, metrics.classification_report(label, predicted))
    print "Confusion matrix:\n%s" % metrics.confusion_matrix(label, predicted)

    return predicted

### Superpixels Features Extraction

def Center_Boundary(segments,segments_label):
    row,col = segments.shape
    Center_Boundary_Features = np.zeros((len(segments_label),2))

    # Check Center
    Center_Boundary_Features[segments[int(row/2)][int(col/2)]][0] = 1

    # Check Boundary
    for i in range(row):
        Center_Boundary_Features[segments[i][0]][1] = 1
        Center_Boundary_Features[segments[i][col-1]][1] = 1
    for i in range(col):
        Center_Boundary_Features[segments[0][i]][1] = 1
        Center_Boundary_Features[segments[row-1][i]][1] = 1

    return Center_Boundary_Features.tolist()

def Segment_Mask(segments,label):
    # Make mask for each segment
    # mark 1 for particular label of segments
    # mark 0 for other pixels

    mask = [1 if i^label==0 else 0 for i in segments.flatten()]
    return np.array(mask, np.uint8).reshape(segments.shape)

def Location_Shape_Color(img,segments,segments_label):
    # 72-D Feature
    row,col = segments.shape
    location_block_row = int(math.ceil(row/6.))
    location_block_col = int(math.ceil(col/6.))

    Location_Shape_Color_Features = []
    for label in range(len(segments_label)):
        # Make mask for each segment
        seg_mask = Segment_Mask(segments, label)

        ### Get Location Features
        # Downsample to 6*6
        downsample = block_reduce(seg_mask, block_size=(location_block_row, location_block_col), cval = 0, func=np.max)

        # Convert to 36-D Location Features
        Location_Features = downsample.flatten().tolist()

        ### Get Shape Features
        # Bounding Box
        left,up,right,down = Image.fromarray(np.uint8(seg_mask)).getbbox()

        # Cropped the mask
        cropped_mask =  seg_mask[up:down,left:right]

        # Downsample to 6*6
        cropped_row,cropped_col = cropped_mask.shape
        cropped_block_row = int(math.ceil(cropped_row/6.))
        cropped_block_col = int(math.ceil(cropped_col/6.))
        downsample = block_reduce(cropped_mask, block_size=(cropped_block_row, cropped_block_col), cval = 0, func=np.max)


        # Convert to 36-D Shape Features
        Shape_Features = downsample.flatten().tolist()

        ### Get Color Hist Features
        Color_Hist_Features = []
        hist_mask_b = cv2.calcHist([img],[0],seg_mask,[64],[0,256])
        hist_mask_g = cv2.calcHist([img],[1],seg_mask,[64],[0,256])
        hist_mask_r = cv2.calcHist([img],[2],seg_mask,[64],[0,256])

        Color_Hist_Features.append(hist_mask_b)
        Color_Hist_Features.append(hist_mask_g)
        Color_Hist_Features.append(hist_mask_r)
        Color_Hist_Features = np.array(Color_Hist_Features).flatten().tolist()
        Location_Shape_Color_Features.append(Location_Features+Shape_Features+Color_Hist_Features)

    return Location_Shape_Color_Features

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []

    for point in keypoints:
        temp_array.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i]))
        i = i + 1

    return np.array(temp_array)

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []

    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)

    return keypoints, np.array(descriptors)

def SIFT_extract(img):
    sift = cv2.xfeatures2d.SIFT_create()

    kp,des = sift.detectAndCompute(img,None)
    temp = pickle_keypoints(kp, des)

    # img = cv2.drawKeypoints(img,kp,img)
    # plt.imshow(img),plt.show()
    return temp

def Superpixel_SIFT(image_SIFT,segments,segments_label):
    SIFT_Features = [[] for y in range(len(segments_label))]
    for index, input_vector in enumerate(image_SIFT):
        x = int(round(input_vector[0][1]))
        y = int(round(input_vector[0][0]))
        SIFT_Features[segments[x][y]].append(input_vector[6])

    for index, vectors in enumerate(SIFT_Features):
        len_v = 128
        mean_vector = np.zeros(len_v)
        if len(vectors) != 0:
            for i in range(len(vectors)):
                mean_vector += vectors[i]
            SIFT_Features[index] = (mean_vector/(len(vectors))).tolist()
        else:
            SIFT_Features[index] = mean_vector.tolist()

    return SIFT_Features


# main function
if __name__ == "__main__":
    img = 'image.jpg'

    img_cv2 = cv2.imread(img)

    segments,segments_pixels,segments_label = Label_Super_Pixels(Super_Pixels(img),Grab_Cut(img))

    Center_Boundary_Features = Center_Boundary(segments,segments_label)
    Location_Shape_Color_Features = Location_Shape_Color(img_cv2,segments,segments_label)
    SIFT_Features = Superpixel_SIFT(SIFT_extract(img_cv2),segments,segments_label)

    Superpixel_Features = []
    for i in range(len(segments_label)):
        Features = Center_Boundary_Features[i] + Location_Shape_Color_Features[i] + SIFT_Features[i]
        Superpixel_Features.append(map(int,Features))

    print Superpixel_Features