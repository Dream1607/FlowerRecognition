# -*- coding: utf-8 -*-
import cv2
import cPickle as pickle

import numpy as np

import os
import sys
import math
import datetime

from PIL import Image
from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.measure import block_reduce

from sklearn import svm
from sklearn import datasets,metrics


#################### 整合Grab_Cut里的画图方法
def draw(img,mask,save_name = False):
    rows,columns,rgb = img.shape

    for i in range(rows):
        for j in range(columns):
            img[i][j]*=mask[i][j]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if save_name == False:
        plt.show()
    else:
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=None, hspace=None)
        plt.savefig(save_name+'_result.jpg')

def Grab_Cut(img, segmentation_mask = None, save_name = False):
    height, weight, rgb = img.shape

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    if segmentation_mask == None:
        # mask initialized to PR_BG
        mask = np.zeros(img.shape[:2],np.uint8)

        # the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
        rect = (int(0.15 * weight),int(0.15 * height),int(0.7 * weight),int(0.7 * height))
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    else:
        mask = np.array(segmentation_mask,np.uint8) * 3
        rect = (50,50,weight - 150,height - 100)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    result = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    if segmentation_mask != None:
        draw(img,result,save_name = save_name)

    return result

def Super_Pixels(img_name):
    # load the image and convert it to a floating point data type
    image = img_as_float(io.imread(img_name))

	# apply SLIC and extract (approximately) the supplied number
    numSegments = 200
    
#################### slic参数待定

    # of segments
    segments = slic(image, n_segments = numSegments, sigma = 5)

    return segments

def Label_Super_Pixels(segments, grabcut):
    segments_num = max(max(row) for row in segments) + 1
    segments_cnt = np.zeros(segments_num)

    # count the majority of 0/1
    for seg, value in zip(np.array(segments).flatten(),grabcut.flatten()):
        if value==0:
            segments_cnt[seg]-=1
        else:
            segments_cnt[seg]+=1
    segments_label = [1 if cnt>0 else 0 for cnt in segments_cnt]

    rows,columns = np.array(segments).shape
    segments_pixels = [[0 for col in range(columns)] for row in range(rows)]
    for i in range(rows):
        for j in range(columns):
            segments_pixels[i][j] = segments_label[segments[i][j]]

    return segments,segments_pixels,segments_label

def SuperPixels_Segmentation_Adjust(features, label):
#################### svm参数待定

    # features are all the superpixels' features of the same class
    clf = svm.LinearSVC(loss='l1',C=10)
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

def Location_Shape(img,segments,segments_label):
    # 72-D Feature
    row,col = segments.shape
    location_block_row = int(math.ceil(row/6.))
    location_block_col = int(math.ceil(col/6.))

    Location_Shape_Features = []
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

        ### When the number is too small, there would be a bug
        ### Consider this special situation
        if cropped_row < 26:
            cropped_mask = cropped_mask[:(cropped_row-cropped_row%6),:]
        if cropped_col < 26:
            cropped_mask = cropped_mask[:,:(cropped_col-cropped_col%6)]

        cropped_row,cropped_col = cropped_mask.shape
        cropped_block_row = int(math.ceil(cropped_row/6.))
        cropped_block_col = int(math.ceil(cropped_col/6.))
        downsample = block_reduce(cropped_mask, block_size=(cropped_block_row, cropped_block_col), cval = 0, func=np.max)

        # Convert to 36-D Shape Features
        Shape_Features = downsample.flatten().tolist()

        Location_Shape_Features.append(Location_Features+Shape_Features)

    return Location_Shape_Features

def Class_Location_Shape_CB_Features_Extract(img_folder):
    print "Class_Location_Shape_CB_Features_Extract Start"
    starttime = datetime.datetime.now()

    Class_Location_Shape_CB_Features = []
    Labels = []
    ### IMAGES SHOULD BE READ IN ORDER!!!
    for index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") + image_name
        img = cv2.imread(image_path)

        ### GrabCut & SLIC
        segments,segments_pixels,segments_label = Label_Super_Pixels(Super_Pixels(image_path),Grab_Cut(img))
        Labels+=segments_label

        ### Get Single Image's Superpixels Location, Shape, Center & Boundary Features
        Center_Boundary_Features = Center_Boundary(segments,segments_label)
        Location_Shape_Features = Location_Shape(img,segments,segments_label)

        for i in range(len(segments_label)):
            Features = Center_Boundary_Features[i] + Location_Shape_Features[i]
            Class_Location_Shape_CB_Features.append(map(int,Features))

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Class_Location_Shape_CB_Features_Extract End"

    return Class_Location_Shape_CB_Features, Labels

def Class_Size_Features_Extract(img_folder):
    print "Class_Size_Features_Extract Start"
    starttime = datetime.datetime.now()

    Class_Superpixels_Num = [0 for x in range(len(os.listdir(img_folder)))]
    Class_Size_Features = []

    for index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name

        segments = Super_Pixels(image_path)

        Class_Size_Features.append(np.histogram([y for sublist in segments for y in sublist], bins = max(max(row) for row in segments) + 1)[0].tolist())

        Class_Superpixels_Num[index] = max(max(row) for row in segments) + 1

    # Format        
    Class_Size_Features = [[x] for x in [y for sublist in Class_Size_Features for y in sublist]]

    # Get CodeBook of Class_Color_Features
    Class_Size_Features = np.float32(Class_Size_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

#################### kmeans参数待定，下同

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Class_Size_Features,2,None,criteria,10,flags)

    Superpixel_Size_Features = labels.tolist()

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Class_Size_Features_Extract End"

    return Superpixel_Size_Features

def Class_Color_Features_Extract(img_folder):
    print "Class_Color_Features_Extract Start"
    starttime = datetime.datetime.now()

    Class_Superpixels_Num = [0 for x in range(len(os.listdir(img_folder)))]
    Class_Pixels_Num = [0 for x in range(len(os.listdir(img_folder)))]
    Class_Color_Features = []

    for index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name

        image = cv2.imread(image_path)
        rows, columns, bgr = image.shape

        # Make densely-sampling color features
        pixel_index = 0
        densely_sampling_pixels_number = 0

        for x in range(rows):
            for y in range(columns):
                if pixel_index % 6 == 0:
                    Class_Color_Features.append(image[x][y].tolist())
                    densely_sampling_pixels_number += 1
                pixel_index += 1

        Class_Pixels_Num[index] = densely_sampling_pixels_number
        Class_Superpixels_Num[index] = max(max(row) for row in Super_Pixels(image_path)) + 1
    
    # Get CodeBook of Class_Color_Features
    Class_Color_Features = np.float32(Class_Color_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Class_Color_Features,200,None,criteria,10,flags)

    Superpixel_Color_Features = [[0 for x in range(200)] for y in range(sum(Class_Superpixels_Num))]

    for image_index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name
        img =  cv2.imread(image_path)

        segments,segments_pixels,segments_label = Label_Super_Pixels(Super_Pixels(image_path),Grab_Cut(img))
        rows, columns = np.array(segments).shape
        superpixels_num = sum(Class_Superpixels_Num[0:image_index])
        pixels_num = sum(Class_Pixels_Num[0:image_index])
        
        pixel_index = 0
        densely_sampling_pixels_number = 0

        for x in range(rows):
            for y in range(columns):
                if pixel_index % 6 == 0:
                    Superpixel_Color_Features[segments[x][y] + superpixels_num][labels[densely_sampling_pixels_number + pixels_num]] += 1
                    densely_sampling_pixels_number += 1
                pixel_index += 1

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Class_Color_Features_Extract End"

    return Superpixel_Color_Features

def Class_SIFT_Features_Extract(img_folder):
    print "Class_SIFT_Features_Extract Start"
    starttime = datetime.datetime.now()

    Class_Superpixels_Num = [0 for x in range(len(os.listdir(img_folder)))]
    Class_SIFT_Points = []
    Class_SIFT_Features = []

    for index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name
        img =  cv2.imread(image_path)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints,des = sift.detectAndCompute(img,None)

        k = 0

        for point in keypoints:
            Class_SIFT_Points += [point.pt]
            Class_SIFT_Features.append(des[k])
            k += 1

        Class_Superpixels_Num[index] = max(max(row) for row in Super_Pixels(image_path)) + 1

    # Get CodeBook of Class_SIFT_Features
    Class_SIFT_Features = np.float32(Class_SIFT_Features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(Class_SIFT_Features,800,None,criteria,10,flags)

    Superpixel_SIFT_Features = [[0 for x in range(800)] for y in range(sum(Class_Superpixels_Num))]

    for image_index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") +image_name
        img =  cv2.imread(image_path)

        segments,segments_pixels,segments_label = Label_Super_Pixels(Super_Pixels(image_path),Grab_Cut(img))
        rows, columns = np.array(segments).shape
        num = sum(Class_Superpixels_Num[0:image_index])
        
        for index, input_vector in enumerate(Class_SIFT_Points):
            x = int(round(input_vector[1])) if int(round(input_vector[1])) < rows else rows - 1
            y = int(round(input_vector[0])) if int(round(input_vector[0])) < columns else columns - 1

            Superpixel_SIFT_Features[segments[x][y] + num][labels[index]] += 1
        print "image_" + str(image_index)

    endtime = datetime.datetime.now()
    print "Time: " + str((endtime - starttime).seconds) + "s"
    print "Class_SIFT_Features_Extract End"

    return Superpixel_SIFT_Features

def Get_Group_Features(img_folder):
    # Given a path of folder, return all the super pixels' features(1076-D) List

    Class_Location_Shape_CB_Features, Labels = Class_Location_Shape_CB_Features_Extract(img_folder)

    Class_Size_Features = Class_Size_Features_Extract(img_folder)

    Class_Color_Features = Class_Color_Features_Extract(img_folder)

    Class_SIFT_Features = Class_SIFT_Features_Extract(img_folder)

    ### Merge Features
    Superpixel_Features = [ x + y + z + w for x,y,z,w in zip(Class_Location_Shape_CB_Features,Class_Size_Features,Class_Color_Features,Class_SIFT_Features)]

    # print np.array(Superpixel_Features)
    # print np.array(Superpixel_Features).shape
    ### Save to database
    return Superpixel_Features,Labels

# main function
if __name__ == "__main__":

    img_folder = 'image'

    Superpixel_Features, Labels = Get_Group_Features(img_folder)
    predicted = SuperPixels_Segmentation_Adjust(Superpixel_Features, Labels)

    offset = 0

    for image_index, image_name in enumerate(os.listdir(img_folder)):
        image_path = img_folder + str("/") + image_name
        img =  cv2.imread(image_path)
        segments,segments_pixels,segments_label = Label_Super_Pixels(Super_Pixels(image_path),Grab_Cut(img))
        mask = []
        for seg in segments.flatten():
            mask.append(predicted[offset + seg])
        mask = np.array(mask).reshape(segments.shape).tolist()

        Grab_Cut(img,mask,save_name = "seg_image/" + image_name.split('.')[0] + "_final_")

        offset += len(segments_label)