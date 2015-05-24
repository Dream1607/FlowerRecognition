import cv2

import numpy as np

from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

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

# main function
if __name__ == "__main__":
    img = 'image.jpg'
    
    segments,segments_pixels,segments_label = Label_Super_Pixels(Super_Pixels(img), Grab_Cut(img))

    draw(io.imread(img), segments_pixels)