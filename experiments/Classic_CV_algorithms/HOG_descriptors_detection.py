from utils.utils import convert_and_trim_bb
import imutils
import time
import dlib
import cv2

################################## HOG + SVM ############################

# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
image = cv2.imread('C:/Users/glebr/Desktop/pict_5.jpg')
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# perform face detection using dlib's face detector
start = time.time()
print("[INFO[ performing face detection with dlib...")
rects = detector(rgb, 1)
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))

# convert the resulting dlib rectangle objects to bounding boxes,
# then ensure the bounding boxes are all within the bounds of the
# input image
boxes = [convert_and_trim_bb(image, r) for r in rects]
# loop over the bounding boxes
for (x, y, w, h) in boxes:
    # draw the bounding box on our image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

######################################### SIFT ####################################

import numpy as np
import cv2 as cv
# img = cv.imread('C:/Users/glebr/Desktop/pict_5.jpg')
# gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# sift = cv.SIFT_create()
# kp = sift.detect(gray,None)
# img=cv.drawKeypoints(gray,kp,img)
# cv.imwrite('sift_keypoints.jpg',img)
#
# img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints.jpg',img)

# sift = cv.SIFT_create()
# kp, des = sift.detectAndCompute(gray,None)

