from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

######################## MAKING EMBEDDINGS #####################
proto = "C:/Users/glebr/PycharmProjects/Brouton_course/experiments/NN_for_face_recognition/assets/MobileNetSSD_deploy.prototxt"
model = "C:/Users/glebr/PycharmProjects/Brouton_course/experiments/NN_for_face_recognition/assets/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(proto, model)

embedder = cv2.dnn.readNetFromTorch(
    'C:/Users/glebr/PycharmProjects/Brouton_course/experiments/NN_for_face_recognition/assets/openface.nn4.small2.v1.t7')

imagePaths = list(paths.list_images('C:/Users/glebr/Desktop/train_set/'))
print(imagePaths)
knownEmbeddings = []
knownNames = []
# q = cv2.imread("C:/Users/glebr/Desktop/train_set/Keanu_Reeves/15.jpg")
# cv2.imshow("qq", q)
# cv2.waitKey(0)
# initialize the total number of faces processed
total = 0
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    print(imagePath)
    name = imagePath.split('/')[-1].split('\\')[0]
    #print(name)
    #print(imagePath)
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)
    # cv2.imshow("img",image)
    # cv2.waitKey()
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # add the name of the person + corresponding face
            # embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1
# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open('embeddings', "wb")
f.write(pickle.dumps(data))
f.close()
