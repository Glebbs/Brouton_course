from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

######################## MAKING EMBEDDINGS #####################

# proto = "C:/Users/glebr/Desktop/MobileNetSSD_deploy.prototxt"
# model = "C:/Users/glebr/Desktop/res10_300x300_ssd_iter_140000.caffemodel"
# detector = cv2.dnn.readNetFromCaffe(proto,model)
#
# embedder = cv2.dnn.readNetFromTorch('C:/Users/glebr/Desktop/openface.nn4.small2.v1.t7')
#
# imagePaths = list(paths.list_images('C:/Users/glebr/Desktop/train_set'))
#
# knownEmbeddings = []
# knownNames = []
# # initialize the total number of faces processed
# total = 0
# for (i, imagePath) in enumerate(imagePaths):
#     # extract the person name from the image path
#     print("[INFO] processing image {}/{}".format(i + 1,
#         len(imagePaths)))
#     name = imagePath.split(os.path.sep)[-2]
#     # load the image, resize it to have a width of 600 pixels (while
#     # maintaining the aspect ratio), and then grab the image
#     # dimensions
#     image = cv2.imread(imagePath)
#     image = imutils.resize(image, width=600)
#     (h, w) = image.shape[:2]
#     imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
#                                       (104.0, 177.0, 123.0), swapRB=False, crop=False)
#     # apply OpenCV's deep learning-based face detector to localize
#     # faces in the input image
#     detector.setInput(imageBlob)
#     detections = detector.forward()
#     if len(detections) > 0:
#         # we're making the assumption that each image has only ONE
#         # face, so find the bounding box with the largest probability
#         i = np.argmax(detections[0, 0, :, 2])
#         confidence = detections[0, 0, i, 2]
#         # ensure that the detection with the largest probability also
#         # means our minimum probability test (thus helping filter out
#         # weak detections)
#         if confidence > 0.5:
#             # compute the (x, y)-coordinates of the bounding box for
#             # the face
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             # extract the face ROI and grab the ROI dimensions
#             face = image[startY:endY, startX:endX]
#             (fH, fW) = face.shape[:2]
#             # ensure the face width and height are sufficiently large
#             if fW < 20 or fH < 20:
#                 continue
#                 # construct a blob for the face ROI, then pass the blob
#                 # through our face embedding model to obtain the 128-d
#                 # quantification of the face
#             faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
#                                              (96, 96), (0, 0, 0), swapRB=True, crop=False)
#             embedder.setInput(faceBlob)
#             vec = embedder.forward()
#             # add the name of the person + corresponding face
#             # embedding to their respective lists
#             knownNames.append(name)
#             knownEmbeddings.append(vec.flatten())
#             total += 1
# # dump the facial embeddings + names to disk
# print("[INFO] serializing {} encodings...".format(total))
# data = {"embeddings": knownEmbeddings, "names": knownNames}
# f = open('embeddings', "wb")
# f.write(pickle.dumps(data))
# f.close()

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

#################### TRAIN SVM #################

# data = pickle.loads(open('embeddings', "rb").read())
# # encode the labels
# print("[INFO] encoding labels...")
# le = LabelEncoder()
# labels = le.fit_transform(data["names"])
#
# print("[INFO] training model...")
# recognizer = SVC(C=1.0, kernel="linear", probability=True)
# recognizer.fit(data["embeddings"], labels)
#
# f = open("recognizer", "wb")
# f.write(pickle.dumps(recognizer))
# f.close()
# # write the label encoder to disk
# f = open("le", "wb")
# f.write(pickle.dumps(le))
# f.close()

import numpy as np
import imutils
import pickle
import cv2
import os

proto = "C:/Users/glebr/Desktop/MobileNetSSD_deploy.prototxt"
model = "C:/Users/glebr/Desktop/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(proto, model)
embedder = cv2.dnn.readNetFromTorch('C:/Users/glebr/Desktop/openface.nn4.small2.v1.t7')
recognizer = pickle.loads(open("recognizer", "rb").read())
le = pickle.loads(open("le", "rb").read())

correct = 0
amount_of_photos = 0

test_image_paths = list(paths.list_images('C:/Users/glebr/Desktop/test_set'))
for (i, ip) in enumerate(test_image_paths):
    image = cv2.imread(ip)
    answer = ip.split('\\')[-2]
    amount_of_photos += 1
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction

        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.8:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            if name.lower() == answer.lower():
                correct += 1
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

print(correct / amount_of_photos * 100)
