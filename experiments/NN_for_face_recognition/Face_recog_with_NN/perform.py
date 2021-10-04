import numpy as np
import imutils
import pickle
import cv2

from imutils import paths

proto = "C:/Users/glebr/PycharmProjects/Brouton_course/experiments/NN_for_face_recognition/assets/MobileNetSSD_deploy.prototxt"
model = "C:/Users/glebr/PycharmProjects/Brouton_course/experiments/NN_for_face_recognition/assets/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(proto, model)
embedder = cv2.dnn.readNetFromTorch(
    'C:/Users/glebr/PycharmProjects/Brouton_course/experiments/NN_for_face_recognition/assets/openface.nn4.small2.v1.t7')
recognizer = pickle.loads(open("recognizer", "rb").read())
le = pickle.loads(open("le", "rb").read())

correct = 0
amount_of_photos = 0

def check_one_image(path):
    image = cv2.imread(path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return name
#
# test_image_paths = list(paths.list_images('C:/Users/glebr/Desktop/test_set'))
# for (i, ip) in enumerate(test_image_paths):
#     image = cv2.imread(ip)
#     answer = ip.split('\\')[-2]
#     amount_of_photos += 1
#     image = imutils.resize(image, width=600)
#     (h, w) = image.shape[:2]
#     # construct a blob from the image
#     imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
#                                       (104.0, 177.0, 123.0), swapRB=False, crop=False)
#     # apply OpenCV's deep learning-based face detector to localize
#     # faces in the input image
#     detector.setInput(imageBlob)
#     detections = detector.forward()
#
#     for i in range(0, detections.shape[2]):
#         # extract the confidence (i.e., probability) associated with the
#         # prediction
#
#         confidence = detections[0, 0, i, 2]
#         # filter out weak detections
#         if confidence > 0.8:
#             # compute the (x, y)-coordinates of the bounding box for the
#             # face
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             # extract the face ROI
#             face = image[startY:endY, startX:endX]
#             (fH, fW) = face.shape[:2]
#             # ensure the face width and height are sufficiently large
#             if fW < 20 or fH < 20:
#                 continue
#             faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
#             embedder.setInput(faceBlob)
#             vec = embedder.forward()
#             # perform classification to recognize the face
#             preds = recognizer.predict_proba(vec)[0]
#             j = np.argmax(preds)
#             proba = preds[j]
#             name = le.classes_[j]
#             if name.lower() == answer.lower():
#                 correct += 1
#             text = "{}: {:.2f}%".format(name, proba * 100)
#             y = startY - 10 if startY - 10 > 10 else startY + 10
#             cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#             cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#     # show the output image
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)

#print(correct / amount_of_photos * 100)
