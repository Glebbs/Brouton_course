import cv2

img = cv2.imread('C:/Users/glebr/Desktop/pict_5.jpg', cv2.IMREAD_UNCHANGED) # load a picture

print('Original Dimensions : ', img.shape)

face_classifier = cv2.CascadeClassifier('C:/Users/glebr/Desktop/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
''' Our classifier returns the ROI of the detected face as a tuple,
It stores the top left coordinate and the bottom right coordiantes'''
faces = face_classifier.detectMultiScale(gray, 1.05, 7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
'''When no faces detected, face_classifier returns and empty tuple'''
if faces is ():
    print("No faces found")
'''We iterate through our faces array and draw a rectangle over each face in faces'''
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
