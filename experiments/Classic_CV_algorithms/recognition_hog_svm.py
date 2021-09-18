from sklearn.svm import LinearSVC
from skimage import feature
import os
import cv2
from imutils import paths

images = []
labels = []

image_paths = list(paths.list_images('C:/Users/glebr/Desktop/train_set'))

for (i, ip) in enumerate(image_paths):
    image = cv2.imread(ip)
    image = cv2.resize(image, (128, 256))
    # get the HOG descriptor for the image
    hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

    # update the data and labels
    name = ip.split(os.path.sep)[-2]
    images.append(hog_desc)
    labels.append(name)

svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)

test_image_paths = list(paths.list_images('C:/Users/glebr/Desktop/test_set'))
for (i, ip) in enumerate(test_image_paths):
    image = cv2.imread(ip)
    resized_image = cv2.resize(image, (128, 256))
    # get the HOG descriptor for the test image
    (hog_desc, hog_image) = feature.hog(resized_image, orientations=9,
                                        pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2),
                                        transform_sqrt=True,
                                        block_norm='L2-Hys',
                                        visualize=True)
    # prediction
    pred = svm_model.predict(hog_desc.reshape(1, -1))[0]
    # convert the HOG image to appropriate data type. We do...
    # ... this instead of rescaling the pixels from 0. to 255.
    hog_image = hog_image.astype('float64')
    # show thw HOG image
    cv2.imshow('HOG Image', hog_image)
    # put the predicted text on the test image
    cv2.putText(image, pred.title(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)
    cv2.imshow('Test Image', image)
    # cv2.imwrite(f"outputs/{args['path']}_hog_{i}.jpg",
    #             hog_image * 255.)  # multiply by 255. to bring to OpenCV pixel range
    # cv2.imwrite(f"outputs/{args['path']}_pred_{i}.jpg", image)
    cv2.waitKey(0)










