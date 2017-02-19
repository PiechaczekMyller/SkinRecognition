import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

image_path = "Images/1.jpg"
image_save_path = "12.png"
features = []
labels = []
data_path = 'Datasets/subset'
max_values = [235.0, 211.0, 223.0]

def LoadData(path, features, labels):
    index = 0
    for i in range(1, 11, 1):
        name = path + str(i) + '.txt'
        file = open(name, 'r')
        feature = []
        for line in file:
            if index == 3:
                labels.append(float(line))
                features.append(feature)
                feature = []
                index = 0
            else:
                feature.append(float(line))
                index = index + 1
    file.close()

def CreateDetectedImage(classifier, pixels, image_save_path, image_height, image_width):
    prob_pixels = []
    prob_pixel = []
    for pixel in pixels:
        pixel = np.array(pixel).reshape(1, -1)
        prediction = classifier.predict_proba(pixel)
        prob_pixel.append(255.0 - 255.0 * prediction[0, 1])
        prob_pixel.append(255.0 - 255 * prediction[0, 1])
        prob_pixel.append(255.0 - 255 * prediction[0, 1])
        prob_pixels.append(prob_pixel)
        prob_pixel = []

    prob_pixels = np.array(prob_pixels).reshape((image_height, image_width, 3))
    formatted = prob_pixels.astype('uint8')
    img = Image.fromarray(formatted)
    img.save(image_save_path)

im = Image.open(image_path)
data = list(im.getdata())
pixels = [(pixel[0]/max_values[0], pixel[1]/max_values[1], pixel[2]/max_values[2]) for pixel in data]
LoadData(data_path, features, labels)

features = np.array(features)
labels = np.array(labels)

classifier = LogisticRegression(max_iter=400, tol=0.000001, solver='liblinear')
classifier.fit(features, labels)

CreateDetectedImage(classifier, pixels, image_save_path, im.size[1], im.size[0])
