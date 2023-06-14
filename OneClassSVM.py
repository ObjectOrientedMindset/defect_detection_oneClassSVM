import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
from skimage.io import imread
from skimage.filters import prewitt
import matplotlib.pyplot as plt
from skimage import measure


def load_images(directory):
    image_size = (256, 256)
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            img = imread(os.path.join(directory, filename), as_gray=True) # Convert to grayscale
            resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            images.append(resized)
    return images

def extract_features(images):
    features = []
    for img in images:
        # Extracting Edge Features
        edges_prewitt = prewitt(img)
        features.append(edges_prewitt)
    return features

def train_one_class_classifier(features):
    clf = OneClassSVM(gamma='auto', nu=0.01)
    features_reshaped = np.array(features).reshape(len(features), -1)
    clf.fit(features_reshaped)
    return clf


def detect_defects(image, clf):
    # Extract features from the test image
    test_feature = prewitt(image) 
    

    # plt.imshow(test_feature, cmap='gray')

    # Reshape the feature to match the shape of training features
    test_feature_reshaped = test_feature.reshape(1, -1)

    # Predict the anomaly score for the test image
    anomaly_score = clf.decision_function(test_feature_reshaped)[0]

    print(f"Anomaly Score: {anomaly_score}")

    # Set a threshold to determine if the image is defective or not
    if anomaly_score < 0.002:
        return "flawless"
    elif anomaly_score > 0.002 and anomaly_score < 0.004:
        return "good"
    elif anomaly_score > 0.004 and anomaly_score < 0.006:
        return "average"
    elif anomaly_score > 0.006 and anomaly_score < 0.008:
        return "bad"
    elif anomaly_score > 0.008:
        return "critical"


# Set the paths to your dataset
train_directory = './train'
test_directory = './test'

# Load train dataset
train_images = load_images(train_directory)

# Extract features from the train images
train_features = extract_features(train_images)

# Train the one-class classifier
clf = train_one_class_classifier(train_features)


# Program run loop
while True:
    print('Enter image id: (0, 1, 2, 3...) \nEnter x to exit!')
    user_input = input()
    if user_input == 'x':
        break
    # Load test image
    image_id = int(user_input)
    image_path = os.path.join(test_directory, f'{image_id}.jpg')
    img = imread(image_path, as_gray=True) # Convert to grayscale
    test_image = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Print test image
    test_img = cv2.imread(image_path)
    
    # Detect defects
    result = detect_defects(test_image, clf)
    print(f"Defect Degree: {result}")
    cv2.imshow(f"Defect Degree: {result}",test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()










