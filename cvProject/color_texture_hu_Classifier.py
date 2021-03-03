#importing libraries
import h5py
import numpy as np
import os
import pickle
import glob
import cv2
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import mahotas
from sklearn.ensemble import AdaBoostClassifier
import requests

warnings.filterwarnings('ignore')
#--------------------
# tunable-parameters
#--------------------

num_trees = 100
test_size = 0.10
seed      = 9
train_path = "dataset/train"
test_path  = "dataset/test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"
bins = 8

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# creating the feature extraction functions

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
##################################################################

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()



# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                         test_size=test_size,
                                                                                          random_state=seed)
#X_train, X_test, Y_train, Y_test = train_test_split(np.array(global_features), np.array(global_labels), test_size=0.3, random_state=100)

# Split into train and test
X = trainDataGlobal
Y = trainLabelsGlobal


########################## MODEL ##########################################
#Assign to neigh the different models 
#neigh = KNeighborsClassifier(n_neighbors=3) 
#neigh = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), algorithm='SAMME', random_state=100)
#neigh = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, random_state=100)
#neigh = SVC()
#neigh = SVC(kernel='poly')
neigh = RandomForestClassifier(n_estimators=100)
clf = neigh.fit(trainDataGlobal, trainLabelsGlobal)
prediction = neigh.predict(testDataGlobal)
#print_metrics(trainDataGlobal, trainLabelsGlobal, testDataGlobal, testLabelsGlobal, prediction)

######################### Saving model #####################################
saved_model = pickle.dumps(clf)

######################### Loading model ###################################
knn_from_pickle = pickle.loads(saved_model)



global_features = []
#url for image
url = "http://192.168.1.3:8080/shot.jpg"

while True:
    # getting the image
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    # resizing the image
    img = cv2.resize(img, tuple((500, 500)))
    blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
    #hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # making image gray
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(blurred_frame)
    fv_haralick = fd_haralick(blurred_frame)
    fv_histogram = fd_histogram(blurred_frame)

    ###################################
    # Concatenate the Realtime Features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    global_features.append(global_feature)
    global_features = global_feature.reshape(-1, 1)

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_features)

    # predict label of test image
    prediction = knn_from_pickle.predict(rescaled_feature.reshape(1, -1))[0]

    print(train_labels[prediction])
    # show predicted label on image
    # cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    #Emptying features for the next image
    global_features = []


    # Building Color Mask
    low = np.array([100, 60, 60])
    high = np.array([180, 255, 255])


    #Combining masks into 1
    mask = cv2.inRange(img, low, high)


    #Finding Countours of the object based on the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #Drawing the rectangle based on the contours found
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 5000 < area < 500000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)


    #showing the image
    cv2.imshow('img', img)


    #the program will stop running after hitting "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#To test on the test folder

'''
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, tuple((500, 500)))

   
   #Real Time image Feature Extraction
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

   #Concatenate Real time global features
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    global_features.append(global_feature)
    global_features = global_feature.reshape(-1, 1)

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_features)

    # predict label of test image
    prediction = knn_from_pickle.predict(rescaled_feature.reshape(1,-1))[0]

    print (train_labels[prediction])

    # show predicted label on image
    #cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    global_features = []


'''















