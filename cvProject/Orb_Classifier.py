import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import h5py
import mahotas
import matplotlib as plt
import requests
import glob
from skimage import feature, exposure
from scipy.stats import itemfreq
import numpy as np
import cv2
from matplotlib import pyplot as plt



bins = 8
numPoints = 8
radius= 2


#Orb feature detection function
def orb(image):
   orb = cv2.ORB_create()
   kp, desc = orb.detectAndCompute(image, None)
   return kp, desc

def load_images_from_path(path):
    image_list  = []
    for filename in os.listdir(path):
        name_list = os.listdir(path)
        name_list.sort()
        #load image and make it gray
        img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)
        image_list.append(img)
    return image_list

#Got the lists of images
apples = load_images_from_path("dataset/train/Apple/")
pepper = load_images_from_path("dataset/train/Peper/")
potato = load_images_from_path("dataset/train/Potato")


#Applying orb function to the images in the lists to get keypoints and desctiptors
for a in apples:
    kp1, desc1 = orb(a)
for p in pepper:
    kp2, desc2 = orb(p)
for p in potato:
    kp3, desc3 = orb(p)

#Accesing the url of the ip webcam app
url = "http://192.168.1.3:8080/shot.jpg"

while True:
    #getting the image
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)


    #resizing the image
    img = cv2.resize(img, (500, 500))

    #making image gray
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Applying orb function to the image
    kpimg, descimg = orb(grayimage)

    #Applying BruteForce matching function
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    #Trying to determine which of the classes has the most "good" matches.
    best = "nothing"
    if len(kpimg) > 0:
        matches1 = bf.match(desc1, descimg)
        matches2 = bf.match(desc2, descimg)
        matches3 = bf.match(desc3, descimg)
        good1 = []
        good2 = []
        good3 = []
        for m in matches1:
            if m.distance < 55:
                good1.append([m])
        for m in matches2:
            if m.distance < 55:
                good2.append([m])
        for m in matches3:
            if m.distance < 50:
                good3.append([m])

        #matches1 = sorted(matches1, key=lambda x: x.distance)
        #matches2 = sorted(matches2, key=lambda x: x.distance)
        #matches3 = sorted(matches3, key=lambda x: x.distance)


        matching_result = 0
        if len(good1) > matching_result:
            best = "Apple"
            matching_result = len(matches1)

        if len(good2) > matching_result:
            best = "Potato"
            matching_result = len(matches2)

        if len(good3) > matching_result:
            best = "Pepper"
            matching_result = len(matches3)

    #printing the class with the most good matches.
    print(best)



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



