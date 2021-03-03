import requests
import cv2
import numpy as np

def detection():
    #getting the image
    url = "http://192.168.1.3:8080/shot.jpg"

    while True:

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        #blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
        #hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Assigning low and high HSV values to the mask
        low = np.array([100, 60, 60])
        high = np.array([180, 255, 255])
        

        #combing low and high
        mask = cv2.inRange(img, low, high)
        #res = cv2.bitwise_and(img, img, mask=mask)

        #Finding Contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #Drawing rectangle based on Contours and area
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if 5000 < area < 500000:
                cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)

        #Showing the image
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



















