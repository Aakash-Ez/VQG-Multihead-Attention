import numpy as np
import cv2
import requests

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    cv2.imshow('img',img)
    if cv2.waitKey(10) == 32:
        # SPACE pressed
        img_name = "opencv_frame.jpg"
        cv2.imwrite(img_name, img)
        with open("opencv_frame.jpg", 'rb') as f:
            r = requests.post('http://127.0.0.1:5000/sergio', files={"fileupload": f})
            print(r.text)
    if cv2.waitKey(10) == 27:#Use ESC to close the webcam
        break
cap.release()
cv2.destroyAllWindows()