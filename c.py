import cv2
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

x = width//2
y = height//2

w = width//4
h = height//4


while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0,0,255),thickness= 4)
    image = frame[y:y+h,x:x+w]
    if(int(np.average(image))<50):
        cv2.putText(frame,"JUMP",(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,2,2)
        #pyautogui.press('space')
    elif (int(np.average(image))<80):
        cv2.putText(frame,"DOWN",(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,2,2)
        pyautogui.press('down')                                  
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()