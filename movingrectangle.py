import cv2
import numpy as np
import copy

drawing = False # True if mouse is pressed
ix,iy = 1,1

def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,mode,tmp

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            tmp=np.copy(img)
            cv2.rectangle(tmp,(ix,iy),(x,y),(0,255,0),1)
           

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        
        

img = np.zeros((1000,1000,3), np.uint8)
tmp=np.copy(img)
cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing',draw_rectangle)

while True: 
    if drawing == True:
        cv2.imshow('my_drawing',tmp)
    else:
        cv2.imshow('my_drawing',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()