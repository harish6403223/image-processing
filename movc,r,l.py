import cv2
import numpy as np
import copy

drawing = False
ix,iy = 1,1
no=1

def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,mode,tmp,no

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            tmp=np.copy(img)
            if(no==1):
                cv2.rectangle(tmp,(ix,iy),(x,y),(0,255,0),1)
            elif(no==2):
                if abs(ix-x)>abs(iy-y):
                    l=abs(ix-x)
                else:
                    l=abs(iy-y)
                cv2.circle(tmp,(ix,iy),l,(0,255,0),1)
            else:
                cv2.line(tmp,(ix,iy),(x,y),(0,255,0),1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if(no==1):
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        elif(no==2):
            if abs(ix-x)>abs(iy-y):
            	l=abs(ix-x)
            else:
                l=abs(iy-y)
            cv2.circle(img,(ix,iy),l,(0,255,0),-1)
        else:
            cv2.line(img,(ix,iy),(x,y),(0,255,0),-1)
        
        

img = np.zeros((1000,1000,3), np.uint8)
tmp=np.copy(img)
cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing',draw_rectangle)

while True: 
    if drawing == True:
        cv2.imshow('my_drawing',tmp)
    else:
        cv2.imshow('my_drawing',img)
    k=cv2.waitKey(1) & 0xFF
    if(k==27):
        break
    elif(k==ord('c')):
        no=2
    elif(k==ord('l')):
        no=3
    elif(k==ord('r')):
        no=1

cv2.destroyAllWindows()