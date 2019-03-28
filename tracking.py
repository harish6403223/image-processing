import cv2,copy
import numpy as np
import pyautogui,os
# Parameters for ShiTomasi corner detection (good features to track paper)
corner_track_params = dict(maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (200,200),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))
#os.chdir('C://Users//P. Harish Kumar//Desktop//Python//facedetect')

# Capture the video
cap = cv2.VideoCapture(0)

#frame1 = pyautogui.screenshot()
#frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
#height, width, channels = frame1.shape
#writer = cv2.VideoWriter('dinotracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'),20.0, (width, height))

# Grab the very first frame of the stream


face_cascade = cv2.CascadeClassifier('C://Users//P. Harish Kumar//Desktop//Python//facedetect//haarcascade_frontalface_default.xml')
x=0
while(x==0):
    ret, prev_frame = cap.read()
    face_rects = face_cascade.detectMultiScale(prev_frame,scaleFactor=1.2, minNeighbors=10)
    for (x,y,w,h) in face_rects:
        print(x,y,w,h)
s=0
s1=0
# Grab a grayscale image (We will refer to this as the previous frame)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Grabbing the corners
prevPts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **corner_track_params)
sl=copy.deepcopy(prevPts)
points=[(new.ravel())[0] for i,(new) in enumerate(prevPts) if(new.ravel())[0]>x and (new.ravel())[0]<(x+w)]

# Create a matching mask of the previous frame for drawing on later
mask = np.zeros_like(prev_frame)


while True:
    
    # Grab current frame
    ret,frame = cap.read()
    
    # Grab gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Optical Flow on the Gray Scale Frame
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
    
    # Using the returned status array (the status output)
    # status output status vector (of unsigned chars); each element of the vector is set to 1 if
    # the flow for the corresponding features has been found, otherwise, it is set to 0.
    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]
    
    # Use ravel to get points to draw lines and circles
    for i,(new,prev) in enumerate(zip(good_new,good_prev)):
        
        x_new,y_new = new.ravel()
        x_prev,y_prev = prev.ravel()
        
        # Lines will be drawn using the mask created from the first frame
        mask = cv2.line(mask, (x_new,y_new),(x_prev,y_prev), (0,255,0), 3)
        cv2.rectangle(mask, (x,y), (x+w,y+h), (255,255,255), 10)

        # Draw red circles at corner points
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
    points1=[(new1.ravel())[0] for i,(new,new1) in enumerate(zip(sl,nextPts)) if(new.ravel())[0]>x and (new.ravel())[0]<(x+w)]
    for i in points1:
        if(i<x+h and i>x):
            s1=0
        else:
            s1=1
            if(i<x and s==0):
                s=1
                pyautogui.press('space')
            elif(s==0):
                pyautogui.keyDown('down')
                s=2
            break
    if s1==0:      
        if(s==2):
            pyautogui.keyUp('down') 
        s=0

    # Display the image along with the mask we drew the line on.
    img = cv2.add(frame,mask)
    frame2 = cv2.flip(img, 1)
    if(s==0):
        cv2.putText(frame2,"MID",(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,2,2)
    elif(s==1):
        cv2.putText(frame2,"RIGHT",(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,2,2)
    else:
        cv2.putText(frame2,"LEFT",(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,2,2)
    #frame1 = pyautogui.screenshot()
    #frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
    #writer.write(frame1)
    cv2.imshow('frame',frame2)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
   
    # Now update the previous frame and previous points
    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1,1,2)
    
   
cv2.destroyAllWindows()
cap.release()
#writer.release()