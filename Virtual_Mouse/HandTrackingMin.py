import cv2
import mediapipe as mp
import time             #to check the framerate

cap = cv2.VideoCapture(0)       #to create vedio object

mpHands = mp.solutions.hands                                    # for hand detection
hands = mpHands.Hands()         #check parameter
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark): # giving id no to the landmark points
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)    #will give position
                print(id,cx,cy)
                if id == 4:         # for specific id or landmark we can edit it
                    cv2.circle(img, (cx,cy), 15, (255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # will draw connections in traking points

    cTime = time.time() # will give current time
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

