import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils #gives all drawing utilities, and visualize the poses
mp_pose = mp.solutions.pose #importing pose estimation models, face detection, etc

#video feed set up 
cap = cv2.VideoCapture(0) #webcam, any camera that relates to your machine, 0 is the number that represents your web camera
while cap.isOpened():
    #give me the current feed for my webcam
    ret, frame = cap.read() #ret is return variable, frame will give the imagefrom the came
    cv2.imshow('Smile! Youre on Camera!', frame) #cv2 lets you visualize the code, the stuff, pops up on screen
    #Mediapipe Feed is what you want the frame to be called 
    #frame is the image for the webcam

    #next is what do we do if we break out of our feed
    if cv2.waitKey(10) & 0xFF==ord('q'): #checks if we hit q to close out screen, if we do, it stops the loop
        break # breaks out of the while loop
cap.release() #releases out of webcam
cv2.destroyAllWindows() #so when you start you don't have to deal with existing windows

cap = cv2.VideoCapture(0) 
# setting up the media pipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:# if you want more accurate, bump up, high confidence bump it up, you want to maintain state, this whole line will be accesible to pose variable
    while cap.isOpened():
        ret, frame = cap.read() 
        #detect stuff and render, recolouring the images to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #frame is webcam
        image.flags.writeable=False #recoloring image
       
        #making the detection
        results = pose.process(image) #accessing the pose variable we set, storing the visualisation into results
    
        #recoloring back to BGR 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #rendering detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #draws detections to images 
        #results.pose_landmarks gives us the landmarks, gives coordinates for each landmark
        #mp_pose.POSE_CONNECTIONS passes through pose connections
        cv2.imshow('Camera', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 

cap.release() #releases out of webcam
cv2.destroyAllWindows() #so when you start you don't have to deal with existing windows

cap = cv2.VideoCapture(0) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:# if you want more accurate, bump up, high confidence bump it up, you want to maintain state, this whole line will be accesible to pose variable
    while cap.isOpened():
        ret, frame = cap.read() 
        #detect stuff and render, recolouring the images to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #frame is webcam
        image.flags.writeable=False #recoloring image
       
        #make detection
        results = pose.process(image) #accessing the pose variable we set, storing the visualisation into results
    
        #recolor back to BGR 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #landmarks hold landmarks, this will give us the landmark
        except:
            #if it doesn't work we just step out
            pass


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #draws detections to images 
        #results.pose_landmarks gives us the landmarks, gives coordinates for each landmark
        #mp_pose.POSE_CONNECTIONS passes through pose connections
        cv2.imshow('Mediapipe Feed', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 

cap.release() #releases out of webcam
cv2.destroyAllWindows() #so when you start you don't have to deal with existing windows

'''Calculating Angle Between Body Parts'''
def calculate_angle(a,b,c): 
    a = np.array(a) #first, converting to numpy arrays, so you can calculate angle
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi) #convert r to º

    if angle > 180.0:
        angle = 360-angle

    return angle



shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]



#shoulder:
# x: 0.72693515 - a[0]
# y: 0.9739203 - a[1]
# z: -0.3003548 - a[2]
# visibility: 0.9977697

# left elbow: 
# x: 0.8228488 - b[0]
# y: 1.2781793 - b[1]
# z: -0.29505315 - b[2]
# visibility: 0.30260402

# left wrist: 
# x: 0.81951475 - c[0]
# y: 1.7374046 - c[1]]
# z: -0.46688765 - c[2]
# visibility: 0.08473396



#updating it now so we can visualize the angle 


cap = cv2.VideoCapture(0) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:# if you want more accurate, bump up, high confidence bump it up, you want to maintain state, this whole line will be accesible to pose variable
    while cap.isOpened():
        ret, frame = cap.read() 
        #detect stuff and render, recolouring the images to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #frame is webcam
        image.flags.writeable=False #recoloring image
       
        #make detection
        results = pose.process(image) #accessing the pose variable we set, storing the visualisation into results
    
        #recolor back to BGR 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # Get Coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculating Angle
            angle = calculate_angle(wrist, elbow, shoulder)

            # Visualize Angle
            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [300, 230]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) 
        except:
            pass


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #draws detections to images 
        cv2.imshow('Smile, Youre On Camera!', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 

cap.release() #releases out of webcam
cv2.destroyAllWindows() #so when you start you don't have to deal with existing windows








cap = cv2.VideoCapture(0) 

#curl counter variables
counter = 0 
stage = None # whether or not we're at the down curl or the up part of the curl

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:# if you want more accurate, bump up, high confidence bump it up, you want to maintain state, this whole line will be accesible to pose variable
    while cap.isOpened():
        ret, frame = cap.read() 
        #detect stuff and render, recolouring the images to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #frame is webcam
        image.flags.writeable=False #recoloring image
       
        #make detection
        results = pose.process(image) #accessing the pose variable we set, storing the visualisation into results
    
        #recolor back to BGR 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # Get Coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculating Angle
            angle = calculate_angle(wrist, elbow, shoulder)

            # Visualize Angle
            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) 

            #Curl Counter Logic
            if angle > 140:
                stage = "down"
            elif angle < 50 and stage == "down":
                stage = "up"
                counter += 1
            print(counter)
        except:
            pass
       

        # Render Curl Counter
        # Set Up Status Box - in the top left hand corner
        cv2.rectangle(image, (0,0), (225, 73), (230,215,255), -1)
        # (0,0) - start point
        # (225, 73) - end point of rect
        # -1: fill the box with colour

        # Rep Data
        cv2.putText(image, 'Reps', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image,str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)

        # cv2.putText(image, 'Stage', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image,stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #draws detections to images 
        cv2.imshow('Smile, Youre On Camera!', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 
        elif cv2.waitKey(10) & 0xFF==ord('r'):
            counter = 0

cap.release() #releases out of webcam
cv2.destroyAllWindows() #so when you start you don't have to deal with existing windows