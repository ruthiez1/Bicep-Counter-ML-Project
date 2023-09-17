import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 

cap = cv2.VideoCapture(0) 
while cap.isOpened():
    ret, frame = cap.read() 
    cv2.imshow('Smile! Youre on Camera!', frame) 

    if cv2.waitKey(10) & 0xFF==ord('q'): 
        break 
cap.release() 
cv2.destroyAllWindows() 

cap = cv2.VideoCapture(0) 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read() 
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        image.flags.writeable=False
        results = pose.process(image) 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 
        cv2.imshow('Camera', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 

cap.release() 
cv2.destroyAllWindows() 

cap = cv2.VideoCapture(0) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read() 
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        image.flags.writeable=False 
        results = pose.process(image) 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) s 
        cv2.imshow('Mediapipe Feed', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 

cap.release() 
cv2.destroyAllWindows() 

def calculate_angle(a,b,c): 
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle



shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]


cap = cv2.VideoCapture(0) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read() 
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        image.flags.writeable=False 
        results = pose.process(image) 
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(wrist, elbow, shoulder)
            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [300, 230]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) 
        except:
            pass


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #draws detections to images 
        cv2.imshow('Smile, Youre On Camera!', image ) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 

cap.release() 
cv2.destroyAllWindows() 


cap = cv2.VideoCapture(0) 


counter = 0 
stage = None 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read() 
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        image.flags.writeable=False 
        results = pose.process(image) #accessing the pose variable we set, storing the visualisation into results
        image.flags.writeable=True 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # Get Coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(wrist, elbow, shoulder)

            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) 

            if angle > 140:
                stage = "down"
            elif angle < 50 and stage == "down":
                stage = "up"
                counter += 1
            print(counter)
        except:
            pass

        cv2.rectangle(image, (0,0), (225, 73), (230,215,255), -1)

        cv2.putText(image, 'Reps', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image,str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 
        
        if cv2.waitKey(10) & 0xFF==ord('q'): 
            break 
        elif cv2.waitKey(10) & 0xFF==ord('r'):
            counter = 0

cap.release() 
cv2.destroyAllWindows() 
