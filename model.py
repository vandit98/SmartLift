#  importing all libraries
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from ast import Str
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
# creating instance of the fastapi class
app = FastAPI()


mp_pose = mp.solutions.pose
# setting the parameter for the pose detection
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils 
@app.get("/knee")
def pose():


# this function will set landmarks on the image and return the coordinates of landmark and final marked image

# this function will calculate the angle between the three pts-ankle,knee and hip


    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # capturing the video-put your video here
    # video_path='/home/vandit/Downloads/pose_detector_mediapipe/knee_bend_filckering_removed.mp4'
    # cap = cv2.VideoCapture(video_path)
    cap=cv2.VideoCapture(0)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    fps = 0
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
    sec = 0
    period = '00'
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpuutttt.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


    """"
    Res calculating the time
        I have used the fps to calculate the time

    """
    res=0
    start_time = time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        # print("captured_frame")
        # setting the timer
        now=time.time()-start_time
        if ok==True:    
            cfn = cap.get(1)
            
            if int(cfn)%int(fps)==0:
                if int(now) >= 2:
                    res+=1
                    start_time = time.time()
                    now=0
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,str(int(now)),(10,30), font, 1,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,str(res),(10,60), font, 1,(255,0,0),2,cv2.LINE_AA)

            # frame = cv2.flip(frame, 1)
            
            frame_height, frame_width, _ =  frame.shape
            
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
            
            # getting the landmarked frame 
            frame, landmarks = detectPose(frame, pose, display=False)
        
            if landmarks:
                
                frame, _ ,right_angle,left_angle,label= classifyPose(landmarks, frame, display=False)
                print(right_angle,left_angle)
                right_angle='{0:.2f}'.format(right_angle)
                left_angle='{0:.2f}'.format(left_angle)
                angle_detail=f'l-angle:{left_angle} r-angle:{right_angle} res:{res}'
                cv2.putText(frame,str(angle_detail),(10,60),font, 1,(255,0,0),2,cv2.LINE_AA)
            cv2.imshow('frame',frame)
            plt.imshow(frame)
            # writing the frame into the video
            out.write(frame)



            key = cv2.waitKey(1) & 0xFF
            # if q is pressed video reading is stopped
            if key == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))    
    if angle < 0:
        angle += 360    
    return angle




# this function will  put the condition on how much angle will start coundown
def classifyPose(landmarks, output_image, display=False):
   

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    label = 'continue'
                
    # if angle is b/w [210,325] we will start the coundown
    # if (right_knee_angle > 210 and right_knee_angle < 325) or (left_knee_angle > 210 and left_knee_angle < 325):   
    # cv2.putText(str(right_knee_angle),str(left_knee_angle),(20,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    if (right_knee_angle > 200) or (left_knee_angle > 200):  
        cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                # to get the real time angle add this
                    # label=str(right_knee_angle)
                    # cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(frame,period,(10,30), font, 1,(0,0,255),2,cv2.LINE_AA)
        
    # if angle dont fulfill the required angle we will show feedback
    else:
        label='\b keep your knee bent'
        cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
      
    
    
    if display:
    
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        return output_image, label,right_knee_angle,left_knee_angle,label
def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    
    if results.pose_landmarks:
    
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    if display:
    
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        # to get the image of pose in a 3-d plane
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    else:
        
        return output_image, landmarks

# @app.get('/curl')
# def arm():
#     # capturing the video-put your video here
#     # video_path='/home/vandit/Downloads/pose_detector_mediapipe/knee_bend_filckering_removed.mp4'
#     # cap = cv2.VideoCapture(video_path)
#     cap = cv2.VideoCapture(0)
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#     fps = 0
#     if int(major_ver) < 3:
#         fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#         print(
#             "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
#     else:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#     sec = 0
#     period = '00'
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     # out = cv2.VideoWriter('outpuutttt.avi', cv2.VideoWriter_fourcc(
#     #     'M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

#     res = 0

#     """"
#     Res calculating the time
#         I have used the fps to calculate the time

#     """

#     while cap.isOpened():
#         ok, frame = cap.read()
#         # print("captured_frame")
#         # setting the timer
#         if ok == True:
#             cfn = cap.get(1)
#             if int(cfn) % int(fps) == 0:
#                 if sec >= 8:
#                     sec = 0
#                     res += 1
#                 if label == '\b good lets try again keep you arm bent':
#                     sec = 0
#                 period = "{:02d}".format(sec)
#                 sec = sec + 1

#             font = cv2.FONT_HERSHEY_SIMPLEX
            

#             frame = cv2.flip(frame, 1)

#             frame_height, frame_width, _ = frame.shape

#             frame = cv2.resize(
#                 frame, (int(frame_width * (640 / frame_height)), 640))

#             # getting the landmarked frame
#             frame, landmarks = detectPose(frame, pose, display=False)

#             if landmarks:

#                 frame, _, right_angle, left_angle, label = armpose(
#                     landmarks, frame, display=False)
#                 print(right_angle, left_angle)
#                 right_angle = '{0:.2f}'.format(right_angle)
#                 left_angle = '{0:.2f}'.format(left_angle)
#                 angle_detail = f'l-angle:{left_angle} r-angle:{right_angle} res:{res}'
#                 cv2.putText(frame, str(angle_detail), (10, 60),
#                             font, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 cv2.putText(frame, str(sec), (10, 30), font,
#                         1, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.imshow('frame', frame)
#             plt.imshow(frame)
#             # writing the frame into the video
#             # out.write(frame)

#             key = cv2.waitKey(1) & 0xFF
#             # if q is pressed video reading is stopped
#             if key == ord("q"):

#                 break

#     cap.release()
#     cv2.destroyAllWindows()
#     return {"succesfull res":res}

#     # cap = cv2.VideoCapture(0)

#     # # process_video(cap) like this we can call our functions
#     # cap.release()
#     # cv2.destroyAllWindows()


# mp_pose = mp.solutions.pose
# # setting the parameter for the pose detection
# pose = mp_pose.Pose(static_image_mode=True,
#                     min_detection_confidence=0.3, model_complexity=2)
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=False,
#                     min_detection_confidence=0.5, model_complexity=1)

# # this function will set landmarks on the image and return the coordinates of landmark and final marked image


# def detectPose(image, pose, display=True):
#     output_image = image.copy()
#     imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(imageRGB)
#     height, width, _ = image.shape
#     landmarks = []

#     if results.pose_landmarks:

#         mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
#                                   connections=mp_pose.POSE_CONNECTIONS)

#         for landmark in results.pose_landmarks.landmark:
#             landmarks.append((int(landmark.x * width), int(landmark.y * height),
#                               (landmark.z * width)))

#     if display:

#         plt.figure(figsize=[22, 22])
#         plt.subplot(121)
#         plt.imshow(image[:, :, ::-1])
#         plt.title("Original Image")
#         plt.axis('off')
#         plt.subplot(122)
#         plt.imshow(output_image[:, :, ::-1])
#         plt.title("Output Image")
#         plt.axis('off')
#         # to get the image of pose in a 3-d plane
#         mp_drawing.plot_landmarks(
#             results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

#     else:

#         return output_image, landmarks
# # this function will calculate the angle between the three pts-ankle,knee and hip


# def calculateAngle(landmark1, landmark2, landmark3):
#     x1, y1, _ = landmark1
#     x2, y2, _ = landmark2
#     x3, y3, _ = landmark3
#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                          math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360
#     return angle


# # this function will  put the condition on how much angle will start coundown
# def armpose(landmarks, output_image, display=False):
#     right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
#     left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
#     if right_elbow_angle>90:
#         right_elbow_angle=180-right_elbow_angle
#     # right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#     #                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
#     #                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
#     label = 'continue'
                
#     # if angle is b/w [210,325] we will start the coundown
#     # if (right_knee_angle > 210 and right_knee_angle < 325) or (left_knee_angle > 210 and left_knee_angle < 325):   
#     # cv2.putText(str(right_knee_angle),str(left_knee_angle),(20,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
#     # if (right_elbow_angle<30) or (left_elbow_angle > 150):  
#     if (left_elbow_angle <90):
#         cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
#                 # to get the real time angle add this
#                     # label=str(right_knee_angle)
#                     # cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         # cv2.putText(frame,period,(10,30), font, 1,(0,0,255),2,cv2.LINE_AA)
        
#     # if angle dont fulfill the required angle we will show feedback
#     else:
#         label='\b next attempt please'
#         cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
      
    
    
#     if display:
    
#         plt.figure(figsize=[10,10])
#         plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
#     else:
        
#         return output_image, label,right_elbow_angle,left_elbow_angle,label


  

# #   integrating face detection with fast_api

# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# from ast import Str
# import math
# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt
# import time
# app = FastAPI()



@app.get("/arm")
def curl():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose


# getting the landmark
    begin_time=time.time()
    now=0
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                print(landmarks)
            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or time.time()-begin_time>5:
                break
            cv2.putText(image,'checking camera',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cap.release()
        cv2.destroyAllWindows()
    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    calculate_curl_angle(shoulder, elbow, wrist)
    tuple(np.multiply(elbow, [640, 480]).astype(int))
    cap = cv2.VideoCapture(0)
## Setup mediapipe instance


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_curl_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        
            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or time.time()-begin_time>10:
                break

        cap.release()
        cv2.destroyAllWindows()


    # curl counter
    cap = cv2.VideoCapture(0)

# Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle=calculate_curl_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    if counter>10:
        return {f"you have completed {counter} curls great job!"}
    elif 0<counter<10:
        return {f"you have completed {counter} curls, keep going!"}
    else:
        return {"you have not completed any curls yet, keep trying!"}
    
# defining few function
def calculate_curl_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 