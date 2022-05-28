import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import winsound
# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

map_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture(0)


def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]

    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance


def blinkRatio(img, landmarks, right_indices, left_indices):
    
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]

    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

haar_face = cv.CascadeClassifier('haar_face.xml')
haar_eye = cv.CascadeClassifier('haar_eye.xml')


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    start_time = time.time()

    count = 0

    while True:
        frame_counter +=1
        ret, frame = camera.read()
        mask_frame = frame.copy()
        blank_frame = np. zeros(shape=[512,712,3],dtype=np. uint8)
        face_rect = haar_face.detectMultiScale(mask_frame, scaleFactor = 1.1, minNeighbors=20)
        eye_rect = haar_eye.detectMultiScale(mask_frame, scaleFactor = 1.1, minNeighbors=7)
        if not ret: 
            break
        
        frame = cv.resize(frame, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, (0,255,0), 2)

            if ratio >4.6:
                CEF_COUNTER +=1
                cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, (0,0,255), 2)

            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
            cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, (0,255,0), 2)
            
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv.LINE_AA)



        end_time = time.time()-start_time
        fps = frame_counter/end_time
        #mask program
        cv.rectangle(blank_frame, (20, 20), (180, 60), (255,0,0), thickness=3)
        cv.rectangle(blank_frame, (160, 200), (480, 295), (255,0,0), thickness=3)
        if len(face_rect) == 0 and len(eye_rect) != 0: 
            cv.putText(blank_frame, "MASK FOUND", (220, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), thickness=2)
            count = 0
        elif len(face_rect) == 0 and len(eye_rect) == 0:
            cv.putText(blank_frame, "FACE NOT FOUND", (170, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), thickness=2)
        else:
            cv.putText(blank_frame, "MASK NOT FOUND", (170, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), thickness=2)
            count +=1
            if (count > 30 and count%10 == 0):
                winsound.Beep(800,150)

        for (x, y, w, h) in face_rect:
            cv.rectangle(mask_frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        for (x, y, w, h) in eye_rect:
            cv.rectangle(mask_frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

        cv.imshow('Face Detection', mask_frame)

        cv.putText(blank_frame,f'FPS: {round(fps,1)}', (30,50), FONTS, 1, (0,0,255), thickness=2)
        
        cv.imshow('frame', frame)

        #DISPLAY
        cv.imshow("Blank", blank_frame)

        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()