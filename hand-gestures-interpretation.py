import os               # OS library is used to open the software
import cv2              # OpenCV library is used to open the camera
import numpy as np      # Numpy and Math library is used to make the calculations
import math

# Constants
LOWER_SKIN = np.array([0, 20, 70], dtype=np.uint8)          # Defines color range of the skin in HSV
UPPER_SKIN = np.array([20, 255, 255], dtype=np.uint8)       # Defines color range of the skin in HSV
KERNEL = np.ones((3, 3), np.uint8)                          # Creates a matrix of 3x3 with 8 bits
ROI_COORDINATES = (400, 90, 600, 290)                       # Top left corner (x1, y1) and bottom right corner (x2, y2)

# Preprocess the frame (frame inversion)
def preprocess_frame(frame):
    return cv2.flip(frame, 1)

# Function to defines roi (region of interest) of the object analysis
def define_roi(frame):

    x1, y1, x2, y2 = ROI_COORDINATES
    
    # Define the region of interest - mask of the object analysis
    roi = frame[y1:y2, x1:x2]                           # mask size

    # Draw a rectangle in the region of interest
    pt1 = x1,y1                                         # Top left corner
    pt2 = x2,y2                                         # Bottom right corner
    color = (0,255,0)                                   # Color of the rectangle
    thickness = 3                                       # Thickness of the rectangle
    cv2.rectangle(frame, pt1, pt2, color, thickness)    # mask reading

    return roi

# Function to detect the object
def detect_object(roi):

    # Apply color-based skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)          # Convert of the RGB pattern to HSV

    # Extract image from the skin contour in relation to the object background
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    return mask if np.any(mask) else None               # Return the mask if it is not empty

# Function to process the object mask (e.g., dilation, blur, contour detection)
def process_object(mask):
    if mask is None:
        return None
    
    if not np.any(mask):  # Check if mask is empty (contains all zeros)
        return None

    dilated = cv2.dilate(mask, KERNEL, iterations=4)    # Dilation in the hand to fill dark spots inside
    blurred = cv2.GaussianBlur(dilated, (5, 5), 100)    # Fill hand with "blur" in the image
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours if contours else None

# Function to recognize the object motion based on contours and defects
def recognize_motion(contours, roi):
    if contours is None or len(contours) == 0:
        return [0, 0, 0]  # No motion detected

    # Find the maximum hand contour
    cnt = max(contours, key = lambda x: cv2.contourArea(x)) # function applied to detect the contour of the binarized object

    # Contour approximation to the object
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Make a convex object around the hand
    hull = cv2.convexHull(cnt)

    # Define hull area
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    # Find the percentage of area not covered by the hand in convex hull
    arearatio = ((areahull - areacnt) / areacnt) * 100

    # Find the defects in the convex hull with respect to the hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    if defects is None:
        return [0, areacnt, arearatio]  # No motion detected

    # motion_label = no defects
    motion_label = 0

    # Apply Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])

        # Find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

        # Distance between point and convex hull
        d = (2 * ar) / a

        # Apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

        # Ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            motion_label += 1
            cv2.circle(roi, far, 3, [0,0,255], -1)

        # Draw green lines around hand
        cv2.line(roi,start, end, [0,255,0], 2)      # roi = region of interest

    motion_label += 1

    return [motion_label, areacnt, arearatio]

def main():
    
    # Open the camera
    cap = cv2.VideoCapture(1)

    x1, y1, x2, y2 = ROI_COORDINATES

    while True:
        try:
            # Read the camera frame
            ret, frame = cap.read()
            
            if not ret:
                break

            frame = preprocess_frame(frame)
            roi = define_roi(frame)
            mask = detect_object(roi)

            if mask is not None:
                contours = process_object(mask)

                if mask is not None:
                    motion_label, areacnt, arearatio = recognize_motion(contours, roi)

                    font = cv2.FONT_HERSHEY_SIMPLEX                    

                    if motion_label == 1:
                        if areacnt < 2000:
                            cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        elif arearatio < 12:
                            cv2.putText(frame, '0 = Zero', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, '1 = One', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                    elif motion_label == 2:
                        cv2.putText(frame,'2 = Two',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                    elif motion_label == 3:
                        if arearatio < 27:
                            cv2.putText(frame,'3 = Three',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                    elif motion_label == 4:
                        cv2.putText(frame,'4 = Four',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        
                    elif motion_label == 5:
                        cv2.putText(frame,'5 = Five',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        
                    else :
                        cv2.putText(frame,'Reposition your hand',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        continue
                
                else:
                    cv2.putText(frame,'No hand detected',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    continue

            # Show the mask and frame only if they are not empty
            if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
                cv2.imshow('mask', mask)

            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                cv2.imshow('frame', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        except Exception as e:
            print("Error:", e)
            pass

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()