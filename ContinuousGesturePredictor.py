import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import imutils
import argparse
import glob
import os
from os.path import join, split

# Argument parser for easy modifications
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold',
                    required=False, default=4,
                    help="Prediction threshold for confidence filter")
arguments = vars(parser.parse_args())

# Get a list with the name of the gestures in alphabetical order
gestureNames = []

for name in glob.glob(join(join('Dataset', 'Train'), '*')):
    gestureNames.append(split(name)[-1])
gestureNames.sort()

# global variables
bg = None
size = 350

# Confidence filter, it receives a vector with predictions for each 'n' frames
# if all are the same, return True, else return False
def samePredictions(new_predictions):
    for i in range(len(new_predictions)-1):
        if(new_predictions[0] != new_predictions[i+1]):
            return False
    return True


def direction(ret, frame, clone, firstGray, start_recording):
    if start_recording:
        #resize the first frame to the shape of the frame
        firstGray=cv2.resize(firstGray,(frame.shape[1],frame.shape[0]))

        #crop a specific part of the first frame 
        firstGRight=firstGray[10:225,590:620]
        firstGLeft=firstGray[10:225, 320:350]

        #transforms the frame in grayscale and blur it
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)

        #crop a specific part of the frame
        frameGRight= grayFrame[10:225,590:620]
        frameGLeft= grayFrame[10:225, 320:350]

        #comapare the parts of the first frame with the actual frame 
        differenceR = cv2.absdiff(firstGRight,frameGRight)
        differenceL = cv2.absdiff(frameGLeft,firstGLeft)

        a,differenceR = cv2.threshold(differenceR, 25, 255, cv2.THRESH_BINARY)
        b,differenceL = cv2.threshold(differenceL, 25, 555, cv2.THRESH_BINARY)

        #count the number of white pixels to determinate the moviment
        countRight = np.count_nonzero(differenceR)
        countLeft = np.count_nonzero(differenceL)

        #print the results of countage 
        # if countRight>50:
        #     print("Right")
        # if countLeft> 50:
        #     print("Left")

    #draw a square on the clone frame 
    cv2.rectangle(clone, (620, 10), (590, 225), (255,0,0), 2)
    cv2.rectangle(clone, (350, 10), (320, 225), (255,0,0), 2)


def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # Initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # Compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # Find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # Threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # Get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # Based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # Method global variables
    first_pass = True
    past_predictions = []
    new_predictions = []
    predictions_count = 0
    predictionsThreshold = arguments['threshold']

    # Show gestures found in the Dataset folder
    print("\nGestures found: ")
    for index, gesture in enumerate(gestureNames):
        print("{}- {} ".format(index+1, gesture))
    print("\n[WARNING] If during the predictions a gesture is not shown, verify if you have trained for it.")

    # Initialize weight for running average
    aWeight = 0.5

    # Get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # Region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # Initialize num of frames
    num_frames = 0
    start_recording = False

    #init frirtGray frame 
    firstGray = 0

    # keep looping, until interrupted
    while(True):
        # Get the current frame
        (grabbed, frame) = camera.read()

        #get the frame for direction function
        ret, frame = camera.read() 

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # Flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # Clone the frame
        clone = frame.copy()

        # counts the white pixels of the areas next to the green box
        direction(ret,frame,clone,firstGray,start_recording)

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # Get the ROI
        roi = frame[top:bottom, right:left]

        # Convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # To get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # Segment the hand region
            hand = segment(gray)

            # Check if user started the predictions,
            # otherwise show instructions
            if start_recording:

                # Check if there's a hand detection
                if hand is not None:

                    # If yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    
                    # Get predicted class and confidence
                    predictedClass, confidence = getPredictedClass()

                    # Resize threshold by size x size for better visualization
                    thresholded = cv2.resize(thresholded, (size,size), interpolation=cv2.INTER_CUBIC)
                
                else:
                    # As it has no hand make thresholded all black and class 'Null'
                    thresholded = np.ones((size,size), np.uint8)
                    predictedClass = -1
                    
                # For the first 'n' predictions, save the detections in a vector
                # the one detected will be shown
                if first_pass:
                    past_predictions.append(predictedClass)
                    predictions_count+=1
                    if predictions_count == predictionsThreshold:
                        predictions_count = 0
                        first_pass = False

                # For each 'n' frames save the predictions in a vector and check if
                # all are the same, if they do shown detection, else shown previous 
                else:
                    new_predictions.append(predictedClass)
                    predictions_count+=1
                    if predictions_count == predictionsThreshold:
                        if samePredictions(new_predictions):
                            predictedClass = new_predictions[-1]
                            past_predictions = new_predictions
                        else:
                            predictedClass = past_predictions[-1]
                        new_predictions = []
                        predictions_count = 0
                    else:
                        predictedClass = past_predictions[-1] 

                if hand is not None:
                    # Show text with class and confidence
                    textImage = showStatistics(predictedClass, confidence)

                else:
                    # If does not detect hand, show class Null
                    textImage = showStatistics(None, None, None)
                
            else:
                # Info about needing to press 's' to start the predictions 
                textImage = showWaitingStatistics()

                # Make thresholded all black
                thresholded = np.ones((size,size), np.uint8)
            
            # Show the detections made
            cv2.imshow("Thresholded and Statistics", np.concatenate((thresholded, textImage), 1))

        # Draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # Increment the number of frames
        num_frames += 1

        # Display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # Observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # If the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True
            
            #get the first frame for the function direction()
            _, firstFrame = camera.read()

            #flip the fist frame so its not a mirror view 
            firstFrame=cv2.flip(firstFrame,1)

            #transforms the first frame in grayscale and blur it
            firstGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            firstGray = cv2.GaussianBlur(firstGray, (5, 5), 0)

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.reshape(1, 89, 100, 1)
    prediction = model.predict([gray_image])
    return np.argmax(prediction), np.amax(prediction)

def showWaitingStatistics():
    textImage = np.zeros((size,size), np.uint8)

    cv2.putText(textImage, "Press 's' to start the predictions", 
    (0, size//2), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6,
    (255, 255, 255),
    2)
    return textImage

def showStatistics(predictedClass, confidence, containGesture = True):

    textImage = np.zeros((size,size), np.uint8)
    className = ""

    # Check if the prediction contains a gesture,
    # if does not contain the gesture will be named 'Null'
    if containGesture:

        # Get gesture name from the nameGesture variable 
        className = gestureNames[predictedClass]

    else:
        className = "Null"
        confidence = 1

    cv2.putText(textImage,"Pedicted Class : " + className, 
    (30, size//2 - 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(round(confidence * 100, 3)) + '%', 
    (30, size//2 + 20), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.7,
    (255, 255, 255),
    2)
    return textImage

# Load model weights
model = tf.keras.models.load_model(join("ModelWeights","GestureRecogModel_tf.tfl"))

try:
    main()
except KeyboardInterrupt:
    # Remove temp backgroud image if the program is abruptly terminated 
    os.remove("Temp.png")
    
# Remove temp backgroud image
try:
    os.remove("Temp.png")
except:
    pass
