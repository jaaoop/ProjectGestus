# organize imports
import cv2
import imutils
import numpy as np
import os
from os.path import join, split
import argparse

# Argument parser for easy modifications
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name',
                    required=True,
                    help="Name of the gesture")
parser.add_argument('-t', '--training',
                    required=False, default=1000,
                    help="Number of training images")
arguments = vars(parser.parse_args())

# global variables
bg = None
gestureTrainFolder = join(join('Dataset', 'Train'), arguments['name'])
gestureTestFolder = join(join('Dataset', 'Test'), arguments['name'])

def createGestureFolders():

    # Create gesture train folder if does not exist
    if not os.path.exists(gestureTrainFolder):
        try:
            os.mkdir(gestureTrainFolder)
        except:
            print("[ERROR] Dataset/Train does not exists")

    # Create gesture test folder if does not exist
    if not os.path.exists(gestureTestFolder):
        try:
            os.mkdir(gestureTestFolder)
        except:
            print("[ERROR] Dataset/Test does not exists")

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # Method global variables
    size = 350

    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    image_num = 0

    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if (grabbed == True):

            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if num_frames < 30:
                run_avg(gray, aWeight)
                print("[INFO] Taking background ({}/30)".format(num_frames+1))

            elif num_frames == 30:
                print("[INFO] Ready for recording")

            else:
                # segment the hand region
                hand = segment(gray)

                # Check if user started the predictions,
                # otherwise show instructions
                if start_recording:

                    # Check if there's a hand detection
                    if hand is not None:
                        # if yes, unpack the thresholded image and
                        # segmented region
                        (thresholded, segmented) = hand

                        # Resize image since the model requires images with width=100 and height=89
                        thresholded = cv2.resize(thresholded, (100, 89), interpolation=cv2.INTER_AREA)

                    else:
                        # As it has no hand make thresholded all black
                        thresholded = np.ones((89, 100), np.uint8)

                    # Save train images
                    if image_num<arguments['training']:
                        cv2.imwrite(join(gestureTrainFolder, arguments['name'].lower()+'_') + str(image_num) + '.png', thresholded)
                    
                    # Save test images
                    else:
                        cv2.imwrite(join(gestureTestFolder, arguments['name'].lower()+'_') + str(image_num - arguments['training']) + '.png', 
                                    thresholded)

                    # Show generator progress
                    print("Progress: {}%".format((image_num+1)*100 // (arguments['training'] + int(arguments['training']*0.1))))
                    image_num += 1
                    
                    # Increase image size for showing to the user
                    thresholded = cv2.resize(thresholded, (size,size), interpolation=cv2.INTER_CUBIC)

                else:
                    # Info about needing to press 's' to start the generator
                    thresholded = cv2.putText(np.zeros((size,size), np.uint8),
                                                "Press 's' to start the generator", 
                                                (20, size//2), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.6,
                                                (255, 255, 255),
                                                2)

                # Show Thresholded image
                cv2.imshow("Thesholded", thresholded)

            # Draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

            # Increment the number of frames
            num_frames += 1

            # Display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # Observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q") or image_num >= arguments['training'] + int(arguments['training']*0.1):
                print('[INFO] The process ended successfully')
                break
        
            if keypress == ord("s"):
                #Create gesture test and train folders
                createGestureFolders()
                start_recording = True

        else:
            print("[WARNING] Error input. Please check your(camera or video)")
            break

    # Free up memory
    camera.release()
    cv2.destroyAllWindows()

main()


