import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
from pynput.keyboard import Key, Controller
from yolo import YOLO
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=256, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

keyboard = Controller()

# global variables
bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,89), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, top, left, bottom, right, threshold=20):

    image = image[top:bottom, left:right]

    image = cv2.resize(image, (215, 240), interpolation=cv2.INTER_AREA)
    background = bg[top:bottom, left:right]
    background = cv2.resize(background, (215, 240), interpolation=cv2.INTER_AREA)
    cv2.imshow("teste", image)

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
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
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # initialize num of frames
    num_frames = 0
    start_recording = False

    (grabbed, frame) = camera.read()
    # keep looping, until interrupted
    while grabbed:
        time.sleep(0.025)
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # region of interest (ROI) coordinates
        top, left, bottom, right = 10, 350, 225, 590

        # get the ROI
        roi = frame.copy() #[top:bottom, right:left]

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            run_avg(gray, aWeight)
        else:
            width, height, inference_time, results = yolo.inference(roi)
            for detection in results:
                id, name, confidence, x, y, w, h = detection
                if x<0 : x=0
                if y<0 : y=0
                top, left, bottom, right = y, x, y+h, x+w
                cv2.rectangle(clone, (x-15, y-15), (x + w + 15, y + h + 15), (0, 255, 255), 2)
            
            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)                

            # segment the hand region
            hand = segment(gray, top, left, bottom, right)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    print("Started prediction\nStarted prediction\nStarted prediction")
                    # predictedClass, confidence = getPredictedClass()
                    # showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        # cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True

# def getPredictedClass():
#     # Predict
#     image = cv2.imread('Temp.png')
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     prediction = model.predict([gray_image.reshape(89, 100, 1)])
#     return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

# def showStatistics(predictedClass, confidence):

#     textImage = np.zeros((300,512,3), np.uint8)
#     className = ""

#     if predictedClass == 0:
#         className = "Swing"
#         keyboard.release('w')
#         keyboard.release('d')
#         keyboard.press('a')
#     elif predictedClass == 1:
#         className = "Palm"
#         keyboard.release('a')
#         keyboard.release('d')
#         keyboard.press('w')
#     elif predictedClass == 2:
#         className = "Fist"
#         keyboard.release('a')
#         keyboard.release('w')
#         keyboard.press('d')

#     cv2.putText(textImage,"Pedicted Class : " + className, 
#     (30, 30), 
#     cv2.FONT_HERSHEY_SIMPLEX, 
#     1,
#     (255, 255, 255),
#     2)

#     cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
#     (30, 100), 
#     cv2.FONT_HERSHEY_SIMPLEX, 
#     1,
#     (255, 255, 255),
#     2)
#     cv2.imshow("Statistics", textImage)




# # Model defined
# tf.reset_default_graph()
# convnet=input_data(shape=[None,89,100,1],name='input')
# convnet=conv_2d(convnet,32,2,activation='relu')
# convnet=max_pool_2d(convnet,2)
# convnet=conv_2d(convnet,64,2,activation='relu')
# convnet=max_pool_2d(convnet,2)

# convnet=conv_2d(convnet,128,2,activation='relu')
# convnet=max_pool_2d(convnet,2)

# convnet=conv_2d(convnet,256,2,activation='relu')
# convnet=max_pool_2d(convnet,2)

# convnet=conv_2d(convnet,256,2,activation='relu')
# convnet=max_pool_2d(convnet,2)

# convnet=conv_2d(convnet,128,2,activation='relu')
# convnet=max_pool_2d(convnet,2)

# convnet=conv_2d(convnet,64,2,activation='relu')
# convnet=max_pool_2d(convnet,2)

# convnet=fully_connected(convnet,1000,activation='relu')
# convnet=dropout(convnet,0.75)

# convnet=fully_connected(convnet,3,activation='softmax')

# convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

# model=tflearn.DNN(convnet,tensorboard_verbose=0)

# # Load Saved Model
# model.load("TrainedModel/GestureRecogModel.tfl")

main()
