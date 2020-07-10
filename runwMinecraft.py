import cv2
import keyboard
from ContinuousGesturePredictor import continuousGesturePredictor
#Open minecraft and test inputs
#Requires keyboard in base -> pip install keyboard
#The codes must be installed and running in base as well because of keyboard library
camera=cv2.VideoCapture(0)
predizGesto=continuousGesturePredictor()
while(True):
    _, frame = camera.read()
    predizGesto.main(frame)
    prediction = predizGesto.className

    if predizGesto.keypress == ord("q"): #Press 'q' to exit
        break

    #Assign actions for each prediction
    if prediction == "Fist":
        keyboard.press('d')
        keyboard.release('space')
        keyboard.release('s')
        keyboard.release('w')
        keyboard.release('a')

    elif prediction == "Joinha":
        keyboard.release('s')
        keyboard.release('a')
        keyboard.release('d')
        keyboard.release('w')
        keyboard.press('space')

    elif prediction == "Ok":
        keyboard.press('s')
        keyboard.release('space')
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('d')

    elif prediction == "Palm":
        keyboard.press('w')
        keyboard.release('space')
        keyboard.release('s')
        keyboard.release('a')
        keyboard.release('d')

    elif prediction == "Swing":
        keyboard.press('a')
        keyboard.release('space')
        keyboard.release('w')
        keyboard.release('s')
        keyboard.release('d')

    elif predicition == "Null":
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.release('space')