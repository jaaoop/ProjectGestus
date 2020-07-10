import cv2
import pynput
from pynput.keyboard import Key, Controller
from ContinuousGesturePredictor import continuousGesturePredictor
#requires pynput installed -> pip install pynput
#Instead of pynput, other libraries can be used according to your preference: keyboard, pyautogui
keyboard = Controller()
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
        keyboard.press('a')
        keyboard.release('a')

    elif prediction == "Joinha":
        keyboard.press('b')
        keyboard.release('b')

    elif prediction == "Ok":
        keyboard.press('c')
        keyboard.release('c')

    elif prediction == "Palm":
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        
    elif prediction == "Swing":
        keyboard.press('d')
        keyboard.release('d')