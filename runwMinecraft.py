import cv2
# import keyboard
import pyautogui
from ContinuousGesturePredictor import continuousGesturePredictor
#requires keyboard in the environment -> pip install keyboard
#requires PyAutoGUI in the env -> pip install pyautogui
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
        pyautogui.press('w')

    elif prediction == "Joinha":
        pyautogui.press('a')

    elif prediction == "Ok":
        pyautogui.press('s')

    elif prediction == "Palm":
        pyautogui.press('space')

    elif prediction == "Swing":
        pyautogui.press('d')

    elif prediction == "Null":
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('s')
        pyautogui.keyUp('d')
        pyautogui.keyUp('space')