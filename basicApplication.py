import cv2
from ContinuousGesturePredictor import continuousGesturePredictor
#Basic code layout for editing
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
        pass

    elif prediction == "Joinha":
        pass

    elif prediction == "Ok":
        pass

    elif prediction == "Palm":
        pass

    elif prediction == "Swing":
        pass