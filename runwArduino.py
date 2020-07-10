import cv2
from ContinuousGesturePredictor import continuousGesturePredictor

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
        arduino(port, "a")

    elif prediction == "Joinha":
        arduino(port, "b")

    elif prediction == "Ok":
        arduino(port, "c")

    elif prediction == "Palm":
        arduino(port, "d")

    elif prediction == "Swing":
        arduino(port, "e")


#Arduino function
def arduino(channel, char):
    char -> letra
    example: channel = '/dev/ttyACM0'
    arduino = serial.Serial(port=channel, baudrate=9600, timeout=1)
    arduino.write(bytes(char, encoding='utf-8'))
    arduino.close()