import cv2
import serial 
from ContinuousGesturePredictor import continuousGesturePredictor

camera=cv2.VideoCapture(0)
predizGesto=continuousGesturePredictor()

#Arduino variable 
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)  # Open serial port
​
while(True):
    _, frame = camera.read()
    predizGesto.main(frame)
    prediction = predizGesto.className
​
    if predizGesto.keypress == ord("q"): #Press 'q' to exit
        arduino.close() #Close serial port 
        break
        
    #Assign actions for each prediction
    if prediction == "Fist":
        arduino.write(bytes('a', encoding='utf-8'))         # writes a string
​
    elif prediction == "Joinha":
        arduino.write(bytes('b', encoding='utf-8'))         # writes a string
                                        
    elif prediction == "Ok":
        arduino.write(bytes('c', encoding='utf-8'))         # writes a string
​
    elif prediction == "Palm":
        arduino.write(bytes('d', encoding='utf-8'))         # writes a string
​
    elif prediction == "Swing":
        arduino.write(bytes('f', encoding='utf-8'))         # writes a string
​