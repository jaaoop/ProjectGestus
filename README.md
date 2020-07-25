# Gesture detector with computer vision

<img src="https://i.imgur.com/x8p2Ygk.jpg" width="250" height="250">

*Read this in other languages: [English](https://github.com/jaaoop/ProjectGestus), [Portuguese](https://github.com/jaaoop/ProjectGestus/blob/master/README.pt.md).*

## Project operation

### Description

The following project has the purpose to detect hand gestures and apply functionalities to the detections, such as moving a video game character or controlling hardware parts. 

## Recommendations for using the repository
- Use conda to create virtual environments.
- Use cuda and cudnn for better performance (verify if GPU is compatible).
- Linux is recommended for all procedures.

## Installation

### Python 

Recommended version 3.7.

### Create a conda virtual environment

`conda create -n <name-of-environment> python=3.7`<br/>
`conda activate <name-of-environment>` 

### Necessary dependencies

**One line installation**<br/>
`pip install tensorflow-gpu==1.15.2 opencv-python numpy Pillow imutils scipy matplotlib`

- ### TensorFlow (recommended version <= 1.15.2)
`pip install tensorflow==1.15.2` (CPU)<br/>
`pip install tensorflow-gpu==1.15.2` (GPU)

- ### OpenCV
`pip install opencv-python`

- ### Numpy
`pip install numpy`

- ### Pillow
`pip install Pillow`

- ### Imutils
`pip install imutils`

- ### SciPy
`pip install scipy`

- ### MatPlotLib
`pip install matplotlib`

### Download the repository
Download the repository or clone by executing in the shell `git clone https://github.com/jaaoop/ProjectGestus.git`. After this steps it will be ready to use.

### Files guide

[**ContinuousGesturePredictor.py**](https://github.com/jaaoop/ProjectGestus/blob/master/ContinuousGesturePredictor.py) makes gesture detection in real time. 
1. Execute in the shell `python ContinuousGesturePredictor.py` .
2. When the file is open, the webcam will start and the recording will be shown for the user.
3. In the open window, a square will be drawn and, during 30 frames, will take the area as background. Taking this to count, leave this area free from the hand for better results.
4. After the 30 first frames, a *Thresholded and Statistics* window will appear, at this moment the user must press '**s**' to start the detection and, then, positionate the hand in the drawn square.
5. The window *Thresholded and Statistics* will show the detected gesture, the user is free to move and test new detections.

[**LabelGenerator.py**](https://github.com/jaaoop/ProjectGestus/blob/master/LabelGenerator.py) generates new training gestures.
1. Execute in the shell `python LabelGenerator.py -n <gesture-name>`.
2. When the file is open, the webcam will start and the recording will be shown for the user.
3. In the open window a square will be drawn and, during 30 frames, will take the area as background. Taking this to count, leave this area free from the hand for better results.
4. After the 30 first frames, a *Thresholded and Statistics* window will appear, in this moment the user must press '**s**' to start generating training and testing gesture pictures. **Suggestion:** Move the hand for diverse results.
5. In the process of creating the new gesture, the shell will show the progress. When finished, two folders will be created, one of [Train](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Train) and one of [Test](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Test), both with the gesture name.

>**Note:** One additional parameter from LabelGenerator is `-t <image-number>` where the ammount of training images is defined, the test ones are 10% of this value. By default the parameter is set to `-t 1000`.

[**ModelTrainer.py**](https://github.com/jaaoop/ProjectGestus/blob/master/ModelTrainer.py) trains the model to detect new gestures.
1. Certify that you have the same folders in [Train](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Train) and [Test](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Test).
2. Execute in the shell `python ModelTrainer.py`.
3. Wait until the end of the training.

>**Note:** One additional parameter from ModelTrainer is `-c True` that allows to save the training chart. By default the parameter is set to `-c False`.

### Code implementations

The following files will behave in the same way as `ContinousGesturePredictor.py`, with the exception that for each gesture an action is 
assigned, like moving a video game character or pressing a keyboard key. The file `basicApplication.py` is the implementations code structure without the assignments, the user is able to set actions accordingly to his necessities. 

[**basicApplication.py**](https://github.com/jaaoop/ProjectGestus/blob/master/basicApplication.py) is a template for possible applications.
1. Execute in the shell `python basicApplication.py` .
2. The file will behave in the same way as `ContinousGesturePredictor.py` if the user doesn't make any gesture assignments.

[**runwNotes.py**](https://github.com/jaaoop/ProjectGestus/blob/master/runwNotes.py) is application demo for your basic text editor.
1. Execute in the shell `python runwNotes.py`.
2. If the text editor window is open, the user might see the designed characters being written according to the detection.

>**Note:** Additional dependencies might be needed for some applications.

[**runwMinecraft.py**](https://github.com/jaaoop/ProjectGestus/blob/master/runwMinecraft.py) is application demo that controls character movements in Minecraft.
1. Execute in the shell `python runwMinecraft.py`.
2. If the Minecraft window is open, the character will move according to the detection.

>**Note:** Additional dependencies might be needed for some applications.

[**runwArduino.py**](https://github.com/jaaoop/ProjectGestus/blob/master/runwArduino.py) is application demo for Arduino.
1. Execute in the shell `python runwArduino.py` . 
2. If an Arduino is connected, the user might see commands being given according to the detection.

>**Note:** Additional dependencies might be needed for some applications.

## Informations
This project is part of the RAS Unesp Bauru projects. For more information about this and other projects, access: https://sites.google.com/unesp.br/rasunespbauru/home.

## Authors

- [**Artur Starling**](https://github.com/ArturStarling)
- [**Fabrício Amoroso**](https://github.com/FabricioAmoroso)
- [**Gustavo Stahl**](https://github.com/GustavoStah)
- [**João Gouvêa**](https://github.com/jaaoop)

## License

This project is free and non-profit.

## Credits

The project is based in the repository [Hand Gesture Recognition using Convolution Neural Network built using Tensorflow, OpenCV and python](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network).
