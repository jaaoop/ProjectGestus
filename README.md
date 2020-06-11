# Gesture detector with computer vision

<img src="https://i.imgur.com/x8p2Ygk.jpg" width="250" height="250">

*Read this in other languages: [English](https://github.com/jaaoop/ProjectGestus), [Portuguese](https://github.com/jaaoop/ProjectGestus/tree/master/Lang/README.pt.md).*

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

`conda create -n <nome-do-ambiente> python=3.7`<br/>
`conda activate <nome-do-ambiente>` 

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

[**ContinuousGesturePredictor.py**](https://github.com/jaaoop/ProjectGestus/blob/master/ContinuousGesturePredictor.py) make gestures detection in real time. 
1. Execute in the shell `python ContinuousGesturePredictor.py` .
2. When the file is open, the webcam will start and the recording will be shown for the user.
3. In the open window, a square will be drawn and, during 30 frames, will take the area as background. Taking this to count, leave this area free from the hand for better results.
4. After the 30 first frames, a *Thresholded and Statistics* window will appear, at this moment the user must press '**s**' to start the detection and, then, positionate the hand in the drawn square.
5. The window *Thresholded and Statistics* will show the detected gesture, the user is free to move and test new detections.

[**LabelGenerator.py**](https://github.com/jaaoop/ProjectGestus/blob/master/LabelGenerator.py) generate new training gestures.
1. Execute in the shell `python LabelGenerator.py -n <gesture-name>`.
2. When the file is open, the webcam will start and the recording will be shown for the user.
3. In the open window a square will be drawn and, during 30 frames, will take the area as background. Taking this to count, leave this area free from the hand for better results.
4. After the 30 first frames, a *Thresholded and Statistics* window will appear, in this moment the user must press '**s**' to start generating training and testing gesture pictures. **Suggestion:** Move the hand for diverse results.
5. In the process of creating the new gesture, the shell will show the progress. When finished, two folders will be created, one of [Train](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Train) and one of [Test](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Test), both with the gesture name.

>**Note:** One additional parameter from LabelGenerator is `-t <image-number>` where the ammount of training images is defined, the test ones are 10% of this value. By default the parameter is set to `-t 1000`.

[**ModelTrainer.py**](https://github.com/jaaoop/ProjectGestus/blob/master/ModelTrainer.py) train the model to detect new gestures.
1. Certify that you have the same folders in [Train](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Train) and [Test](https://github.com/jaaoop/ProjectGestus/tree/master/Dataset/Test).
2. Execute in the shell `python ModelTrainer.py`.
3. Wait until the end of the training.

>**Note:** One additional parameter from ModelTrainer is `-c True` that allows to save the training chart. By default the parameter is set to `-c False`.

## Informations
This project is part of the RAS Unesp Bauru projects. For more information about this and other projects, access: https://sites.google.com/unesp.br/rasunespbauru/home.

## Authors

- [**Artur Starling**](https://github.com/ArturStarling)
- [**Fabrício Amoroso**](https://github.com/lefabricion)
- [**Gustavo Stahl**](https://github.com/GustavoStah)
- [**João Gouvêa**](https://github.com/jaaoop)

## License

This project is free and non-profit.

## Credits

The project is based in the repository [Hand Gesture Recognition using Convolution Neural Network built using Tensorflow, OpenCV and python](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network).
