# Game mode
This is the game mode of the Edtech application. <br> It contains both Taoism and Buddhism playing methods. 
Other processes like data collection and model building are also includes in this project. 
<br> The program recognizes player's hand gesture and hand positions to play around the game.

### Requirements
- Python 3.9.17
- PyQt6 6.7.0
- Mediapipe 0.9.0.1
- OpenCV 4.8.0
- Tensorflow 2.13.0

### Running method
1. Open the terminal
2. Run python app.py 
3. The game window is shown and player can choose from Taoism and Buddhism.

Taoism:
The game consists of 3 levels, in each level the player need to perform the correct hand gesture based on the requirement.
<br> When player performs the correct hand gesture, he/she need to follow the dots on the screen in correct sequcne to perform the movement.
<br> If the current level is completed, player can go to the next level.
<br> After all three levels are completed, the player wins the game.

Buddhism:
This game requires player to perform the correct hand gesture in a random sequence within 30 seconds.
<br> On the left side of the screen, it displays which hand gesture player need to perform now.
<br> If player performs the wrong hand gesture for more than 3 seconds, the game will reduce the timer by 5 seconds as a punishment.
<br> After all the hand gestures successfully completed, the player wins the game.

### Project structure
│  app.py
│  budgame.py
│  taogame.py
|  data_collection.py
│  test.py
│  
├─model
│  ├─bud_classifier
│  │  │  bud_classification.ipynb
│  │  │  bud_classifier.hdf5
│  │  │  bud_classifier.py
│  │  │  bud_classifier.tflite
│  │  │  gesture_label.csv
│  │  └─ keypoint.csv
│  │          
│  └─tao_classifier
│      │  tao_classification.ipynb
│      │  tao_classification.hdf5
│      │  tao_classification.py
│      │  tao_classification.tflite
│      │  gesture_label.csv
│      └─ keypoint.csv
│      
├─module
│  ├─buddhism.py
│  └─taoism.py
│  
├─resources
│  ├─buddhism
│  │  │  0.jpg
│  │  │  1.jpg
│  │  └─ ...
│  │          
│  └─taoism
│      │  0.jpg
│      │  1.jpg
│      └─ ...
│ 
└─utils
    └─cvfpscalc.py

### app.py
The main program to enter the game mode, it invokes taoism module or buddhism module based on user's selection.

### budgame.py
The implementation of buddhism play mode.

### taogame.py
The implementation of taoism play mode.

### data_collection.py
The python file for data collection process. 
<br> Press 'r' can enter the record mode to record the keypoint positions, then input the gesture id that want to record.
<br> Press 'c' can clear the recording status.

### test.py
The python file to test the model, it displays the hand gesture label that the model classified and the corresponding classification rate.

### module/buddhism.py and module/taoism.py
These two python files contains the neccessary functions to play during the process.

### bud_classification.ipynb and tao_classification.ipynb
The jupyter notebook for model training, they also evaluate the model performance.

### Reference
https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe

