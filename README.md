
<div align="center">
  <h1>Count number of fingers are raised Using OpenCV and MediaPipe</h1>
 </div>

> This Project uses OpenCV and MediaPipe to Count the number of fingers displayed


## Features

- **Hand Gesture Recognition**: Leverages the MediaPipe library for accurate hand tracking and gesture recognition.
- **Finger Count**: It counts the Number of finger which can be further used in many area. By initializing different finger position a sign can be understood and further evaluated to fullfill required needs
- **Real-Time Feedback**: Provides immediate response to hand movements, ensuring a seamless user experience.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or above installed on your computer.
- A webcam or any compatible camera module for hand tracking.

## REQUIREMENTS
+ opencv-python
+ mediapipe
+ numpy

```bash
pip install -r requirements.txt
```
***
### MEDIAPIPE
> MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media.
> 
## ML Pipeline

MediaPipe Hands utilizes an ML pipeline consisting of multiple models working
together: A palm detection model that operates on the full image and returns an
oriented hand bounding box. A hand landmark model that operates on the cropped
image region defined by the palm detector and returns high-fidelity 3D hand
keypoints. This strategy is similar to that employed in our
[MediaPipe Face Mesh](./face_mesh.md) solution, which uses a face detector
together with a face landmark model.

Providing the accurately cropped hand image to the hand landmark model
drastically reduces the need for data augmentation (e.g. rotations, translation
and scale) and instead allows the network to dedicate most of its capacity
towards coordinate prediction accuracy. In addition, in our pipeline the crops
can also be generated based on the hand landmarks identified in the previous
frame, and only when the landmark model could no longer identify hand presence
is palm detection invoked to relocalize the hand.

#### Hand Landmark Model
After the palm detection over the whole image our subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression, that is direct coordinate prediction. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions.

To obtain ground truth data, we have manually annotated ~30K real-world images with 21 3D coordinates, as shown below (we take Z-value from image depth map, if it exists per corresponding coordinate). To better cover the possible hand poses and provide additional supervision on the nature of hand geometry, we also render a high-quality synthetic hand model over various backgrounds and map it to the corresponding 3D coordinates.<br>

#### Solution APIs
##### Configuration Options
> Naming style and availability may differ slightly across platforms/languages.

+ <b>STATIC_IMAGE_MODE</b><br>
If set to false, the solution treats the input images as a video stream. It will try to detect hands in the first input images, and upon a successful detection further localizes the hand landmarks. In subsequent images, once all max_num_hands hands are detected and the corresponding hand landmarks are localized, it simply tracks those landmarks without invoking another detection until it loses track of any of the hands. This reduces latency and is ideal for processing video frames. If set to true, hand detection runs on every input image, ideal for processing a batch of static, possibly unrelated, images. Default to false.

+ <b>MAX_NUM_HANDS</b><br>
Maximum number of hands to detect. Default to 2. But for changing the Volume it is set 1 as default so that stability is high and error is less

+ <b>MODEL_COMPLEXITY</b><br>
Complexity of the hand landmark model: 0 or 1. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.

+ <b>MIN_DETECTION_CONFIDENCE</b><br>
Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. Default to 0.5 and for volume control default is 0.7

+ <b>MIN_TRACKING_CONFIDENCE:</b><br>
Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully, or otherwise hand detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. Ignored if static_image_mode is true, where hand detection simply runs on every image. Default to 0.7.



## CODE EXPLANATION
<b>Importing Libraries</b>
```py
import cv2
import mediapipe as mp
import math
import numpy as np

```
***
Solution APIs 
```py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
```
***
Setting up webCam using OpenCV
```py
wCam, hCam = 1280, 720
cam = cv2.VideoCapture(0)
#Use 0 for deafult web cam and increase value based on connected web cam to its index
```
***
Using MediaPipe Hand Landmark Model for identifying Hands 
```py
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cam.isOpened():
    success, image = cam.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )
```
***
Using multi_hand_landmarks method for Finding postion of Hand landmarks
```py
lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])    
```
***
Assigning Every Finger tip for a index to evaluate Whether it is up or down
```py
tipIds = [4, 8, 12, 16, 20]
```
***
Marking all fingers landmarks and storing them
```py
if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
```

***
Checking in Landmark List which fingers are up based on tipids array and making boolean every for every finger
```py
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
```
***
Displaying Output using `cv2.putText` and displaying the fingers Count
```py
for finger in fingers:
    if finger:
        fingercnt += 1
cv2.putText(img, "Count: {:.2f}".format(int(fingercnt)), (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, colorVol)
fingercnt = 0
```
***
## Acknowledgments

- This project is built using the [MediaPipe](https://google.github.io/mediapipe/) framework for hand tracking and pose estimation.
- Special thanks to the [pycaw](https://github.com/AndreMiras/pycaw) library for controlling the system volume.

Closing webCam
```py
cam.release()
```
***
## Future Plan
  -Currently it requires to change the code based on the number of hands Working to fix this issue
