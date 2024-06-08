import cv2 as cv
import numpy as np
import csv
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TaoClassifier
tao_gesture_classifier = TaoClassifier()
gesture_1 = 0
gesture_2 = 0
gesture_3 = 0

def get_gesture_label(number):
    with open("model/tao_classifier/gesture_label.csv", encoding="utf-8-sig") as f:
        tao_gesture_labels = csv.reader(f)
        tao_gesture_labels = [row[0] for row in tao_gesture_labels]
        return tao_gesture_labels[number]

def get_gesture_info(image, hand_area, handedness, pre_process_landmark_list):
    hand_gesture_id = tao_gesture_classifier(pre_process_landmark_list)
    gesture = get_gesture_label(hand_gesture_id) 
    cv.rectangle(
        image,
        (hand_area[0], hand_area[1]),
        (hand_area[2], hand_area[1] - 22),
        (0, 0, 0),
        -1,
    )

    info_text = handedness.classification[0].label[0:]
    if gesture != "":
        info_text = info_text + ":" + gesture
    cv.putText(
        image,
        info_text,
        (hand_area[0] + 5, hand_area[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return image

def display_target_gesture(image, current_level):
    # Get images and variables of the current frame
    resource_folder = 'resources/taoism'
    images = [cv.imread(os.path.join(resource_folder, f"{i}.jpg")) for i in range(10)]

    target_hand_gesture = 0
    if current_level == 1:
        target_hand_gesture = gesture_1
    elif current_level == 2:
        target_hand_gesture = gesture_2
    else:
        target_hand_gesture = gesture_3

    current_image = images[target_hand_gesture]

    gesture_height, gesture_width = current_image.shape[:2]
    image_height, image_width = image.shape[:2]

    # Target geature name
    text = "Gesture:"
    x = 10
    y = (image_height - gesture_height) // 2 - 45
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 2
    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    text = get_gesture_label(target_hand_gesture)
    y = (image_height - gesture_height) // 2 - 15
    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    # Calculate position to place the image on the left middle side of the frame
    x_offset = 10 
    y_offset = (image_height - gesture_height) // 2 # Put it in the middle of the height

    # Overlay the current image onto the frame at the calculated position
    y1, y2 = y_offset, y_offset + gesture_height
    x1, x2 = x_offset, x_offset + gesture_width

    if x2 <= image_width and y2 <= image_height:  # Check if the image fits within the frame
        alpha_s = current_image[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            image[y1:y2, x1:x2, c] = (alpha_s * current_image[:, :, c] + alpha_l * image[y1:y2, x1:x2, c])

    return image

def get_level_1_positions(image):
    # Get center position
    x = image.shape[1] // 2
    y = image.shape[0] // 2 - 40
    # 20 points
    positions = [(x+100, y-200), (x+75, y-100), (x+50, y-50), (x, y-10), (x-70, y+20),
                 (x-140, y+30), (x-190, y-20), (x-120, y-40), (x-50, y-30), (x+90, y-5),
                 (x+200, y+5), (x+140, y+80), (x+40, y+140), (x-40, y+190), (x-100, y+210), 
                 (x-190, y+190), (x-140, y+140), (x-50, y+130), (x+150, y+175), (x+230, y+210)]
    return positions

def get_level_2_positions(image):
    # Get center position
    x = image.shape[1] // 2
    y = image.shape[0] // 2 - 40
    # 22 points
    positions = [(x-220, y-200), (x-200, y-140), (x-80, y-150), (x+50, y-170), (x+150, y-120),
                 (x+45, y-85), (x-70, y-75), (x-150, y-30), (x-60, y+10), (x+55, y-5), 
                 (x+170, y-15), (x+230, y+30), (x+150, y+80), (x+60, y+100), (x-20, y+110), 
                 (x-130, y+150), (x-80, y+230), (x-10, y+180), (x+60, y+220), (x+120, y+170), 
                 (x+190, y+210), (x+260, y+150)]
    return positions

def get_level_3_positions(image):
    # Get center position
    x = image.shape[1] // 2
    y = image.shape[0] // 2 - 20
    # 30 points
    positions = [(x-90, y-50), (x-130, y-120), (x-130, y-200), (x-40, y-260), (x+60, y-240),
                 (x+110, y-180), (x+120, y-110), (x+80, y-40) ,(x, y), (x-80, y+20),
                 (x-160, y+60), (x-200, y+120), (x-170, y+180), (x-100, y+210), (x-20, y+160),
                 (x-55, y+120), (x-100, y+150), (x-50, y+210), (x+30, y+200), (x+110, y+170),
                 (x+150, y+110), (x+100, y+80), (x+70, y+130), (x+160, y+145), (x+240, y+110),
                 (x+300, y+60), (x+250, y+20), (x+210, y+60), (x+280, y+100), (x+360, y+50)]
    return positions

def init_game_setting():
    global gesture_1
    global gesture_2
    global gesture_3
    
    numbers = list(range(10))
    gesture_1 = random.choice(numbers)
    
    numbers.remove(gesture_1)
    gesture_2 = random.choice(numbers)

    numbers.remove(gesture_2)
    gesture_3 = random.choice(numbers)

def is_gesture_matching(pre_process_landmark_list, current_level):
    target_hand_gesture = 0
    if current_level == 1:
        target_hand_gesture = gesture_1
    elif current_level == 2:
        target_hand_gesture = gesture_2
    else:
        target_hand_gesture = gesture_3
    current_hand_gesture = tao_gesture_classifier(pre_process_landmark_list)
    return current_hand_gesture == target_hand_gesture

def is_position_matching(hand_x, hand_y, point_x, point_y, threshold=15):
    return np.sqrt((hand_x - point_x) ** 2 + (hand_y - point_y) ** 2) < threshold

def apply_module(image, hand_area, handedness, pre_process_landmark_list):
    return get_gesture_info(image, hand_area, handedness, pre_process_landmark_list)
