import cv2 as cv
import csv
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import BudClassifier
bud_gesture_classifier = BudClassifier()
sequence = []

def get_gesture_info(image, hand_area, handedness, pre_process_landmark_list):
    hand_gesture_id = bud_gesture_classifier(pre_process_landmark_list)
    gesture = get_gesture_label(hand_gesture_id)
    cv.rectangle(
        image,
        (hand_area[0], hand_area[1]),
        (hand_area[2], hand_area[1] - 22),
        (0, 0, 0),
        -1,
    )

    info_text = handedness.classification[0].label[0:] # left hand or right hand
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


def get_random_gesture_sequence():
    global sequence
    sequence = random.sample(range(8), 8)

def get_gesture_label(number):
    bud_gesture_labels = []
    with open("model/bud_classifier/gesture_label.csv", encoding="utf-8-sig") as f:
        bud_gesture_labels = csv.reader(f)
        bud_gesture_labels = [row[0] for row in bud_gesture_labels]
    return bud_gesture_labels[number]  

def init_game_setting():
    get_random_gesture_sequence()

def display_target_gesture(image, current_index):
    # Get images and variables of the current frame
    resource_folder = 'resources/buddhism'
    images = [cv.imread(os.path.join(resource_folder, f"{i}.jpg")) for i in range(8)]

    current_image = images[sequence[current_index]]

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

    text = get_gesture_label(sequence[current_index])
    y = (image_height - gesture_height) // 2 - 15
    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

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

def is_gesture_matching(pre_process_landmark_list, target_hand_gesture):
    current_hand_gesture = bud_gesture_classifier(pre_process_landmark_list)
    return current_hand_gesture == target_hand_gesture

def apply_module(image, hand_area, handedness, pre_process_landmark_list):
    return get_gesture_info(image, hand_area, handedness, pre_process_landmark_list)
