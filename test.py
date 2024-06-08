import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools

from utils import CvFpsCalc
from model import TaoClassifier
from model import BudClassifier

def select_mode(key, mode):
    if key == ord('t'):  # t, taoism
        mode = 0
        print("Using Taoism model")
    if key == ord('b'):  # b, buddhism
        mode = 1
        print("Using Buddhism model")
    return mode


def get_hand_area(image, hand_landmarks):
    image_height, image_width = image.shape[0], image.shape[1]
    coordinates_array = np.empty((0, 2), int)
    for _, landmark in enumerate(hand_landmarks.landmark):
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        hand_coordinates = [np.array((x, y))]
        coordinates_array = np.append(coordinates_array, hand_coordinates, axis=0)
    x, y, w, h = cv.boundingRect(coordinates_array)
    return [x, y, x + w, y + h]


def add_rectange(image, hand_area):
    cv.rectangle(
        image, (hand_area[0], hand_area[1]), (hand_area[2], hand_area[3]), (0, 0, 0), 1
    )
    return image


def get_landmark_point(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Get the coordinates
    for _, landmark in enumerate(landmarks.landmark):
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])
    return landmark_point


def pre_process_landmarks(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize(n):
        return n / max_value

    temp_landmark_list = list(map(normalize, temp_landmark_list))
    return temp_landmark_list


def get_gesture_info(image, hand_area, handedness, gesture, hand_gesture_id):
    # Add the description of the hand
    if hand_gesture_id == 3:
        cv.rectangle(
            image,
            (hand_area[0], hand_area[1]),
            (hand_area[2], hand_area[1] - 22),
            (255, 0, 0),
            -1,
        )
    else:
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

    # Add a rectangle to hightlight where is the hand
    cv.rectangle(
        image,
        (hand_area[0], hand_area[1]),
        (hand_area[2], hand_area[3]),
        (160, 0, 160),
        2,
    )
    return image


def info_frame(image, fps):
    # Show FPS
    cv.putText(
        image,
        "FPS: " + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return image

def calculate_percentages(count_dict):
    total = sum(count_dict.values())
    if total == 0:
        return {k: 0 for k in count_dict}
    return {k: (v / total) * 100 for k, v in count_dict.items()}

def display_percentages(image, percentages, start_y):
    for gesture, percent in percentages.items():
        text = f"{gesture}: {percent:.2f}%"
        cv.putText(
            image,
            text,
            (10, start_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        start_y += 20
    return image

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    tao_gesture_classifier = TaoClassifier()
    with open("model/tao_classifier/gesture_label.csv", encoding="utf-8-sig") as f:
        tao_gesture_labels = csv.reader(f)
        tao_gesture_labels = [row[0] for row in tao_gesture_labels]

    bud_gesture_classifier = BudClassifier()
    with open("model/bud_classifier/gesture_label.csv", encoding="utf-8-sig") as f:
        bud_gesture_labels = csv.reader(f)
        bud_gesture_labels = [row[0] for row in bud_gesture_labels]

    mode = 0  # Mode is set to Taoism originally
    hand_area = []

    tao_gesture_counts = {label: 0 for label in tao_gesture_labels}
    bud_gesture_counts = {label: 0 for label in bud_gesture_labels}

    while cap.isOpened():
        # Check whether the real-time captured image stable
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Press 'q' to end the program
        key = cv.waitKey(5)
        if key == ord('q'):
            break

        success, image = cap.read()
        if not success:
            print("No frame captured.")
            continue

        mode = select_mode(key, mode)

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True

        fps = cvFpsCalc.get()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        processed_image = image

        if results.multi_hand_landmarks is not None:  # Hand is detected
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                hand_area = get_hand_area(image, hand_landmarks)
                landmark_list = get_landmark_point(image, hand_landmarks)

                pre_process_landmark_list = pre_process_landmarks(landmark_list)

                processed_image = image
                if mode == 0:
                    hand_gesture_id = tao_gesture_classifier(pre_process_landmark_list)
                    gesture = tao_gesture_labels[hand_gesture_id]
                    tao_gesture_counts[gesture] += 1
                    processed_image = get_gesture_info(
                        processed_image,
                        hand_area,
                        handedness,
                        tao_gesture_labels[hand_gesture_id],
                        hand_gesture_id,
                    )
                    cv.putText(
                        image,
                        "Taoism",
                        (10, 70),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv.LINE_AA,
                    )
                else:
                    hand_gesture_id = bud_gesture_classifier(pre_process_landmark_list)
                    gesture = bud_gesture_labels[hand_gesture_id]
                    bud_gesture_counts[gesture] += 1                    
                    processed_image = get_gesture_info(
                        processed_image,
                        hand_area,
                        handedness,
                        bud_gesture_labels[hand_gesture_id],
                        hand_gesture_id,
                    )
                    cv.putText(
                        image,
                        "Buddhism",
                        (10, 70),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv.LINE_AA,
                    )                    


        if mode == 0:
            tao_percentages = calculate_percentages(tao_gesture_counts)
            processed_image = display_percentages(processed_image, tao_percentages, 100)
        else:
            bud_percentages = calculate_percentages(bud_gesture_counts)
            processed_image = display_percentages(processed_image, bud_percentages, 100)

        processed_image = info_frame(image, fps)
        cv.imshow("Test", processed_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
