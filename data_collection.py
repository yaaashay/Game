import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools

from utils import CvFpsCalc

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


def get_gesture_id(gesture_id):
    gesture_id = input("Enter the hand gesture ID: ")
    return int(gesture_id)

def select_status(key, status, gesture_id):
    if key == 99:  # c
        status = 0  # Clear
        gesture_id = 0
        print("Status clear")
    if key == 114:  # r
        status = 1  # Record
        gesture_id = get_gesture_id(gesture_id)
        print("Record status, current hand gesture id is " + str(gesture_id))
    return status, gesture_id

def log_to_csv(gesture_id, status, landmark_list):
    if status == 1:  # Record mode
        path_csv = "model/tao_classifier/keypoint.csv" # Modify the path for Taoism/Buddhism
        with open(path_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([int(gesture_id), *landmark_list])
    return

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
        if index == 0: # Coordinates of wrist as the base
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    #Some value is negative -> take the absolute value
    max_value = max(list(map(abs, temp_landmark_list))) 

    def normalize(n):
        return n / max_value

    temp_landmark_list = list(map(normalize, temp_landmark_list))
    return temp_landmark_list

def info_frame(image, fps, status, gesture_id):
    # Show FPS
    text = "FPS: " + str(fps)
    x = 10
    y = 30

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 2

    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    # Show current recording hand gesture
    if status == 1:
        text = "Recording coordinates"
        x = 10
        y = 70
        cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

        text = "Current hand gesture ID: " + str(gesture_id)
        x = 10
        y = 110
        cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    return image

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    status = 0  # No status initially
    gesture_id = 0
    hand_area = []

    while cap.isOpened():
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Press 'q' to end the data collection process
        key = cv.waitKey(5)
        if key == ord('q'):
            break

        success, image = cap.read()
        if not success:
            print("No frame captured.")
            continue

        # Successfully capture the frame
        status, gesture_id = select_status(key, status, gesture_id)

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True

        fps = cvFpsCalc.get()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

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

                log_to_csv(int(gesture_id), status, pre_process_landmark_list)

                processed_image = cv.rectangle(
                    image,
                    (hand_area[0], hand_area[1]),
                    (hand_area[2], hand_area[3]),
                    (160, 0, 160),
                    2,
                )

        processed_image = info_frame(image, fps, status, gesture_id)
        cv.imshow("Coordinates capture", processed_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
