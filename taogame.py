import cv2 as cv
import mediapipe as mp
import numpy as np
import copy
import itertools
import time

from module import taoism

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

def add_rectange(image, hand_area, color):
    cv.rectangle(
        image, (hand_area[0], hand_area[1]), (hand_area[2], hand_area[3]), color, 3
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

def info_frame(image, current_level):
    text = f'Level {current_level}'
    x = 10
    y = 30
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 2

    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    text = 'Press Q to quit the game'
    y = 65
    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)
    
    text = 'Press P to pause the game'
    y = 95
    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    return image

def process_level(positions, pre_process_landmark_list, current_level, index_finger_x, index_finger_y, circle_touched):
    # Check if the hand is near the current circle in sequence
    for i, (x, y) in enumerate(positions):
        if (taoism.is_gesture_matching(pre_process_landmark_list, current_level) 
            and taoism.is_position_matching(index_finger_x, index_finger_y, x, y) 
            and (i == 0 or circle_touched[i - 1])
        ):
            circle_touched[i] = True
            break  # Move to the next frame once the current circle is touched

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    hand_area = []
    paused = False
    total_paused_duration = 0

    taoism.init_game_setting()
    current_level = 1

    circle_touched_1 = [False] * 20
    circle_touched_2 = [False] * 22
    circle_touched_3 = [False] * 30
    
    positions_level_1 = []
    positions_level_2 = []
    positions_level_3 = []

    game_over = False
    game_started = False
    game_win = False

    while cap.isOpened():

        # Control setting
        key = cv.waitKey(5)
        if key == ord('q'): # Press 'q' to quit the game
            break
        elif key == ord('p'):  # Press 'p' to pause the game
            paused = not paused
            if paused:
                pause_start_time = time.time()
            else:
                total_paused_duration += time.time() - pause_start_time

        if paused:

            text = 'Paused'
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 3)
            x = (image.shape[1] - text_width) // 2
            y = (image.shape[0] + text_height) // 2

            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (0, 0, 255)
            thickness = 3

            cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA, )
            cv.imshow("Game", image)
            continue

        success, image = cap.read()
        if not success:
            print("No frame captured.")
            continue

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        positions_level_1 = taoism.get_level_1_positions(image)
        positions_level_2 = taoism.get_level_2_positions(image)
        positions_level_3 = taoism.get_level_3_positions(image)

        if not game_over and not game_win:
                
            if results.multi_hand_landmarks is not None:  # Hand is detected
                game_started = True

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    hand_area = get_hand_area(image, hand_landmarks)
                    landmark_list = get_landmark_point(image, hand_landmarks)                    
                    pre_process_landmark_list = pre_process_landmarks(landmark_list)

                    index_finger = hand_landmarks.landmark[8]
                    index_finger_x = int(index_finger.x * image.shape[1])
                    index_finger_y = int(index_finger.y * image.shape[0])

                    if current_level == 1:
                        process_level(positions_level_1, pre_process_landmark_list, current_level, index_finger_x, index_finger_y, circle_touched_1)
                    elif current_level == 2:
                        process_level(positions_level_2, pre_process_landmark_list, current_level, index_finger_x, index_finger_y, circle_touched_2)
                    elif current_level == 3:
                        process_level(positions_level_3, pre_process_landmark_list, current_level, index_finger_x, index_finger_y, circle_touched_3)

                    add_rectange(image, hand_area, (0, 0, 255)) # Not correct gesture
                    # Check if the gesture matches the current target gesture
                    if taoism.is_gesture_matching(pre_process_landmark_list, current_level):
                        add_rectange(image, hand_area, (0, 255, 0)) # Correct hand gesture                      
                   
                    taoism.apply_module(image, hand_area, handedness, pre_process_landmark_list)

            # Display the hand gesture image of current number
            image = taoism.display_target_gesture(image, current_level)

        elif game_win:
            # Display "You win!" message
            text = 'You win!'
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 3)
            x = (image.shape[1] - text_width) // 2
            y = (image.shape[0] + text_height) // 2

            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (0, 255, 0)
            thickness = 3

            cv.putText(image, text, (x, y), font, font_scale, color, thickness)

        # Display "Game starts when the hand is detected" message before the game starts
        if not game_started:
            start_message = 'Game starts when the hand is detected'
            (text_w, text_h), _ = cv.getTextSize(start_message, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = (image.shape[1] - text_w) // 2
            text_y = (image.shape[0] + text_h) // 2

            cv.putText(image, start_message, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw the circles and numbers on the screen based on the current level
        if current_level == 1:
            if game_started:
                for idx, (x, y) in enumerate(positions_level_1):
                    color = (0, 255, 0) if circle_touched_1[idx] else (0, 0, 255)
                    cv.circle(image, (x, y), 15, color, -1)
                    cv.putText(image, str(idx + 1), (x - 10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            if all(circle_touched_1):
                current_level = 2  # Move to the next level

        elif current_level == 2:
            for idx, (x, y) in enumerate(positions_level_2):
                color = (0, 255, 0) if circle_touched_2[idx] else (0, 0, 255)
                cv.circle(image, (x, y), 15, color, -1)
                cv.putText(image, str(idx + 1), (x - 10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if all(circle_touched_2):
                current_level = 3  # Move to the next level
   
        elif current_level == 3:
            if not all(circle_touched_3):
                for idx, (x, y) in enumerate(positions_level_3):
                    color = (0, 255, 0) if circle_touched_3[idx] else (0, 0, 255)
                    cv.circle(image, (x, y), 15, color, -1)
                    cv.putText(image, str(idx + 1), (x - 10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                game_win = True

        processed_image = info_frame(image, current_level)
        cv.imshow("Game", processed_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
