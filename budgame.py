import cv2 as cv
import mediapipe as mp
import numpy as np
import copy
import itertools
import time

from module import buddhism

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

def info_frame(image, current_number):
    text = f'Gesture No. {current_number}'
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

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    hand_area = []

    buddhism.init_game_setting()
    current_index = 0

    # Initialize countdown timer
    countdown_time = 30
    timer_started = False
    start_time = None
    remaining_time = 30
    display_duration = 2
    last_update_time = time.time()
    non_match_start_time = None
    non_match_duration = 3  # Time in seconds for non-matching gestures to penalize    
    pause_start_time = None # Use for calculate the remaining time
    total_paused_duration = 0

    game_over = False
    game_started = False
    game_win = False
    paused = False

    while cap.isOpened():

        # Control setting
        key = cv.waitKey(5)
        if key == ord('q'): # Press 'q' to quit the game
            break
        elif key == ord('p') and not game_over:  # Press 'p' to pause the game
            paused = not paused # toggle
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
        image.flags.writeable = True # Draw the hand annotations on the image.

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if not game_over and not game_win:

            # Check if hands are detected and start the timer if not started
            if results.multi_hand_landmarks is not None and not timer_started:
                timer_started = True
                game_started = True
                start_time = time.time()

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

                    add_rectange(image, hand_area, (0, 0, 255)) # Not correct gesture                    

                    # Check if the gesture matches the current index
                    if buddhism.is_gesture_matching(pre_process_landmark_list, buddhism.sequence[current_index]) and not paused:
                        non_match_start_time = None  # Reset non-match start time
                        current_time = time.time()
                        if current_time - last_update_time > display_duration:
                            current_index = current_index + 1
                            last_update_time = current_time
                            # Check if the user has completed all gestures
                            if current_index == 8:
                                current_index = 7
                                game_win = True

                        add_rectange(image, hand_area, (0, 255, 0)) # Correct hand gesture

                    elif not paused:
                        if non_match_start_time is None:
                            non_match_start_time = time.time()
                        elif time.time() - non_match_start_time - total_paused_duration >= non_match_duration:
                            countdown_time = max(0, countdown_time - 5)
                            non_match_start_time = None  # Reset after penalty                        
                   
                    buddhism.apply_module(image, hand_area, handedness, pre_process_landmark_list)
            elif not paused:
                non_match_start_time = None  # Reset non-match start time if no hand detected

            # Display the hand gesture image of current number
            image = buddhism.display_target_gesture(image, current_index)

            # Calculate remaining time for the countdown if timer started
            if timer_started and not paused:
                elapsed_time = time.time() - start_time - total_paused_duration
                remaining_time = max(0, countdown_time - int(elapsed_time))

                # Display the countdown timer on the top middle of the screen
                timer_text = f'Time Left: {remaining_time}s'
                (text_width, text_height), _ = cv.getTextSize(timer_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = (image.shape[1] - text_width) // 2
                text_y = 50  # Distance from the top of the screen

                cv.putText(image, timer_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            # Break the loop if countdown reaches zero
            if remaining_time <= 0:
                game_over = True

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


        else:
            # Display "Game Over" message
            text = 'Game Over'
            (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 3)
            text_x = (image.shape[1] - text_w) // 2
            text_y = (image.shape[0] + text_h) // 2

            cv.putText(image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Display "Game starts when the hand is detected" message before the game starts
        if not game_started:
            start_message = 'Game starts when the hand is detected'
            (text_w, text_h), _ = cv.getTextSize(start_message, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = (image.shape[1] - text_w) // 2
            text_y = (image.shape[0] + text_h) // 2

            cv.putText(image, start_message, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        processed_image = info_frame(image, current_index + 1)
        cv.imshow("Game", processed_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
