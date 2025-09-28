import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Queue


def hand_finger_counter(input_queue: Queue, output_queue: Queue) -> None:
    """Detect hands, count fingers, and create visual mask/contours using Mediapipe."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    while True:
        frame = input_queue.get()
        if frame is None:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        finger_count = 0
        mask_visual = np.zeros(frame.shape[:2], dtype=np.uint8)
        contour_visual = np.zeros(frame.shape, dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to pixel coordinates
                h, w, _ = frame.shape
                landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # Create convex hull for visual mask
                hull = cv2.convexHull(np.array(landmark_points))
                cv2.drawContours(contour_visual, [hull], -1, (0, 255, 0), 2)
                cv2.fillPoly(mask_visual, [hull], 255)

                # Count fingers
                if landmark_points[finger_tips[0]][0] < landmark_points[finger_tips[0] - 1][0]:  # Thumb
                    finger_count += 1
                for tip in finger_tips[1:]:
                    if landmark_points[tip][1] < landmark_points[tip - 2][1]:
                        finger_count += 1

        # Convert mask to 3 channels for display consistency
        mask_3ch = cv2.merge([mask_visual] * 3)

        output_queue.put((frame, mask_3ch, contour_visual, finger_count))
