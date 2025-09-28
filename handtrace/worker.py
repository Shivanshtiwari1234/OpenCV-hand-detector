import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Queue
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def hand_finger_counter(input_queue: Queue, output_queue: Queue) -> None:
    logging.info("Worker process started")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    finger_tips = [4, 8, 12, 16, 20]

    processed_frames = 0

    while True:
        try:
            frame = input_queue.get(timeout=1)
        except Exception as e:
            logging.warning(f"Queue get timeout: {e}")
            continue

        if frame is None:
            logging.info("Worker received exit signal")
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            finger_count = 0
            mask_visual = np.zeros(frame.shape[:2], dtype=np.uint8)
            contour_visual = np.zeros(frame.shape, dtype=np.uint8)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape
                    landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                    hull = cv2.convexHull(np.array(landmark_points))
                    cv2.drawContours(contour_visual, [hull], -1, (0, 255, 0), 2)
                    cv2.fillPoly(mask_visual, [hull], 255)

                    hand_label = results.multi_handedness[idx].classification[0].label
                    thumb_tip_x = landmark_points[finger_tips[0]][0]
                    thumb_mcp_x = landmark_points[finger_tips[0] - 1][0]

                    if (hand_label == "Right" and thumb_tip_x < thumb_mcp_x) or \
                       (hand_label == "Left" and thumb_tip_x > thumb_mcp_x):
                        finger_count += 1

                    for tip in finger_tips[1:]:
                        if landmark_points[tip][1] < landmark_points[tip - 2][1]:
                            finger_count += 1

            mask_3ch = cv2.merge([mask_visual]*3)

            try:
                output_queue.put((frame, mask_3ch, contour_visual, finger_count), timeout=0.1)
            except Exception as e:
                logging.warning(f"Output queue full, dropped frame: {e}")

            processed_frames += 1
            if processed_frames % 50 == 0:
                logging.info(f"Processed {processed_frames} frames")

        except Exception as e:
            logging.error(f"Worker exception: {e}")

    logging.info("Worker process terminated")
