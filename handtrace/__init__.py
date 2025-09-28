from multiprocessing import Process, Queue
import cv2
from handtrace.worker import hand_finger_counter
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def resize_frame_to_height(frame, target_height):
    """Resize frame keeping aspect ratio."""
    h, w = frame.shape[:2]
    new_w = int(w * target_height / h)
    return cv2.resize(frame, (new_w, target_height))


def main():
    logging.info("Starting main process")

    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)

    # Start the worker process
    worker_proc = Process(target=hand_finger_counter, args=(input_queue, output_queue))
    worker_proc.start()
    logging.info(f"Worker process started (PID: {worker_proc.pid})")

    cap = cv2.VideoCapture(0)
    target_height = 480
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                break

            frame_flipped = cv2.flip(frame, 1)
            frame_small = cv2.resize(frame_flipped, (320, 240))  # reduce load for worker

            # Put frame into worker queue with timeout
            try:
                input_queue.put(frame_small.copy(), timeout=0.1)
            except:
                logging.warning("Input queue full, dropped frame")

            # Get processed frame from worker
            try:
                if not output_queue.empty():
                    processed_frame, mask, contour_visual, fingers = output_queue.get(timeout=0.1)

                    # Calculate FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time

                    cv2.putText(
                        processed_frame,
                        f"Fingers: {fingers} | FPS: {int(fps)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    # Resize frames for dashboard
                    processed_resized = resize_frame_to_height(processed_frame, target_height)
                    mask_resized = resize_frame_to_height(mask, target_height)
                    contour_resized = resize_frame_to_height(contour_visual, target_height)

                    combined = cv2.hconcat([processed_resized, mask_resized, contour_resized])
                    cv2.imshow("Hand Detection Dashboard", combined)

            except Exception as e:
                logging.warning(f"Output queue get exception: {e}")

            # Exit on 'q' or ESC
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                logging.info("Exit key pressed")
                break

            # Monitor worker health
            if not worker_proc.is_alive():
                logging.error("Worker process died unexpectedly")
                break

    except Exception as e:
        logging.error(f"Exception in main loop: {e}")

    finally:
        logging.info("Stopping worker process")
        try:
            input_queue.put(None, timeout=1)
        except:
            logging.warning("Failed to send exit signal to worker")
        worker_proc.join()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Main process terminated")
