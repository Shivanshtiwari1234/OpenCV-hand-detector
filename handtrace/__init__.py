from multiprocessing import Process, Queue
import cv2
from handtrace.worker import hand_finger_counter
import time
import logging
from collections import deque

# Log everything to file, overwrite each run
logging.basicConfig(
    filename="hand_detection.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


def main() -> None:
    logging.info("Starting main process")

    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)

    worker_proc = Process(target=hand_finger_counter, args=(input_queue, output_queue), daemon=True)
    worker_proc.start()
    logging.info(f"Worker process started (PID: {worker_proc.pid})")

    cap = cv2.VideoCapture(0)
    fps_window = deque(maxlen=10)
    prev_time = time.time()

    # Create a fixed-size, non-resizable window
    cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Detection", 640, 480)
    cv2.setWindowProperty("Hand Detection", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                break

            frame_flipped = cv2.flip(frame, 1)
            frame_small = cv2.resize(frame_flipped, (320, 240))

            try:
                input_queue.put_nowait(frame_small.copy())
            except:
                logging.debug("Input queue full, dropped frame")

            try:
                if not output_queue.empty():
                    processed_frame, _, _, fingers = output_queue.get_nowait()

                    curr_time = time.time()
                    fps_window.append(1 / max(curr_time - prev_time, 1e-6))
                    prev_time = curr_time
                    fps = sum(fps_window) / len(fps_window)

                    cv2.putText(
                        processed_frame,
                        f"Fingers: {fingers} | FPS: {int(fps)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    cv2.imshow("Hand Detection", processed_frame)

            except Exception as e:
                logging.warning(f"Output queue exception: {e}")

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                logging.info("Exit key pressed")
                break

            if not worker_proc.is_alive():
                logging.error("Worker process died unexpectedly")
                break

    except Exception as e:
        logging.error(f"Exception in main loop: {e}")

    finally:
        logging.info("Cleaning up...")
        try:
            input_queue.put_nowait(None)
        except:
            logging.warning("Failed to send exit signal to worker")

        if worker_proc.is_alive():
            worker_proc.terminate()
        worker_proc.join()

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Main process terminated")


if __name__ == "__main__":
    main()
