from multiprocessing import Process, Queue
import cv2
from handtrace.worker import hand_finger_counter
import logging


# ------------------- LOGGING CONFIG -------------------
logging.basicConfig(
    level=logging.DEBUG,  # MAX VERBOSITY
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# no file handler, console only
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


# ------------------- SETUP FUNCTIONS -------------------
def setup_camera() -> cv2.VideoCapture:
    logging.debug("Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        logging.critical("Failed to open camera.")
        raise RuntimeError("Camera not accessible.")
    logging.debug("Camera initialized successfully.")
    return cap


def setup_window(window_name: str = "Hand Detection") -> None:
    logging.debug("Creating non-resizable window...")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    logging.debug("Window setup complete.")


def start_worker(input_queue: Queue, output_queue: Queue) -> Process:
    logging.debug("Starting worker process...")
    worker_proc = Process(
        target=hand_finger_counter,
        args=(input_queue, output_queue),
        daemon=True
    )
    worker_proc.start()
    logging.info(f"Worker process started (PID: {worker_proc.pid})")
    return worker_proc


# ------------------- PROCESSING LOGIC -------------------
def handle_frame(frame, input_queue, output_queue):
    logging.debug("Handling new frame...")
    frame_flipped = cv2.flip(frame, 1)

    try:
        input_queue.put_nowait(frame_flipped.copy())
        logging.debug("Frame sent to input queue.")
    except Exception as e:
        logging.debug(f"Input queue full, dropped frame: {e}")

    try:
        if not output_queue.empty():
            processed_frame, _, _, fingers = output_queue.get_nowait()
            logging.debug(f"Frame received from output queue. Fingers detected: {fingers}")

            cv2.putText(
                processed_frame,
                f"Fingers: {fingers}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4
            )
            cv2.imshow("Hand Detection", processed_frame)
        else:
            logging.debug("Output queue empty.")
    except Exception as e:
        logging.warning(f"Output queue exception: {e}")


# ------------------- CLEANUP -------------------
def cleanup(cap, worker_proc, input_queue):
    logging.info("Cleaning up resources...")
    try:
        input_queue.put_nowait(None)
        logging.debug("Exit signal sent to worker.")
    except Exception as e:
        logging.warning(f"Failed to send exit signal to worker: {e}")

    if worker_proc.is_alive():
        logging.debug("Terminating worker process...")
        worker_proc.terminate()
    worker_proc.join()
    logging.debug("Worker process joined successfully.")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Main process terminated cleanly.")


# ------------------- MAIN LOOP -------------------
def main() -> None:
    logging.info("Starting main process (MAX VERBOSITY, CONSOLE MODE)")

    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)

    worker_proc = start_worker(input_queue, output_queue)
    cap = setup_camera()
    setup_window("Hand Detection")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                break

            handle_frame(frame, input_queue, output_queue)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logging.info("Exit key pressed by user.")
                break

            if not worker_proc.is_alive():
                logging.error("Worker process died unexpectedly.")
                break

    except Exception as e:
        logging.exception(f"Exception in main loop: {e}")

    finally:
        cleanup(cap, worker_proc, input_queue)


# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    main()
