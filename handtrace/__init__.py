from multiprocessing import Process, Queue
import cv2
from handtrace.worker import hand_finger_counter


def main() -> None:
    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)

    worker_proc = Process(target=hand_finger_counter, args=(input_queue, output_queue))
    worker_proc.start()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_flipped = cv2.flip(frame, 1)

        # Keep only the latest frame in the queue
        if input_queue.full():
            try:
                input_queue.get_nowait()  # drop old frame
            except:
                pass
        input_queue.put(frame_flipped.copy())

        if not output_queue.empty():
            processed_frame, mask, contour_visual, fingers = output_queue.get()

            cv2.putText(
                processed_frame,
                f"Fingers: {fingers}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            # Resize all frames to same height
            height = 480
            width = int(processed_frame.shape[1] * height / processed_frame.shape[0])
            frame_resized = cv2.resize(processed_frame, (width, height))
            mask_resized = cv2.resize(mask, (width, height))
            contour_resized = cv2.resize(contour_visual, (width, height))

            # Combine horizontally
            combined = cv2.hconcat([frame_resized, mask_resized, contour_resized])
            cv2.imshow("Hand Detection Dashboard", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    input_queue.put(None)
    worker_proc.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
