# Multiprocess Hand & Finger Detection

This project detects hands from a webcam feed, counts the number of fingers shown, and displays real-time results using OpenCV and Python multiprocessing. The detection uses HSV skin color segmentation and convexity defects to estimate fingers.

## Features

- Real-time webcam hand detection

- Counts 0–5 fingers

- Shows original frame, skin mask, and contours separately

- Uses multiprocessing to keep detection smooth

- Modular structure for easy expansion

## File Structure
hand_finger_counter/
│
├─ main.py        # Main program, handles webcam input and output display
├─ worker.py      # Worker process for hand detection and finger counting
└─ README.md      # This file

## Requirements

- Python 3.10+

- OpenCV (opencv-python)

- NumPy

## Install dependencies

``` bash
pip install opencv-python numpy
```

How to Run

Clone the repo:

``` bash
git clone https://github.com/shivanshtiwari1234/OpenCV-hand-detector
cd hand_finger_counter
```

Run the main program:

``` bash
python main.py
```

- Show your hand to the webcam.

- Press ```q``` to quit.

## How it Works

- **Webcam Capture**: Captures frames in real-time.

- **Worker Process**: Processes frames in a separate process for smooth performance:

- Converts frame to HSV

- Detects skin using color range

- Finds contours and convex hull

- Counts fingers using convexity defects

- **Display**: Shows:

  1. Original frame with bounding boxes

  2. Skin mask

  3. Contour visualization

  4. Number of fingers detected

## Notes

- Only works for 0–5 fingers per hand

- Lighting affects skin detection accuracy

- Adjust HSV ranges for different skin tones if needed

## License

**MIT License** – free to use, modify, and distribute.